from typing import Any, Callable, List
import csv
from collections import OrderedDict
from sklearn.model_selection import train_test_split

import torch
import esm
import pytorch_lightning as pl

from munis.seqs import MIN_LEN, SEQUENCES


class DataView(object):
    def __init__(self, data: Any, process: Callable, split: str):
        self.data = data
        self.process = process
        self.split = split

    def __getitem__(self, index: int) -> Any:
        return self.process(self.data[index], split=self.split)

    def __len__(self) -> int:
        return len(self.data)


class MunisDataset(pl.LightningDataModule):
    def __init__(
        self,
        path: str,
        batch_size: int = 32,
        num_workers: int = 0,
        pin_memory: bool = True,
        min_epitope_len: int = 8,
        max_epitope_len: int = 15,
        esm_model_name: str = "esm2_t6_8M_UR50D",
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self._train: List = []
        self._val: List = []
        self._test: List = []
        self._data: List = []
        self.min_epitope_len = min_epitope_len
        self.max_epitope_len = max_epitope_len

        # Load EL data
        with open(path, "r") as f:
            raw = list(csv.DictReader(f))

        # Filter by length
        raw = [
            sample
            for sample in raw
            if min_epitope_len <= len(sample["pep"]) <= max_epitope_len
        ]

        # Split into train/val
        train, val = train_test_split(raw, test_size=0.05, random_state=42)

        # Load LM vocab
        _, alphabet = esm.pretrained.load_model_and_alphabet(esm_model_name)
        self.alphabet = alphabet
        self._vocab_size = len(alphabet.all_toks)
        self.pad_id = alphabet.padding_idx
        self.cls_id = alphabet.cls_idx
        self.eos_id = alphabet.eos_idx
        self.mask_id = alphabet.mask_idx

        # Load HLA groups
        self.sequences = SEQUENCES
        self.group_map = OrderedDict(
            (i, k) for i, k in enumerate(self.sequences.keys())
        )
        self.group_map_inv = OrderedDict((k, i) for i, k in self.group_map.items())

        # Prepare splits
        self._data = raw
        self._train = train
        self._val = val
        self._test = []

        print("Train length: ", len(self._train))
        print("Val length: ", len(self._val))

    @property
    def train_data(self) -> List:
        """The validation data."""
        return self._train

    @property
    def val_data(self) -> List:
        """The validation data."""
        return self._val

    @property
    def test_data(self) -> List:
        """The testing data."""
        return self._test

    def prepare_data(self):
        """Only happens on single GPU, ATTENTION: do no assign states."""
        pass

    def setup(self, stage: str = None):
        """Prepares the data for training, validation, and testing."""
        pass

    def process(self, sample: Any, split: str) -> Any:
        """Processes a single data sample."""
        # Load sequence features
        pep_seq = sample["pep"]
        mhc_seq = self.sequences[sample["mhc"].replace("*", "")][:MIN_LEN]
        groups = self.group_map_inv[sample["mhc"].replace("*", "")]

        # Prepare labels
        label = float(sample["label"])

        mhc_toks = self.alphabet.encode(mhc_seq)
        pep_toks = self.alphabet.encode(pep_seq)

        # LM features
        pad_len = self.max_epitope_len - len(pep_seq)
        lm_seq = (
            [self.cls_id]
            + mhc_toks
            + [self.mask_id]
            + pep_toks
            + [self.eos_id]
            + [self.pad_id] * pad_len
        )

        # Flank features
        left = sample["left"]
        right = sample["right"]

        prot_seq = self.alphabet.encode(left) + pep_toks + self.alphabet.encode(right)
        prot_type = [0] * len(left) + [1] * len(pep_seq) + [0] * len(right)
        pad_len = (self.max_epitope_len - len(pep_seq)) + (10 - len(left) - len(right))
        prot_seq = prot_seq + [self.pad_id] * pad_len
        prot_type = prot_type + [0] * pad_len

        return {
            "seq": torch.tensor(lm_seq, dtype=torch.long),
            "prot_seq": torch.tensor(prot_seq, dtype=torch.long),
            "prot_type": torch.tensor(prot_type, dtype=torch.long),
            "label": torch.tensor(label, dtype=torch.float),
            "groups": torch.tensor(groups, dtype=torch.long),
        }

    def train_dataloader(self):
        train_data = DataView(self.train_data, self.process, split="train")
        train_data_loader = torch.utils.data.DataLoader(
            train_data,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            batch_size=self.batch_size,
            shuffle=True,
        )

        return train_data_loader

    def val_dataloader(self):
        val_data = DataView(self.val_data, self.process, split="val")
        data_loader = torch.utils.data.DataLoader(
            val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            shuffle=False,
        )
        return data_loader
