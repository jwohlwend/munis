from typing import Callable, List, Any
import os
import argparse
from tqdm import tqdm

import esm
import pandas as pd
import torch
from Bio import SeqIO

from munis.seqs import MIN_LEN, SEQUENCES
from munis.model import EnsembleMunisModel


def clean_mhc_name(mhc):
    """Clean HLA."""
    mhc = mhc.replace("*", "")
    if len(mhc.split(":")) > 1:
        mhc = ":".join(mhc.split(":")[:2])
    return mhc


class DataView(object):
    def __init__(self, data: Any, process: Callable, split: str):
        self.data = data
        self.process = process
        self.split = split

    def __getitem__(self, index: int) -> Any:
        return self.process(self.data[index], split=self.split)

    def __len__(self) -> int:
        return len(self.data)


class PredictionDataset:
    def __init__(
        self,
        data,
        batch_size: int = 32,
        num_workers: int = 0,
        balance: bool = False,
        pin_memory: bool = True,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.balance = balance
        self.pin_memory = pin_memory
        self._train: List = []
        self._val: List = []
        self._test: List = []
        self._data: List = []

        # Load data and apply filters
        raw = data.to_dict("records")

        # Encode sequences
        _, alphabet = esm.pretrained.esm2_t6_8M_UR50D()
        self.alphabet = alphabet
        self._vocab_size = len(alphabet.all_toks)
        self.pad_id = alphabet.padding_idx
        self.cls_id = alphabet.cls_idx
        self.eos_id = alphabet.eos_idx
        self.mask_id = alphabet.mask_idx

        # Load HLA groups
        self.sequences = SEQUENCES

        # Prepare splits
        self._train = []
        self._val = []
        self._test = raw
        print("Test length: ", len(self._test))

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
        pep_seq = sample["pep"]
        mhc_seq = self.sequences[sample["mhc"].replace("*", "")][:MIN_LEN]

        mhc_toks = self.alphabet.encode(mhc_seq)
        pep_toks = self.alphabet.encode(pep_seq)
        left = sample["left"]
        right = sample["right"]

        # Replace null by empty string
        left = "" if pd.isnull(left) else left
        right = "" if pd.isnull(right) else right

        # LM features
        max_len = 15
        pad_len = max_len - len(pep_seq)

        lm_seq = (
            [self.cls_id]
            + mhc_toks
            + [self.mask_id]
            + pep_toks
            + [self.eos_id]
            + [self.pad_id] * pad_len
        )
        prot_seq = self.alphabet.encode(left) + pep_toks + self.alphabet.encode(right)
        prot_type = [0] * len(left) + [1] * len(pep_seq) + [0] * len(right)
        pad_len = (max_len - len(pep_seq)) + (10 - len(left) - len(right))
        prot_seq = prot_seq + [self.pad_id] * pad_len
        prot_type = prot_type + [0] * pad_len

        return {
            "seq": torch.tensor(lm_seq, dtype=torch.long),
            "prot_seq": torch.tensor(prot_seq, dtype=torch.long),
            "prot_type": torch.tensor(prot_type, dtype=torch.long),
        }

    def test_dataloader(self):
        test_data = DataView(self.test_data, self.process, split="test")
        data_loader = torch.utils.data.DataLoader(
            test_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=False,
            shuffle=False,
        )
        return data_loader


def predict(data, model, device, batch_size):
    # Run inference
    dataset = PredictionDataset(data, batch_size=batch_size, num_workers=4)
    data_loader = dataset.test_dataloader()

    # Run model predictions
    out = []
    with torch.no_grad(), torch.cuda.amp.autocast(enabled=True):
        for batch in tqdm(data_loader):
            seq = batch["seq"].to(device)
            prot = batch["prot_seq"].to(device)
            prot_type = batch["prot_type"].to(device)
            logits = model(seq, prot, prot_type, return_feats=False)
            logits = torch.sigmoid(logits).squeeze(-1).cpu()
            logits = logits.mean(dim=0)
            out.append(logits)

    # Save predictions
    out = torch.cat(out, dim=0).numpy().tolist()
    data["score"] = out
    return data


def peptides_from_fasta(fasta_path, min_len, max_len):
    with open(fasta_path, "r") as f:
        seqs = list(SeqIO.parse(f, "fasta"))

    peptides = []
    for entry in seqs:
        seq = str(entry.seq)
        for i in range(len(seq) - min_len + 1):
            for j in range(min_len, max_len + 1):
                if i + j <= len(seq):
                    pep = seq[i : i + j]
                    left = seq[max(0, i - 5) : i]
                    right = seq[i + j : min(len(seq), i + j + 5)]
                    peptides.append(
                        {
                            "prot": str(entry.id),
                            "pep": pep,
                            "left": left,
                            "right": right,
                        }
                    )
    return peptides


def main(args):
    # Check input
    if args.fasta is None and args.peptides is None:
        raise ValueError("Either --fasta or --peptides must be provided.")
    elif args.fasta is not None and args.peptides is not None:
        raise ValueError("Only one of --fasta or --peptides can be provided.")

    if args.fasta is not None:
        # Check MHC's were given as input
        if args.mhc is None:
            raise ValueError("Please provide a list of MHC alleles.")

        # Load peptides from fasta
        peptides = peptides_from_fasta(args.fasta, args.min_len, args.max_len)

        # Check MHC's
        mhcs = [clean_mhc_name(m) for m in args.mhc.split(",")]
        for mhc in mhcs:
            if mhc not in SEQUENCES:
                raise ValueError(f"Invalid MHC allele: {mhc}")

        # Prepare data
        data = []
        for mhc in mhcs:
            for p in peptides:
                new_row = p.copy()
                new_row["mhc"] = mhc
                data.append(new_row)
        data = pd.DataFrame(data)

    else:
        if args.mhc is not None:
            raise ValueError(
                "MHC alleles are expecte in the input csv when using the peptide format."
            )

        data = pd.read_csv(args.peptides)
        if "pep" not in data.columns:
            raise ValueError("Peptide column not found.")
        if "left" not in data.columns:
            raise ValueError("Left-flank column not found.")
        if "right" not in data.columns:
            raise ValueError("Right-flank column not found.")

    # Replace flanks if to be ignored
    if not args.use_flanks:
        data["left"] = "GGGGG"
        data["right"] = "GGGGG"

    # Set torch hub cache path
    os.makedirs(args.cache, exist_ok=True)
    torch.hub.set_dir(args.cache)

    # Load model ensemble
    if args.checkpoint is not None:
        checkpoints = [args.checkpoint]
    elif args.use_flanks:
        checkpoints = [
            "models/flanks/model1.ckpt",
            "models/flanks/model2.ckpt",
            "models/flanks/model3.ckpt",
            "models/flanks/model4.ckpt",
            "models/flanks/model5.ckpt",
        ]
    else:
        checkpoints = [
            "models/no-flanks/model1.ckpt",
            "models/no-flanks/model2.ckpt",
            "models/no-flanks/model3.ckpt",
            "models/no-flanks/model4.ckpt",
            "models/no-flanks/model5.ckpt",
        ]

    model = EnsembleMunisModel(checkpoints, torch.device(args.device))
    model.eval()
    model.to(args.device)

    # Prepare output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Run predictions
    preds = predict(data, model, args.device, args.batch_size)

    # Save
    if args.fasta is not None:
        name = os.path.basename(args.fasta).split(".")[0]
    elif args.peptides is not None:
        name = os.path.basename(args.peptides).split(".")[0]

    name = f"{name}_munis_predictions.csv"
    out_path = os.path.join(args.outdir, name)
    preds.to_csv(out_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory for predictions.",
        default="./predictions",
    )
    parser.add_argument(
        "--cache",
        help="Cache directory for ESM pretrained models.",
        default="./cache",
    )
    parser.add_argument(
        "--fasta",
        help="Path to a fasta file.",
    )
    parser.add_argument(
        "--peptides",
        help="Path to a CSV file.",
    )
    parser.add_argument(
        "--mhc",
        help="Comma separated list of MHC's.",
    )
    parser.add_argument(
        "--min_len",
        type=int,
        default=8,
        help="Minimum peptide length.",
    )
    parser.add_argument(
        "--max_len",
        type=int,
        default=15,
        help="Maximum peptide length.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size when running prediction.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="The device to use for prediction.",
    )
    parser.add_argument(
        "--use_flanks",
        action="store_true",
        help="Use flanking regions.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to a model checkpoint, uses default 5 model ensemble if not passed.",
    )
    args = parser.parse_args()
    main(args)
