from typing import Dict, Optional
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import esm
import pytorch_lightning as pl

from munis.metrics import GroupAveragePrecision
from munis.seqs import SEQUENCES


class MunisModel(pl.LightningModule):
    def __init__(
        self,
        hidden_dim: int = 128,
        node_dim: int = 32,
        lr: float = 1e-3,
        lr_lm: float = 1e-4,
        pad_id: int = 0,
        weight_decay: float = 0,
        use_flanks: bool = True,
        esm_model_name: str = "esm2_t6_8M_UR50D",
        reset_lm: bool = False,
        pretrained: Optional[str] = None,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lr = lr
        self.lr_lm = lr_lm
        self.weight_decay = weight_decay
        self.pad_id = pad_id
        self.hidden_dim = hidden_dim
        self.use_flanks = use_flanks

        # Language model module
        lm, _ = esm.pretrained.load_model_and_alphabet(esm_model_name)
        self.lm = lm
        if reset_lm:
            self.lm._init_submodules()
        self.lm.contact_head = nn.Identity()
        self.lm.lm_head = nn.Identity()

        # Antigen processing module
        self.feats_dim = lm.embed_dim
        if self.use_flanks:
            self.feats_dim = lm.embed_dim + hidden_dim
            self.prot_type = nn.Embedding(2, node_dim)
            self.lstm = nn.LSTM(
                input_size=lm.embed_dim + node_dim,
                hidden_size=hidden_dim // 2,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )

        # Classification layers
        in_dim = lm.embed_dim + hidden_dim if self.use_flanks else lm.embed_dim
        self.fc_el = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        # Metrics
        seq_map = SEQUENCES
        group_map = OrderedDict((i, k) for i, k in enumerate(seq_map.keys()))

        # Load pretrained model
        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained)["state_dict"], strict=False)

        self.el_avgp = GroupAveragePrecision(groups=group_map)

    @classmethod
    def add_extra_args(cls, dataset) -> Dict:
        extra_args = {"pad_id": dataset.pad_id}
        return extra_args

    @classmethod
    def selection_metric(cls):
        return "val/el_avgp", "max"

    def lm_encoding(self, seq):
        idx = self.lm.num_layers
        seq_emb = self.lm(seq, repr_layers=[idx])
        seq_emb = seq_emb["representations"][idx]
        return seq_emb[:, 0]

    def flank_encoding(self, prot, prot_type):
        prot_feats = self.lm.embed_tokens(prot)
        prot_type = self.prot_type(prot_type)
        prot_feats = torch.cat([prot_feats, prot_type], dim=-1)

        pad_mask = prot != self.pad_id
        prot_lens = pad_mask.sum(dim=-1)
        prot_feats = torch.nn.utils.rnn.pack_padded_sequence(
            prot_feats, prot_lens.cpu(), batch_first=True, enforce_sorted=False
        )
        feats, _ = self.lstm(prot_feats)
        feats = torch.nn.utils.rnn.pad_packed_sequence(feats, batch_first=True)[0]

        # Mean pooling
        feats = feats.sum(dim=1) / prot_lens.unsqueeze(-1)
        return feats

    def forward(self, seq, prot, prot_type, return_feats=False):
        # Compute features
        lm_feats = self.lm_encoding(seq)

        # Compute flanks
        if self.use_flanks:
            flank_feats = self.flank_encoding(prot, prot_type)
            full_feats = torch.cat([lm_feats, flank_feats], dim=-1)
        else:
            full_feats = lm_feats

        # Compue logits
        logits = self.fc_el(full_feats)

        # Combine and return
        if return_feats:
            return logits, full_feats
        else:
            return logits

    def compute_loss(self, logits, label):
        logits = logits.view(-1)
        label = label.view(-1)
        loss = F.binary_cross_entropy_with_logits(logits, label)
        return loss

    def training_step(self, batch, batch_idx):
        seq = batch["seq"]
        prot = batch.get("prot_seq", None)
        prot_type = batch.get("prot_type", None)
        label = batch["label"]

        # Forward pass
        logits = self(seq, prot, prot_type)

        # Compute loss
        loss = self.compute_loss(logits, label)
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/grad_norm", self.gradient_norm, prog_bar=False)
        self.log("train/param_norm", self.parameter_norm, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        seq = batch["seq"]
        prot = batch.get("prot_seq", None)
        prot_type = batch.get("prot_type", None)
        label = batch["label"]
        groups = batch["groups"]

        # Forward pass
        logits = self(seq, prot, prot_type)

        # Compute loss
        loss = self.compute_loss(logits, label)
        self.log("val/loss", loss, prog_bar=True)

        # Compute EL metrics
        el_pred = torch.sigmoid(logits[:, 0])
        self.el_avgp(el_pred, label.long(), groups)

    def on_validation_epoch_end(self):
        # Compute EL AUC
        try:
            el_avgp = self.el_avgp.compute()
            avg_el_avgp = np.mean(list(el_avgp.values()))
        except Exception:
            el_avgp = {}
            avg_el_avgp = -1

        # Log metrics
        self.log("val/el_avgp", avg_el_avgp, prog_bar=True)
        self.el_avgp.reset()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {
                    "params": [
                        p
                        for k, p in self.named_parameters()
                        if p.requires_grad and ("lm" in k)
                    ],
                    "lr": self.lr_lm,
                },
                {
                    "params": [
                        p
                        for k, p in self.named_parameters()
                        if p.requires_grad and ("lm" not in k)
                    ],
                    "lr": self.lr,
                },
            ],
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer

    @property
    def gradient_norm(self) -> float:
        """Compute the average gradient norm.
        Returns
        -------
        float
            The current average gradient norm
        """
        # Only compute over parameters that are being trained
        parameters = filter(
            lambda p: p.requires_grad and p.grad is not None, self.parameters()
        )
        norm = (
            torch.tensor([param.grad.norm(p=2) ** 2 for param in parameters])
            .sum()
            .sqrt()
        )
        return norm

    @property
    def parameter_norm(self) -> float:
        """Compute the average parameter norm.

        Returns
        -------
        float
            The current average parameter norm

        """
        # Only compute over parameters that are being trained
        parameters = filter(lambda p: p.requires_grad, self.parameters())
        norm = torch.tensor([param.norm(p=2) ** 2 for param in parameters]).sum().sqrt()
        return norm


class EnsembleMunisModel(nn.Module):
    def __init__(self, checkpoints):
        super().__init__()
        models = []
        for checkpoint in checkpoints:
            model = MunisModel.load_from_checkpoint(checkpoint)
            models.append(model)
        self.models = nn.ModuleList(models)
        self.feats_dim = models[0].feats_dim

    def forward(self, seq, prot, prot_type, return_feats=False):
        # Prepare outputs
        all_logits = []
        all_feats = []

        # Run models
        for model in self.models:
            logits, feats = model(seq, prot, prot_type, return_feats=True)
            all_logits.append(logits)
            all_feats.append(feats)

        # Concat and return
        all_logits = torch.stack(all_logits, dim=0)
        if return_feats:
            all_feats = torch.stack(all_feats, dim=0)
            return all_logits, all_feats
        else:
            return all_logits
