from typing import Any, List, Optional, Callable, Dict

import torch
from torch import Tensor
from torchmetrics.functional.classification.average_precision import (
    binary_average_precision,
)
from torchmetrics.metric import Metric
from torchmetrics.utilities import rank_zero_warn
from torchmetrics.utilities.data import dim_zero_cat
from torchmetrics.utilities.enums import DataType
from sklearn.metrics import roc_auc_score


class GroupAUROC(Metric):
    preds: List[Tensor]
    target: List[Tensor]
    group: List[Tensor]
    mask: List[Tensor]
    is_differentiable = False

    def __init__(
        self,
        groups: Optional[Dict[int, str]] = None,
        pos_label: Optional[int] = None,
        max_fpr: Optional[float] = None,
        compute_on_step: bool = True,
        dist_sync_on_step: bool = False,
        process_group: Optional[Any] = None,
        dist_sync_fn: Callable = None,
        sync_on_compute: bool = False,
    ) -> None:
        super().__init__(
            compute_on_step=compute_on_step,
            dist_sync_on_step=dist_sync_on_step,
            process_group=process_group,
            dist_sync_fn=dist_sync_fn,
            sync_on_compute=sync_on_compute,
        )

        self.pos_label = pos_label
        self.max_fpr = max_fpr
        self.group_names = groups

        if self.max_fpr is not None:
            if not isinstance(max_fpr, float) or not 0 < max_fpr <= 1:
                raise ValueError(
                    f"`max_fpr` should be a float in range (0, 1], got: {max_fpr}"
                )

        self.mode: DataType = None  # type: ignore
        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("group", default=[], dist_reduce_fx="cat")
        self.add_state("mask", default=[], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `AUROC` will save all targets and predictions in buffer."
            " For large datasets this may lead to large memory footprint."
        )

    def update(  # type: ignore
        self,
        preds: Tensor,
        target: Tensor,
        group: Tensor,
        mask: Optional[Tensor] = None,
    ) -> None:
        """
        Update state with predictions and targets.

        Args:
            preds: Predictions from model (probabilities, or labels)
            target: Ground truth labels
        """
        if mask is None:
            mask = torch.ones_like(target).bool()

        self.preds.append(preds)
        self.target.append(target)
        self.group.append(group)
        self.mask.append(mask)

    def compute(self) -> Dict[str, float]:
        """
        Computes AUROC based on inputs passed in to ``update`` previously.
        """
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        group = dim_zero_cat(self.group)
        mask = dim_zero_cat(self.mask).bool()

        preds = preds[mask]
        target = target[mask]
        group = group[mask]

        unique = set(group.cpu().numpy().tolist())
        unique = sorted(list(unique))

        out = {}
        for idx in unique:
            preds_select = preds[group == idx]
            target_select = target[group == idx]
            if (
                (len(preds_select) < 2)
                or (target_select.sum() == 0)
                or (target_select.sum() == len(target_select))
            ):
                continue
            auc = roc_auc_score(
                target_select.cpu().numpy(),
                preds_select.cpu().numpy(),
                max_fpr=self.max_fpr,
            )
            if self.group_names is not None:
                out[self.group_names[idx]] = auc
            else:
                out[idx] = auc
        return out


class GroupAveragePrecision(Metric):
    is_differentiable = False
    preds: List[Tensor]
    target: List[Tensor]
    group: List[Tensor]
    mask: List[Tensor]

    def __init__(
        self,
        groups: Dict[int, str],
    ) -> None:
        super().__init__()
        self.group_names = groups

        self.add_state("preds", default=[], dist_reduce_fx="cat")
        self.add_state("target", default=[], dist_reduce_fx="cat")
        self.add_state("group", default=[], dist_reduce_fx="cat")
        self.add_state("mask", default=[], dist_reduce_fx="cat")

        rank_zero_warn(
            "Metric `AveragePrecision` will save all targets and predictions in buffer."
            " For large datasets this may lead to large memory footprint."
        )

    def update(  # type: ignore
        self,
        preds: Tensor,
        target: Tensor,
        group: Tensor,
        mask: Optional[Tensor] = None,
    ) -> None:
        """Update state with predictions and targets.

        Args:
            preds: Predictions from model
            target: Ground truth values
        """
        if mask is None:
            mask = torch.ones_like(target).bool()

        self.preds.append(preds)
        self.target.append(target)
        self.group.append(group)
        self.mask.append(mask)

    def compute(self) -> Dict[str, float]:
        """Compute the average precision score.

        Returns:
            tensor with average precision. If multiclass will return list
            of such tensors, one for each class
        """
        preds = dim_zero_cat(self.preds)
        target = dim_zero_cat(self.target)
        group = dim_zero_cat(self.group)
        mask = dim_zero_cat(self.mask).bool()

        preds = preds[mask]
        target = target[mask]
        group = group[mask]

        unique = set(group.cpu().numpy().tolist())
        out = {}
        for idx in unique:
            preds_select = preds[group == idx]
            target_select = target[group == idx]
            if (
                (len(preds_select) < 2)
                or (target_select.sum() == 0)
                or target_select.sum() == len(target_select)
            ):
                continue
            avgp = binary_average_precision(preds_select, target_select)
            out[self.group_names[idx]] = avgp.cpu().item()  # type: ignore
        return out
