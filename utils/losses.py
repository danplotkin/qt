import torch
import torch.nn as nn
from abc import ABC, abstractmethod


class BaseLoss(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the loss"""
        pass

    @abstractmethod
    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss.

        Args:
            logits (torch.Tensor): Shape (B, T, V) — raw model outputs.
            targets (torch.Tensor): Shape (B, T) — target token IDs.

        Returns:
            torch.Tensor: scalar loss
        """
        pass


class SequenceLoss(BaseLoss):
    def __init__(self, ignore_index: int = -100):
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self._name = "XE sequence loss"

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        B, T, V = logits.shape
        loss = self.loss_fn(logits.view(B * T, V), targets.view(B * T))
        return loss


class LastTokenLoss(BaseLoss):
    def __init__(self, ignore_index: int = -100):
        self.ignore_index = ignore_index
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self._name = "XE last token loss"

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # logits: (B, T, V) → take last token
        # targets: (B, T) → take last token
        loss = self.loss_fn(logits[:, -1, :], targets[:, -1])
        return loss