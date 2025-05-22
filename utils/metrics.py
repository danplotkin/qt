import torch
from abc import abstractmethod, ABC


class BaseMetric(ABC):
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the metric"""
        pass

    @abstractmethod
    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> float:
        pass


class MaskedAccuracy(BaseMetric):
    def __init__(self, padding_token_id: int):
        self.padding_token_id = padding_token_id
        self._name = "masked accuracy"

    @property
    def name(self) -> str:
        return self._name

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> float:
        mask = target.ne(self.padding_token_id)
        output = output.argmax(-1).masked_select(mask)
        target = target.masked_select(mask)
        return (output == target).float().mean()