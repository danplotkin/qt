import torch


class MaskedAccuracy:
    def __init__(self, padding_token_id: int):
        self.padding_token_id = padding_token_id

    def __call__(self, output: torch.Tensor, target: torch.Tensor) -> float:
        mask = target.ne(self.padding_token_id)
        output = output.argmax(-1).masked_select(mask)
        target = target.masked_select(mask)
        return (output == target).float().mean()