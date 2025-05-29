import os
import sys
sys.path.append(os.getcwd())
from utils.configs import load_configs, TrainingConfigs, TransformerConfigs
from utils.training import Trainer
from utils.transformer.model import QT
from utils.losses import LastTokenLoss
from utils.metrics import MaskedAccuracy
from utils.tokenizer import get_tokenizer
from utils.torch_datasets import MiniPileDataset

from torch.utils.data import DataLoader
import torch


def configure_trainer() -> Trainer:
    tokenizer = get_tokenizer()
    configs = load_configs('unittests/test_config.yaml')
    training_configs: TrainingConfigs = configs['training']
    transformer_configs: TransformerConfigs = configs['transformer']
    model = QT(config=transformer_configs, tokenizer=tokenizer, device='cpu')
    train_ds = MiniPileDataset(path='data/flattened_corpa/minipile/train.pt', block_size=transformer_configs.max_seq_length)
    val_ds = MiniPileDataset(path='data/flattened_corpa/minipile/validation.pt', block_size=transformer_configs.max_seq_length)
    train_loader = DataLoader(train_ds, batch_size=training_configs.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=training_configs.batch_size, shuffle=False)
    # model.initialize_output_bias(train_ds.tokens)

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_configs,
        criterion=LastTokenLoss(ignore_index=tokenizer.pad_token_id),
        metric=MaskedAccuracy(padding_token_id=tokenizer.pad_token_id)
    )
    return trainer


def main():
    trainer = configure_trainer()
    trainer.train()


if __name__ == '__main__':
    main()