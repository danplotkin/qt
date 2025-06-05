import os
import sys
sys.path.append(os.getcwd())
from utils.tokenizer import get_tokenizer
from utils.configs import load_configs, TrainingConfigs, TransformerConfigs
from utils.training import Trainer
from utils.losses import SequenceLoss
from utils.metrics import MaskedAccuracy
from utils.transformer.model import QT
from utils.torch_datasets import FineTuneCorpusDataset

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

PRETRAINED_WEIGHTS_PATH='experiments/qt-pretrain/checkpoints/qt-pretrain_best.pt'

def configure_trainer() -> tuple[Trainer, DataLoader]:
    logging.info("Loading tokenizer and configurations...")
    tokenizer = get_tokenizer()
    configs = load_configs()
    training_configs: TrainingConfigs = configs['training']
    transformer_configs: TransformerConfigs = configs['transformer']
    model = QT(config=transformer_configs, tokenizer=tokenizer, device='cpu')
    pretrained_weights = torch.load(PRETRAINED_WEIGHTS_PATH)['model_state_dict']
    model.load_state_dict(pretrained_weights)
    model.resize_token_embeddings(len(tokenizer))
    logging.info("Loaded pretrained weights from %s", PRETRAINED_WEIGHTS_PATH)
    train_ds = FineTuneCorpusDataset(split='train', block_size=transformer_configs.max_seq_length)
    val_ds = FineTuneCorpusDataset(split='validation', block_size=transformer_configs.max_seq_length)
    test_ds = FineTuneCorpusDataset(split='test', block_size=transformer_configs.max_seq_length)
    logging.info("Prepared NoRobotsDataset for train, validation, and test splits.")
    train_loader = DataLoader(train_ds, batch_size=training_configs.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=training_configs.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=training_configs.batch_size, shuffle=False)
    logging.info("Constructed DataLoaders for training and validation.")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_configs,
        criterion=SequenceLoss(ignore_index=tokenizer.pad_token_id),
        metric=MaskedAccuracy(padding_token_id=tokenizer.pad_token_id)
    )
    return trainer, test_loader


def main():
    trainer, test_loader = configure_trainer()
    logging.info("Starting fine-tuning process...")
    trainer.train()
    logging.info("Training complete. Beginning testing...")
    trainer.test(test_loader=test_loader)
    logging.info("Testing complete.")


if __name__ == '__main__':
    main()