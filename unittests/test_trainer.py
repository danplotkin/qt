import os
import sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils.configs import TrainingConfigs, TransformerConfigs
from utils.transformer.model import QT
from utils.metrics import MaskedAccuracy
from utils.training import Trainer
from utils.tokenizer import get_tokenizer
from utils.losses import SequenceLoss, LastTokenLoss


def test_trainer_runs_without_error():
    # Create dummy data
    input_ids = torch.randint(0, 100, (8, 32))  # batch_size=8, seq_len=32
    target_ids = input_ids.clone()
    dataset = TensorDataset(input_ids, target_ids)
    loader = DataLoader(dataset, batch_size=2)

    # Configs
    config = TrainingConfigs(epochs=1)
    transformer_config = TransformerConfigs(tgt_vocab_size=100, d_model=32, num_heads=2, num_layers=2, d_ff=64, max_seq_length=32, dropout=0.1)

    # Tokenizer 
    tokenizer = get_tokenizer()

    # Model
    model = QT(
        config = transformer_config,
        tokenizer=tokenizer,
        device=torch.device("cpu")
    )

    # Metric
    metric = MaskedAccuracy(padding_token_id=tokenizer.pad_token_id)
    criterion = LastTokenLoss(ignore_index=tokenizer.pad_token_id)

    # Test both loss functions
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=config,
        criterion=criterion,
        metric=metric,
        device="cpu"
    )

    trainer.train()

    assert len(trainer.history['train_loss']) == config.epochs
    assert len(trainer.history['train_acc']) == config.epochs


if __name__ == '__main__':
    test_trainer_runs_without_error()