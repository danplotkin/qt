import os
import sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils.configs import TrainingConfigs, TransformerConfigs
from utils.transformer.models import QT
from utils.metrics import MaskedAccuracy
from utils.training import Trainer
from utils.tokenizer import get_tokenizer

def test_trainer_runs_without_error():
    # Create dummy data
    input_ids = torch.randint(0, 100, (8, 32))  # batch_size=8, seq_len=32
    target_ids = input_ids.clone()
    dataset = TensorDataset(input_ids, target_ids)
    loader = DataLoader(dataset, batch_size=2)

    # Configs
    config = TrainingConfigs(num_epochs=1)
    transformer_config = TransformerConfigs(tgt_vocab_size=100, d_model=32, num_heads=2, num_layers=2, d_ff=64, max_seq_length=32, dropout=0.1)

    # Tokenizer 
    tokenizer = get_tokenizer()

    # Model
    model = QT(
        tgt_vocab_size=transformer_config.tgt_vocab_size,
        d_model=transformer_config.d_model,
        num_heads=transformer_config.num_heads,
        num_layers=transformer_config.num_layers,
        d_ff=transformer_config.d_ff,
        max_seq_length=transformer_config.max_seq_length,
        dropout=transformer_config.dropout,
        tokenizer=tokenizer,
        device=torch.device("cpu")
    )

    # Loss and metric
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    metric = MaskedAccuracy(padding_token_id=tokenizer.pad_token_id)

    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=loader,
        val_loader=loader,
        config=config,
        criterion=criterion,
        metric=metric,
        device="cpu"
    )

    # Run training
    trainer.train()

    # Assert model trained and history recorded
    assert len(trainer.history['train_loss']) == config.num_epochs
    assert len(trainer.history['train_acc']) == config.num_epochs


if __name__ == '__main__':
    test_trainer_runs_without_error()