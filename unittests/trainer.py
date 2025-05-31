import os
import sys
sys.path.append(os.getcwd())
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from utils.torch_datasets import ExampleCorpusDataset
from utils.configs import TrainingConfigs, TransformerConfigs
from utils.transformer.model import QT
from utils.metrics import MaskedAccuracy
from utils.training import Trainer
from utils.tokenizer import get_tokenizer
from utils.losses import SequenceLoss, LastTokenLoss
import time
import shutil
import copy

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def test_trainer_runs_without_error():
    # Create dummy data
    # Tokenizer 
    tokenizer = get_tokenizer()
    train_dataset = ExampleCorpusDataset(split='train', block_size=20, stride=1)
    val_dataset = ExampleCorpusDataset(split='val', block_size=20, stride=1)
    train_loader = DataLoader(train_dataset, batch_size=2)
    val_loader = DataLoader(val_dataset, batch_size=2)

    # Configs
    config = TrainingConfigs(epochs=15, model_name='unittest')
    transformer_config = TransformerConfigs(tgt_vocab_size=50_000, d_model=32, num_heads=2, num_layers=2, d_ff=64, max_seq_length=32, dropout=0.1)

    shutil.rmtree(config.output_dir, ignore_errors=True)

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
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        criterion=criterion,
        metric=metric,
        device="cpu"
    )

    trainer.train()

    print('\n\nSimilating stop....', '\n\n')
    time.sleep(2)

    config.epochs = 30
    model_reloaded = QT(
        config=transformer_config,
        tokenizer=tokenizer,
        device=torch.device("cpu")
    )
    trainer = Trainer(
        model=model_reloaded,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        criterion=criterion,
        metric=metric,
        device="cpu"
    )

    trainer.train()


if __name__ == '__main__':
    test_trainer_runs_without_error()