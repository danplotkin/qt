from dataclasses import dataclass
from torch.optim import Optimizer, AdamW


@dataclass
class TrainingConfigs:
    model_name: str = "qt"
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    batch_size: int = 32
    num_epochs: int = 3
    output_dir: str = "./checkpoints"
    optimizer: type[Optimizer] = AdamW
    # Early stopping configurations
    early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    early_stopping_mode: str = 'min'  # 'min' for loss, 'max' for accuracy
    restore_best_model: bool = True  # restore model weights from best epoch upon early stopping


@dataclass
class TransformerConfigs:
    tgt_vocab_size: int = 50257
    d_model: int = 768
    num_heads: int = 12
    num_layers: int = 12
    d_ff: int = 3072
    max_seq_length: int = 512
    dropout: float = 0.1
