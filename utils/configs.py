import yaml
from dataclasses import dataclass
from torch.optim import Optimizer, AdamW
from typing import Optional


@dataclass
class TrainingConfigs:
    model_name: str = "qt"
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    batch_size: int = 32
    epochs: int = 3
    output_dir: str = "./experiments"
    s3_bucket: Optional[str] = None
    s3_prefix: str = ""
    optimizer: type[Optimizer] = AdamW
    # Early stopping configurations
    early_stopping: bool = False
    early_stopping_patience: int = 3
    early_stopping_min_delta: float = 0.0
    early_stopping_mode: str = 'min'  # 'min' for loss, 'max' for accuracy
    restore_best_model: bool = True  # restore model weights from best epoch upon early stopping
    logging_dir: str = "./logs"

    def __post_init__(self):
        if isinstance(self.learning_rate, str):
            self.learning_rate = eval(self.learning_rate)
        if isinstance(self.weight_decay, str):
            self.weight_decay = eval(self.weight_decay)


@dataclass
class TransformerConfigs:
    tgt_vocab_size: int = 50257
    d_model: int = 2048
    num_heads: int = 16
    num_layers: int = 14
    d_ff: int = 8192
    max_seq_length: int = 2048
    dropout: float = 0.1


def load_configs(path: str = "config.yaml") -> dict[str, object]:
    """
    Load configuration values from a YAML file and return TrainingConfigs and TransformerConfigs instances.
    Expects the YAML structure:
    training:
      <TrainingConfigs fields>
    transformer:
      <TransformerConfigs fields>
    """
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    train_cfg = cfg.get("training", {})
    trans_cfg = cfg.get("transformer", {})
    training = TrainingConfigs(**train_cfg)
    transformer = TransformerConfigs(**trans_cfg)
    return {
        "training": training,
        "transformer": transformer
    }
