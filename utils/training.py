import os
import shutil
import glob
import re
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import logging
from time import time
import boto3
from logging.handlers import RotatingFileHandler
import yaml
from dataclasses import asdict
import json

from utils.configs import TrainingConfigs, TransformerConfigs
from utils.metrics import BaseMetric
from utils.losses import BaseLoss


logger = logging.getLogger(__name__)
# Set module logger level to DEBUG
logger.setLevel(logging.DEBUG)


class EarlyStopping:
    """
    Stops training if a monitored metric stops improving.
    """
    def __init__(self, patience: int = 3, min_delta: float = 0.0, mode: str = 'min') -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_state: Optional[dict] = None
        if mode == 'min':
            self.monitor_op = lambda current, best: current < best - self.min_delta
        else:
            self.monitor_op = lambda current, best: current > best + self.min_delta

    def __call__(self, current_score: float, model: nn.Module) -> None:
        if self.best_score is None:
            self.best_score = current_score
            self.best_state = copy.deepcopy(model.state_dict())
        elif self.monitor_op(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class Trainer:
    """
    Trainer class for training and evaluating a PyTorch model using provided configurations.

    Attributes:
        model (nn.Module): The model to train and evaluate.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (Optional[DataLoader]): DataLoader for validation data.
        config (TrainingConfigs): Training configuration parameters.
        criterion (nn.Module): Loss function.
        metric (callable): Function to compute accuracy or another evaluation metric.
        device (str): Device to run training on ('cuda' or 'cpu').
    """
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        config: TrainingConfigs,
        criterion: BaseLoss,
        metric: BaseMetric,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.transformer_config = self.model.config
        self.device = device
        self.optimizer = config.optimizer(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        self.criterion = criterion
        self.metric = metric
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': []
        }
        # Initialize S3 client if bucket is specified
        if self.config.s3_bucket:
            self.s3_client = boto3.client('s3')
        else:
            self.s3_client = None
        # Setup unified experiment directory structure
        self.experiment_dir = os.path.join(self.config.output_dir, self.config.model_name)
        self.checkpoint_dir = os.path.join(self.experiment_dir, "checkpoints")
        self.log_dir = os.path.join(self.experiment_dir, "logs")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

        # Save a plain-text summary of training config and transformer config
        config_txt_path = os.path.join(self.experiment_dir, "config.txt")
        with open(config_txt_path, "w") as f:
            f.write("TrainingConfigs:\n")
            for key, value in asdict(self.config).items():
                f.write(f"{key}: {value}\n")
            f.write("\nTransformerConfigs:\n")
            for key, value in asdict(self.transformer_config).items():
                f.write(f"{key}: {value}\n")
        if self.s3_client:
            txt_s3_key = f"{self.config.s3_prefix}{self.config.model_name}/config.txt"
            self.s3_client.upload_file(config_txt_path, self.config.s3_bucket, txt_s3_key)

        # Download existing artifacts from S3 to local dirs
        if self.s3_client:
            try:
                bucket = self.config.s3_bucket
                prefix = self.config.s3_prefix or ""
                # Paginate through S3 objects under prefix
                paginator = self.s3_client.get_paginator('list_objects_v2')
                pages = paginator.paginate(Bucket=bucket, Prefix=prefix)
                for page in pages:
                    for obj in page.get('Contents', []):
                        key = obj['Key']
                        rel = key[len(prefix):] if key.startswith(prefix) else key
                        if rel.startswith(f"{self.config.model_name}/checkpoints") and rel.endswith('.pt'):
                            dest = os.path.join(self.checkpoint_dir, os.path.basename(rel))
                        elif rel.startswith(f"{self.config.model_name}/logs") and rel.endswith('.log'):
                            dest = os.path.join(self.log_dir, os.path.basename(rel))
                        elif rel == f"{self.config.model_name}/config.yaml" or rel.endswith('best_epoch.json'):
                            dest = os.path.join(self.experiment_dir, os.path.basename(rel))
                        else:
                            continue
                        # Download only if not existing locally
                        if not os.path.exists(dest):
                            self.s3_client.download_file(bucket, key, dest)
            except Exception as e:
                logger.warning(f"Could not download existing artifacts from S3: {e}")
        # Load history if exists
        history_path = os.path.join(self.experiment_dir, "history.json")
        s3_history_key = f"{self.config.s3_prefix}{self.config.model_name}/history.json"
        # Attempt to download from S3
        if self.s3_client:
            try:
                self.s3_client.download_file(bucket, s3_history_key, history_path)
            except Exception:
                pass
        # Load local history if present
        if os.path.exists(history_path):
            try:
                with open(history_path, "r") as hf:
                    self.history = json.load(hf)
            except Exception as e:
                logger.warning(f"Could not load history.json: {e}")
        # Initialize early stopping if configured
        if self.config.early_stopping:
            self.early_stopping = EarlyStopping(
                patience=self.config.early_stopping_patience,
                min_delta=self.config.early_stopping_min_delta,
                mode=self.config.early_stopping_mode
            )
        else:
            self.early_stopping = None
        # Track best validation loss for uploading best model
        self.best_val_loss = float('inf')
        # Load existing weights: prefer best, else latest epoch checkpoint
        try:
            output_dir = self.config.output_dir
            model_name = self.config.model_name
            best_path = os.path.join(output_dir, model_name, f"{model_name}_best.pt")
            if os.path.exists(best_path):
                logger.info(f"Loading best model weights from {best_path}")
                self.model.load_state_dict(torch.load(best_path, map_location=self.device))
            else:
                # find epoch checkpoints
                pattern = os.path.join(output_dir, model_name, f"{model_name}_epoch*.pt")
                files = glob.glob(pattern)
                if files:
                    # extract epoch numbers
                    epochs = [
                        (int(re.search(r"_epoch(\d+)\.pt$", f).group(1)), f)
                        for f in files
                        if re.search(r"_epoch(\d+)\.pt$", f)
                    ]
                    if epochs:
                        latest = max(epochs, key=lambda x: x[0])[1]
                        logger.info(f"Loading latest checkpoint weights from {latest}")
                        self.model.load_state_dict(torch.load(latest, map_location=self.device))
        except Exception as e:
            logger.warning(f"Could not load existing weights: {e}")


    def train(self) -> None:
        """
        Runs the training loop over the dataset for the specified number of epochs.
        Logs average loss and accuracy per epoch.
        """
        self.model.train()
        # Determine starting epoch based on loaded history
        last_epoch = len(self.history.get('train_loss', []))
        start_epoch = last_epoch + 1
        if start_epoch > self.config.epochs:
            logger.info(f"All {last_epoch} epochs already completed. Nothing to train.")
            return
        for epoch in range(start_epoch, self.config.epochs + 1):
            # Configure logging for this epoch
            epoch_log_path = os.path.join(self.log_dir, f"epoch{epoch}.log")
            # Truncate existing log file so it doesn't append
            open(epoch_log_path, 'w').close()
            handler = RotatingFileHandler(epoch_log_path, mode='w', maxBytes=10_000_000, backupCount=5)
            formatter = logging.Formatter('[%(asctime)s] %(levelname)s %(message)s')
            handler.setFormatter(formatter)
            root_logger = logging.getLogger()
            # Ensure logger captures debug and info messages
            root_logger.setLevel(logging.DEBUG)
            handler.setLevel(logging.DEBUG)
            root_logger.addHandler(handler)
            ## training
            train_start = time()
            epoch_loss = 0
            sum_acc = 0.0
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for i, (inputs, targets) in pbar:
                # targets is shaped (batch, seq_len)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()

                # Forward pass
                logits = self.model(inputs) # batch, seq_len, vocab_size
                loss = self.criterion(logits, targets)

                # Backward pass
                loss.backward()
                self.optimizer.step()

                # TODO gradient clipping
                # TODO fp32 stuff, casting or at init?

                epoch_loss += loss.item()

                batch_acc = self.metric(logits, targets)
                sum_acc += batch_acc.item()

                train_loss = epoch_loss / (i + 1)
                train_acc = sum_acc / (i + 1)
                # train batch logging
                train_batch_str = f"Epoch {epoch}/{self.config.epochs} Batch {i} :: Train {self.criterion.name}: {train_loss:.4f}, Train {self.metric.name}: {train_acc:.4f}"
                pbar.set_description(train_batch_str)
                logging.debug(train_batch_str)
            
            avg_loss = epoch_loss / len(self.train_loader)
            avg_accuracy = sum_acc / len(self.train_loader)

            ## train logging
            train_str = f'Epoch {epoch}/{self.config.epochs} training done with loss: {avg_loss:.4f} in {time()-train_start:.2f} s.'
            print(train_str)
            logging.info(train_str)

            # Validation
            if self.val_loader is not None:
                val_loss = self.evaluate()
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(self.last_val_accuracy)

            self.history['train_loss'].append(avg_loss)
            self.history['train_acc'].append(avg_accuracy)

            # Save checkpoint and upload
            self._save_checkpoint(epoch=epoch)
            # Save best model if validation improves
            if self.val_loader is not None and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_checkpoint(is_best=True)
                # Write best epoch info
                best_info = {
                    "epoch": epoch,
                    "val_loss": val_loss,
                    "val_accuracy": self.last_val_accuracy
                }
                best_info_path = os.path.join(self.experiment_dir, "best_epoch.json")
                with open(best_info_path, "w") as f:
                    json.dump(best_info, f, indent=2)
                # Upload best epoch info file to S3
                if self.s3_client:
                    self.s3_client.upload_file(
                        best_info_path,
                        self.config.s3_bucket,
                        f"{self.config.s3_prefix}{self.config.model_name}/best_epoch.json"
                    )

            # Check early stopping based on validation loss
            if self.early_stopping is not None and self.val_loader is not None:
                self.early_stopping(val_loss, self.model)
                if self.early_stopping.early_stop:
                    print("Early stopping triggered.")
                    if self.config.restore_best_model and self.early_stopping.best_state is not None:
                        print("Restoring best model weights from epoch with optimal metric.")
                        self.model.load_state_dict(self.early_stopping.best_state)
                    # Remove epoch log handler
                    root_logger.removeHandler(handler)
                    handler.close()
                    break
            # Save history to JSON
            history_path = os.path.join(self.experiment_dir, "history.json")
            with open(history_path, "w") as hf:
                json.dump(self.history, hf, indent=2)
            # Upload history to S3
            if self.s3_client:
                s3_history_key = f"{self.config.s3_prefix}{self.config.model_name}/history.json"
                self.s3_client.upload_file(history_path, self.config.s3_bucket, s3_history_key)
            # Remove epoch log handler
            root_logger.removeHandler(handler)
            handler.close()

    def evaluate(self) -> float:
        """
        Evaluates the model on the validation set.
        Logs and returns average loss and accuracy over the validation set.

        Returns:
            float: Average loss on the validation set.
        """
        self.model.eval()
        total_loss = 0
        sum_acc = 0.0
        with torch.no_grad():
            pbar = tqdm(enumerate(self.val_loader), total=len(self.val_loader))
            for i, (inputs, targets) in pbar:
                batch_num = i + 1
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()

                batch_acc = self.metric(outputs, targets)
                sum_acc += batch_acc.item()

                avg_loss = total_loss / batch_num
                avg_accuracy = sum_acc / batch_num
                pbar.set_description(f"Eval {self.criterion.name}: {avg_loss:.4f}, Eval {self.metric.name}: {avg_accuracy:.4f}")

        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = sum_acc / len(self.val_loader)
        self.last_val_accuracy = avg_accuracy
        print('*'*50)
        return avg_loss

    def save_model(self) -> None:
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        model_path = os.path.join(self.checkpoint_dir, f"{self.config.model_name}.pt")
        torch.save(self.model.state_dict(), model_path)
        # Upload to S3 if configured
        if self.s3_client:
            s3_key = f"{self.config.s3_prefix}{self.config.model_name}/checkpoints/{self.config.model_name}.pt"
            self.s3_client.upload_file(model_path, self.config.s3_bucket, s3_key)
            # Upload log files if they exist
            if os.path.exists(self.log_dir):
                for fname in os.listdir(self.log_dir):
                    self.s3_client.upload_file(
                        os.path.join(self.log_dir, fname),
                        self.config.s3_bucket,
                        f"{self.config.s3_prefix}{self.config.model_name}/logs/{fname}"
                    )

    def _save_checkpoint(self, epoch: Optional[int] = None, is_best: bool = False) -> None:
        """
        Save model checkpoint locally and upload to S3 if configured.
        If epoch is specified, saves as <model_name>_epoch{epoch}.pt.
        If is_best is True, saves as <model_name>_best.pt.
        """
        suffix = "_best" if is_best else f"_epoch{epoch}"
        checkpoint_path = os.path.join(self.checkpoint_dir, f"{self.config.model_name}{suffix}.pt")
        torch.save(self.model.state_dict(), checkpoint_path)
        if self.s3_client:
            s3_key = f"{self.config.s3_prefix}{self.config.model_name}/checkpoints/{self.config.model_name}{suffix}.pt"
            self.s3_client.upload_file(checkpoint_path, self.config.s3_bucket, s3_key)
            # Upload log files if they exist
            log_path = self.log_dir
            if os.path.exists(log_path):
                for fname in os.listdir(log_path):
                    log_key = f"{self.config.s3_prefix}{self.config.model_name}/logs/{os.path.basename(fname)}"
                    self.s3_client.upload_file(
                        os.path.join(log_path, fname),
                        self.config.s3_bucket,
                        log_key
                    )

    def plot_history(self) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)
        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.legend()

        plt.tight_layout()
        plt.savefig(os.path.join(self.config.output_dir, f"{self.config.model_name}_training_curves.png"))
        plt.close()
