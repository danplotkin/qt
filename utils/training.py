import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from tqdm import tqdm
from utils.configs import TrainingConfigs


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
        criterion: nn.Module,
        metric: callable,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ) -> None:
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
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

    def train(self) -> None:
        """
        Runs the training loop over the dataset for the specified number of epochs.
        Logs average loss and accuracy per epoch.
        """
        self.model.train()
        for epoch in range(1, self.config.num_epochs + 1):
            print(f'Epoch {epoch}/{self.config.num_epochs}:')
            epoch_loss = 0
            sum_acc = 0.0
            pbar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
            for i, (inputs, targets) in pbar:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                # Shift for next-token prediction
                logits = outputs[:, :-1, :]
                target_tokens = targets[:, 1:]
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_tokens.reshape(-1))
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()

                batch_acc = self.metric(logits, target_tokens)
                sum_acc += batch_acc.item()

                train_loss = epoch_loss / (i + 1)
                train_acc = sum_acc / (i + 1)
                pbar.set_description(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            avg_loss = epoch_loss / len(self.train_loader)
            avg_accuracy = sum_acc / len(self.train_loader)
            print()

            if self.val_loader is not None:
                val_loss = self.evaluate()
                self.history['val_loss'].append(val_loss)
                self.history['val_acc'].append(self.last_val_accuracy)
            self.history['train_loss'].append(avg_loss)
            self.history['train_acc'].append(avg_accuracy)

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
                logits = outputs[:, :-1, :]
                target_tokens = targets[:, 1:]
                loss = self.criterion(logits.reshape(-1, logits.size(-1)), target_tokens.reshape(-1))
                total_loss += loss.item()

                batch_acc = self.metric(logits, target_tokens)
                sum_acc += batch_acc.item()

                avg_loss = total_loss / batch_num
                avg_accuracy = sum_acc / batch_num
                pbar.set_description(f"Eval Loss: {avg_loss:.4f}, Eval Acc: {avg_accuracy:.4f}")

        avg_loss = total_loss / len(self.val_loader)
        avg_accuracy = sum_acc / len(self.val_loader)
        self.last_val_accuracy = avg_accuracy
        print('*'*50)
        return avg_loss

    def save_model(self) -> None:
        os.makedirs(self.config.output_dir, exist_ok=True)
        model_path = os.path.join(self.config.output_dir, f"{self.config.model_name}.pt")
        torch.save(self.model.state_dict(), model_path)

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
