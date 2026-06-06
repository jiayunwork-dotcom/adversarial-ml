import os
import time
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision.transforms as transforms
from PIL import Image

from config import DEVICE, CHECKPOINT_DIR
from src.utils.helpers import normalize_image, generate_id, save_json, load_json


@dataclass
class TrainingConfig:
    method: str = "pgd_at"
    epochs: int = 10
    learning_rate: float = 0.01
    batch_size: int = 32
    epsilon: float = 8.0 / 255.0
    attack_steps: int = 10
    beta: float = 6.0
    input_size: int = 32
    num_classes: int = 10
    early_stopping: bool = False
    patience: int = 5

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "epsilon": self.epsilon,
            "attack_steps": self.attack_steps,
            "beta": self.beta,
            "input_size": self.input_size,
            "num_classes": self.num_classes,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
        }


@dataclass
class TrainingLog:
    epoch: int
    total_epochs: int
    batch: int
    total_batches: int
    loss: float
    clean_acc: Optional[float] = None
    robust_acc: Optional[float] = None
    timestamp: str = field(default_factory=lambda: time.strftime("%Y-%m-%d %H:%M:%S"))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "epoch": self.epoch,
            "total_epochs": self.total_epochs,
            "batch": self.batch,
            "total_batches": self.total_batches,
            "loss": self.loss,
            "clean_acc": self.clean_acc,
            "robust_acc": self.robust_acc,
            "timestamp": self.timestamp,
        }


class PGDAdversary:
    def __init__(self, model: nn.Module, epsilon: float = 8.0 / 255.0,
                 alpha: float = 2.0 / 255.0, steps: int = 10,
                 random_start: bool = True):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start

    def generate(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        images = images.clone().detach().to(DEVICE)
        labels = labels.clone().detach().to(DEVICE)

        adv_images = images.clone()

        if self.random_start:
            delta = torch.empty_like(adv_images).uniform_(-self.epsilon, self.epsilon)
            adv_images = adv_images + delta
            adv_images = torch.clamp(adv_images, 0.0, 1.0)

        for _ in range(self.steps):
            adv_images.requires_grad = True
            outputs = self.model(normalize_image(adv_images))
            loss = nn.CrossEntropyLoss()(outputs, labels)

            self.model.zero_grad()
            loss.backward()

            grad = adv_images.grad.data
            adv_images = adv_images.detach() + self.alpha * grad.sign()

            adv_images = torch.clamp(adv_images, images - self.epsilon, images + self.epsilon)
            adv_images = torch.clamp(adv_images, 0.0, 1.0)

        return adv_images.detach()


class TRADESAdversary:
    def __init__(self, model: nn.Module, epsilon: float = 8.0 / 255.0,
                 alpha: float = 2.0 / 255.0, steps: int = 10,
                 beta: float = 6.0):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.beta = beta

    def generate(self, images: torch.Tensor) -> torch.Tensor:
        images = images.clone().detach().to(DEVICE)
        batch_size = images.size(0)

        with torch.no_grad():
            clean_logits = self.model(normalize_image(images))
            clean_probs = F.softmax(clean_logits, dim=1)

        adv_images = images.clone().detach() + 0.001 * torch.randn_like(images)

        for _ in range(self.steps):
            adv_images.requires_grad = True
            adv_logits = self.model(normalize_image(adv_images))
            adv_probs = F.softmax(adv_logits, dim=1)

            loss = F.kl_div(F.log_softmax(adv_logits, dim=1), clean_probs, reduction="batchmean")

            self.model.zero_grad()
            if adv_images.grad is not None:
                adv_images.grad.data.zero_()
            loss.backward()

            grad = adv_images.grad.data
            adv_images = adv_images.detach() + self.alpha * grad.sign()

            adv_images = torch.clamp(adv_images, images - self.epsilon, images + self.epsilon)
            adv_images = torch.clamp(adv_images, 0.0, 1.0)

        return adv_images.detach()

    def compute_loss(self, images: torch.Tensor, labels: torch.Tensor,
                     adv_images: torch.Tensor) -> torch.Tensor:
        clean_logits = self.model(normalize_image(images))
        adv_logits = self.model(normalize_image(adv_images))

        clean_loss = F.cross_entropy(clean_logits, labels)

        clean_probs = F.softmax(clean_logits, dim=1)
        adv_log_probs = F.log_softmax(adv_logits, dim=1)
        kl_loss = F.kl_div(adv_log_probs, clean_probs, reduction="batchmean")

        return clean_loss + self.beta * kl_loss


class AdversarialTrainer:
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model.to(DEVICE)
        self.config = config
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=config.learning_rate,
            momentum=0.9,
            weight_decay=2e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=config.epochs // 3, gamma=0.1
        )

        if config.method == "pgd_at":
            self.adversary = PGDAdversary(
                self.model,
                epsilon=config.epsilon,
                alpha=config.epsilon / 4.0,
                steps=config.attack_steps
            )
        elif config.method == "trades":
            self.adversary = TRADESAdversary(
                self.model,
                epsilon=config.epsilon,
                alpha=config.epsilon / 4.0,
                steps=config.attack_steps,
                beta=config.beta
            )
        else:
            raise ValueError(f"Unknown training method: {config.method}")

        self.eval_adversary = PGDAdversary(
            self.model,
            epsilon=config.epsilon,
            alpha=config.epsilon / 4.0,
            steps=10
        )

        self.checkpoint_dir = CHECKPOINT_DIR
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self._is_running = False
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    def is_running(self) -> bool:
        return self._is_running

    def save_checkpoint(self, epoch: int, training_id: str) -> str:
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"{training_id}_epoch_{epoch}.pt"
        )
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "config": self.config.to_dict(),
            "training_id": training_id,
        }
        torch.save(checkpoint, checkpoint_path)
        return checkpoint_path

    def load_checkpoint(self, checkpoint_path: str) -> int:
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        return checkpoint["epoch"]

    def find_latest_checkpoint(self, training_id: str) -> Optional[str]:
        checkpoints = []
        for f in os.listdir(self.checkpoint_dir):
            if f.startswith(training_id) and f.endswith(".pt"):
                checkpoints.append(f)
        if not checkpoints:
            return None
        checkpoints.sort()
        return os.path.join(self.checkpoint_dir, checkpoints[-1])

    def evaluate(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        clean_correct = 0
        robust_correct = 0
        total = 0

        for images, labels in dataloader:
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            batch_size = labels.size(0)

            with torch.no_grad():
                clean_outputs = self.model(normalize_image(images))
                clean_preds = clean_outputs.argmax(1)
                clean_mask = clean_preds == labels
                clean_correct += clean_mask.sum().item()
                total += batch_size

            if clean_mask.sum() > 0:
                clean_images = images[clean_mask]
                clean_labels = labels[clean_mask]

                self.eval_adversary.model = self.model
                adv_images = self.eval_adversary.generate(clean_images, clean_labels)

                with torch.no_grad():
                    adv_outputs = self.model(normalize_image(adv_images))
                    adv_preds = adv_outputs.argmax(1)
                    robust_correct += (adv_preds == clean_labels).sum().item()

        clean_acc = 100.0 * clean_correct / total if total > 0 else 0.0
        robust_acc = 100.0 * robust_correct / max(clean_correct, 1) if clean_correct > 0 else 0.0

        return clean_acc, robust_acc

    def train_epoch(self, train_loader: DataLoader, epoch: int,
                    total_epochs: int, training_id: str,
                    log_callback: Optional[Callable[[TrainingLog], None]] = None
                    ) -> List[TrainingLog]:
        self.model.train()
        logs = []
        total_batches = len(train_loader)

        for batch_idx, (images, labels) in enumerate(train_loader):
            if self._stop_requested:
                break

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            batch_size = labels.size(0)

            self.optimizer.zero_grad()

            if self.config.method == "pgd_at":
                adv_images = self.adversary.generate(images, labels)
                outputs = self.model(normalize_image(adv_images))
                loss = nn.CrossEntropyLoss()(outputs, labels)
            elif self.config.method == "trades":
                adv_images = self.adversary.generate(images)
                loss = self.adversary.compute_loss(images, labels, adv_images)

            loss.backward()
            self.optimizer.step()

            log_entry = TrainingLog(
                epoch=epoch + 1,
                total_epochs=total_epochs,
                batch=batch_idx + 1,
                total_batches=total_batches,
                loss=loss.item()
            )
            logs.append(log_entry)

            if log_callback:
                log_callback(log_entry)

            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{total_epochs}, Batch {batch_idx+1}/{total_batches}, Loss: {loss.item():.4f}")

        self.scheduler.step()

        return logs

    def train(self, train_dataset: Dataset, val_dataset: Dataset, training_id: str,
              log_callback: Optional[Callable[[TrainingLog], None]] = None,
              start_epoch: int = 0) -> Tuple[nn.Module, List[TrainingLog], Dict[str, Any]]:
        self._is_running = True
        self._stop_requested = False
        all_logs = []

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )

        best_robust_acc = 0.0
        best_epoch = 0
        patience_counter = 0
        early_stopped = False

        for epoch in range(start_epoch, self.config.epochs):
            if self._stop_requested:
                print(f"Training stopped at epoch {epoch}")
                break

            print(f"\n=== Epoch {epoch + 1}/{self.config.epochs} ===")
            epoch_logs = self.train_epoch(
                train_loader, epoch, self.config.epochs, training_id, log_callback
            )
            all_logs.extend(epoch_logs)

            clean_acc, robust_acc = self.evaluate(val_loader)

            epoch_loss = np.mean([log.loss for log in epoch_logs]) if epoch_logs else 0.0
            eval_log = TrainingLog(
                epoch=epoch + 1,
                total_epochs=self.config.epochs,
                batch=len(train_loader),
                total_batches=len(train_loader),
                loss=epoch_loss,
                clean_acc=clean_acc,
                robust_acc=robust_acc
            )
            all_logs.append(eval_log)

            if log_callback:
                log_callback(eval_log)

            print(f"Epoch {epoch+1} - Clean Acc: {clean_acc:.2f}%, Robust Acc: {robust_acc:.2f}%")

            self.save_checkpoint(epoch + 1, training_id)

            if self.config.early_stopping:
                if robust_acc > best_robust_acc:
                    best_robust_acc = robust_acc
                    best_epoch = epoch + 1
                    patience_counter = 0
                    print(f"New best robust accuracy: {best_robust_acc:.2f}% at epoch {best_epoch}")
                else:
                    patience_counter += 1
                    print(f"Patience: {patience_counter}/{self.config.patience}")

                if patience_counter >= self.config.patience:
                    early_stopped = True
                    stop_msg = f"Early stopping triggered at epoch {epoch + 1}, best robust accuracy: {best_robust_acc:.2f}%"
                    print(stop_msg)
                    if log_callback:
                        stop_log = TrainingLog(
                            epoch=epoch + 1,
                            total_epochs=self.config.epochs,
                            batch=len(train_loader),
                            total_batches=len(train_loader),
                            loss=epoch_loss,
                            clean_acc=clean_acc,
                            robust_acc=robust_acc
                        )
                        all_logs.append(stop_log)
                    break

        self._is_running = False

        training_info = {
            "best_robust_acc": best_robust_acc,
            "best_epoch": best_epoch,
            "early_stopped": early_stopped,
            "final_epoch": epoch + 1 if early_stopped or self._stop_requested else self.config.epochs
        }

        return self.model, all_logs, training_info
