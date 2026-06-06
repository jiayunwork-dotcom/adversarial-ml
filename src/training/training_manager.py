import os
import time
import threading
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, random_split

from config import DEVICE, TRAINING_DIR, MODEL_DIR, CHECKPOINT_DIR
from src.training.adversarial_trainer import (
    AdversarialTrainer,
    TrainingConfig,
    TrainingLog,
)
from src.utils.helpers import generate_id, save_json, load_json
from src.datasets.dataset_manager import ImageFolderDataset


class TrainingStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class TrainingTask:
    id: str
    original_model_id: str
    original_model_name: str
    dataset_id: str
    config: TrainingConfig
    status: TrainingStatus = TrainingStatus.PENDING
    logs: List[TrainingLog] = field(default_factory=list)
    result_model_id: Optional[str] = None
    original_metrics: Optional[Dict[str, Any]] = None
    trained_metrics: Optional[Dict[str, Any]] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    error_message: Optional[str] = None
    resume_from_checkpoint: bool = False
    training_info: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "original_model_id": self.original_model_id,
            "original_model_name": self.original_model_name,
            "dataset_id": self.dataset_id,
            "config": self.config.to_dict(),
            "status": self.status.value,
            "logs": [log.to_dict() for log in self.logs],
            "result_model_id": self.result_model_id,
            "original_metrics": self.original_metrics,
            "trained_metrics": self.trained_metrics,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "error_message": self.error_message,
            "resume_from_checkpoint": self.resume_from_checkpoint,
            "training_info": self.training_info,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingTask":
        config = TrainingConfig(**data["config"])
        logs = [TrainingLog(**log_data) for log_data in data["logs"]]
        status = TrainingStatus(data["status"])
        task = cls(
            id=data["id"],
            original_model_id=data["original_model_id"],
            original_model_name=data["original_model_name"],
            dataset_id=data["dataset_id"],
            config=config,
            status=status,
            logs=logs,
            result_model_id=data.get("result_model_id"),
            original_metrics=data.get("original_metrics"),
            trained_metrics=data.get("trained_metrics"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            error_message=data.get("error_message"),
            resume_from_checkpoint=data.get("resume_from_checkpoint", False),
            training_info=data.get("training_info"),
        )
        return task


class AdversarialDataset(Dataset):
    def __init__(self, root_dir: str, input_size: int = 32, train: bool = True):
        self.input_size = input_size
        self.train = train

        if train:
            self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.RandomCrop(input_size, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
            ])

        self._base_dataset = ImageFolderDataset(
            root_dir, transform=None, input_size=input_size
        )

    def __len__(self) -> int:
        return len(self._base_dataset)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self._base_dataset.image_paths[idx]
        from PIL import Image
        image = Image.open(img_path).convert("RGB")
        label = self._base_dataset.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class TrainingManager:
    def __init__(self, model_manager, dataset_manager, robustness_metrics):
        self.model_manager = model_manager
        self.dataset_manager = dataset_manager
        self.robustness_metrics = robustness_metrics
        self.tasks_file = os.path.join(TRAINING_DIR, "tasks.json")
        self._tasks: Dict[str, TrainingTask] = {}
        self._trainers: Dict[str, AdversarialTrainer] = {}
        self._threads: Dict[str, threading.Thread] = {}
        self._load_tasks()

    def _load_tasks(self) -> None:
        data = load_json(self.tasks_file)
        for task_id, task_data in data.items():
            self._tasks[task_id] = TrainingTask.from_dict(task_data)

    def _save_tasks(self) -> None:
        data = {tid: task.to_dict() for tid, task in self._tasks.items()}
        save_json(data, self.tasks_file)

    def _append_log(self, task_id: str, log: TrainingLog) -> None:
        if task_id in self._tasks:
            self._tasks[task_id].logs.append(log)
            self._save_tasks()

    def _prepare_datasets(self, dataset_id: str, input_size: int
                          ) -> Tuple[Dataset, Dataset]:
        ds_info = self.dataset_manager.get_dataset(dataset_id)
        if not ds_info:
            raise ValueError(f"Dataset {dataset_id} not found")

        full_dataset = AdversarialDataset(
            ds_info.path, input_size=input_size, train=True
        )

        total_size = len(full_dataset)
        val_size = max(1, int(total_size * 0.2))
        train_size = total_size - val_size

        if train_size <= 0 or val_size <= 0:
            train_size = max(1, total_size - 1)
            val_size = total_size - train_size

        train_dataset, val_dataset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )

        val_dataset.dataset.train = False

        return train_dataset, val_dataset

    def _get_original_metrics(self, model_id: str, dataset_id: str
                              ) -> Dict[str, Any]:
        model_info = self.model_manager.get_model(model_id)
        input_size = model_info.input_size

        dataloader = self.dataset_manager.get_dataloader(
            dataset_id, batch_size=8, input_size=input_size, shuffle=False
        )

        def attack_fn(model, images, labels, **kwargs):
            from src.attacks.pgd import PGD
            attack = PGD(model, **kwargs)
            return attack.generate(images, labels)

        attack_params = {
            "epsilon": 8.0 / 255.0,
            "norm": "Linf",
            "iterations": 20,
            "alpha": 2.0 / 255.0,
            "random_start": True,
        }

        metrics = self.robustness_metrics.evaluate(
            model_id, dataloader, attack_fn, attack_params
        )

        return {
            "clean_accuracy": metrics["clean_accuracy"],
            "robust_accuracy": metrics["robust_accuracy"],
        }

    def _save_trained_model(self, model: torch.nn.Module, original_model_id: str,
                            config: TrainingConfig) -> str:
        import shutil
        original_model_info = self.model_manager.get_model(original_model_id)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        new_model_name = f"{original_model_info.name}_AT_{timestamp}"
        new_model_id = generate_id()

        model_path = os.path.join(MODEL_DIR, f"{new_model_id}.pt")
        torch.save(model.state_dict(), model_path)

        from src.models.model_manager import ModelInfo
        model_info = ModelInfo(
            id=new_model_id,
            name=new_model_name,
            architecture=original_model_info.architecture,
            num_classes=original_model_info.num_classes,
            input_size=original_model_info.input_size,
            upload_time=time.strftime("%Y-%m-%d %H:%M:%S"),
            file_path=model_path,
            file_type="pt",
            labels=original_model_info.labels,
            is_onnx=False,
            is_state_dict=True,
        )

        self.model_manager._models[new_model_id] = model_info
        self.model_manager._save_models_to_file()

        return new_model_id

    def _evaluate_trained_model(self, model_id: str, dataset_id: str
                                ) -> Dict[str, Any]:
        return self._get_original_metrics(model_id, dataset_id)

    def create_training_task(self, model_id: str, dataset_id: str,
                             method: str, epochs: int, learning_rate: float,
                             batch_size: int, attack_steps: int,
                             epsilon: float, beta: float = 6.0,
                             resume_from_checkpoint: bool = False,
                             early_stopping: bool = False,
                             patience: int = 5
                             ) -> TrainingTask:
        model_info = self.model_manager.get_model(model_id)
        if not model_info:
            raise ValueError(f"Model {model_id} not found")

        if model_info.is_onnx:
            raise ValueError("ONNX models are not supported for adversarial training")

        ds_info = self.dataset_manager.get_dataset(dataset_id)
        if not ds_info:
            raise ValueError(f"Dataset {dataset_id} not found")

        config = TrainingConfig(
            method=method,
            epochs=epochs,
            learning_rate=learning_rate,
            batch_size=batch_size,
            epsilon=epsilon,
            attack_steps=attack_steps,
            beta=beta,
            input_size=model_info.input_size,
            num_classes=model_info.num_classes,
            early_stopping=early_stopping,
            patience=patience,
        )

        task_id = generate_id()
        task = TrainingTask(
            id=task_id,
            original_model_id=model_id,
            original_model_name=model_info.name,
            dataset_id=dataset_id,
            config=config,
            resume_from_checkpoint=resume_from_checkpoint,
        )

        self._tasks[task_id] = task
        self._save_tasks()

        return task

    def get_task(self, task_id: str) -> Optional[TrainingTask]:
        return self._tasks.get(task_id)

    def list_tasks(self) -> List[TrainingTask]:
        return list(self._tasks.values())

    def stop_training(self, task_id: str) -> bool:
        if task_id in self._trainers:
            self._trainers[task_id].stop()
            return True
        return False

    def _run_training(self, task_id: str) -> None:
        try:
            task = self._tasks[task_id]
            task.status = TrainingStatus.RUNNING
            task.start_time = time.strftime("%Y-%m-%d %H:%M:%S")
            self._save_tasks()

            model = self.model_manager.load_model(task.original_model_id)
            model = model.to(DEVICE)
            model.train()

            trainer = AdversarialTrainer(model, task.config)
            self._trainers[task_id] = trainer

            start_epoch = 0
            if task.resume_from_checkpoint:
                latest_checkpoint = trainer.find_latest_checkpoint(task_id)
                if latest_checkpoint:
                    start_epoch = trainer.load_checkpoint(latest_checkpoint)
                    print(f"Resuming training from epoch {start_epoch}")

            task.original_metrics = self._get_original_metrics(
                task.original_model_id, task.dataset_id
            )
            self._save_tasks()

            train_dataset, val_dataset = self._prepare_datasets(
                task.dataset_id, task.config.input_size
            )

            def log_callback(log: TrainingLog):
                self._append_log(task_id, log)

            trained_model, all_logs, training_info = trainer.train(
                train_dataset, val_dataset, task_id,
                log_callback=log_callback,
                start_epoch=start_epoch
            )

            task.training_info = training_info

            if training_info["early_stopped"]:
                task.status = TrainingStatus.STOPPED
                best_epoch = training_info["best_epoch"]
                print(f"Loading best checkpoint from epoch {best_epoch}")
                best_checkpoint_path = os.path.join(
                    CHECKPOINT_DIR, f"{task_id}_epoch_{best_epoch}.pt"
                )
                if os.path.exists(best_checkpoint_path):
                    trainer.load_checkpoint(best_checkpoint_path)
                    trained_model = trainer.model
            elif trainer._stop_requested:
                task.status = TrainingStatus.STOPPED
            else:
                task.status = TrainingStatus.COMPLETED

            task.end_time = time.strftime("%Y-%m-%d %H:%M:%S")

            if task.status in [TrainingStatus.COMPLETED, TrainingStatus.STOPPED] and trained_model is not None:
                trained_model.eval()
                result_model_id = self._save_trained_model(
                    trained_model, task.original_model_id, task.config
                )
                task.result_model_id = result_model_id

                task.trained_metrics = self._evaluate_trained_model(
                    result_model_id, task.dataset_id
                )

            self._save_tasks()

        except Exception as e:
            import traceback
            error_msg = f"{str(e)}\n{traceback.format_exc()}"
            print(f"Training error: {error_msg}")
            if task_id in self._tasks:
                self._tasks[task_id].status = TrainingStatus.ERROR
                self._tasks[task_id].error_message = str(e)
                self._tasks[task_id].end_time = time.strftime("%Y-%m-%d %H:%M:%S")
                self._save_tasks()
        finally:
            if task_id in self._trainers:
                del self._trainers[task_id]
            if task_id in self._threads:
                del self._threads[task_id]

    def start_training(self, task_id: str) -> None:
        if task_id not in self._tasks:
            raise ValueError(f"Task {task_id} not found")

        if task_id in self._threads and self._threads[task_id].is_alive():
            raise RuntimeError("Training is already running for this task")

        thread = threading.Thread(
            target=self._run_training,
            args=(task_id,),
            daemon=True
        )
        self._threads[task_id] = thread
        thread.start()

    def get_training_logs(self, task_id: str) -> List[TrainingLog]:
        task = self._tasks.get(task_id)
        return task.logs if task else []

    def get_training_status(self, task_id: str) -> Optional[TrainingStatus]:
        task = self._tasks.get(task_id)
        return task.status if task else None

    def get_comparison_results(self, task_id: str) -> Optional[Dict[str, Any]]:
        task = self._tasks.get(task_id)
        if not task or not task.original_metrics or not task.trained_metrics:
            return None

        return {
            "original": task.original_metrics,
            "trained": task.trained_metrics,
            "improvements": {
                "clean_accuracy": task.trained_metrics["clean_accuracy"] - task.original_metrics["clean_accuracy"],
                "robust_accuracy": task.trained_metrics["robust_accuracy"] - task.original_metrics["robust_accuracy"],
            },
            "result_model_id": task.result_model_id,
            "original_model_name": task.original_model_name,
        }

    def format_log_message(self, log: TrainingLog) -> str:
        msg = f"[{log.timestamp}] Epoch {log.epoch}/{log.total_epochs}, Batch {log.batch}/{log.total_batches}, Loss: {log.loss:.4f}"
        if log.clean_acc is not None:
            msg += f", Clean Acc: {log.clean_acc:.2f}%"
        if log.robust_acc is not None:
            msg += f", Robust Acc: {log.robust_acc:.2f}%"
        return msg

    def get_training_history(self, task_id: str) -> Dict[str, Any]:
        task = self._tasks.get(task_id)
        if not task:
            return {"epochs": [], "losses": [], "clean_accs": [], "robust_accs": []}

        epochs = []
        losses = []
        clean_accs = []
        robust_accs = []

        epoch_data = {}
        for log in task.logs:
            if log.clean_acc is not None and log.robust_acc is not None:
                epoch_data[log.epoch] = {
                    "loss": log.loss,
                    "clean_acc": log.clean_acc,
                    "robust_acc": log.robust_acc
                }

        for epoch in sorted(epoch_data.keys()):
            epochs.append(epoch)
            losses.append(epoch_data[epoch]["loss"])
            clean_accs.append(epoch_data[epoch]["clean_acc"])
            robust_accs.append(epoch_data[epoch]["robust_acc"])

        return {
            "epochs": epochs,
            "losses": losses,
            "clean_accs": clean_accs,
            "robust_accs": robust_accs,
            "task_info": {
                "id": task.id,
                "model_name": task.original_model_name,
                "method": task.config.method,
                "epochs": task.config.epochs,
                "status": task.status.value,
                "start_time": task.start_time,
                "end_time": task.end_time,
                "training_info": task.training_info,
                "trained_metrics": task.trained_metrics,
            }
        }
