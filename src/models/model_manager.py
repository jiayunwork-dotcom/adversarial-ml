import os
import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import MODEL_DIR, DEVICE
from src.utils.helpers import (
    get_model_architecture,
    generate_id,
    save_json,
    load_json,
    get_imagenet_labels,
    normalize_image,
)


@dataclass
class ModelInfo:
    id: str
    name: str
    architecture: str
    num_classes: int
    input_size: int
    upload_time: str
    file_path: str
    file_type: str
    labels: List[str] = field(default_factory=list)
    clean_accuracy: Optional[float] = None
    is_onnx: bool = False
    is_state_dict: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "architecture": self.architecture,
            "num_classes": self.num_classes,
            "input_size": self.input_size,
            "upload_time": self.upload_time,
            "file_path": self.file_path,
            "file_type": self.file_type,
            "labels": self.labels,
            "clean_accuracy": self.clean_accuracy,
            "is_onnx": self.is_onnx,
            "is_state_dict": self.is_state_dict,
        }


class ModelManager:
    def __init__(self):
        self.models_file = os.path.join(MODEL_DIR, "models.json")
        self._models: Dict[str, ModelInfo] = {}
        self._loaded_models: Dict[str, Any] = {}
        self._onnx_sessions: Dict[str, Any] = {}
        self._load_models_from_file()

    def _load_models_from_file(self) -> None:
        data = load_json(self.models_file)
        for model_id, model_data in data.items():
            self._models[model_id] = ModelInfo(**model_data)

    def _save_models_to_file(self) -> None:
        data = {mid: m.to_dict() for mid, m in self._models.items()}
        save_json(data, self.models_file)

    def upload_model(self, file_path: str, name: str, architecture: str,
                     is_state_dict: bool = False, labels_file: Optional[str] = None) -> ModelInfo:
        model_id = generate_id()
        file_ext = os.path.splitext(file_path)[1].lower()
        is_onnx = file_ext == ".onnx"

        dest_path = os.path.join(MODEL_DIR, f"{model_id}{file_ext}")
        import shutil
        shutil.copy(file_path, dest_path)

        labels = self._load_labels(labels_file) if labels_file else get_imagenet_labels()

        model_info = ModelInfo(
            id=model_id,
            name=name,
            architecture=architecture,
            num_classes=len(labels),
            input_size=224,
            upload_time=time.strftime("%Y-%m-%d %H:%M:%S"),
            file_path=dest_path,
            file_type=file_ext[1:],
            labels=labels,
            is_onnx=is_onnx,
            is_state_dict=is_state_dict,
        )

        self._detect_model_properties(model_info)

        self._models[model_id] = model_info
        self._save_models_to_file()

        return model_info

    def _load_labels(self, labels_file: str) -> List[str]:
        labels = []
        with open(labels_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    labels.append(line)
        return labels

    def _detect_model_properties(self, model_info: ModelInfo) -> None:
        try:
            if model_info.is_onnx:
                import onnx
                import onnxruntime as ort
                onnx_model = onnx.load(model_info.file_path)
                onnx.checker.check_model(onnx_model)
                input_shape = onnx_model.graph.input[0].type.tensor_type.shape.dim
                if len(input_shape) >= 3:
                    model_info.input_size = input_shape[2].dim_value if input_shape[2].dim_value else 224
                output_shape = onnx_model.graph.output[0].type.tensor_type.shape.dim
                model_info.num_classes = output_shape[-1].dim_value if output_shape[-1].dim_value else model_info.num_classes
            else:
                checkpoint = torch.load(model_info.file_path, map_location=DEVICE, weights_only=False)
                if model_info.is_state_dict:
                    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                        state_dict = checkpoint["state_dict"]
                    else:
                        state_dict = checkpoint
                    last_key = list(state_dict.keys())[-1]
                    if "classifier" in last_key or "fc" in last_key:
                        model_info.num_classes = state_dict[last_key].shape[0]
                else:
                    if hasattr(checkpoint, "fc"):
                        model_info.num_classes = checkpoint.fc.out_features
                    elif hasattr(checkpoint, "classifier"):
                        if isinstance(checkpoint.classifier, nn.Sequential):
                            model_info.num_classes = checkpoint.classifier[-1].out_features
                        else:
                            model_info.num_classes = checkpoint.classifier.out_features
        except Exception as e:
            print(f"Warning: Could not fully detect model properties: {e}")

    def load_model(self, model_id: str) -> Any:
        if model_id in self._loaded_models:
            return self._loaded_models[model_id]

        model_info = self._models.get(model_id)
        if not model_info:
            raise ValueError(f"Model {model_id} not found")

        if model_info.is_onnx:
            return self._load_onnx_model(model_info)

        model = get_model_architecture(
            model_info.architecture,
            num_classes=model_info.num_classes,
            pretrained=False
        )

        if model_info.is_state_dict:
            checkpoint = torch.load(model_info.file_path, map_location=DEVICE, weights_only=False)
            if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint
            model.load_state_dict(state_dict)
        else:
            model = torch.load(model_info.file_path, map_location=DEVICE, weights_only=False)

        model = model.to(DEVICE)
        model.eval()

        self._loaded_models[model_id] = model
        return model

    def _load_onnx_model(self, model_info: ModelInfo):
        import onnxruntime as ort
        if model_info.id in self._onnx_sessions:
            return ONNXWrapper(self._onnx_sessions[model_info.id], model_info.num_classes, DEVICE)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if DEVICE.type == "cuda" else ["CPUExecutionProvider"]
        session = ort.InferenceSession(model_info.file_path, providers=providers)
        self._onnx_sessions[model_info.id] = session

        return ONNXWrapper(session, model_info.num_classes, DEVICE)

    def predict(self, model_id: str, images: torch.Tensor) -> torch.Tensor:
        model_info = self._models.get(model_id)
        if not model_info:
            raise ValueError(f"Model {model_id} not found")

        if model_info.is_onnx:
            return self._onnx_predict(model_id, images)

        model = self.load_model(model_id)
        normalized_images = normalize_image(images)
        with torch.no_grad():
            outputs = model(normalized_images)
        return outputs

    def _onnx_predict(self, model_id: str, images: torch.Tensor) -> torch.Tensor:
        session = self._onnx_sessions.get(model_id)
        if not session:
            model_info = self._models[model_id]
            self.load_model(model_id)
            session = self._onnx_sessions[model_id]

        input_name = session.get_inputs()[0].name
        normalized_images = normalize_image(images)
        outputs = session.run(None, {input_name: normalized_images.cpu().numpy()})
        return torch.tensor(outputs[0], device=DEVICE)

    def evaluate_clean_accuracy(self, model_id: str, dataloader: DataLoader) -> float:
        model_info = self._models.get(model_id)
        if not model_info:
            raise ValueError(f"Model {model_id} not found")

        correct = 0
        total = 0

        for images, labels in tqdm(dataloader, desc="Evaluating clean accuracy"):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = self.predict(model_id, images)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        accuracy = 100.0 * correct / total
        model_info.clean_accuracy = accuracy
        self._save_models_to_file()
        return accuracy

    def list_models(self) -> List[ModelInfo]:
        return list(self._models.values())

    def get_model(self, model_id: str) -> Optional[ModelInfo]:
        return self._models.get(model_id)

    def delete_model(self, model_id: str) -> bool:
        if model_id in self._models:
            model_info = self._models[model_id]
            if os.path.exists(model_info.file_path):
                os.remove(model_info.file_path)
            del self._models[model_id]
            if model_id in self._loaded_models:
                del self._loaded_models[model_id]
            if model_id in self._onnx_sessions:
                del self._onnx_sessions[model_id]
            self._save_models_to_file()
            return True
        return False

    def get_input_size(self, model_id: str) -> int:
        model_info = self._models.get(model_id)
        return model_info.input_size if model_info else 224

    def get_labels(self, model_id: str) -> List[str]:
        model_info = self._models.get(model_id)
        return model_info.labels if model_info else get_imagenet_labels()


class ONNXWrapper(nn.Module):
    def __init__(self, session, num_classes: int, device: torch.device):
        super().__init__()
        self.session = session
        self.num_classes = num_classes
        self.device = device
        self._input_name = session.get_inputs()[0].name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.session.run(None, {self._input_name: x.detach().cpu().numpy()})
        return torch.tensor(outputs[0], device=self.device)

    def eval(self):
        return self

    def to(self, device):
        self.device = device
        return self

    def parameters(self):
        return iter([])
