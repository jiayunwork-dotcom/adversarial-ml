from abc import ABC, abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from config import DEVICE
from src.utils.helpers import normalize_image, denormalize_image


class Attack(ABC):
    def __init__(self, model: nn.Module, epsilon: float = 8.0 / 255.0,
                 norm: str = "Linf", targeted: bool = False):
        self.model = model
        self.epsilon = epsilon
        self.norm = norm
        self.targeted = targeted
        self.device = DEVICE

    @abstractmethod
    def generate(self, images: torch.Tensor, labels: torch.Tensor,
                 target_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        pass

    def _get_loss(self, outputs: torch.Tensor, labels: torch.Tensor,
                  target_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        if self.targeted and target_labels is not None:
            return -nn.CrossEntropyLoss()(outputs, target_labels)
        else:
            return nn.CrossEntropyLoss()(outputs, labels)

    def _clamp(self, adv_images: torch.Tensor, original_images: torch.Tensor) -> torch.Tensor:
        if self.norm == "Linf":
            adv_images = torch.clamp(adv_images, original_images - self.epsilon, original_images + self.epsilon)
        elif self.norm == "L2":
            delta = adv_images - original_images
            delta_flat = delta.view(delta.shape[0], -1)
            norms = torch.norm(delta_flat, p=2, dim=1)
            mask = norms > self.epsilon
            if mask.any():
                scale = self.epsilon / norms[mask]
                delta[mask] = delta[mask] * scale.view(-1, 1, 1, 1)
            adv_images = original_images + delta
        return torch.clamp(adv_images, 0.0, 1.0)

    def _project(self, adv_images: torch.Tensor, original_images: torch.Tensor) -> torch.Tensor:
        return self._clamp(adv_images, original_images)

    def _model_forward(self, images: torch.Tensor) -> torch.Tensor:
        normalized = normalize_image(images)
        return self.model(normalized)

    def __call__(self, *args, **kwargs) -> torch.Tensor:
        return self.generate(*args, **kwargs)
