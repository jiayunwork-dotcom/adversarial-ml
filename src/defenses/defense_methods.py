import io
from typing import Dict, Any, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageFilter

from config import DEVICE
from src.utils.helpers import normalize_image, denormalize_image


class DefenseMethods:
    def __init__(self):
        self.defenses = {
            "jpeg_compression": self.jpeg_compression,
            "random_resize_padding": self.random_resize_padding,
            "bit_depth_reduction": self.bit_depth_reduction,
            "median_filter": self.median_filter,
        }

    def apply_defense(self, images: torch.Tensor, defense_type: str,
                      **params) -> torch.Tensor:
        if defense_type not in self.defenses:
            raise ValueError(f"Unknown defense: {defense_type}")
        return self.defenses[defense_type](images, **params)

    def jpeg_compression(self, images: torch.Tensor, quality: int = 50) -> torch.Tensor:
        batch_size = images.shape[0]
        results = []

        for i in range(batch_size):
            img_tensor = images[i]
            img_np = (img_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)

            buffer = io.BytesIO()
            img_pil.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            compressed = Image.open(buffer)

            compressed_np = np.array(compressed).astype(np.float32) / 255.0
            compressed_tensor = torch.from_numpy(compressed_np).permute(2, 0, 1).to(images.device)
            results.append(compressed_tensor)

        return torch.stack(results)

    def random_resize_padding(self, images: torch.Tensor,
                              min_scale: float = 0.8,
                              max_scale: float = 1.2,
                              pad_mode: str = "reflect") -> torch.Tensor:
        batch_size, channels, height, width = images.shape
        results = []

        for i in range(batch_size):
            scale = np.random.uniform(min_scale, max_scale)
            new_h = int(height * scale)
            new_w = int(width * scale)

            img = torch.nn.functional.interpolate(
                images[i:i+1], size=(new_h, new_w), mode='bilinear', align_corners=False
            )

            pad_top = (height - new_h) // 2
            pad_bottom = height - new_h - pad_top
            pad_left = (width - new_w) // 2
            pad_right = width - new_w - pad_left

            if new_h <= height and new_w <= width:
                padding = (pad_left, pad_right, pad_top, pad_bottom)
                padded = torch.nn.functional.pad(img, padding, mode=pad_mode)
            else:
                crop_top = (new_h - height) // 2
                crop_left = (new_w - width) // 2
                padded = img[:, :, crop_top:crop_top+height, crop_left:crop_left+width]

            results.append(padded.squeeze(0))

        return torch.stack(results)

    def bit_depth_reduction(self, images: torch.Tensor, bits: int = 4) -> torch.Tensor:
        levels = 2 ** bits - 1
        return torch.round(images * levels) / levels

    def median_filter(self, images: torch.Tensor, kernel_size: int = 3) -> torch.Tensor:
        batch_size = images.shape[0]
        results = []

        for i in range(batch_size):
            img_np = (images[i].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            img_pil = Image.fromarray(img_np)
            filtered = img_pil.filter(ImageFilter.MedianFilter(size=kernel_size))
            filtered_np = np.array(filtered).astype(np.float32) / 255.0
            filtered_tensor = torch.from_numpy(filtered_np).permute(2, 0, 1).to(images.device)
            results.append(filtered_tensor)

        return torch.stack(results)

    def wrap_model_with_defense(self, model: nn.Module, defense_type: str,
                                **defense_params) -> nn.Module:
        class DefendedModel(nn.Module):
            def __init__(self, base_model, defense_fn, defense_params):
                super().__init__()
                self.base_model = base_model
                self.defense_fn = defense_fn
                self.defense_params = defense_params

            def forward(self, x):
                x_denorm = denormalize_image_batch(x)
                x_defended = self.defense_fn(x_denorm, **self.defense_params)
                x_norm = normalize_image(x_defended)
                return self.base_model(x_norm)

        def denormalize_image_batch(tensor):
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
            return tensor * std + mean

        return DefendedModel(model, self.apply_defense, {"defense_type": defense_type, **defense_params})

    def list_defenses(self) -> Dict[str, Dict[str, Any]]:
        return {
            "jpeg_compression": {
                "name": "JPEG 压缩",
                "params": {"quality": {"type": "int", "default": 50, "min": 1, "max": 100}}
            },
            "random_resize_padding": {
                "name": "随机缩放+填充",
                "params": {
                    "min_scale": {"type": "float", "default": 0.8, "min": 0.5, "max": 1.0},
                    "max_scale": {"type": "float", "default": 1.2, "min": 1.0, "max": 1.5}
                }
            },
            "bit_depth_reduction": {
                "name": "位深度缩减",
                "params": {"bits": {"type": "int", "default": 4, "min": 1, "max": 8}}
            },
            "median_filter": {
                "name": "中值滤波",
                "params": {"kernel_size": {"type": "int", "default": 3, "min": 3, "max": 7}}
            }
        }
