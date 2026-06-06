from typing import Dict, Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from config import DEVICE

try:
    from skimage.metrics import structural_similarity as ssim
    from skimage.metrics import peak_signal_noise_ratio as psnr
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    print("scikit-image not available. Install with: pip install scikit-image")
    ssim = None
    psnr = None


class QualityMetrics:
    def __init__(self):
        self.lpips_model = None
        self._init_lpips()

    def _init_lpips(self):
        try:
            import lpips
            self.lpips_model = lpips.LPIPS(net='vgg').to(DEVICE)
        except ImportError:
            print("LPIPS not available. Install with: pip install lpips")
            self.lpips_model = None

    def compute_all(self, original: torch.Tensor, adversarial: torch.Tensor
                    ) -> Dict[str, Any]:
        ssim_val = self.compute_ssim(original, adversarial)
        psnr_val = self.compute_psnr(original, adversarial)
        lpips_val = self.compute_lpips(original, adversarial)

        low_quality_warning = (ssim_val >= 0 and ssim_val < 0.9)

        return {
            "ssim": ssim_val,
            "psnr": psnr_val,
            "lpips": lpips_val,
            "low_quality_warning": low_quality_warning,
        }

    def compute_ssim(self, original: torch.Tensor, adversarial: torch.Tensor) -> float:
        if not SKIMAGE_AVAILABLE or ssim is None:
            return -1.0
        orig_np = self._tensor_to_numpy(original)
        adv_np = self._tensor_to_numpy(adversarial)

        if orig_np.shape[-1] == 3:
            ssim_values = []
            for c in range(3):
                ssim_c = ssim(orig_np[..., c], adv_np[..., c], data_range=1.0)
                ssim_values.append(ssim_c)
            return float(np.mean(ssim_values))
        else:
            return float(ssim(orig_np, adv_np, data_range=1.0))

    def compute_psnr(self, original: torch.Tensor, adversarial: torch.Tensor) -> float:
        if not SKIMAGE_AVAILABLE or psnr is None:
            return -1.0
        orig_np = self._tensor_to_numpy(original)
        adv_np = self._tensor_to_numpy(adversarial)
        return float(psnr(orig_np, adv_np, data_range=1.0))

    def compute_lpips(self, original: torch.Tensor, adversarial: torch.Tensor) -> float:
        if self.lpips_model is None:
            return -1.0

        orig_norm = original * 2 - 1
        adv_norm = adversarial * 2 - 1

        with torch.no_grad():
            lpips_val = self.lpips_model(orig_norm, adv_norm)

        return float(lpips_val.item())

    def _tensor_to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        if tensor.dim() == 4:
            tensor = tensor.squeeze(0)
        img = tensor.permute(1, 2, 0).cpu().numpy()
        return np.clip(img, 0.0, 1.0)
