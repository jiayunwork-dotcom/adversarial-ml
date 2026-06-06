from typing import Optional

import torch
import torch.nn as nn

from .attack_base import Attack


class DeepFool(Attack):
    def __init__(self, model, epsilon: float = 8.0 / 255.0,
                 norm: str = "L2", targeted: bool = False,
                 max_iter: int = 50, overshoot: float = 1.02,
                 num_classes: Optional[int] = None):
        super().__init__(model, epsilon, norm, targeted)
        self.max_iter = max_iter
        self.overshoot = overshoot
        self.num_classes = num_classes

    def generate(self, images: torch.Tensor, labels: torch.Tensor,
                 target_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        images = images.clone().detach().to(self.device)
        batch_size = images.shape[0]

        adv_images = images.clone()
        labels = labels.clone().detach().to(self.device)

        with torch.no_grad():
            outputs = self._model_forward(adv_images)
            if self.num_classes is None:
                self.num_classes = outputs.shape[1]

        for i in range(batch_size):
            x = adv_images[i:i+1].clone().detach()
            original_label = labels[i].item()

            for _ in range(self.max_iter):
                x.requires_grad = True
                outputs = self._model_forward(x)

                current_pred = outputs.argmax(1).item()
                if current_pred != original_label and not self.targeted:
                    break
                if self.targeted and target_labels is not None and current_pred == target_labels[i].item():
                    break

                grads = []
                for k in range(self.num_classes):
                    self.model.zero_grad()
                    if x.grad is not None:
                        x.grad.zero_()
                    outputs[0, k].backward(retain_graph=True)
                    grads.append(x.grad.data.clone().view(-1))

                grads = torch.stack(grads)
                f_k = outputs[0].detach()

                f_0 = f_k[original_label]
                grad_0 = grads[original_label]

                w_k = grads - grad_0
                f_k_diff = f_k - f_0

                mask = torch.ones(self.num_classes, dtype=torch.bool, device=self.device)
                mask[original_label] = False
                if self.targeted and target_labels is not None:
                    target = target_labels[i].item()
                    mask = torch.zeros(self.num_classes, dtype=torch.bool, device=self.device)
                    mask[target] = True

                w_k = w_k[mask]
                f_k_diff = f_k_diff[mask]

                norm_w = torch.norm(w_k.view(w_k.shape[0], -1), p=2, dim=1) + 1e-8
                distances = torch.abs(f_k_diff) / norm_w

                min_idx = torch.argmin(distances)
                r_i = distances[min_idx] * w_k[min_idx] / norm_w[min_idx]

                x = x.detach() + (1 + self.overshoot) * r_i.view(x.shape)
                x = torch.clamp(x, 0.0, 1.0)

                with torch.no_grad():
                    delta = x - images[i:i+1]
                    delta_norm = torch.norm(delta.view(-1), p=2)
                    if self.norm == "L2" and delta_norm > self.epsilon:
                        delta = delta * self.epsilon / delta_norm
                        x = images[i:i+1] + delta
                        x = torch.clamp(x, 0.0, 1.0)
                        break

            adv_images[i] = x.squeeze(0)

        adv_images = self._clamp(adv_images, images)
        return adv_images.detach()
