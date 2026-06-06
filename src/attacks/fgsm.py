from typing import Optional

import torch

from .attack_base import Attack


class FGSM(Attack):
    def __init__(self, model, epsilon: float = 8.0 / 255.0,
                 norm: str = "Linf", targeted: bool = False):
        super().__init__(model, epsilon, norm, targeted)

    def generate(self, images: torch.Tensor, labels: torch.Tensor,
                 target_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        if target_labels is not None:
            target_labels = target_labels.clone().detach().to(self.device)

        images.requires_grad = True

        outputs = self._model_forward(images)
        loss = self._get_loss(outputs, labels, target_labels)

        self.model.zero_grad()
        loss.backward()

        grad = images.grad.data

        if self.norm == "Linf":
            perturbation = self.epsilon * grad.sign()
        elif self.norm == "L2":
            grad_flat = grad.view(grad.shape[0], -1)
            grad_norm = torch.norm(grad_flat, p=2, dim=1).view(-1, 1, 1, 1)
            perturbation = self.epsilon * grad / (grad_norm + 1e-10)
        else:
            perturbation = self.epsilon * grad.sign()

        if self.targeted:
            adv_images = images - perturbation
        else:
            adv_images = images + perturbation

        adv_images = self._clamp(adv_images, images)
        return adv_images.detach()
