from typing import Optional

import torch

from .attack_base import Attack


class PGD(Attack):
    def __init__(self, model, epsilon: float = 8.0 / 255.0,
                 alpha: float = 2.0 / 255.0, iterations: int = 20,
                 norm: str = "Linf", targeted: bool = False,
                 random_start: bool = True):
        super().__init__(model, epsilon, norm, targeted)
        self.alpha = alpha
        self.iterations = iterations
        self.random_start = random_start

    def generate(self, images: torch.Tensor, labels: torch.Tensor,
                 target_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        if target_labels is not None:
            target_labels = target_labels.clone().detach().to(self.device)

        adv_images = images.clone()

        if self.random_start:
            if self.norm == "Linf":
                delta = torch.empty_like(adv_images).uniform_(-self.epsilon, self.epsilon)
            elif self.norm == "L2":
                delta = torch.randn_like(adv_images)
                delta_flat = delta.view(delta.shape[0], -1)
                delta_norm = torch.norm(delta_flat, p=2, dim=1).view(-1, 1, 1, 1)
                delta = delta * self.epsilon / (delta_norm + 1e-10)
            else:
                delta = torch.empty_like(adv_images).uniform_(-self.epsilon, self.epsilon)
            adv_images = adv_images + delta
            adv_images = self._clamp(adv_images, images)

        for _ in range(self.iterations):
            adv_images.requires_grad = True

            outputs = self._model_forward(adv_images)
            loss = self._get_loss(outputs, labels, target_labels)

            self.model.zero_grad()
            loss.backward()

            grad = adv_images.grad.data

            if self.norm == "Linf":
                adv_images = adv_images.detach() + self.alpha * grad.sign()
            elif self.norm == "L2":
                grad_flat = grad.view(grad.shape[0], -1)
                grad_norm = torch.norm(grad_flat, p=2, dim=1).view(-1, 1, 1, 1)
                adv_images = adv_images.detach() + self.alpha * grad / (grad_norm + 1e-10)

            adv_images = self._project(adv_images, images)

        return adv_images.detach()
