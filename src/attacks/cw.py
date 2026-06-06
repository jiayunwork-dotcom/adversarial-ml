from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

from .attack_base import Attack


class CarliniWagner(Attack):
    def __init__(self, model, epsilon: float = 8.0 / 255.0,
                 norm: str = "L2", targeted: bool = True,
                 c: float = 1e-3, kappa: float = 0.0,
                 max_iter: int = 1000, lr: float = 0.01,
                 binary_search_steps: int = 9,
                 initial_c: float = 1e-3):
        super().__init__(model, epsilon, norm, targeted)
        self.c = c
        self.kappa = kappa
        self.max_iter = max_iter
        self.lr = lr
        self.binary_search_steps = binary_search_steps
        self.initial_c = initial_c

    def _f_function(self, outputs: torch.Tensor, labels: torch.Tensor,
                    target_labels: Optional[torch.Tensor]) -> torch.Tensor:
        batch_size = outputs.shape[0]
        num_classes = outputs.shape[1]

        one_hot_labels = torch.eye(num_classes, device=self.device)[labels]
        real = torch.sum(one_hot_labels * outputs, dim=1)

        if self.targeted and target_labels is not None:
            one_hot_targets = torch.eye(num_classes, device=self.device)[target_labels]
            target = torch.sum(one_hot_targets * outputs, dim=1)
            other = torch.max((1 - one_hot_targets) * outputs - one_hot_targets * 1e4, dim=1)[0]
            return torch.clamp(target - other + self.kappa, min=0)
        else:
            other = torch.max((1 - one_hot_labels) * outputs - one_hot_labels * 1e4, dim=1)[0]
            return torch.clamp(other - real + self.kappa, min=0)

    def generate(self, images: torch.Tensor, labels: torch.Tensor,
                 target_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        batch_size = images.shape[0]

        if self.targeted and target_labels is None:
            target_labels = (labels + 1) % self.model._modules.get('fc', self.model._modules.get('classifier', nn.Linear(10, 10))).out_features
            target_labels = target_labels.to(self.device)
        elif target_labels is not None:
            target_labels = target_labels.clone().detach().to(self.device)

        lower_bound = torch.zeros(batch_size, device=self.device)
        upper_bound = torch.ones(batch_size, device=self.device) * 1e10
        c = torch.ones(batch_size, device=self.device) * self.initial_c

        best_adv = images.clone()
        best_l2 = torch.full((batch_size,), float("inf"), device=self.device)

        for _ in range(self.binary_search_steps):
            w = torch.zeros_like(images, requires_grad=True)
            optimizer = optim.Adam([w], lr=self.lr)

            for _ in range(self.max_iter):
                adv_images = 0.5 * (torch.tanh(w) + 1)
                delta = adv_images - images
                l2_loss = torch.sum(delta.view(batch_size, -1) ** 2, dim=1)

                outputs = self._model_forward(adv_images)
                f_loss = self._f_function(outputs, labels, target_labels)

                loss = torch.sum(l2_loss + c * f_loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    preds = outputs.argmax(1)
                    if self.targeted:
                        success = preds == target_labels
                    else:
                        success = preds != labels

                    for i in range(batch_size):
                        if success[i] and l2_loss[i] < best_l2[i]:
                            best_l2[i] = l2_loss[i]
                            best_adv[i] = adv_images[i]

            for i in range(batch_size):
                if best_l2[i] < float("inf"):
                    upper_bound[i] = min(upper_bound[i], c[i])
                    c[i] = (lower_bound[i] + upper_bound[i]) / 2
                else:
                    lower_bound[i] = max(lower_bound[i], c[i])
                    if upper_bound[i] < 1e9:
                        c[i] = (lower_bound[i] + upper_bound[i]) / 2
                    else:
                        c[i] *= 10

        return self._clamp(best_adv, images).detach()
