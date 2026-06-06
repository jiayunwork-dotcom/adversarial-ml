from typing import Optional, List

import torch
import torch.nn as nn

from .attack_base import Attack
from .pgd import PGD


class APGD(Attack):
    def __init__(self, model, epsilon: float = 8.0 / 255.0,
                 norm: str = "Linf", targeted: bool = False,
                 max_iter: int = 100, rho: float = 0.75,
                 alpha: float = 0.01):
        super().__init__(model, epsilon, norm, targeted)
        self.max_iter = max_iter
        self.rho = rho
        self.alpha = alpha

    def generate(self, images: torch.Tensor, labels: torch.Tensor,
                 target_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        batch_size = images.shape[0]

        if self.targeted and target_labels is not None:
            target_labels = target_labels.clone().detach().to(self.device)

        best_adv = images.clone()
        best_loss = torch.full((batch_size,), float("-inf"), device=self.device)

        adv_images = images.clone()
        if self.norm == "Linf":
            delta = torch.empty_like(adv_images).uniform_(-self.epsilon, self.epsilon)
        else:
            delta = torch.randn_like(adv_images)
            delta_flat = delta.view(delta.shape[0], -1)
            delta_norm = torch.norm(delta_flat, p=2, dim=1).view(-1, 1, 1, 1)
            delta = delta * self.epsilon / (delta_norm + 1e-10)
        adv_images = images + delta
        adv_images = self._clamp(adv_images, images)

        step_size = self.alpha * self.epsilon
        momentum = torch.zeros_like(adv_images)

        for i in range(self.max_iter):
            adv_images.requires_grad = True
            outputs = self._model_forward(adv_images)

            with torch.no_grad():
                if self.targeted and target_labels is not None:
                    loss = -nn.CrossEntropyLoss(reduction="none")(outputs, target_labels)
                else:
                    loss = nn.CrossEntropyLoss(reduction="none")(outputs, labels)

            improve_mask = loss > best_loss
            best_adv[improve_mask] = adv_images.detach()[improve_mask]
            best_loss = torch.max(best_loss, loss)

            if i % max(1, self.max_iter // 5) == 0 and i > 0:
                if self.norm == "Linf":
                    delta = torch.empty_like(adv_images).uniform_(-self.epsilon, self.epsilon)
                else:
                    delta = torch.randn_like(adv_images)
                    delta_flat = delta.view(delta.shape[0], -1)
                    delta_norm = torch.norm(delta_flat, p=2, dim=1).view(-1, 1, 1, 1)
                    delta = delta * self.epsilon / (delta_norm + 1e-10)
                adv_images = images + delta
                adv_images = self._clamp(adv_images, images)
                step_size *= 0.9
                momentum = torch.zeros_like(adv_images)

            self.model.zero_grad()
            if adv_images.grad is not None:
                adv_images.grad.zero_()

            total_loss = loss.sum()
            total_loss.backward()

            grad = adv_images.grad.data
            grad_norm = torch.norm(grad.view(batch_size, -1), p=2, dim=1).view(-1, 1, 1, 1) + 1e-8
            grad = grad / grad_norm

            momentum = 0.9 * momentum + grad

            if self.norm == "Linf":
                adv_images = adv_images.detach() + step_size * momentum.sign()
            else:
                momentum_norm = torch.norm(momentum.view(batch_size, -1), p=2, dim=1).view(-1, 1, 1, 1) + 1e-8
                adv_images = adv_images.detach() + step_size * momentum / momentum_norm

            adv_images = self._project(adv_images, images)

        return best_adv.detach()


class FAB(Attack):
    def __init__(self, model, epsilon: float = 8.0 / 255.0,
                 norm: str = "Linf", targeted: bool = False,
                 max_iter: int = 100, restarts: int = 1):
        super().__init__(model, epsilon, norm, targeted)
        self.max_iter = max_iter
        self.restarts = restarts

    def generate(self, images: torch.Tensor, labels: torch.Tensor,
                 target_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        batch_size = images.shape[0]

        best_adv = images.clone()
        best_dist = torch.full((batch_size,), float("inf"), device=self.device)

        for restart in range(self.restarts):
            adv_images = images.clone()
            if restart > 0:
                if self.norm == "Linf":
                    delta = torch.empty_like(adv_images).uniform_(-self.epsilon * 0.1, self.epsilon * 0.1)
                else:
                    delta = torch.randn_like(adv_images) * 0.01
                adv_images = adv_images + delta
                adv_images = self._clamp(adv_images, images)

            for _ in range(self.max_iter):
                adv_images.requires_grad = True
                outputs = self._model_forward(adv_images)

                pred = outputs.argmax(1)
                if self.targeted and target_labels is not None:
                    success = pred == target_labels
                else:
                    success = pred != labels

                if success.all():
                    break

                self.model.zero_grad()
                if adv_images.grad is not None:
                    adv_images.grad.zero_()

                if self.targeted and target_labels is not None:
                    loss = nn.CrossEntropyLoss()(outputs, target_labels)
                else:
                    loss = nn.CrossEntropyLoss()(outputs, labels)

                loss.backward()
                grad = adv_images.grad.data

                with torch.no_grad():
                    if self.norm == "Linf":
                        grad_sign = grad.sign()
                        step_size = self.epsilon / self.max_iter
                        adv_images = adv_images.detach() - step_size * grad_sign
                    else:
                        grad_norm = torch.norm(grad.view(batch_size, -1), p=2, dim=1).view(-1, 1, 1, 1) + 1e-8
                        step_size = self.epsilon / (2 * self.max_iter)
                        adv_images = adv_images.detach() - step_size * grad / grad_norm

                    adv_images = self._clamp(adv_images, images)

                    delta = adv_images - images
                    if self.norm == "Linf":
                        dist = torch.max(torch.abs(delta.view(batch_size, -1)), dim=1)[0]
                    else:
                        dist = torch.norm(delta.view(batch_size, -1), p=2, dim=1)

                    update_mask = (dist < best_dist) & success
                    best_adv[update_mask] = adv_images[update_mask]
                    best_dist[update_mask] = dist[update_mask]

        return best_adv.detach()


class AutoAttack(Attack):
    def __init__(self, model, epsilon: float = 8.0 / 255.0,
                 norm: str = "Linf", targeted: bool = False,
                 attacks: Optional[List[str]] = None,
                 verbose: bool = False):
        super().__init__(model, epsilon, norm, targeted)
        self.verbose = verbose
        self.attacks = attacks or ["apgd_ce", "apgd_t", "fab", "square"]

    def generate(self, images: torch.Tensor, labels: torch.Tensor,
                 target_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        batch_size = images.shape[0]

        best_adv = images.clone()
        best_success = torch.zeros(batch_size, dtype=torch.bool, device=self.device)

        with torch.no_grad():
            initial_preds = self._model_forward(images).argmax(1)
            clean_correct = initial_preds == labels

        for attack_name in self.attacks:
            if clean_correct.sum() == 0 or best_success.all():
                break

            remaining_mask = clean_correct & ~best_success

            if remaining_mask.sum() == 0:
                break

            target_images = images[remaining_mask]
            target_labels_img = labels[remaining_mask]

            if target_labels is not None:
                target_t = target_labels[remaining_mask]
            else:
                target_t = None

            attack = self._get_attack(attack_name)
            adv_samples = attack.generate(target_images, target_labels_img, target_t)

            with torch.no_grad():
                adv_preds = self._model_forward(adv_samples).argmax(1)
                if self.targeted and target_labels is not None:
                    success = adv_preds == target_labels[remaining_mask]
                else:
                    success = adv_preds != target_labels_img

            remaining_indices = torch.where(remaining_mask)[0]
            for i, success_flag in enumerate(success):
                if success_flag and not best_success[remaining_indices[i]]:
                    best_adv[remaining_indices[i]] = adv_samples[i]
                    best_success[remaining_indices[i]] = True

            if self.verbose:
                print(f"Attack {attack_name}: {success.sum().item()}/{len(success)} successful")

        for i in range(batch_size):
            if not best_success[i]:
                pgd = PGD(self.model, epsilon=self.epsilon, norm=self.norm,
                         targeted=self.targeted, iterations=40, random_start=True)
                best_adv[i] = pgd.generate(images[i:i+1], labels[i:i+1],
                                          target_labels[i:i+1] if target_labels is not None else None)

        return best_adv.detach()

    def _get_attack(self, attack_name: str) -> Attack:
        if attack_name == "apgd_ce":
            return APGD(self.model, epsilon=self.epsilon, norm=self.norm,
                       targeted=self.targeted, max_iter=100)
        elif attack_name == "apgd_t":
            return APGD(self.model, epsilon=self.epsilon, norm=self.norm,
                       targeted=True, max_iter=100)
        elif attack_name == "fab":
            return FAB(self.model, epsilon=self.epsilon, norm=self.norm,
                      targeted=self.targeted, max_iter=100)
        elif attack_name == "square":
            from .square_attack import SquareAttack
            return SquareAttack(self.model, epsilon=self.epsilon, norm=self.norm,
                               targeted=self.targeted, max_queries=5000)
        else:
            return PGD(self.model, epsilon=self.epsilon, norm=self.norm,
                      targeted=self.targeted, iterations=20)
