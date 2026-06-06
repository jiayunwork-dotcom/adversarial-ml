from typing import Optional

import torch
import torch.nn as nn

from .attack_base import Attack


class SquareAttack(Attack):
    def __init__(self, model, epsilon: float = 8.0 / 255.0,
                 norm: str = "Linf", targeted: bool = False,
                 max_queries: int = 5000, p_init: float = 0.8):
        super().__init__(model, epsilon, norm, targeted)
        self.max_queries = max_queries
        self.p_init = p_init

    def generate(self, images: torch.Tensor, labels: torch.Tensor,
                 target_labels: Optional[torch.Tensor] = None) -> torch.Tensor:
        images = images.clone().detach().to(self.device)
        labels = labels.clone().detach().to(self.device)
        batch_size = images.shape[0]

        if self.targeted and target_labels is not None:
            target_labels = target_labels.clone().detach().to(self.device)

        best_adv = images.clone()
        best_logits = self._get_logits(images)
        best_score = self._get_score(best_logits, labels, target_labels)
        queries = torch.ones(batch_size, device=self.device)

        h, w = images.shape[-2], images.shape[-1]
        n_features = h * w

        for i in range(self.max_queries):
            if i % 100 == 0:
                with torch.no_grad():
                    preds = best_logits.argmax(1)
                    if self.targeted and target_labels is not None:
                        success = preds == target_labels
                    else:
                        success = preds != labels
                    if success.all():
                        break

            p = self._get_p(i, n_features)
            s = max(int(round(torch.sqrt(torch.tensor(p * n_features)).item())), 1)

            for b in range(batch_size):
                if success[b] if i >= 100 else False:
                    continue

                vh = torch.randint(0, h - s + 1, (1,)).item()
                vw = torch.randint(0, w - s + 1, (1,)).item()

                delta = torch.zeros_like(images[b])
                if self.norm == "Linf":
                    sign = 2 * torch.randint(0, 2, (1,)).item() - 1
                    delta[:, vh:vh+s, vw:vw+s] = sign * self.epsilon
                else:
                    delta[:, vh:vh+s, vw:vw+s] = torch.randn(3, s, s, device=self.device)
                    norm = torch.norm(delta.view(-1), p=2)
                    if norm > 0:
                        delta = delta * self.epsilon * 0.1 / norm

                new_adv = best_adv[b:b+1] + delta
                new_adv = self._clamp(new_adv, images[b:b+1])

                with torch.no_grad():
                    new_logits = self._get_logits(new_adv)
                    new_score = self._get_score(new_logits, labels[b:b+1],
                                               target_labels[b:b+1] if target_labels is not None else None)

                if new_score.item() > best_score[b].item():
                    best_adv[b] = new_adv.squeeze(0)
                    best_logits[b] = new_logits.squeeze(0)
                    best_score[b] = new_score.squeeze(0)

                queries[b] += 1

        return best_adv.detach()

    def _get_logits(self, images: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self._model_forward(images)

    def _get_score(self, logits: torch.Tensor, labels: torch.Tensor,
                   target_labels: Optional[torch.Tensor]) -> torch.Tensor:
        if self.targeted and target_labels is not None:
            target_logits = logits.gather(1, target_labels.unsqueeze(1)).squeeze(1)
            max_other = logits.scatter(1, target_labels.unsqueeze(1), float("-inf")).max(1)[0]
            return target_logits - max_other
        else:
            label_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            max_other = logits.scatter(1, labels.unsqueeze(1), float("-inf")).max(1)[0]
            return max_other - label_logits

    def _get_p(self, iteration: int, n_features: int) -> float:
        return max(self.p_init * (1 - iteration / self.max_queries), 1.0 / n_features)
