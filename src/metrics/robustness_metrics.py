from typing import Dict, Any, List, Tuple, Optional, Callable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import DEVICE
from src.utils.helpers import normalize_image, denormalize_image, compute_perturbation_metrics


class RobustnessMetrics:
    def __init__(self, model_manager):
        self.model_manager = model_manager

    def evaluate(self, model_id: str, dataloader: DataLoader,
                 attack_fn: Callable, attack_params: Dict[str, Any],
                 progress_callback: Optional[Callable] = None,
                 model: Optional[nn.Module] = None,
                 attack_model: Optional[nn.Module] = None) -> Dict[str, Any]:
        model_info = self.model_manager.get_model(model_id)
        if model is None:
            model = self.model_manager.load_model(model_id)
        if attack_model is None:
            attack_model = model
        model.eval()
        attack_model.eval()

        clean_correct = 0
        robust_correct = 0
        attack_success = 0
        total_samples = 0

        all_clean_preds = []
        all_adv_preds = []
        all_labels = []
        all_confidences_clean = []
        all_confidences_adv = []
        all_perts_linf = []
        all_perts_l2 = []

        class_clean_correct = {}
        class_robust_correct = {}
        class_total = {}

        total_batches = len(dataloader)

        for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Evaluating robustness")):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)
            batch_size = labels.size(0)

            with torch.no_grad():
                clean_outputs = model(normalize_image(images))
                clean_probs = torch.softmax(clean_outputs, dim=1)
                clean_preds = clean_outputs.argmax(1)
                clean_confidence = clean_probs.gather(1, clean_preds.unsqueeze(1)).squeeze(1)

            clean_mask = clean_preds == labels
            clean_correct += clean_mask.sum().item()
            total_samples += batch_size

            for i in range(batch_size):
                label = labels[i].item()
                class_total[label] = class_total.get(label, 0) + 1
                if clean_mask[i].item():
                    class_clean_correct[label] = class_clean_correct.get(label, 0) + 1

            if clean_mask.sum() > 0:
                clean_images = images[clean_mask]
                clean_labels = labels[clean_mask]

                adv_images = attack_fn(attack_model, clean_images, clean_labels, **attack_params)

                with torch.no_grad():
                    adv_outputs = model(normalize_image(adv_images))
                    adv_probs = torch.softmax(adv_outputs, dim=1)
                    adv_preds = adv_outputs.argmax(1)
                    adv_confidence = adv_probs.gather(1, clean_labels.unsqueeze(1)).squeeze(1)

                adv_mask = adv_preds == clean_labels
                robust_correct += adv_mask.sum().item()
                attack_success += (~adv_mask).sum().item()

                for i in range(adv_images.size(0)):
                    label = clean_labels[i].item()
                    if adv_mask[i].item():
                        class_robust_correct[label] = class_robust_correct.get(label, 0) + 1

                    linf, l2 = compute_perturbation_metrics(
                        clean_images[i:i+1], adv_images[i:i+1]
                    )
                    all_perts_linf.append(linf)
                    all_perts_l2.append(l2)

                all_adv_preds.extend(adv_preds.cpu().numpy())
                all_confidences_adv.extend(adv_confidence.cpu().numpy())

            all_clean_preds.extend(clean_preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences_clean.extend(clean_confidence.cpu().numpy())

            if progress_callback:
                progress_callback(batch_idx + 1, total_batches)

        clean_acc = 100.0 * clean_correct / total_samples if total_samples > 0 else 0.0
        robust_acc = 100.0 * robust_correct / max(clean_correct, 1) if clean_correct > 0 else 0.0
        attack_success_rate = 100.0 * attack_success / max(clean_correct, 1) if clean_correct > 0 else 0.0

        avg_pert_linf = np.mean(all_perts_linf) if all_perts_linf else 0.0
        avg_pert_l2 = np.mean(all_perts_l2) if all_perts_l2 else 0.0

        per_class_robustness = {}
        for cls in class_total:
            total_cls = class_clean_correct.get(cls, 0)
            robust_cls = class_robust_correct.get(cls, 0)
            per_class_robustness[cls] = 100.0 * robust_cls / max(total_cls, 1) if total_cls > 0 else 0.0

        clean_labels_torch = torch.tensor(all_labels)
        clean_preds_torch = torch.tensor(all_clean_preds)
        clean_correct_mask = clean_preds_torch == clean_labels_torch

        if clean_correct_mask.sum() > 0 and len(all_confidences_adv) > 0:
            conf_clean = torch.tensor(all_confidences_clean)[clean_correct_mask].numpy()
            conf_adv = torch.tensor(all_confidences_adv).numpy()
            if len(conf_clean) > 0 and len(conf_adv) > 0:
                min_len = min(len(conf_clean), len(conf_adv))
                confidence_drop = 100.0 * np.mean(conf_clean[:min_len] - conf_adv[:min_len])
            else:
                confidence_drop = 0.0
        else:
            confidence_drop = 0.0

        metrics = {
            "clean_accuracy": clean_acc,
            "robust_accuracy": robust_acc,
            "attack_success_rate": attack_success_rate,
            "average_perturbation_linf": avg_pert_linf,
            "average_perturbation_l2": avg_pert_l2,
            "per_class_robustness": per_class_robustness,
            "confidence_drop": confidence_drop,
            "total_samples": total_samples,
            "clean_correct": clean_correct,
            "robust_correct": robust_correct,
            "attack_success": attack_success,
        }

        return metrics

    def evaluate_defense_comparison(self, model_id: str, defense_model_id: Optional[str],
                                    dataloader: DataLoader, attack_fn: Callable,
                                    attack_params: Dict[str, Any]) -> Dict[str, Any]:
        original_metrics = self.evaluate(model_id, dataloader, attack_fn, attack_params)

        if defense_model_id:
            defense_metrics = self.evaluate(defense_model_id, dataloader, attack_fn, attack_params)
        else:
            defense_metrics = None

        comparison = {
            "original": original_metrics,
            "defense": defense_metrics,
            "improvements": {}
        }

        if defense_metrics:
            comparison["improvements"] = {
                "clean_accuracy": defense_metrics["clean_accuracy"] - original_metrics["clean_accuracy"],
                "robust_accuracy": defense_metrics["robust_accuracy"] - original_metrics["robust_accuracy"],
                "attack_success_rate": original_metrics["attack_success_rate"] - defense_metrics["attack_success_rate"],
            }

        return comparison
