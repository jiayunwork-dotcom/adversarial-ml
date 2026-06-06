import numpy as np
import torch
from typing import List, Dict, Any, Tuple, Callable
from tqdm import tqdm

from config import DEVICE
from .helpers import denormalize_image, normalize_image


class TransferabilityAnalyzer:
    def __init__(self, model_manager):
        self.model_manager = model_manager

    def compute_transferability_matrix(self, model_ids: List[str], dataset,
                                       attack_fn: Callable, attack_params: Dict[str, Any],
                                       progress_callback=None) -> Tuple[np.ndarray, List[str]]:
        n = len(model_ids)
        matrix = np.zeros((n, n))
        model_names = []

        for i, source_id in enumerate(model_ids):
            source_model_info = self.model_manager.get_model(source_id)
            model_names.append(source_model_info["name"])
            source_model = self.model_manager.load_model(source_id)
            source_model.eval()

            for j, target_id in enumerate(model_ids):
                target_model_info = self.model_manager.get_model(target_id)
                target_model = self.model_manager.load_model(target_id)
                target_model.eval()

                if i == j:
                    success_rate = self._evaluate_whitebox(
                        source_model, dataset, attack_fn, attack_params
                    )
                else:
                    success_rate = self._evaluate_transfer(
                        source_model, target_model, dataset, attack_fn, attack_params
                    )

                matrix[i, j] = success_rate
                if progress_callback:
                    progress_callback(i * n + j + 1, n * n)

                del target_model
                torch.cuda.empty_cache()

            del source_model
            torch.cuda.empty_cache()

        return matrix, model_names

    def _evaluate_whitebox(self, model, dataset, attack_fn, attack_params) -> float:
        correct_clean = 0
        success_attack = 0
        total = 0

        for images, labels in tqdm(dataset, desc="Whitebox attack", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.no_grad():
                clean_preds = model(normalize_image(images)).argmax(1)
            correct_mask = clean_preds == labels
            correct_clean += correct_mask.sum().item()
            total += len(labels)

            if correct_mask.sum() > 0:
                clean_correct_images = images[correct_mask]
                clean_correct_labels = labels[correct_mask]

                adv_images = attack_fn(
                    model, clean_correct_images, clean_correct_labels, **attack_params
                )

                with torch.no_grad():
                    adv_preds = model(normalize_image(adv_images)).argmax(1)

                success_attack += (adv_preds != clean_correct_labels).sum().item()

        return success_attack / max(correct_clean, 1) if correct_clean > 0 else 0.0

    def _evaluate_transfer(self, source_model, target_model, dataset,
                           attack_fn, attack_params) -> float:
        correct_clean = 0
        transfer_success = 0
        total = 0

        for images, labels in tqdm(dataset, desc="Transfer attack", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            with torch.no_grad():
                target_clean_preds = target_model(normalize_image(images)).argmax(1)
            correct_mask = target_clean_preds == labels
            correct_clean += correct_mask.sum().item()
            total += len(labels)

            if correct_mask.sum() > 0:
                clean_correct_images = images[correct_mask]
                clean_correct_labels = labels[correct_mask]

                adv_images = attack_fn(
                    source_model, clean_correct_images, clean_correct_labels, **attack_params
                )

                with torch.no_grad():
                    target_adv_preds = target_model(normalize_image(adv_images)).argmax(1)

                transfer_success += (target_adv_preds != clean_correct_labels).sum().item()

        return transfer_success / max(correct_clean, 1) if correct_clean > 0 else 0.0
