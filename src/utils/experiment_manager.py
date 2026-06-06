import os
import time
from typing import Dict, Any, List, Optional

from config import EXPERIMENT_DIR
from .helpers import save_json, load_json, generate_id


class ExperimentManager:
    def __init__(self):
        self.experiment_file = os.path.join(EXPERIMENT_DIR, "experiments.json")
        if not os.path.exists(self.experiment_file):
            save_json([], self.experiment_file)

    def record_experiment(self, model_id: str, dataset_id: str, attack_method: str,
                          attack_params: Dict[str, Any], metrics: Dict[str, Any],
                          experiment_type: str = "single") -> str:
        experiments = self._load_experiments()
        exp_id = generate_id()
        experiment = {
            "id": exp_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_id": model_id,
            "dataset_id": dataset_id,
            "attack_method": attack_method,
            "attack_params": attack_params,
            "metrics": metrics,
            "experiment_type": experiment_type,
        }
        experiments.append(experiment)
        self._save_experiments(experiments)
        return exp_id

    def record_comparison_experiment(self, model_id: str, dataset_id: str,
                                     attack_methods: List[str],
                                     attack_params: Dict[str, Any],
                                     results: List[Dict[str, Any]]) -> str:
        experiments = self._load_experiments()
        exp_id = generate_id()
        experiment = {
            "id": exp_id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_id": model_id,
            "dataset_id": dataset_id,
            "attack_method": attack_methods,
            "attack_params": attack_params,
            "metrics": results,
            "experiment_type": "comparison",
        }
        experiments.append(experiment)
        self._save_experiments(experiments)
        return exp_id

    def list_experiments(self, filter_by: Optional[Dict[str, Any]] = None,
                         sort_by: str = "timestamp", reverse: bool = True) -> List[Dict[str, Any]]:
        experiments = self._load_experiments()
        if filter_by:
            for key, value in filter_by.items():
                experiments = [exp for exp in experiments if exp.get(key) == value]
        experiments.sort(key=lambda x: x.get(sort_by, ""), reverse=reverse)
        return experiments

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        experiments = self._load_experiments()
        for exp in experiments:
            if exp["id"] == exp_id:
                return exp
        return None

    def delete_experiment(self, exp_id: str) -> bool:
        experiments = self._load_experiments()
        original_len = len(experiments)
        experiments = [exp for exp in experiments if exp["id"] != exp_id]
        if len(experiments) < original_len:
            self._save_experiments(experiments)
            return True
        return False

    def compare_experiments(self, exp_ids: List[str]) -> List[Dict[str, Any]]:
        return [self.get_experiment(exp_id) for exp_id in exp_ids if self.get_experiment(exp_id)]

    def _load_experiments(self) -> List[Dict[str, Any]]:
        return load_json(self.experiment_file)

    def _save_experiments(self, experiments: List[Dict[str, Any]]) -> None:
        save_json(experiments, self.experiment_file)
