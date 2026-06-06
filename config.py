import os
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(BASE_DIR, "storage", "models")
DATASET_DIR = os.path.join(BASE_DIR, "storage", "datasets")
RESULT_DIR = os.path.join(BASE_DIR, "storage", "results")
PRESET_DIR = os.path.join(BASE_DIR, "storage", "presets")
EXPERIMENT_DIR = os.path.join(BASE_DIR, "storage", "experiments")
TEMP_DIR = os.path.join(BASE_DIR, "storage", "temp")

for dir_path in [MODEL_DIR, DATASET_DIR, RESULT_DIR, PRESET_DIR, EXPERIMENT_DIR, TEMP_DIR]:
    os.makedirs(dir_path, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SUPPORTED_ARCHITECTURES = [
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "vgg11", "vgg13", "vgg16", "vgg19",
    "mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small",
    "densenet121", "densenet161", "densenet169", "densenet201",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
    "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
    "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32",
]

DEFAULT_EPSILON = 8.0 / 255.0
DEFAULT_ALPHA = 2.0 / 255.0
DEFAULT_ITERATIONS = 20

PRESET_ATTACK_CONFIGS = {
    "weak_fgsm": {
        "name": "弱攻击 FGSM",
        "attack": "fgsm",
        "params": {"epsilon": 4.0 / 255.0, "norm": "Linf", "targeted": False}
    },
    "strong_autoattack": {
        "name": "强攻击 AutoAttack",
        "attack": "autoattack",
        "params": {"epsilon": 8.0 / 255.0, "norm": "Linf"}
    },
    "pgd_standard": {
        "name": "标准 PGD 攻击",
        "attack": "pgd",
        "params": {"epsilon": 8.0 / 255.0, "alpha": 2.0 / 255.0, "iterations": 20, "norm": "Linf", "random_start": True}
    }
}
