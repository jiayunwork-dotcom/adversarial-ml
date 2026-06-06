import os
import time
import zipfile
import shutil
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from collections import Counter

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from config import DATASET_DIR, TEMP_DIR, DEVICE
from src.utils.helpers import generate_id, save_json, load_json, get_imagenet_labels


@dataclass
class DatasetInfo:
    id: str
    name: str
    path: str
    total_images: int
    class_distribution: Dict[str, int]
    image_size_range: Tuple[int, int, int, int]
    upload_time: str
    is_builtin: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "path": self.path,
            "total_images": self.total_images,
            "class_distribution": self.class_distribution,
            "image_size_range": list(self.image_size_range),
            "upload_time": self.upload_time,
            "is_builtin": self.is_builtin,
        }


class ImageFolderDataset(Dataset):
    def __init__(self, root_dir: str, transform=None, input_size: int = 224):
        self.root_dir = root_dir
        self.transform = transform
        self.input_size = input_size
        self.image_paths = []
        self.labels = []
        self.class_to_idx = {}

        class_dirs = sorted([d for d in os.listdir(root_dir)
                             if os.path.isdir(os.path.join(root_dir, d))])

        if not class_dirs:
            for img_file in os.listdir(root_dir):
                if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    self.image_paths.append(os.path.join(root_dir, img_file))
                    self.labels.append(0)
            self.class_to_idx = {"unknown": 0}
        else:
            for idx, class_name in enumerate(class_dirs):
                self.class_to_idx[class_name] = idx
                class_path = os.path.join(root_dir, class_name)
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                        self.image_paths.append(os.path.join(class_path, img_file))
                        self.labels.append(idx)

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
            ])

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class SingleImageDataset(Dataset):
    def __init__(self, image: Image.Image, transform=None, input_size: int = 224):
        self.image = image
        self.transform = transform
        self.input_size = input_size

        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((input_size, input_size)),
                transforms.ToTensor(),
            ])

    def __len__(self) -> int:
        return 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = self.transform(self.image)
        return image, 0


class DatasetManager:
    def __init__(self):
        self.datasets_file = os.path.join(DATASET_DIR, "datasets.json")
        self._datasets: Dict[str, DatasetInfo] = {}
        self._load_datasets_from_file()
        self._ensure_builtin_dataset()

    def _load_datasets_from_file(self) -> None:
        data = load_json(self.datasets_file)
        for ds_id, ds_data in data.items():
            ds_data["image_size_range"] = tuple(ds_data["image_size_range"])
            self._datasets[ds_id] = DatasetInfo(**ds_data)

    def _save_datasets_to_file(self) -> None:
        data = {did: d.to_dict() for did, d in self._datasets.items()}
        save_json(data, self.datasets_file)

    def _ensure_builtin_dataset(self) -> None:
        builtin_id = "builtin_imagenet_100"
        builtin_path = os.path.join(DATASET_DIR, builtin_id)
        version_file = os.path.join(builtin_path, ".dataset_version")
        current_version = "2.0"

        needs_rebuild = False
        if builtin_id not in self._datasets:
            needs_rebuild = True
        elif not os.path.exists(version_file):
            needs_rebuild = True
        else:
            try:
                with open(version_file, "r") as f:
                    saved_version = f.read().strip()
                if saved_version != current_version:
                    needs_rebuild = True
            except:
                needs_rebuild = True

        if needs_rebuild:
            import shutil
            if os.path.exists(builtin_path):
                shutil.rmtree(builtin_path)
            if builtin_id in self._datasets:
                del self._datasets[builtin_id]

            os.makedirs(builtin_path, exist_ok=True)
            self._create_sample_dataset(builtin_path)

            with open(version_file, "w") as f:
                f.write(current_version)

            class_dist = self._analyze_class_distribution(builtin_path)
            size_range = self._analyze_image_sizes(builtin_path)

            ds_info = DatasetInfo(
                id=builtin_id,
                name="ImageNet 验证子集 (100张, CIFAR10)",
                path=builtin_path,
                total_images=100,
                class_distribution=class_dist,
                image_size_range=size_range,
                upload_time=time.strftime("%Y-%m-%d %H:%M:%S"),
                is_builtin=True,
            )
            self._datasets[builtin_id] = ds_info
            self._save_datasets_to_file()

    def _create_sample_dataset(self, path: str) -> None:
        labels = get_imagenet_labels()
        np.random.seed(42)
        selected_classes = np.random.choice(100, 10, replace=False)

        try:
            from torchvision import datasets, transforms
            import torchvision.transforms.functional as TF

            cifar10 = datasets.CIFAR10(root=os.path.join(TEMP_DIR, "cifar10"),
                                       train=False, download=True)
            cifar_images = cifar10.data
            cifar_labels = cifar10.targets

            cifar_class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                                 'dog', 'frog', 'horse', 'ship', 'truck']

            for i, class_idx in enumerate(selected_classes):
                class_name = labels[class_idx]
                class_dir = os.path.join(path, f"{class_idx:04d}_{class_name}")
                os.makedirs(class_dir, exist_ok=True)

                cifar_class_idx = i % 10
                class_mask = np.array(cifar_labels) == cifar_class_idx
                class_images = cifar_images[class_mask]

                if len(class_images) >= 10:
                    indices = np.random.choice(len(class_images), 10, replace=False)
                else:
                    indices = np.random.choice(len(class_images), 10, replace=True)

                for j, idx in enumerate(indices):
                    img_array = class_images[idx]
                    img = Image.fromarray(img_array)
                    img = img.resize((224, 224), Image.BILINEAR)
                    img.save(os.path.join(class_dir, f"image_{j:02d}.jpg"))

        except Exception as e:
            print(f"Warning: Could not download CIFAR10, using structured patterns instead: {e}")
            for i, class_idx in enumerate(selected_classes):
                class_name = labels[class_idx]
                class_dir = os.path.join(path, f"{class_idx:04d}_{class_name}")
                os.makedirs(class_dir, exist_ok=True)

                for j in range(10):
                    img_array = self._create_structured_image(i, j)
                    img = Image.fromarray(img_array)
                    img.save(os.path.join(class_dir, f"image_{j:02d}.jpg"))

    def _create_structured_image(self, class_idx: int, img_idx: int) -> np.ndarray:
        img = np.zeros((224, 224, 3), dtype=np.uint8)
        base_color = [(class_idx * 25) % 255, (class_idx * 50) % 255, (class_idx * 75) % 255]

        for y in range(224):
            for x in range(224):
                pattern = (x // 32 + y // 32 + img_idx) % 3
                if pattern == 0:
                    img[y, x] = base_color
                elif pattern == 1:
                    img[y, x] = [min(255, c + 50) for c in base_color]
                else:
                    img[y, x] = [max(0, c - 50) for c in base_color]

        for k in range(5 + img_idx):
            cx = (img_idx * 37 + k * 53) % 200 + 12
            cy = (k * 41 + img_idx * 29) % 200 + 12
            r = 8 + (k % 3) * 4
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    if dx * dx + dy * dy <= r * r:
                        ny, nx = cy + dy, cx + dx
                        if 0 <= ny < 224 and 0 <= nx < 224:
                            img[ny, nx] = [255 - base_color[0], 255 - base_color[1], 255 - base_color[2]]

        return img

    def upload_dataset(self, zip_path: str, name: str) -> DatasetInfo:
        dataset_id = generate_id()
        dataset_path = os.path.join(DATASET_DIR, dataset_id)
        os.makedirs(dataset_path, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(dataset_path)

        extracted_dirs = os.listdir(dataset_path)
        if len(extracted_dirs) == 1 and os.path.isdir(os.path.join(dataset_path, extracted_dirs[0])):
            inner_dir = os.path.join(dataset_path, extracted_dirs[0])
            for item in os.listdir(inner_dir):
                shutil.move(os.path.join(inner_dir, item), dataset_path)
            os.rmdir(inner_dir)

        class_dist = self._analyze_class_distribution(dataset_path)
        size_range = self._analyze_image_sizes(dataset_path)
        total_images = sum(class_dist.values())

        ds_info = DatasetInfo(
            id=dataset_id,
            name=name,
            path=dataset_path,
            total_images=total_images,
            class_distribution=class_dist,
            image_size_range=size_range,
            upload_time=time.strftime("%Y-%m-%d %H:%M:%S"),
            is_builtin=False,
        )

        self._datasets[dataset_id] = ds_info
        self._save_datasets_to_file()

        return ds_info

    def _analyze_class_distribution(self, path: str) -> Dict[str, int]:
        class_dist = {}
        class_dirs = sorted([d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))])

        if class_dirs:
            for class_dir in class_dirs:
                class_path = os.path.join(path, class_dir)
                count = len([f for f in os.listdir(class_path)
                             if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])
                class_dist[class_dir] = count
        else:
            count = len([f for f in os.listdir(path)
                         if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif'))])
            class_dist["images"] = count

        return class_dist

    def _analyze_image_sizes(self, path: str) -> Tuple[int, int, int, int]:
        min_w, min_h = float("inf"), float("inf")
        max_w, max_h = 0, 0

        for root, _, files in os.walk(path):
            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')):
                    try:
                        img = Image.open(os.path.join(root, file))
                        w, h = img.size
                        min_w = min(min_w, w)
                        min_h = min(min_h, h)
                        max_w = max(max_w, w)
                        max_h = max(max_h, h)
                    except:
                        continue

        if min_w == float("inf"):
            min_w, min_h, max_w, max_h = 224, 224, 224, 224

        return (min_w, min_h, max_w, max_h)

    def get_dataloader(self, dataset_id: str, batch_size: int = 8,
                       input_size: int = 224, shuffle: bool = False) -> DataLoader:
        ds_info = self._datasets.get(dataset_id)
        if not ds_info:
            raise ValueError(f"Dataset {dataset_id} not found")

        dataset = ImageFolderDataset(ds_info.path, input_size=input_size)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

    def get_single_image_dataloader(self, image: Image.Image,
                                    input_size: int = 224) -> DataLoader:
        dataset = SingleImageDataset(image, input_size=input_size)
        return DataLoader(dataset, batch_size=1, shuffle=False)

    def list_datasets(self) -> List[DatasetInfo]:
        return list(self._datasets.values())

    def get_dataset(self, dataset_id: str) -> Optional[DatasetInfo]:
        return self._datasets.get(dataset_id)

    def delete_dataset(self, dataset_id: str) -> bool:
        if dataset_id in self._datasets and not self._datasets[dataset_id].is_builtin:
            ds_info = self._datasets[dataset_id]
            if os.path.exists(ds_info.path):
                shutil.rmtree(ds_info.path)
            del self._datasets[dataset_id]
            self._save_datasets_to_file()
            return True
        return False

    def get_class_names(self, dataset_id: str) -> List[str]:
        ds_info = self._datasets.get(dataset_id)
        if not ds_info:
            return []
        return sorted(ds_info.class_distribution.keys())
