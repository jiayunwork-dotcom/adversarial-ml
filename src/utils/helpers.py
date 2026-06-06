import json
import os
import uuid
from typing import Tuple

import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image

from config import DEVICE, SUPPORTED_ARCHITECTURES


def generate_id() -> str:
    return str(uuid.uuid4())[:8]


def save_json(data: dict, filepath: str) -> None:
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(filepath: str) -> dict:
    if not os.path.exists(filepath):
        return {}
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)


def get_imagenet_labels() -> list:
    labels = [
        "tench", "goldfish", "great white shark", "tiger shark", "hammerhead",
        "electric ray", "stingray", "cock", "hen", "ostrich", "brambling",
        "goldfinch", "house finch", "junco", "indigo bunting", "robin",
        "bulbul", "jay", "magpie", "chickadee", "water ouzel", "kite",
        "bald eagle", "vulture", "great grey owl", "European fire salamander",
        "common newt", "eft", "spotted salamander", "axolotl", "bullfrog",
        "tree frog", "tailed frog", "loggerhead", "leatherback turtle",
        "mud turtle", "terrapin", "box turtle", "banded gecko", "common iguana",
        "American chameleon", "whiptail", "agama", "frilled lizard", "alligator lizard",
        "Gila monster", "green lizard", "African chameleon", "Komodo dragon",
        "African crocodile", "American alligator", "triceratops", "thunder snake",
        "ringneck snake", "hognose snake", "green snake", "king snake",
        "garter snake", "water snake", "vine snake", "night snake",
        "boa constrictor", "rock python", "Indian cobra", "green mamba",
        "sea snake", "horned viper", "diamondback", "sidewinder", "trilobite",
        "harvestman", "scorpion", "black and gold garden spider", "barn spider",
        "garden spider", "black widow", "tarantula", "wolf spider", "tick",
        "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie chicken",
        "peacock", "quail", "partridge", "African grey", "macaw", "sulphur-crested cockatoo",
        "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird",
        "jacamar", "toucan", "drake", "red-breasted merganser", "goose",
        "black swan", "tusker", "echidna", "platypus", "wallaby", "koala",
        "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm",
        "nematode", "conch", "snail", "slug", "sea slug", "chiton",
        "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab",
        "king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab",
        "isopod", "white stork", "black stork", "spoonbill", "flamingo",
        "little blue heron", "American egret", "bittern", "crane", "limpkin",
        "European gallinule", "American coot", "bustard", "ruddy turnstone",
        "red-backed sandpiper", "redshank", "dowitcher", "oystercatcher",
        "pelican", "king penguin", "albatross", "grey whale", "killer whale",
        "dugong", "sea lion", "Chihuahua", "Japanese spaniel", "Maltese dog",
        "Pekinese", "Shih-Tzu", "Blenheim spaniel", "papillon", "toy terrier",
        "Rhodesian ridgeback", "Afghan hound", "basset", "beagle", "bloodhound",
        "bluetick", "black-and-tan coonhound", "Walker hound", "English foxhound",
        "redbone", "borzoi", "Irish wolfhound", "Italian greyhound", "whippet",
        "Ibizan hound", "Norwegian elkhound", "otterhound", "Saluki",
        "Scottish deerhound", "Weimaraner", "Staffordshire bullterrier",
        "American Staffordshire terrier", "Bedlington terrier", "Border terrier",
        "Kerry blue terrier", "Irish terrier", "Norfolk terrier", "Norwich terrier",
        "Yorkshire terrier", "wire-haired fox terrier", "Lakeland terrier",
        "Sealyham terrier", "Airedale", "cairn", "Australian terrier",
        "Dandie Dinmont", "Boston bull", "miniature schnauzer", "giant schnauzer",
        "standard schnauzer", "Scotch terrier", "Tibetan terrier", "silky terrier",
        "soft-coated wheaten terrier", "West Highland white terrier", "Lhasa",
        "flat-coated retriever", "curly-coated retriever", "golden retriever",
        "Labrador retriever", "Chesapeake Bay retriever", "German short-haired pointer",
        "vizsla", "English setter", "Irish setter", "Gordon setter",
        "Brittany spaniel", "clumber", "English springer", "Welsh springer spaniel",
        "cocker spaniel", "Sussex spaniel", "Irish water spaniel", "kuvasz",
        "schipperke", "groenendael dog", "malinois", "briard", "kelpie",
        "komondor", "Old English sheepdog", "Shetland sheepdog", "collie",
        "Border collie", "Bouvier des Flandres", "Rottweiler", "German shepherd",
        "Doberman", "miniature pinscher", "Greater Swiss Mountain dog",
        "Bernese mountain dog", "Appenzeller", "EntleBucher", "boxer",
        "bull mastiff", "Tibetan mastiff", "French bulldog", "Great Dane",
        "Saint Bernard", "Eskimo dog", "malamute", "Siberian husky",
        "dalmatian", "affenpinscher", "basenji", "pug", "Leonberg",
        "Newfoundland", "Great Pyrenees", "Samoyed", "Pomeranian", "chow",
        "keeshond", "Brabancon griffon", "Pembroke", "Cardigan", "toy poodle",
        "miniature poodle", "standard poodle", "Mexican hairless", "timber wolf",
        "white wolf", "red wolf", "coyote", "dingo", "dhole",
        "African hunting dog", "hyena", "red fox", "kit fox", "Arctic fox",
        "grey fox", "tabby", "tiger cat", "Persian cat", "Siamese cat",
        "Egyptian cat", "cougar", "lynx", "leopard", "snow leopard",
        "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear",
        "ice bear", "sloth bear", "mongoose", "meerkat", "tiger beetle",
        "ladybug", "ground beetle", "long-horned beetle", "leaf beetle",
        "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee",
        "ant", "grasshopper", "cricket", "walking stick", "cockroach",
        "mantis", "cicada", "leafhopper", "lacewing", "dragonfly",
        "damselfly", "admiral", "ringlet", "monarch", "cabbage butterfly",
        "sulphur butterfly", "lycaenid", "starfish", "sea urchin",
        "sea cucumber", "cottonwood", "valley", "oak", "sycamore",
        "bottlebrush", "buckeye", "coral fungus", "agaric",
        "bolete", "stinkhorn", "earthstar", "hen-of-the-woods",
        "bolete", "coral fungus", "toilet tissue",
    ]
    return labels + [f"class_{i}" for i in range(len(labels), 1000)]


def get_model_architecture(arch_name: str, num_classes: int = 1000, pretrained: bool = False):
    if arch_name not in SUPPORTED_ARCHITECTURES:
        raise ValueError(f"Unsupported architecture: {arch_name}")

    model_fn = getattr(models, arch_name)
    if pretrained:
        model = model_fn(weights="DEFAULT")
        if num_classes != 1000:
            if arch_name.startswith("resnet"):
                in_features = model.fc.in_features
                model.fc = torch.nn.Linear(in_features, num_classes)
            elif arch_name.startswith("vgg"):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
            elif arch_name.startswith("mobilenet"):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
            elif arch_name.startswith("densenet"):
                in_features = model.classifier.in_features
                model.classifier = torch.nn.Linear(in_features, num_classes)
            elif arch_name.startswith("efficientnet"):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = torch.nn.Linear(in_features, num_classes)
            elif arch_name.startswith("vit"):
                in_features = model.heads.head.in_features
                model.heads.head = torch.nn.Linear(in_features, num_classes)
    else:
        if arch_name.startswith("resnet"):
            model = model_fn(num_classes=num_classes)
        else:
            model = model_fn(num_classes=num_classes)

    return model


def image_to_tensor(image: Image.Image, input_size: int = 224, normalize: bool = True) -> torch.Tensor:
    transform_list = [
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
    ]
    if normalize:
        transform_list.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    transform = transforms.Compose(transform_list)
    return transform(image).unsqueeze(0)


def tensor_to_image(tensor: torch.Tensor, denormalize: bool = True) -> np.ndarray:
    if denormalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
        tensor = tensor * std + mean
    tensor = torch.clamp(tensor, 0, 1)
    img = tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return (img * 255).astype(np.uint8)


def normalize_image(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    return (tensor - mean) / std


def denormalize_image(tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(tensor.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean


def clamp_to_valid_range(tensor: torch.Tensor, epsilon: float, original: torch.Tensor,
                         norm: str = "Linf") -> torch.Tensor:
    if norm == "Linf":
        return torch.clamp(tensor, original - epsilon, original + epsilon)
    elif norm == "L2":
        delta = tensor - original
        norm_val = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1)
        mask = norm_val > epsilon
        if mask.any():
            delta[mask] = delta[mask] * epsilon / norm_val[mask].view(-1, 1, 1, 1)
        return torch.clamp(original + delta, 0, 1)
    return torch.clamp(tensor, 0, 1)


def compute_perturbation_metrics(original: torch.Tensor, adversarial: torch.Tensor
                                 ) -> Tuple[float, float]:
    delta = (adversarial - original).view(-1)
    linf = torch.max(torch.abs(delta)).item()
    l2 = torch.norm(delta, p=2).item()
    return linf, l2
