"""
Neural-network pipeline with regularization, transfer learning, and interpretability.

Dataset: CIFAR-10 or CIFAR-100 (configurable).

Models (PyTorch):
  - Deep MLP: ≥4 hidden layers × 512 units, BatchNorm, Dropout, early stopping.
  - CNN: ≥3 conv blocks + pooling + BatchNorm + Dropout + data augmentation.
  - Transfer: torchvision ResNet18 (ImageNet weights), fine-tune last block (layer4) + fc;
    same augmentation types as CNN (resize/crop/flip/rotation on 224×224 for transfer path).

Hyperparameters: Optuna optimises learning rate, batch size, dropout, weight decay.

Metrics: accuracy, macro-F1, top-5 error, confusion matrix, per-class recall.

Outputs under q5_figures/: training curves, confusion matrices, Grad-CAM (CNN & transfer),
LIME/SHAP for MLP, misclassified galleries, adversarial robustness summary.

Requires: torch, torchvision, optuna, numpy, pandas, matplotlib, scikit-learn;
optional: lime, shap (script degrades gracefully if missing); `pip install datasets` for
Hugging Face CIFAR (used when --data-source huggingface, or when torchvision download fails).
Hyperparameter search uses Optuna (Ray Tune can be substituted similarly for the suggest/search loop).

Usage:
  python q5.py                    # full run (GPU recommended)
  python q5.py --fast           # fewer epochs / trials / PGD steps
  python q5.py --dataset cifar100
  python q5.py --data-source huggingface  # real CIFAR without cs.toronto.edu (503, etc.)
  python q5.py --skip-optuna     # fixed hyperparameters
  python q5.py --dummy-data --fast  # offline / ağsız duman testi (rastgele piksel verisi)
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets as tv_datasets, models, transforms
from PIL import Image

warnings.filterwarnings("ignore", category=UserWarning)

try:
    import optuna

    HAS_OPTUNA = True
except ImportError:
    HAS_OPTUNA = False

try:
    from lime import lime_image

    HAS_LIME = True
except ImportError:
    HAS_LIME = False

try:
    import shap

    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

try:
    from datasets import load_dataset

    HAS_HF_DATASETS = True
except ImportError:
    load_dataset = None  # type: ignore[assignment]
    HAS_HF_DATASETS = False

# Set by load_cifar: "torchvision" | "huggingface" | "dummy" (for consistent new_train/test)
_LAST_CIFAR_BACKEND: str | None = None

# --- Paths & reproducibility ---
RANDOM_STATE = 42
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)
FIG_DIR = Path(__file__).resolve().parent / "q5_figures"
METRICS_CSV = Path(__file__).resolve().parent / "q5_metrics_summary.csv"
DISCUSSION_TXT = Path(__file__).resolve().parent / "q5_discussion.txt"

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2470, 0.2435, 0.2616)


def _ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --- Model definitions ---


class DeepMLP(nn.Module):
    """At least 4 hidden layers with 512 units each (+ BN + dropout after each hidden)."""

    def __init__(
        self,
        in_dim: int,
        num_classes: int,
        dropout: float = 0.4,
        hidden_dim: int = 512,
        n_hidden: int = 4,
    ) -> None:
        super().__init__()
        if n_hidden < 4:
            raise ValueError("DeepMLP requires at least 4 hidden layers.")
        layers: list[nn.Module] = []
        d = in_dim
        for i in range(n_hidden):
            layers.append(nn.Linear(d, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(p=dropout))
            d = hidden_dim
        layers.append(nn.Linear(d, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.size(0), -1)
        return self.net(x)


class CifarCNN(nn.Module):
    """≥3 conv layers + pooling + BN + dropout; global average pool then classifier."""

    def __init__(self, num_classes: int, dropout: float = 0.35) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout * 0.5),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(dropout),
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(256, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        return self.fc(x)


def build_resnet18_transfer(
    num_classes: int,
    freeze_except_last_two: bool = True,
    *,
    pretrained: bool = True,
) -> nn.Module:
    """
    ImageNet-pretrained ResNet18; fine-tune last residual block (layer4) + fc — 'last two layers'
    in the sense of final stage + classifier.
    """
    w = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    m = models.resnet18(weights=w)
    if freeze_except_last_two:
        for p in m.parameters():
            p.requires_grad = False
        for p in m.layer4.parameters():
            p.requires_grad = True
    in_fc = m.fc.in_features
    m.fc = nn.Linear(in_fc, num_classes)
    return m


# --- Data ---


def cifar_train_transform(aug: bool, for_transfer: bool) -> transforms.Compose:
    ops: list[Any] = []
    if for_transfer:
        if aug:
            ops.extend(
                [
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                ]
            )
        else:
            ops.append(transforms.Resize(224))
        ops.extend([transforms.ToTensor(), transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD)])
    else:
        if aug:
            ops.extend(
                [
                    transforms.RandomCrop(32, padding=4),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomRotation(15),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
                ]
            )
        else:
            ops.extend(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
                ]
            )
    return transforms.Compose(ops)


def cifar_test_transform(for_transfer: bool) -> transforms.Compose:
    if for_transfer:
        return transforms.Compose(
            [
                transforms.Resize(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
        ]
    )


class DummyCifarLikeDataset(Dataset):
    """Same interface as torchvision CIFAR (PIL + label); no download — for smoke tests offline."""

    transform: Any | None
    target_transform: Any | None

    def __init__(self, n: int, num_classes: int, seed: int) -> None:
        self.n = n
        self.num_classes = num_classes
        self.transform = None
        self.target_transform = None
        rng = np.random.RandomState(seed)
        self._labels = rng.randint(0, num_classes, size=n)
        self._data = rng.randint(0, 256, size=(n, 32, 32, 3), dtype=np.uint8)
        if num_classes == 10:
            self.classes = [
                "airplane",
                "automobile",
                "bird",
                "cat",
                "deer",
                "dog",
                "frog",
                "horse",
                "ship",
                "truck",
            ]
        else:
            self.classes = [f"class_{i}" for i in range(num_classes)]

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> tuple[Any, int]:
        img = Image.fromarray(self._data[i])
        if self.transform is not None:
            img = self.transform(img)
        target = int(self._labels[i])
        if self.target_transform is not None:
            target = self.target_transform(target)
        return img, target


_HF_CIFAR_DICT: dict[str, tuple[Any, str, list[str]]] = {}


class HuggingFaceCifarDataset(Dataset):
    """CIFAR from Hugging Face `datasets` (PIL `img` + int label)."""

    transform: Any | None
    target_transform: Any | None

    def __init__(self, hf_split: Any, label_key: str, class_names: list[str]) -> None:
        self.hf = hf_split
        self.label_key = label_key
        self.classes = class_names
        self.transform = None
        self.target_transform = None

    def __len__(self) -> int:
        return len(self.hf)

    def __getitem__(self, i: int) -> tuple[Any, int]:
        row = self.hf[i]
        img = row["img"]
        y = int(row[self.label_key])
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            y = self.target_transform(y)
        return img, y


def _get_hf_cifar_dict(name_l: str) -> tuple[Any, str, list[str]]:
    if not HAS_HF_DATASETS or load_dataset is None:
        raise ImportError("Hugging Face CIFAR requires: pip install datasets")
    if name_l not in _HF_CIFAR_DICT:
        mid = "cifar10" if name_l == "cifar10" else "cifar100"
        d: Any = load_dataset(mid)
        label_key = "label" if name_l == "cifar10" else "fine_label"
        cl = list(d["train"].features[label_key].names)  # type: ignore[union-attr]
        _HF_CIFAR_DICT[name_l] = (d, label_key, cl)
    return _HF_CIFAR_DICT[name_l]


def _new_hf_train(name: str) -> HuggingFaceCifarDataset:
    d, label_key, classes = _get_hf_cifar_dict(name.lower())
    return HuggingFaceCifarDataset(d["train"], label_key, classes)


def _new_hf_test(name: str) -> HuggingFaceCifarDataset:
    d, label_key, classes = _get_hf_cifar_dict(name.lower())
    return HuggingFaceCifarDataset(d["test"], label_key, classes)


def _load_cifar_tv(
    name: str,
    data_dir: Path,
    val_fraction: float,
    dummy_train_n: int,
    *,
    dummy: bool,
) -> tuple[Any, np.ndarray, np.ndarray, list[str]]:
    global _LAST_CIFAR_BACKEND
    name_l = name.lower()
    if dummy:
        num_classes = 10 if name_l == "cifar10" else 100
        train_full: Any = DummyCifarLikeDataset(dummy_train_n, num_classes, seed=RANDOM_STATE)
        classes = list(train_full.classes)
        _LAST_CIFAR_BACKEND = "dummy"
    elif name_l == "cifar10":
        train_full = tv_datasets.CIFAR10(
            root=str(data_dir), train=True, download=True, transform=None
        )
        classes = list(train_full.classes)
        _LAST_CIFAR_BACKEND = "torchvision"
    elif name_l == "cifar100":
        train_full = tv_datasets.CIFAR100(
            root=str(data_dir), train=True, download=True, transform=None
        )
        classes = list(train_full.classes)
        _LAST_CIFAR_BACKEND = "torchvision"
    else:
        raise ValueError("dataset must be cifar10 or cifar100")
    n = len(train_full)
    idx = np.arange(n)
    tr_idx, va_idx = train_test_split(idx, test_size=val_fraction, random_state=RANDOM_STATE)
    return train_full, tr_idx, va_idx, classes


def _load_cifar_hf(name: str, val_fraction: float) -> tuple[Any, np.ndarray, np.ndarray, list[str]]:
    global _LAST_CIFAR_BACKEND
    name_l = name.lower()
    if name_l not in ("cifar10", "cifar100"):
        raise ValueError("dataset must be cifar10 or cifar100")
    train_full = _new_hf_train(name)
    classes = list(train_full.classes)
    _LAST_CIFAR_BACKEND = "huggingface"
    n = len(train_full)
    idx = np.arange(n)
    tr_idx, va_idx = train_test_split(idx, test_size=val_fraction, random_state=RANDOM_STATE)
    return train_full, tr_idx, va_idx, classes


def load_cifar(
    name: str,
    data_dir: Path,
    val_fraction: float,
    *,
    dummy: bool = False,
    dummy_train_n: int = 4096,
    data_source: str = "auto",
) -> tuple[Any, np.ndarray, np.ndarray, list[str]]:
    if dummy:
        return _load_cifar_tv(name, data_dir, val_fraction, dummy_train_n, dummy=True)
    if data_source == "huggingface":
        return _load_cifar_hf(name, val_fraction)
    if data_source == "torchvision":
        return _load_cifar_tv(name, data_dir, val_fraction, dummy_train_n, dummy=False)
    try:
        return _load_cifar_tv(name, data_dir, val_fraction, dummy_train_n, dummy=False)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError) as e:
        if HAS_HF_DATASETS:
            warnings.warn(
                f"torchvision CIFAR indirilemedi ({type(e).__name__}: {e}); "
                "Hugging Face `datasets` kullanılıyor.",
                stacklevel=2,
            )
            return _load_cifar_hf(name, val_fraction)
        raise


def load_cifar_test(
    name: str,
    data_dir: Path,
    *,
    dummy: bool = False,
    dummy_test_n: int = 1024,
    data_source: str = "auto",
) -> Any:
    name_l = name.lower()
    if dummy:
        nc = 10 if name_l == "cifar10" else 100
        return DummyCifarLikeDataset(dummy_test_n, nc, seed=RANDOM_STATE + 1)
    if data_source == "huggingface":
        return _new_hf_test(name)
    if data_source == "torchvision":
        if name_l == "cifar10":
            return tv_datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=None)
        return tv_datasets.CIFAR100(root=str(data_dir), train=False, download=True, transform=None)
    if _LAST_CIFAR_BACKEND == "huggingface":
        return _new_hf_test(name)
    if name_l == "cifar10":
        return tv_datasets.CIFAR10(root=str(data_dir), train=False, download=True, transform=None)
    return tv_datasets.CIFAR100(root=str(data_dir), train=False, download=True, transform=None)


def new_train_dataset(
    name: str,
    data_dir: Path,
    *,
    dummy: bool,
    dummy_train_n: int = 4096,
    data_source: str = "auto",
) -> Any:
    """Yeni dataset nesnesi — `attach_transforms` çağrıları birbirinin transform'unu ezmeyelim."""
    name_l = name.lower()
    if dummy:
        nc = 10 if name_l == "cifar10" else 100
        return DummyCifarLikeDataset(dummy_train_n, nc, seed=RANDOM_STATE)
    if data_source == "huggingface":
        return _new_hf_train(name)
    if data_source == "torchvision":
        if name_l == "cifar10":
            return tv_datasets.CIFAR10(
                root=str(data_dir), train=True, download=True, transform=None
            )
        return tv_datasets.CIFAR100(
            root=str(data_dir), train=True, download=True, transform=None
        )
    if _LAST_CIFAR_BACKEND == "huggingface":
        return _new_hf_train(name)
    if name_l == "cifar10":
        return tv_datasets.CIFAR10(
            root=str(data_dir), train=True, download=True, transform=None
        )
    return tv_datasets.CIFAR100(
        root=str(data_dir), train=True, download=True, transform=None
    )


def new_test_dataset(
    name: str,
    data_dir: Path,
    *,
    dummy: bool,
    dummy_test_n: int = 1024,
    data_source: str = "auto",
) -> Any:
    name_l = name.lower()
    if dummy:
        nc = 10 if name_l == "cifar10" else 100
        return DummyCifarLikeDataset(dummy_test_n, nc, seed=RANDOM_STATE + 1)
    if data_source == "huggingface":
        return _new_hf_test(name)
    if data_source == "torchvision":
        if name_l == "cifar10":
            return tv_datasets.CIFAR10(
                root=str(data_dir), train=False, download=True, transform=None
            )
        return tv_datasets.CIFAR100(
            root=str(data_dir), train=False, download=True, transform=None
        )
    if _LAST_CIFAR_BACKEND == "huggingface":
        return _new_hf_test(name)
    if name_l == "cifar10":
        return tv_datasets.CIFAR10(
            root=str(data_dir), train=False, download=True, transform=None
        )
    return tv_datasets.CIFAR100(
        root=str(data_dir), train=False, download=True, transform=None
    )


def attach_transforms(
    base_train: Any,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_raw: Any,
    aug_train: bool,
    for_transfer: bool,
) -> tuple[Subset, Subset, Any]:
    base_train.transform = cifar_train_transform(aug=aug_train, for_transfer=for_transfer)
    test_raw.transform = cifar_test_transform(for_transfer=for_transfer)
    return Subset(base_train, train_idx.tolist()), Subset(base_train, val_idx.tolist()), test_raw


# --- Training ---


@dataclass
class History:
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    train_acc: list[float] = field(default_factory=list)
    val_acc: list[float] = field(default_factory=list)


def accuracy_topk(logits: torch.Tensor, y: torch.Tensor, k: int) -> float:
    """Fraction of samples where true label is in top-k predictions."""
    if k >= logits.size(1):
        return 1.0
    _, pred = logits.topk(k, dim=1)
    y = y.view(-1, 1)
    return float((pred == y).any(dim=1).float().mean().item())


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    criterion: nn.Module,
    train: bool,
) -> tuple[float, float, float]:
    if train:
        model.train()
    else:
        model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    top5_correct = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        if train and optimizer is not None:
            optimizer.zero_grad()
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train and optimizer is not None:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * x.size(0)
        pred = logits.argmax(dim=1)
        correct += (pred == y).sum().item()
        total += x.size(0)
        top5_correct += accuracy_topk(logits.detach(), y, k=5) * x.size(0)
    return total_loss / max(total, 1), correct / max(total, 1), top5_correct / max(total, 1)


def train_with_early_stopping(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    label_smoothing: float = 0.05,
    *,
    run_label: str = "",
    verbose: bool = True,
) -> tuple[History, int]:
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs)
    history = History()
    best_val = -1.0
    best_epoch = 0
    stale = 0
    best_state: dict[str, torch.Tensor] | None = None
    tag = f"[{run_label}] " if run_label else ""
    t_run_start = time.perf_counter()
    for epoch in range(max_epochs):
        t_ep_start = time.perf_counter()
        tr_loss, tr_acc, _ = run_epoch(
            model, train_loader, device, optimizer, criterion, train=True
        )
        va_loss, va_acc, _ = run_epoch(
            model, val_loader, device, None, criterion, train=False
        )
        scheduler.step()
        history.train_loss.append(tr_loss)
        history.val_loss.append(va_loss)
        history.train_acc.append(tr_acc)
        history.val_acc.append(va_acc)
        # Keep the best checkpoint by validation accuracy and stop when it plateaus.
        if va_acc > best_val:
            best_val = va_acc
            best_epoch = epoch
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        epoch_sec = time.perf_counter() - t_ep_start
        total_sec = time.perf_counter() - t_run_start
        if verbose:
            print(
                f"{tag}epoch {epoch + 1}/{max_epochs} | "
                f"bu epoch {epoch_sec:.1f}s · toplam {total_sec:.1f}s | "
                f"tr_loss={tr_loss:.4f} va_loss={va_loss:.4f} | "
                f"tr_acc={tr_acc:.4f} va_acc={va_acc:.4f} | "
                f"en_iyi_va={best_val:.4f} (ep {best_epoch + 1}) · stale {stale}/{patience}",
                flush=True,
            )
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    if verbose:
        print(
            f"{tag}bitti · en iyi val_acc={best_val:.4f} @ epoch {best_epoch + 1} · "
            f"toplam süre {time.perf_counter() - t_run_start:.1f}s",
            flush=True,
        )
    return history, best_epoch


@torch.no_grad()
def evaluate_full(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, Any]:
    model.eval()
    all_y: list[int] = []
    all_p: list[int] = []
    all_logits: list[np.ndarray] = []
    for x, y in loader:
        x = x.to(device)
        logits = model(x).cpu().numpy()
        pred = logits.argmax(axis=1)
        all_y.extend(y.numpy().tolist())
        all_p.extend(pred.tolist())
        all_logits.append(logits)
    y_true = np.array(all_y)
    y_pred = np.array(all_p)
    logits_full = np.concatenate(all_logits, axis=0)
    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    per_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(num_classes))
    top5 = []
    for i in range(len(y_true)):
        row = logits_full[i]
        order = np.argsort(-row)[:5]
        top5.append(1.0 if y_true[i] in order else 0.0)
    top5_acc = float(np.mean(top5))
    top5_err = 1.0 - top5_acc
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "top5_error": top5_err,
        "confusion_matrix": cm,
        "per_class_recall": per_recall,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def plot_history(history: History, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    epochs = range(1, len(history.train_loss) + 1)
    ax[0].plot(epochs, history.train_loss, label="train")
    ax[0].plot(epochs, history.val_loss, label="val")
    ax[0].set_title("Loss")
    ax[0].set_xlabel("epoch")
    ax[0].legend()
    ax[1].plot(epochs, history.train_acc, label="train")
    ax[1].plot(epochs, history.val_acc, label="val")
    ax[1].set_title("Accuracy")
    ax[1].set_xlabel("epoch")
    ax[1].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion(cm: np.ndarray, class_names: list[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax)
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=6)
    ax.set_yticklabels(class_names, fontsize=6)
    ax.set_ylabel("True")
    ax.set_xlabel("Predicted")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


# --- Grad-CAM ---


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self.handles: list[Any] = []

        def fwd_hook(_m: Any, _i: Any, o: Any) -> None:
            self.activations = o.detach()

        def bwd_hook(_m: Any, _gi: Any, go: Any) -> None:
            self.gradients = go[0].detach()

        self.handles.append(target_layer.register_forward_hook(fwd_hook))
        self.handles.append(target_layer.register_full_backward_hook(bwd_hook))

    def remove(self) -> None:
        for h in self.handles:
            h.remove()

    def __call__(self, x: torch.Tensor, class_idx: int | None) -> np.ndarray:
        self.model.eval()
        # Allow gradients to reach conv activations even when lower blocks are frozen (transfer).
        x = x.clone().detach().requires_grad_(True)
        self.model.zero_grad(set_to_none=True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = int(logits.argmax(dim=1).item())
        score = logits[0, class_idx]
        score.backward(retain_graph=False)
        assert self.gradients is not None and self.activations is not None
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


def get_last_conv_for_cam(model: nn.Module, architecture: str) -> nn.Module:
    if architecture == "cnn":
        # last conv in feature extractor
        for m in reversed(list(model.features.modules())):
            if isinstance(m, nn.Conv2d):
                return m
        raise RuntimeError("No conv in CNN")
    if architecture == "resnet":
        return model.layer4[-1].conv2
    raise ValueError(architecture)


def overlay_cam_on_image(
    img_tensor: torch.Tensor,
    cam: np.ndarray,
    mean: tuple[float, ...],
    std: tuple[float, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Denormalize CHW tensor to HWC uint8 and resize cam to H,W."""
    img = img_tensor.detach().cpu().clone()
    for c in range(3):
        img[c] = img[c] * std[c] + mean[c]
    img = torch.clamp(img, 0, 1)
    h, w = img.shape[1], img.shape[2]
    cam_img = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h))) / 255.0
    heat = plt.cm.jet(cam_img)[:, :, :3]
    base = img.numpy().transpose(1, 2, 0)
    blended = 0.5 * base + 0.5 * heat
    return base, np.clip(blended, 0, 1)


# --- Adversarial ---


def fgsm_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    loss_fn: nn.Module,
) -> torch.Tensor:
    x_adv = x.detach().clone().requires_grad_(True)
    logits = model(x_adv)
    loss = loss_fn(logits, y)
    loss.backward()
    adv = x_adv + epsilon * x_adv.grad.sign()
    return adv.detach()


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    loss_fn: nn.Module,
) -> torch.Tensor:
    """L_inf PGD in the same tensor domain as training (normalized CIFAR tensors)."""
    x_orig = x.detach()
    noise = torch.empty_like(x_orig).uniform_(-epsilon, epsilon)
    x_adv = x_orig + torch.clamp(noise, min=-epsilon, max=epsilon)
    for _ in range(steps):
        x_adv = x_adv.detach().requires_grad_(True)
        logits = model(x_adv)
        loss = loss_fn(logits, y)
        loss.backward()
        assert x_adv.grad is not None
        x_adv = x_adv.detach() + alpha * x_adv.grad.sign()
        x_adv = x_orig + torch.clamp(x_adv - x_orig, min=-epsilon, max=epsilon)
    return x_adv.detach()


def accuracy_under_attack(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    attack: str,
    epsilon: float,
    pgd_steps: int,
    pgd_alpha: float,
) -> float:
    model.eval()
    ce = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        if attack == "none":
            with torch.no_grad():
                pred = model(x).argmax(dim=1)
        elif attack == "fgsm":
            x_adv = fgsm_attack(model, x, y, epsilon, ce)
            with torch.no_grad():
                pred = model(x_adv).argmax(dim=1)
        elif attack == "pgd":
            x_adv = pgd_attack(model, x, y, epsilon, pgd_alpha, pgd_steps, ce)
            with torch.no_grad():
                pred = model(x_adv).argmax(dim=1)
        else:
            raise ValueError(attack)
        correct += (pred == y).sum().item()
        total += y.size(0)
    return correct / max(total, 1)


# --- Optuna ---


@dataclass
class HParams:
    lr: float = 3e-4
    batch_size: int = 64
    dropout: float = 0.4
    weight_decay: float = 1e-3


def optuna_optimize(
    build_model: Callable[[float], nn.Module],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    n_trials: int,
    max_epochs: int,
    patience: int,
    fast: bool,
) -> HParams:
    if not HAS_OPTUNA:
        return HParams()

    def objective(trial: "optuna.Trial") -> float:
        lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128] if not fast else [64])
        dropout = trial.suggest_float("dropout", 0.2, 0.55)
        wd = trial.suggest_float("weight_decay", 1e-5, 5e-3, log=True)
        model = build_model(dropout).to(device)
        # rebuild loaders with new batch size — caller passes fixed datasets; we re-wrap
        # Optuna study uses existing loaders' datasets
        from torch.utils.data import DataLoader as DL

        bs = min(batch_size, len(train_loader.dataset))  # type: ignore[arg-type]
        tr_ds = train_loader.dataset
        va_ds = val_loader.dataset
        tl = DL(tr_ds, batch_size=bs, shuffle=True, num_workers=0, pin_memory=False)
        vl = DL(va_ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=False)
        hist, _ = train_with_early_stopping(
            model,
            tl,
            vl,
            device,
            lr=lr,
            weight_decay=wd,
            max_epochs=max_epochs,
            patience=patience,
            verbose=False,
        )
        best = max(hist.val_acc) if hist.val_acc else 0.0
        trial.set_user_attr("val_acc", best)
        return best

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    t = study.best_trial
    return HParams(
        lr=float(t.params["lr"]),
        batch_size=int(t.params["batch_size"]),
        dropout=float(t.params["dropout"]),
        weight_decay=float(t.params["weight_decay"]),
    )


# --- LIME / SHAP helpers ---


def explain_mlp_lime(
    model: nn.Module,
    device: torch.device,
    images_hwc: np.ndarray,
    num_classes: int,
    num_samples: int = 800,
) -> np.ndarray | None:
    if not HAS_LIME:
        return None
    model.eval()

    def predict_fn(imgs: np.ndarray) -> np.ndarray:
        # imgs: N,H,W,C float 0-1
        x = torch.from_numpy(imgs).permute(0, 3, 1, 2).float().to(device)
        # normalize like training
        for c in range(3):
            x[:, c] = (x[:, c] - CIFAR10_MEAN[c]) / CIFAR10_STD[c]
        with torch.no_grad():
            logits = model(x).cpu().numpy()
        out = np.exp(logits - logits.max(axis=1, keepdims=True))
        out /= out.sum(axis=1, keepdims=True)
        return out

    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        images_hwc.astype(np.double),
        classifier_fn=predict_fn,
        top_labels=min(5, num_classes),
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=None,
    )
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=False
    )
    return np.clip(temp, 0, 1)


def explain_mlp_shap(
    model: nn.Module,
    device: torch.device,
    background: torch.Tensor,
    test_img: torch.Tensor,
    num_classes: int,
) -> np.ndarray | None:
    if not HAS_SHAP:
        return None
    model.eval()
    try:
        explainer = shap.GradientExplainer(model, background)
        sv = explainer.shap_values(test_img.unsqueeze(0))
        if isinstance(sv, list):
            arr = np.stack(sv, axis=-1)
        else:
            arr = sv
        # rough relevance map: mean abs over channels
        imp = np.abs(arr).mean(axis=(0, 2))  # simplify
        if imp.ndim == 1:
            side = int(np.sqrt(imp.size))
            imp = imp.reshape(side, side)
        return imp
    except Exception:
        return None


# --- Misclassified gallery ---


def collect_misclassified_indices(
    y_true: np.ndarray, y_pred: np.ndarray, max_n: int
) -> np.ndarray:
    wrong = np.where(y_true != y_pred)[0]
    return wrong[:max_n]


def save_misclassified_gradcam_grid(
    model: nn.Module,
    dataset: Any,
    indices: np.ndarray,
    device: torch.device,
    cam_module: nn.Module,
    class_names: list[str],
    out_path: Path,
    for_transfer: bool,
) -> None:
    model.eval()
    cam = GradCAM(model, cam_module)
    n = min(len(indices), 12)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols * 2, figsize=(cols * 4, rows * 2.2))
    if rows == 1:
        axes = np.array([axes])
    for i in range(n):
        r, c = divmod(i, cols)
        idx = int(indices[i])
        img_pil, y = dataset[idx]
        x = img_pil.unsqueeze(0).to(device)
        pred = int(model(x).argmax(dim=1).item())
        cam_out = cam(x, pred)
        base, blend = overlay_cam_on_image(x[0].cpu(), cam_out, CIFAR10_MEAN, CIFAR10_STD)
        ax_orig = axes[r, c * 2]
        ax_hot = axes[r, c * 2 + 1]
        ax_orig.imshow(base)
        ax_orig.axis("off")
        ax_orig.set_title(f"t:{class_names[y]} p:{class_names[pred]}", fontsize=7)
        ax_hot.imshow(blend)
        ax_hot.axis("off")
    # hide unused
    for j in range(n, rows * cols):
        r, c = divmod(j, cols)
        axes[r, c * 2].axis("off")
        axes[r, c * 2 + 1].axis("off")
    cam.remove()
    fig.suptitle("Misclassified (Grad-CAM on predicted class)" + (" [224]" if for_transfer else ""))
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_discussion_text(
    rows: list[dict[str, Any]],
    out_path: Path,
) -> None:
    lines = [
        "Q5 discussion notes (auto-generated outline; expand if needed)",
        "",
        "Overfitting & regularisation:",
        "- Compare train vs val curves in *_curves.png: a large gap suggests overfitting.",
        "- Dropout + batch norm + weight decay + early stopping reduce gap; transfer learning",
        "  often achieves better val accuracy with fewer trainable parameters (layer4+fc).",
        "",
        "Interpretability:",
        "- Grad-CAM highlights spatial focus; frequent focus on background suggests shortcut learning.",
        "- LIME/SHAP on flat MLP inputs attribute importance over superpixels/ gradients.",
        "",
        "Adversarial robustness:",
        "- FGSM/PGD typically hurt accuracy; CNNs and especially pretrained conv nets may retain",
        "  higher robust accuracy than a flat MLP on raw pixels (inductive bias + smoother decision boundaries),",
        "  though none are robust without adversarial training.",
        "",
        "Trade-offs (complexity vs interpretability vs robustness):",
        "- MLP: flexible but less structured — SHAP/LIME explain tabular-like attributions; often weakest robustness.",
        "- Scratch CNN: spatial Grad-CAM; moderate complexity; augmentation + regularisation critical.",
        "- Pretrained ResNet: strong accuracy and often better robustness under fixed epsilon; interpretability",
        "  still Grad-CAM but over ImageNet features — failures may show texture/background bias.",
        "",
        "Metrics summary:",
    ]
    for r in rows:
        lines.append(json.dumps(r, default=str, ensure_ascii=False))
    out_path.write_text("\n".join(lines), encoding="utf-8")


def save_mlp_lime_mosaic(
    model: nn.Module,
    test_dataset: Any,
    wrong_idx: np.ndarray,
    device: torch.device,
    num_classes: int,
    out_path: Path,
    n_max: int = 10,
    num_samples: int = 500,
) -> None:
    if not HAS_LIME or len(wrong_idx) == 0:
        return
    n = int(min(n_max, len(wrong_idx)))
    cols = 5
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.2, rows * 2.2))
    if rows == 1:
        axes = np.array(axes).reshape(1, -1)
    for k in range(n):
        r, c = divmod(k, cols)
        ax = axes[r, c]
        i0 = int(wrong_idx[k])
        img_t, y0 = test_dataset[i0]
        img_hwc = img_t.permute(1, 2, 0).cpu().numpy()
        img_hwc = np.clip(
            img_hwc * np.array(CIFAR10_STD) + np.array(CIFAR10_MEAN), 0, 1
        )
        lime_map = explain_mlp_lime(
            model, device, img_hwc, num_classes, num_samples=num_samples
        )
        if lime_map is not None:
            ax.imshow(lime_map)
        ax.axis("off")
        ax.set_title(f"t={y0}", fontsize=7)
    for k in range(n, rows * cols):
        r, c = divmod(k, cols)
        axes[r, c].axis("off")
    fig.suptitle("MLP: LIME on misclassified (Top label mask)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("cifar10", "cifar100"), default="cifar10")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--skip-optuna", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--optuna-trials", type=int, default=None)
    parser.add_argument(
        "--dummy-data",
        action="store_true",
        help="Rastgele CIFAR-benzeri veri (ağ yok); sunucu 503 / offline test için.",
    )
    parser.add_argument(
        "--data-source",
        choices=("auto", "torchvision", "huggingface"),
        default="auto",
        help="CIFAR indirme: auto=önce torchvision, 503 vb. olursa Hugging Face (datasets).",
    )
    args = parser.parse_args()
    if args.data_source == "huggingface" and not HAS_HF_DATASETS:
        parser.error("gerçek CIFAR için Hugging Face: pip install datasets")

    fast = args.fast
    # Varsayılan: 10 epoch; --fast: kısa duman testi (8 ep)
    max_epochs = 8 if fast else 10
    patience = 3 if fast else 12
    optuna_trials = 4 if fast else 12
    if args.epochs is not None:
        max_epochs = args.epochs
    if args.optuna_trials is not None:
        optuna_trials = args.optuna_trials

    _ensure_dirs()
    device = get_device()
    print("Device:", device)

    global _LAST_CIFAR_BACKEND
    _LAST_CIFAR_BACKEND = None

    train_full, tr_idx, va_idx, class_names = load_cifar(
        args.dataset,
        args.data_dir,
        val_fraction=0.1,
        dummy=args.dummy_data,
        data_source=args.data_source,
    )
    num_classes = len(class_names)

    results_rows: list[dict[str, Any]] = []

    # ----- MLP -----
    train_mlp, val_mlp, test_mlp = attach_transforms(
        new_train_dataset(
            args.dataset, args.data_dir, dummy=args.dummy_data, data_source=args.data_source
        ),
        tr_idx,
        va_idx,
        new_test_dataset(
            args.dataset, args.data_dir, dummy=args.dummy_data, data_source=args.data_source
        ),
        aug_train=True,
        for_transfer=False,
    )
    in_dim = 32 * 32 * 3

    def build_mlp(drop: float) -> nn.Module:
        return DeepMLP(in_dim, num_classes, dropout=drop, hidden_dim=512, n_hidden=4)

    hp_mlp = HParams()
    if not args.skip_optuna and HAS_OPTUNA:
        tl0 = DataLoader(train_mlp, batch_size=64, shuffle=True, num_workers=0)
        vl0 = DataLoader(val_mlp, batch_size=64, shuffle=False, num_workers=0)
        hp_mlp = optuna_optimize(
            build_mlp,
            tl0,
            vl0,
            device,
            n_trials=optuna_trials,
            max_epochs=min(20, max_epochs),
            patience=min(5, patience),
            fast=fast,
        )
    else:
        hp_mlp = HParams()

    mlp = build_mlp(hp_mlp.dropout).to(device)
    tl = DataLoader(
        train_mlp, batch_size=hp_mlp.batch_size, shuffle=True, num_workers=0
    )
    vl = DataLoader(val_mlp, batch_size=hp_mlp.batch_size, shuffle=False, num_workers=0)
    hist_mlp, _ = train_with_early_stopping(
        mlp,
        tl,
        vl,
        device,
        lr=hp_mlp.lr,
        weight_decay=hp_mlp.weight_decay,
        max_epochs=max_epochs,
        patience=patience,
        run_label="MLP",
    )
    plot_history(hist_mlp, "MLP (CIFAR)", FIG_DIR / "mlp_curves.png")

    te_loader = DataLoader(test_mlp, batch_size=128, shuffle=False, num_workers=0)
    ev_mlp = evaluate_full(mlp, te_loader, device, num_classes)
    plot_confusion(
        ev_mlp["confusion_matrix"], class_names, "MLP Confusion", FIG_DIR / "mlp_confusion.png"
    )
    results_rows.append(
        {
            "model": "MLP",
            "hparams": hp_mlp.__dict__,
            "accuracy": ev_mlp["accuracy"],
            "macro_f1": ev_mlp["macro_f1"],
            "top5_error": ev_mlp["top5_error"],
        }
    )

    wrong_idx = collect_misclassified_indices(ev_mlp["y_true"], ev_mlp["y_pred"], 20)
    # LIME: single example + mosaic of <=10 misclassified samples for MLP explanations
    if HAS_LIME and len(wrong_idx) > 0:
        i0 = int(wrong_idx[0])
        img_t, _ = test_mlp[i0]
        img_hwc = img_t.permute(1, 2, 0).cpu().numpy()
        img_hwc = np.clip(
            img_hwc * np.array(CIFAR10_STD) + np.array(CIFAR10_MEAN), 0, 1
        )
        lime_map = explain_mlp_lime(
            mlp, device, img_hwc, num_classes, num_samples=300 if fast else 1200
        )
        if lime_map is not None:
            plt.imsave(FIG_DIR / "mlp_lime_example.png", lime_map)
        save_mlp_lime_mosaic(
            mlp,
            test_mlp,
            wrong_idx,
            device,
            num_classes,
            FIG_DIR / "mlp_lime_misclassified_10.png",
            n_max=10,
            num_samples=200 if fast else 800,
        )

    # SHAP small background
    if HAS_SHAP:
        bg_idx = np.random.RandomState(RANDOM_STATE).choice(len(train_mlp), size=min(64, len(train_mlp)), replace=False)
        bg_list = [train_mlp[int(j)][0] for j in bg_idx]
        bg = torch.stack(bg_list, dim=0).to(device)
        if len(wrong_idx) > 0:
            timg, _ = test_mlp[int(wrong_idx[0])]
            shap_map = explain_mlp_shap(mlp, device, bg, timg.to(device), num_classes)
            if shap_map is not None:
                sm = np.squeeze(np.asarray(shap_map, dtype=np.float64))
                if sm.ndim == 1 and sm.size == 32 * 32 * 3:
                    sm = sm.reshape(32, 32, 3).mean(axis=-1)
                elif sm.ndim == 3:
                    sm = np.abs(sm).mean(axis=-1)
                if sm.ndim == 2 and sm.size > 0:
                    sm = (sm - sm.min()) / (sm.max() - sm.min() + 1e-8)
                    plt.imsave(FIG_DIR / "mlp_shap_importance.png", sm, cmap="hot")

    # ----- CNN -----
    train_cnn, val_cnn, test_cnn = attach_transforms(
        new_train_dataset(
            args.dataset, args.data_dir, dummy=args.dummy_data, data_source=args.data_source
        ),
        tr_idx,
        va_idx,
        new_test_dataset(
            args.dataset, args.data_dir, dummy=args.dummy_data, data_source=args.data_source
        ),
        aug_train=True,
        for_transfer=False,
    )

    def build_cnn(drop: float) -> nn.Module:
        return CifarCNN(num_classes, dropout=drop)

    hp_cnn = HParams(dropout=0.35, lr=2e-3, batch_size=64, weight_decay=5e-4)
    if not args.skip_optuna and HAS_OPTUNA:
        tl0 = DataLoader(train_cnn, batch_size=64, shuffle=True, num_workers=0)
        vl0 = DataLoader(val_cnn, batch_size=64, shuffle=False, num_workers=0)
        hp_cnn = optuna_optimize(
            build_cnn,
            tl0,
            vl0,
            device,
            n_trials=optuna_trials,
            max_epochs=min(20, max_epochs),
            patience=min(5, patience),
            fast=fast,
        )

    cnn = build_cnn(hp_cnn.dropout).to(device)
    tl = DataLoader(train_cnn, batch_size=hp_cnn.batch_size, shuffle=True, num_workers=0)
    vl = DataLoader(val_cnn, batch_size=hp_cnn.batch_size, shuffle=False, num_workers=0)
    hist_cnn, _ = train_with_early_stopping(
        cnn,
        tl,
        vl,
        device,
        lr=hp_cnn.lr,
        weight_decay=hp_cnn.weight_decay,
        max_epochs=max_epochs,
        patience=patience,
        run_label="CNN",
    )
    plot_history(hist_cnn, "CNN (CIFAR)", FIG_DIR / "cnn_curves.png")

    te_loader_cnn = DataLoader(test_cnn, batch_size=128, shuffle=False, num_workers=0)
    ev_cnn = evaluate_full(cnn, te_loader_cnn, device, num_classes)
    plot_confusion(
        ev_cnn["confusion_matrix"], class_names, "CNN Confusion", FIG_DIR / "cnn_confusion.png"
    )
    results_rows.append(
        {
            "model": "CNN",
            "hparams": hp_cnn.__dict__,
            "accuracy": ev_cnn["accuracy"],
            "macro_f1": ev_cnn["macro_f1"],
            "top5_error": ev_cnn["top5_error"],
        }
    )

    cam_mod_cnn = get_last_conv_for_cam(cnn, "cnn")
    wrong_cnn = collect_misclassified_indices(ev_cnn["y_true"], ev_cnn["y_pred"], 15)
    if len(wrong_cnn) >= 10:
        save_misclassified_gradcam_grid(
            cnn,
            test_cnn,
            wrong_cnn,
            device,
            cam_mod_cnn,
            class_names,
            FIG_DIR / "cnn_misclassified_gradcam.png",
            for_transfer=False,
        )

    # ----- Transfer ResNet18 (same aug pipeline as CNN: cifar_train_transform for_transfer=True) -----
    train_t, val_t, test_t = attach_transforms(
        new_train_dataset(
            args.dataset, args.data_dir, dummy=args.dummy_data, data_source=args.data_source
        ),
        tr_idx,
        va_idx,
        new_test_dataset(
            args.dataset, args.data_dir, dummy=args.dummy_data, data_source=args.data_source
        ),
        aug_train=True,
        for_transfer=True,
    )
    hp_tl = HParams(lr=1e-4, batch_size=32, dropout=0.0, weight_decay=1e-4)
    resnet = build_resnet18_transfer(
        num_classes,
        freeze_except_last_two=True,
        pretrained=not args.dummy_data,
    ).to(device)
    tl = DataLoader(train_t, batch_size=hp_tl.batch_size, shuffle=True, num_workers=0)
    vl = DataLoader(val_t, batch_size=hp_tl.batch_size, shuffle=False, num_workers=0)
    te_t = DataLoader(test_t, batch_size=64, shuffle=False, num_workers=0)
    tl_epochs = max_epochs
    hist_tl, _ = train_with_early_stopping(
        resnet,
        tl,
        vl,
        device,
        lr=hp_tl.lr,
        weight_decay=hp_tl.weight_decay,
        max_epochs=tl_epochs,
        patience=min(patience, 8),
        run_label="ResNet18",
    )
    plot_history(hist_tl, "ResNet18 transfer", FIG_DIR / "resnet_curves.png")

    ev_tl = evaluate_full(resnet, te_t, device, num_classes)
    plot_confusion(
        ev_tl["confusion_matrix"],
        class_names,
        "ResNet18 Confusion",
        FIG_DIR / "resnet_confusion.png",
    )
    results_rows.append(
        {
            "model": "ResNet18_transfer",
            "hparams": hp_tl.__dict__,
            "accuracy": ev_tl["accuracy"],
            "macro_f1": ev_tl["macro_f1"],
            "top5_error": ev_tl["top5_error"],
        }
    )

    cam_mod_rn = get_last_conv_for_cam(resnet, "resnet")
    wrong_tl = collect_misclassified_indices(ev_tl["y_true"], ev_tl["y_pred"], 15)
    if len(wrong_tl) >= 10:
        save_misclassified_gradcam_grid(
            resnet,
            test_t,
            wrong_tl,
            device,
            cam_mod_rn,
            class_names,
            FIG_DIR / "resnet_misclassified_gradcam.png",
            for_transfer=True,
        )

    # ----- Adversarial robustness (use normalized tensors; clip in normalized space approx) -----
    # Note: attacks are applied in the same tensor domain as training; epsilon is small in normalized space.
    eps = 0.03 if not fast else 0.02
    pgd_steps = 7 if not fast else 3
    pgd_alpha = eps / 4

    def adv_eval(name: str, model: nn.Module, loader: DataLoader) -> dict[str, float]:
        clean = accuracy_under_attack(model, loader, device, "none", eps, pgd_steps, pgd_alpha)
        fgsm_a = accuracy_under_attack(model, loader, device, "fgsm", eps, pgd_steps, pgd_alpha)
        pgd_a = accuracy_under_attack(model, loader, device, "pgd", eps, pgd_steps, pgd_alpha)
        return {"clean": clean, "fgsm": fgsm_a, "pgd": pgd_a}

    adv_mlp = adv_eval("mlp", mlp, te_loader)
    adv_cnn = adv_eval("cnn", cnn, te_loader_cnn)
    adv_rn = adv_eval("resnet", resnet, te_t)

    adv_df = pd.DataFrame(
        [
            {"model": "MLP", **adv_mlp},
            {"model": "CNN", **adv_cnn},
            {"model": "ResNet18", **adv_rn},
        ]
    )
    adv_df.to_csv(FIG_DIR / "adversarial_robustness.csv", index=False)

    per_class_df = pd.DataFrame(
        {
            "class": class_names,
            "mlp_recall": ev_mlp["per_class_recall"],
            "cnn_recall": ev_cnn["per_class_recall"],
            "resnet_recall": ev_tl["per_class_recall"],
        }
    )
    per_class_df.to_csv(FIG_DIR / "per_class_recall.csv", index=False)

    pd.DataFrame(results_rows).to_csv(METRICS_CSV, index=False)
    save_discussion_text(
        results_rows + [{"adversarial_csv": str(FIG_DIR / "adversarial_robustness.csv")}],
        DISCUSSION_TXT,
    )

    print("Saved figures to", FIG_DIR)
    print("Metrics:", METRICS_CSV)
    print("Adversarial:\n", adv_df)


if __name__ == "__main__":
    main()
