"""
Cervix Visual AI - pipeline em arquivo único.

Objetivo:
    Pipeline completo de Computer Vision para imagens visuais do colo uterino com o uso de Ácido Acético (VIA), utilizando o conjunto de dados do IARC Cervical Image Bank.

Escopo científico:
    Imagens pós-ácido acético. 
    Diferenciação binária:
        - negative_or_low_grade
        - high_grade_or_cancer
    Resultados experimentais. Não usar para decisão clínica.
"""

from __future__ import annotations
import argparse
import copy
import json
import random
import re
import sys
import time
import zipfile
from io import BytesIO
from pathlib import Path, PurePosixPath
from typing import Any, Callable
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch
import torch.nn as nn
from PIL import Image
from sklearn.calibration import calibration_curve as sk_calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    auc,
    precision_recall_curve,
)
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Configuração padrão
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_IARC_ZIP = Path(r"C:\Users\ueldo\Desktop\IARC EXAME VISUAL.zip")
DEFAULT_DATA_DIR = SCRIPT_DIR / "data" / "iarc"
DEFAULT_MANIFEST = DEFAULT_DATA_DIR / "manifest.csv"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "outputs" / "arquivo_unico"
DEFAULT_CHECKPOINT = DEFAULT_OUTPUT_DIR / "best_model.pt"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
IMAGE_TYPE = "After acetic acid"
LABEL_TO_INDEX = {
    "negative_or_low_grade": 0,
    "high_grade_or_cancer": 1,
}
INDEX_TO_LABEL = {v: k for k, v in LABEL_TO_INDEX.items()}
HIGH_GRADE_TERMS = (
    "hsil", "cin2", "cin3", "high-grade", "high grade",
    "carcinoma", "cancer", "invasion", "invasive", "adenocarcinoma",
)

# ---------------------------------------------------------------------------
# Utilitários
# ---------------------------------------------------------------------------
_ANSI_RESET = "\033[0m"

def show_image_terminal(
    image: str | Path | np.ndarray | "Image.Image",
    width: int = 80,
    title: str = "",
) -> None:    
    # --- carregamento ---
    if isinstance(image, (str, Path)):
        img = Image.open(image).convert("RGB")
    elif isinstance(image, np.ndarray):
        arr = np.clip(image, 0, 255).astype(np.uint8)
        if arr.ndim == 2:                          # grayscale → RGB
            arr = np.stack([arr] * 3, axis=-1)
        img = Image.fromarray(arr)
    else:
        img = image.convert("RGB")

    # --- redimensionamento proporcional ---
    orig_w, orig_h = img.size
    new_w = min(width, orig_w)
    new_h = max(2, int(orig_h * new_w / orig_w))
    if new_h % 2:
        new_h += 1
    img = img.resize((new_w, new_h), Image.LANCZOS)
    px = np.array(img)                             # (H, W, 3) uint8

    # --- impressão ---
    sep = "  " + "─" * new_w
    if title:
        print(f"\n  \033[1m{title}\033[0m")
        print(sep)

    output_lines: list[str] = []
    for row in range(0, new_h - 1, 2):
        line_chars: list[str] = ["  "]
        for col in range(new_w):
            r1, g1, b1 = int(px[row,     col, 0]), int(px[row,     col, 1]), int(px[row,     col, 2])
            r2, g2, b2 = int(px[row + 1, col, 0]), int(px[row + 1, col, 1]), int(px[row + 1, col, 2])
            line_chars.append(
                f"\033[48;2;{r1};{g1};{b1}m"   # bg = pixel superior
                f"\033[38;2;{r2};{g2};{b2}m"   # fg = pixel inferior
                "▄"
            )
        line_chars.append(_ANSI_RESET)
        output_lines.append("".join(line_chars))

    print("\n".join(output_lines))
    if title:
        print(sep + "\n")

def show_images_terminal(
    images: list,
    titles: list[str] | None = None,
    width_per_image: int = 38,
) -> None:

    if not images:
        return
    titles = titles or [""] * len(images)

    # normaliza todas para PIL RGB e mesmo tamanho
    pil_images: list[Image.Image] = []
    for img in images:
        if isinstance(img, (str, Path)):
            pil_images.append(Image.open(img).convert("RGB"))
        elif isinstance(img, np.ndarray):
            arr = np.clip(img, 0, 255).astype(np.uint8)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            pil_images.append(Image.fromarray(arr))
        else:
            pil_images.append(img.convert("RGB"))

    # redimensiona cada imagem para width_per_image
    scaled: list[np.ndarray] = []
    for img in pil_images:
        orig_w, orig_h = img.size
        new_h = max(2, int(orig_h * width_per_image / orig_w))
        if new_h % 2:
            new_h += 1
        scaled.append(np.array(img.resize((width_per_image, new_h), Image.LANCZOS)))

    max_rows = max(px.shape[0] for px in scaled)

    # imprime títulos
    header = "  "
    for t in titles:
        header += t[:width_per_image].ljust(width_per_image) + "  "
    print(f"\n\033[1m{header}\033[0m")

    # imprime linhas de pixel lado a lado
    for row in range(0, max_rows - 1, 2):
        line = "  "
        for px in scaled:
            h = px.shape[0]
            for col in range(width_per_image):
                r1 = int(px[row,     col, 0]) if row     < h else 0
                g1 = int(px[row,     col, 1]) if row     < h else 0
                b1 = int(px[row,     col, 2]) if row     < h else 0
                r2 = int(px[row + 1, col, 0]) if row + 1 < h else 0
                g2 = int(px[row + 1, col, 1]) if row + 1 < h else 0
                b2 = int(px[row + 1, col, 2]) if row + 1 < h else 0
                line += (
                    f"\033[48;2;{r1};{g1};{b1}m"
                    f"\033[38;2;{r2};{g2};{b2}m▄"
                )
            line += _ANSI_RESET + "  "
        print(line)
    print()


def show_gradcam_terminal(
    original: str | Path | np.ndarray | "Image.Image",
    heatmap: np.ndarray,
    prob: float,
    threshold: float = 0.5,
    width: int = 60,
) -> None:
   
    # --- prepara imagem original ---
    if isinstance(original, (str, Path)):
        orig_pil = Image.open(original).convert("RGB")
    elif isinstance(original, np.ndarray):
        orig_pil = Image.fromarray(np.clip(original, 0, 255).astype(np.uint8))
    else:
        orig_pil = original.convert("RGB")

    w_each = width // 2

    # --- cria overlay GradCAM ---
    orig_arr = np.array(orig_pil.convert("RGB")).astype(np.uint8)
    hm_resized = cv2.resize(heatmap, (orig_arr.shape[1], orig_arr.shape[0]))
    hm_u8 = (np.clip(hm_resized, 0, 1) * 255).astype(np.uint8)
    hm_color = cv2.cvtColor(cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(orig_arr, 0.6, hm_color, 0.4, 0)
    label = "ALTO GRAU / CANCER" if prob >= threshold else "NEGATIVO / BAIXO GRAU"
    color_code = "\033[91m" if prob >= threshold else "\033[92m"  # vermelho / verde
    print(f"\n  {color_code}Predição: {label}  |  P(alto grau) = {prob:.3f}\033[0m")

    show_images_terminal(
        [orig_pil, Image.fromarray(overlay)],
        titles=["Original", "GradCAM Overlay"],
        width_per_image=w_each,
    )


def show_dataset_samples_terminal(
    manifest: pd.DataFrame,
    image_root: str | Path,
    n_per_class: int = 3,
    width_per_image: int = 28,
) -> None:
   
    image_root = Path(image_root)
    for label in sorted(manifest["label"].unique()):
        subset = manifest[manifest["label"] == label]
        sample = subset.sample(min(n_per_class, len(subset)), random_state=42)
        imgs = [image_root / row["image_path"] for _, row in sample.iterrows()]
        print(f"\033[1m  Classe: {label}  ({len(subset)} imagens)\033[0m")
        show_images_terminal(
            imgs,
            titles=[Path(p).name[:width_per_image] for p in imgs],
            width_per_image=width_per_image,
        )

def enable_windows_ansi() -> None:
    """Ativa suporte ANSI e UTF-8 no terminal Windows."""
    import ctypes, os
    if sys.platform == "win32":
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        os.system("")
        # força UTF-8 no stdout para suportar caracteres Unicode (═ ▓ ● ★ ▄)
        if hasattr(sys.stdout, "reconfigure"):
            try:
                sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


def ensure_dir(path: str | Path) -> Path:
    d = Path(path)
    d.mkdir(parents=True, exist_ok=True)
    return d


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(requested: str = "auto") -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def write_json(path: str | Path, payload: Any) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    with p.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(v: Any) -> Any:
    if isinstance(v, (np.integer, np.floating)):
        return v.item()
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, Path):
        return str(v)
    raise TypeError(f"Tipo não serializável: {type(v)!r}")

# ---------------------------------------------------------------------------
# Preparação do IARC Cervical Image Bank
# ---------------------------------------------------------------------------

def normalize_text(value: Any) -> str:
    text = "" if value is None else str(value)
    text = re.sub(r"<[^>]+>", " ", text)
    return re.sub(r"\s+", " ", text).strip().lower()

def derive_target(diagnosis: Any) -> int:
    return int(any(t in normalize_text(diagnosis) for t in HIGH_GRADE_TERMS))

def read_excel_from_zip(archive: zipfile.ZipFile, member: str, header: int) -> pd.DataFrame:
    return pd.read_excel(BytesIO(archive.read(member)), header=header)

def index_zip_images(archive: zipfile.ZipFile) -> dict[str, str]:
    index: dict[str, str] = {}
    for member in archive.namelist():
        if PurePosixPath(member).suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        fname = PurePosixPath(member).name.lower()
        if fname in index:
            raise ValueError(f"Nome duplicado no ZIP: {fname}")
        index[fname] = member
    return index

def assign_case_splits(
    metadata: pd.DataFrame,
    seed: int = 42,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> pd.Series:
    train_ids, remaining_ids = train_test_split(
        metadata["patient_id"],
        train_size=train_fraction,
        random_state=seed,
        stratify=metadata["target"],
    )
    remaining = metadata[metadata["patient_id"].isin(remaining_ids)]
    relative_val = val_fraction / (1.0 - train_fraction)
    val_ids, test_ids = train_test_split(
        remaining["patient_id"],
        train_size=relative_val,
        random_state=seed,
        stratify=remaining["target"],
    )
    mapping = {pid: "train" for pid in train_ids}
    mapping.update({pid: "val" for pid in val_ids})
    mapping.update({pid: "test" for pid in test_ids})
    return metadata["patient_id"].map(mapping)

def prepare_iarc_dataset(
    zip_path: str | Path = DEFAULT_IARC_ZIP,
    output_dir: str | Path = DEFAULT_DATA_DIR,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    zip_path = Path(zip_path).resolve()
    output_dir = ensure_dir(output_dir).resolve()
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP não encontrado: {zip_path}")

    print(f"Lendo ZIP: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        images_table = read_excel_from_zip(archive, "Cases - Images.xlsx", header=0)
        metadata = read_excel_from_zip(archive, "Cases Meta data.xlsx", header=1)
        zip_images = index_zip_images(archive)

        metadata = metadata.rename(columns={"\nProvisional diagnosis": "Provisional diagnosis"})
        metadata["Case Number"] = pd.to_numeric(metadata["Case Number"], errors="coerce")
        metadata = metadata.dropna(subset=["Case Number"]).copy()
        metadata["Case Number"] = metadata["Case Number"].astype(int)
        metadata["patient_id"] = metadata["Case Number"].map(lambda v: f"iarc_case_{v:03d}")
        metadata["target"] = metadata["Provisional diagnosis"].map(derive_target)
        metadata["label"] = metadata["target"].map(INDEX_TO_LABEL)

        images_table["Case Number"] = pd.to_numeric(images_table["Case Number"], errors="coerce")
        images_table = images_table.dropna(subset=["Case Number", "File"]).copy()
        images_table["Case Number"] = images_table["Case Number"].astype(int)
        selected = images_table[images_table["Type"] == IMAGE_TYPE].copy()

        selected_cases = set(selected["Case Number"].astype(int))
        metadata = metadata[metadata["Case Number"].isin(selected_cases)].copy()
        metadata["split"] = assign_case_splits(metadata, seed=seed)

        records: list[dict[str, Any]] = []
        for row in selected.to_dict(orient="records"):
            fname = str(row["File"]).strip()
            member = zip_images.get(fname.lower())
            if member is None:
                raise FileNotFoundError(f"Imagem não encontrada: {fname}")
            case_num = int(row["Case Number"])
            rel = Path("images") / f"case_{case_num:03d}" / fname
            dest = output_dir / rel
            ensure_dir(dest.parent)
            if not dest.exists():
                with archive.open(member) as src, dest.open("wb") as tgt:
                    tgt.write(src.read())
            records.append({
                "image_path": rel.as_posix(),
                "file_name": fname,
                "case_number": case_num,
                "image_type": row["Type"],
                "image_sort": row.get("Sort"),
            })

    image_manifest = pd.DataFrame(records).merge(
        metadata[["Case Number", "patient_id", "label", "target", "split",
                   "HPV", "Adequacy", "Transformation zone", "SwedeFinal",
                   "Provisional diagnosis", "Histopathology"]],
        left_on="case_number", right_on="Case Number",
        how="left", validate="many_to_one",
    )
    image_manifest["dataset"] = "IARC_Cervical_Image_Bank_Colposcopy"
    image_manifest["source_role"] = "primary"
    image_manifest = image_manifest.drop(columns=["Case Number"])

    case_manifest = metadata[["Case Number", "patient_id", "label", "target", "split",
                               "HPV", "Adequacy", "Transformation zone", "SwedeFinal",
                               "Provisional diagnosis", "Management", "Histopathology"]].copy()
    case_manifest["dataset"] = "IARC_Cervical_Image_Bank_Colposcopy"

    image_manifest.to_csv(output_dir / "manifest.csv", index=False, encoding="utf-8")
    case_manifest.to_csv(output_dir / "case_manifest.csv", index=False, encoding="utf-8")

    print(f"Manifesto: {output_dir / 'manifest.csv'}")
    print(f"Imagens: {len(image_manifest)} | Casos: {case_manifest['patient_id'].nunique()}")
    print(case_manifest.groupby(["split", "label"])["patient_id"].nunique().to_string())
    return image_manifest, case_manifest

# ---------------------------------------------------------------------------
# Dataset, transforms e DataLoaders
# ---------------------------------------------------------------------------

class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing: float = 0.05, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.smoothing = smoothing
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets * (1 - self.smoothing) + 0.5 * self.smoothing
        return nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )

class FocalLoss(nn.Module):   
    def __init__(self, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        pt = torch.exp(-bce)
        return ((1 - pt) ** self.gamma * bce).mean()

class CervixImageDataset(Dataset):
    def __init__(
        self,
        manifest: pd.DataFrame,
        image_root: str | Path,
        transform: Callable | None = None,
    ) -> None:
        self.manifest = manifest.reset_index(drop=True).copy()
        self.image_root = Path(image_root)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, str]:
        row = self.manifest.iloc[idx]
        img_path = Path(row["image_path"])
        if not img_path.is_absolute():
            img_path = self.image_root / img_path
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
        target = int(row["target"])
        if torch.isnan(img).any() or torch.isinf(img).any():
            raise RuntimeError(f"Tensor inválido na imagem: {img_path}")
        return img, torch.tensor(target, dtype=torch.float32), str(row["patient_id"])

def load_manifest(path: str | Path) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    required = {"image_path", "patient_id", "label", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"Colunas ausentes: {sorted(missing)}")
    if "target" not in manifest.columns:
        manifest["target"] = manifest["label"].map(LABEL_TO_INDEX)
    if manifest["target"].isna().any():
        raise ValueError("Manifesto contém rótulos desconhecidos.")
    leaking = manifest.groupby("patient_id")["split"].nunique()
    leaking = leaking[leaking > 1]
    if not leaking.empty:
        raise ValueError(f"Vazamento de pacientes: {list(leaking.index[:10])}")
    return manifest

def build_transforms(image_size: int) -> dict[str, transforms.Compose]:  
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    pad  = int(image_size * 0.15)
    return {
        "train": transforms.Compose([
            transforms.Resize((image_size + pad, image_size + pad)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.3),
            transforms.RandomRotation(degrees=30),
            transforms.RandomPerspective(distortion_scale=0.35, p=0.5),
            transforms.RandomAffine(degrees=20, translate=(0.12, 0.12),
                                    scale=(0.80, 1.20), shear=12),
            transforms.ColorJitter(brightness=0.45, contrast=0.40,
                                   saturation=0.35, hue=0.12),
            transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.4),
            transforms.RandomAutocontrast(p=0.3),
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            transforms.RandomGrayscale(p=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.35, scale=(0.02, 0.20), ratio=(0.3, 3.3)),
            transforms.RandomErasing(p=0.25, scale=(0.01, 0.10), ratio=(0.5, 2.0)),
        ]),
        "eval": transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]),
    }


def create_dataloaders(
    manifest: pd.DataFrame,
    image_root: str | Path,
    image_size: int,
    batch_size: int,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    tfm = build_transforms(image_size)
    loaders: dict[str, DataLoader] = {}
    for split in ["train", "val", "test", "external_test"]:
        frame = manifest[manifest["split"] == split].copy()
        if frame.empty:
            continue
        ds = CervixImageDataset(frame, image_root=image_root,
                                transform=tfm["train" if split == "train" else "eval"])
        loaders[split] = DataLoader(
            ds, batch_size=batch_size, shuffle=(split == "train"),
            num_workers=num_workers, pin_memory=torch.cuda.is_available(),
        )
    return loaders

# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------

def build_model(
    architecture: str = "efficientnet_b3",
    pretrained: bool = True,
    dropout: float = 0.40,
) -> nn.Module:
    
    def _head(in_f: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_f, 256),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(256, 1),
        )

    if architecture == "resnet34":
        weights = models.ResNet34_Weights.DEFAULT if pretrained else None
        model = models.resnet34(weights=weights)
        model.fc = _head(model.fc.in_features)
        return model

    if architecture == "efficientnet_b3":
        weights = models.EfficientNet_B3_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b3(weights=weights)
        model.classifier = _head(model.classifier[1].in_features)
        return model

    if architecture == "efficientnet_b4":
        weights = models.EfficientNet_B4_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b4(weights=weights)
        model.classifier = _head(model.classifier[1].in_features)
        return model

    if architecture == "convnext_tiny":
        weights = models.ConvNeXt_Tiny_Weights.DEFAULT if pretrained else None
        model = models.convnext_tiny(weights=weights)
        in_f = model.classifier[2].in_features
        model.classifier[2] = _head(in_f)
        return model

    raise ValueError(f"Arquitetura não suportada: {architecture}")


def set_backbone_trainable(model: nn.Module, architecture: str, trainable: bool) -> None:
    for p in model.parameters():
        p.requires_grad = trainable
    if architecture == "resnet34":
        head = model.fc
    elif architecture == "convnext_tiny":
        head = model.classifier
    else:
        head = model.classifier
    for p in head.parameters():
        p.requires_grad = True


def _gradcam_target_layer(model: nn.Module, architecture: str) -> nn.Module:
    if architecture == "resnet34":
        return model.layer4[-1]
    if architecture in ("efficientnet_b3", "efficientnet_b4"):
        return model.features[-1]
    if architecture == "convnext_tiny":
        return model.features[-1][-1]
    raise ValueError(f"GradCAM não suportado para {architecture}")

# ---------------------------------------------------------------------------
# GradCAM
# ---------------------------------------------------------------------------

class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients: torch.Tensor | None = None
        self.activations: torch.Tensor | None = None
        self._handles: list = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        def fwd(module, inp, out):
            self.activations = out.detach()

        def bwd(module, grad_in, grad_out):
            self.gradients = grad_out[0].detach()

        self._handles.append(self.target_layer.register_forward_hook(fwd))
        self._handles.append(self.target_layer.register_full_backward_hook(bwd))

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()

    def generate(self, tensor: torch.Tensor) -> np.ndarray:
        self.gradients = self.activations = None
        self.model.zero_grad()
        out = self.model(tensor)
        out.squeeze().backward()
        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks não capturaram gradientes/ativações.")
        w = self.gradients.mean(dim=[0, 2, 3])
        cam = torch.zeros(self.activations.shape[2:],
                          device=self.activations.device, dtype=torch.float32)
        for i, wi in enumerate(w):
            cam += wi * self.activations[0, i]
        cam = torch.relu(cam)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam.cpu().numpy()


def _overlay_heatmap(
    image_np: np.ndarray,
    heatmap: np.ndarray,
    bbox: tuple[int, int, int, int] | None = None,
    label: str = "",
) -> np.ndarray:
    if image_np.dtype != np.uint8:
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
    if image_np.ndim == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[-1] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    hm_u8 = (np.clip(heatmap, 0.0, 1.0) * 255).astype(np.uint8)
    hm_color = cv2.cvtColor(cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(image_np, 0.6, hm_color, 0.4, 0)

    if bbox is not None:
        x0, y0, x1, y1 = bbox
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (255, 255, 255), 2)
    if label:
        cv2.putText(overlay, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.9, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def get_bbox(heatmap: np.ndarray, threshold: float = 0.5) -> tuple[int, int, int, int] | None:
    mask = heatmap >= threshold
    if not np.any(mask):
        return None
    ys, xs = np.where(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def _make_gradcam_5panel(
    img_np: np.ndarray,
    heatmap: np.ndarray,
    label: str,
    prob: float,
    suptitle: str = "",
    save_path: str | Path | None = None,
) -> None:
 
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    h, w = img_np.shape[:2]
    hm_resized = cv2.resize(heatmap, (w, h))
    hm_u8 = (np.clip(hm_resized, 0, 1) * 255).astype(np.uint8)
    hm_color = cv2.cvtColor(cv2.applyColorMap(hm_u8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay = cv2.addWeighted(img_np, 0.6, hm_color, 0.4, 0)

    bbox = get_bbox(hm_resized, threshold=0.5)
    YELLOW = (255, 220, 0)
    font, fscale, fthick = cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2
    label_y = max(26, h // 20)

    def _draw(base: np.ndarray) -> np.ndarray:
        out = base.copy()
        if bbox is not None:
            cv2.rectangle(out, (bbox[0], bbox[1]), (bbox[2], bbox[3]), YELLOW, 2)
        cv2.putText(out, label, (8, label_y), font, fscale, YELLOW, fthick, cv2.LINE_AA)
        return out

    panels = [img_np, hm_color, overlay, _draw(img_np), _draw(overlay)]
    titles = [
        "Imagem Original", "Grad-CAM Colorido", "Grad-CAM Sobreposto",
        "Bounding Box", "Heatmap + Bounding Box",
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, panel, title in zip(axes.flat, panels, titles):
        ax.imshow(panel)
        ax.set_title(title, fontsize=11)
        ax.axis("off")
    axes.flat[5].set_visible(False)

    fig.suptitle(suptitle or f"GradCAM — {label} | P(alto grau) = {prob:.3f}", fontsize=14)
    plt.tight_layout()
    if save_path is not None:
        ensure_dir(Path(save_path).parent)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def generate_gradcam_for_image(
    model: nn.Module,
    image_path: str | Path,
    image_size: int,
    threshold: float,
    output_file: str | Path,
    device: torch.device,
    architecture: str,
) -> None:
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    tensor = build_transforms(image_size)["eval"](image).unsqueeze(0).to(device)
    target_layer = _gradcam_target_layer(model, architecture)
    gradcam = GradCAM(model, target_layer)

    with torch.no_grad():
        prob = float(torch.sigmoid(model(tensor).squeeze()).item())

    heatmap = gradcam.generate(tensor)
    gradcam.remove_hooks()

    heatmap = cv2.resize(heatmap, (w, h))
    label = "Lesao Alto Grau/Cancer" if prob >= threshold else "Negativo/Baixo Grau"
    img_np = np.asarray(image.convert("RGB"))
    _make_gradcam_5panel(img_np, heatmap, label, prob, save_path=output_file)


def generate_gradcam_exemplars(
    model: nn.Module,
    manifest: pd.DataFrame,
    image_root: Path,
    image_size: int,
    threshold: float,
    output_dir: Path,
    device: torch.device,
    architecture: str,
    case_ids: list[str],
    case_targets: np.ndarray,
    case_probabilities: np.ndarray,
) -> None:
  
    ensure_dir(output_dir)
    preds = (case_probabilities >= threshold).astype(int)

    indices = np.arange(len(case_ids))
    categories = {
        "TP": indices[(case_targets == 1) & (preds == 1)],
        "TN": indices[(case_targets == 0) & (preds == 0)],
        "FP": indices[(case_targets == 0) & (preds == 1)],
        "FN": indices[(case_targets == 1) & (preds == 0)],
    }
    pick = {
        "TP": lambda ix: ix[np.argmax(case_probabilities[ix])],
        "TN": lambda ix: ix[np.argmin(case_probabilities[ix])],
        "FP": lambda ix: ix[np.argmax(case_probabilities[ix])],
        "FN": lambda ix: ix[np.argmin(case_probabilities[ix])],
    }

    panels: dict[str, dict] = {}
    subtitles: dict[str, str] = {}

    model.eval()
    for outcome, idx_array in categories.items():
        if len(idx_array) == 0:
            continue
        chosen = pick[outcome](idx_array)
        pid = case_ids[chosen]
        prob = float(case_probabilities[chosen])
        true_label = INDEX_TO_LABEL[int(case_targets[chosen])]
        pred_label = INDEX_TO_LABEL[int(preds[chosen])]

        patient_rows = manifest[manifest["patient_id"] == pid]
        if patient_rows.empty:
            continue
        img_path = image_root / patient_rows.iloc[0]["image_path"]
        image = Image.open(img_path).convert("RGB")
        w, h = image.size
        tensor = build_transforms(image_size)["eval"](image).unsqueeze(0).to(device)
        target_layer = _gradcam_target_layer(model, architecture)
        gradcam = GradCAM(model, target_layer)
        heatmap = gradcam.generate(tensor)
        gradcam.remove_hooks()
        heatmap = cv2.resize(heatmap, (w, h))
        img_np = np.asarray(image.convert("RGB"))
        overlay = _overlay_heatmap(img_np, heatmap)
        panels[outcome] = {
            "img_np": img_np,
            "heatmap": heatmap,
            "overlay": overlay,
            "prob": prob,
            "pred_label": pred_label,
            "true_label": true_label,
        }
        subtitles[outcome] = (
            f"Predito: {pred_label}\nReal: {true_label}\n"
            f"P(alto grau) = {prob:.3f}"
        )

    titles_map = {
        "TP": "Verdadeiro Positivo (TP)",
        "TN": "Verdadeiro Negativo (TN)",
        "FP": "Falso Positivo (FP)",
        "FN": "Falso Negativo (FN)",
    }
    colors_map = {"TP": "green", "TN": "steelblue", "FP": "crimson", "FN": "darkorange"}

    # --- Painel 5-view individual por categoria ---
    for outcome, panel_data in panels.items():
        hm_path = output_dir / f"gradcam_{outcome.lower()}_5panel.png"
        _make_gradcam_5panel(
            panel_data["img_np"],
            panel_data["heatmap"],
            panel_data["pred_label"],
            panel_data["prob"],
            suptitle=(
                f"[{titles_map[outcome]}]  Predito: {panel_data['pred_label']}  |  "
                f"Real: {panel_data['true_label']}  |  P(alto grau) = {panel_data['prob']:.3f}"
            ),
            save_path=hm_path,
        )
        print(f"  GradCAM 5-panel [{outcome}]: {hm_path}")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))
    axes = axes.flatten()

    for ax, outcome in zip(axes, ["TP", "TN", "FP", "FN"]):
        if outcome in panels:
            ax.imshow(panels[outcome]["overlay"])
            ax.set_title(
                f"{titles_map[outcome]}\n{subtitles[outcome]}",
                color=colors_map[outcome], fontsize=11, pad=6,
            )
        else:
            ax.set_title(f"{titles_map[outcome]}\n(sem exemplar no conjunto de teste)",
                         color="gray", fontsize=10)
        ax.axis("off")

    fig.suptitle("GradCAM — Um Exemplar por Tipo de Classificação", fontsize=15)
    plt.tight_layout()
    out_path = output_dir / "gradcam_exemplars_panel.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Painel GradCAM resumo salvo: {out_path}")


def generate_gradcam_dataset(
    model: nn.Module,
    manifest: pd.DataFrame,
    image_root: str | Path,
    image_size: int,
    threshold: float,
    output_dir: Path,
    device: torch.device,
    architecture: str,
) -> None:
    img_dir = ensure_dir(output_dir / "gradcam_images")
    dataset = CervixImageDataset(
        manifest, image_root=image_root,
        transform=build_transforms(image_size)["eval"],
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    model.eval()
    for idx, (_, _, patient_id) in enumerate(tqdm(loader, desc="Gerando Grad-CAM")):
        pid_str = str(patient_id[0])
        img_name = Path(dataset.manifest.iloc[idx]["image_path"]).stem
        out_file = img_dir / f"{pid_str}_{img_name}_gradcam.png"
        generate_gradcam_for_image(
            model,
            dataset.image_root / dataset.manifest.iloc[idx]["image_path"],
            image_size, threshold, out_file, device, architecture,
        )

# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def binary_metrics(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    targets = np.asarray(targets, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    preds = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(targets, preds, labels=[0, 1]).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    npv = tn / (tn + fn) if (tn + fn) else float("nan")
    m: dict[str, Any] = {
        "n": int(len(targets)),
        "threshold": float(threshold),
        "accuracy": accuracy_score(targets, preds),
        "balanced_accuracy": balanced_accuracy_score(targets, preds),
        "sensitivity_recall": recall_score(targets, preds, zero_division=0),
        "specificity": specificity,
        "precision_ppv": precision_score(targets, preds, zero_division=0),
        "npv": npv,
        "f1": f1_score(targets, preds, zero_division=0),
        "brier_score": brier_score_loss(targets, probabilities),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    if len(np.unique(targets)) == 2:
        m["roc_auc"] = roc_auc_score(targets, probabilities)
        m["average_precision"] = average_precision_score(targets, probabilities)
    else:
        m["roc_auc"] = float("nan")
        m["average_precision"] = float("nan")
    return m


def aggregate_case_predictions(
    patient_ids: list[str],
    targets: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    grouped: dict[str, dict] = {}
    for pid, tgt, prob in zip(patient_ids, targets, probabilities):
        bucket = grouped.setdefault(pid, {"target": int(tgt), "probs": []})
        if bucket["target"] != int(tgt):
            raise ValueError(f"Rótulos conflitantes: {pid}")
        bucket["probs"].append(float(prob))
    ids = sorted(grouped)
    case_targets = np.array([grouped[k]["target"] for k in ids], dtype=int)
    case_probs = np.array([np.mean(grouped[k]["probs"]) for k in ids], dtype=float)
    return ids, case_targets, case_probs


def bootstrap_confidence_intervals(
    targets: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.50,
    iterations: int = 500,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    if iterations <= 0 or len(np.unique(targets)) < 2:
        return {}
    rng = np.random.default_rng(seed)
    names = ["roc_auc", "average_precision", "sensitivity_recall", "specificity",
             "precision_ppv", "npv", "f1", "balanced_accuracy", "brier_score"]
    samples: dict[str, list[float]] = {n: [] for n in names}
    for _ in range(iterations):
        ix = rng.integers(0, len(targets), size=len(targets))
        s_tgt = targets[ix]
        if len(np.unique(s_tgt)) < 2:
            continue
        vals = binary_metrics(s_tgt, probabilities[ix], threshold)
        for n in names:
            v = float(vals[n])
            if np.isfinite(v):
                samples[n].append(v)
    return {
        n: {"lower_95": float(np.percentile(v, 2.5)), "upper_95": float(np.percentile(v, 97.5))}
        for n, v in samples.items() if v
    }


def optimize_threshold(
    targets: np.ndarray,
    probabilities: np.ndarray,
    min_sensitivity: float = 0.80,
) -> tuple[float, float]:
    
    if np.isnan(probabilities).any():
        raise RuntimeError("NaN em probabilidades durante optimize_threshold")

    best_t, best_score = 0.35, -1.0
    for t in np.arange(0.05, 0.95, 0.01):
        m = binary_metrics(targets, probabilities, float(t))
        if m["sensitivity_recall"] < min_sensitivity:
            continue
        score = m["f1"]
        if score > best_score:
            best_score, best_t = score, float(t)
    
    if best_score < 0:
        for t in np.arange(0.05, 0.95, 0.01):
            m = binary_metrics(targets, probabilities, float(t))
            if m["sensitivity_recall"] >= 0.5:
                best_t = float(t)
                best_score = m["f1"]
                break

    return best_t, best_score

# ---------------------------------------------------------------------------
# Treinamento, avaliação e inferência
# ---------------------------------------------------------------------------

def mixup_batch(
    images: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.4,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(images.size(0), device=images.device)
    return (lam * images + (1 - lam) * images[idx],
            lam * targets + (1 - lam) * targets[idx])


def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    temperature: float = 1.0,
) -> dict[str, Any]:
    
    model.eval()
    all_targets: list[float] = []
    all_probs: list[float] = []
    all_pids: list[str] = []
    T = max(temperature, 1e-3)
    with torch.no_grad():
        for images, batch_targets, batch_pids in loader:
            images = images.to(device)
            p_orig   = torch.sigmoid(model(images).squeeze(1)                         / T)
            p_hflip  = torch.sigmoid(model(torch.flip(images, dims=[3])).squeeze(1)   / T)
            p_vflip  = torch.sigmoid(model(torch.flip(images, dims=[2])).squeeze(1)   / T)
            p_hvflip = torch.sigmoid(model(torch.flip(images, dims=[2, 3])).squeeze(1)/ T)
            probs = ((p_orig + p_hflip + p_vflip + p_hvflip) / 4).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_targets.extend(batch_targets.numpy().tolist())
            all_pids.extend(list(batch_pids))
    return {
        "targets": np.asarray(all_targets, dtype=int),
        "probabilities": np.asarray(all_probs, dtype=float),
        "patient_ids": all_pids,
    }


def predict_loader_simple(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    
    model.eval()
    all_targets: list[float] = []
    all_probs: list[float] = []
    all_pids: list[str] = []
    with torch.no_grad():
        for images, batch_targets, batch_pids in loader:
            images = images.to(device)
            probs = torch.sigmoid(model(images).squeeze(1)).cpu().numpy()
            all_probs.extend(probs.tolist())
            all_targets.extend(batch_targets.numpy().tolist())
            all_pids.extend(list(batch_pids))
    return {
        "targets": np.asarray(all_targets, dtype=int),
        "probabilities": np.asarray(all_probs, dtype=float),
        "patient_ids": all_pids,
    }


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    epochs: int,
    mixup_alpha: float = 0.4,
) -> float:
    model.train()
    running_loss = 0.0
    count = 0
    bar = tqdm(loader, desc=f"Treino {epoch + 1}/{epochs}", leave=False)
    for images, targets, _ in bar:
        images, targets = images.to(device), targets.to(device)
        if mixup_alpha > 0:
            images, targets = mixup_batch(images, targets, alpha=mixup_alpha)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images).squeeze(1)
        if not torch.isfinite(logits).all():
            raise RuntimeError(f"Logits inválidos na época {epoch + 1}")
        loss = criterion(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError(f"Loss inválida na época {epoch + 1}: {loss.item()}")
        loss.backward()
        # Gradient clipping: estabiliza treino de fine-tuning
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        bs = images.size(0)
        running_loss += float(loss.item()) * bs
        count += bs
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return running_loss / max(count, 1)


def evaluate_splits(
    model: nn.Module,
    loaders: dict[str, DataLoader],
    device: torch.device,
    threshold: float,
    output_dir: str | Path,
    manifest: pd.DataFrame,
    bootstrap_iterations: int,
    seed: int,
    temperature: float = 1.0,
) -> dict[str, Any]:
    output_dir = ensure_dir(output_dir)
    results: dict[str, Any] = {}
    for split in ["val", "test", "external_test"]:
        if split not in loaders:
            continue
        pred = predict_loader(model, loaders[split], device, temperature=temperature)
        img_m = binary_metrics(pred["targets"], pred["probabilities"], threshold)
        ids, c_tgt, c_prob = aggregate_case_predictions(
            pred["patient_ids"], pred["targets"], pred["probabilities"])
        case_m = binary_metrics(c_tgt, c_prob, threshold)
        ci = bootstrap_confidence_intervals(c_tgt, c_prob, threshold, bootstrap_iterations, seed)
        pd.DataFrame({
            "patient_id": ids,
            "target": c_tgt,
            "probability_high_grade_or_cancer": c_prob,
            "prediction": (c_prob >= threshold).astype(int),
        }).to_csv(output_dir / f"{split}_case_predictions.csv", index=False)
        results[split] = {
            "image_level": img_m,
            "case_level": case_m,
            "case_level_confidence_intervals": ci,
        }
    results["case_distribution"] = (
        manifest.groupby(["split", "label"])["patient_id"]
        .nunique().rename("cases").reset_index().to_dict(orient="records")
    )
    write_json(output_dir / "evaluation_metrics.json", results)
    return results

# ---------------------------------------------------------------------------
# Plots no terminal
# ---------------------------------------------------------------------------

def _ascii_line_chart(
    values: list[float],
    title: str = "",
    color: str = "\033[91m",
    width: int = 56,
    height: int = 8,
    y_decimals: int = 3,
    markers: dict[int, str] | None = None,
) -> list[str]:
  
    R = "\033[0m"
    if not values:
        return []
    n = len(values)
    y_min, y_max = min(values), max(values)
    y_range = max(y_max - y_min, 1e-9)

    grid: list[list[str]] = [[" "] * width for _ in range(height)]

    def _col(i: int) -> int:
        return int(i / max(n - 1, 1) * (width - 1))

    def _row(v: float) -> int:
        return max(0, min(height - 1, height - 1 - int((v - y_min) / y_range * (height - 1))))

    prev_c, prev_r = None, None
    for i, v in enumerate(values):
        c, r = _col(i), _row(v)
        if prev_c is not None and prev_c != c:
            steps = c - prev_c
            for s in range(1, steps + 1):
                ic = prev_c + s
                ir = int(prev_r + (r - prev_r) * s / steps)
                ir = max(0, min(height - 1, ir))
                if 0 <= ic < width and grid[ir][ic] == " ":
                    grid[ir][ic] = color + "▓" + R
        char = (markers or {}).get(i, "●")
        star_color = "\033[93m" if char == "★" else color
        grid[r][c] = star_color + char + R
        prev_c, prev_r = c, r

    lines: list[str] = []
    if title:
        lines.append("  \033[1m" + title.center(width + 10) + R)

    for ri in range(height):
        y_val = y_max - (ri / max(height - 1, 1)) * y_range
        label = f"{y_val:.{y_decimals}f}"
        lines.append(f"  {label:>7} │" + "".join(grid[ri]))

    lines.append("         └" + "─" * width)
    ticks = min(7, n)
    lbl_row = [" "] * width
    for ti in range(ticks):
        pos = int(ti / max(ticks - 1, 1) * (width - 1))
        val = str(int(ti / max(ticks - 1, 1) * (n - 1)) + 1)
        for j, ch in enumerate(val):
            if pos + j < width:
                lbl_row[pos + j] = ch
    lines.append("          " + "".join(lbl_row) + "  época")
    return lines


def _ascii_scatter_chart(
    x_vals: list[float],
    y_vals: list[float],
    title: str = "",
    color: str = "\033[94m",
    width: int = 56,
    height: int = 10,
    x_label: str = "x",
    y_label: str = "y",
    diagonal: bool = False,
) -> list[str]:
    
    R = "\033[0m"
    if not x_vals or not y_vals:
        return []
    x_min, x_max = min(x_vals), max(x_vals)
    y_min, y_max = min(y_vals), max(y_vals)
    x_range = max(x_max - x_min, 1e-9)
    y_range = max(y_max - y_min, 1e-9)

    grid: list[list[str]] = [[" "] * width for _ in range(height)]

    def _c(x: float) -> int:
        return max(0, min(width - 1, int((x - x_min) / x_range * (width - 1))))

    def _r(y: float) -> int:
        return max(0, min(height - 1, height - 1 - int((y - y_min) / y_range * (height - 1))))

    if diagonal:
        dg = "\033[90m"
        for xi in range(width):
            t = xi / (width - 1)
            ri = _r(y_min + t * y_range)
            if grid[ri][xi] == " ":
                grid[ri][xi] = dg + "·" + R

    for i in range(len(x_vals)):
        c, r = _c(x_vals[i]), _r(y_vals[i])
        grid[r][c] = color + "●" + R
        if i > 0:
            pc, pr = _c(x_vals[i - 1]), _r(y_vals[i - 1])
            steps = max(abs(c - pc), abs(r - pr), 1)
            for s in range(1, steps):
                ic = int(pc + s * (c - pc) / steps)
                ir = int(pr + s * (r - pr) / steps)
                ic, ir = max(0, min(width - 1, ic)), max(0, min(height - 1, ir))
                if grid[ir][ic] == " ":
                    grid[ir][ic] = color + "▓" + R

    lines: list[str] = []
    if title:
        lines.append("  \033[1m" + title.center(width + 10) + R)

    for ri in range(height):
        y_val = y_max - (ri / max(height - 1, 1)) * y_range
        lines.append(f"  {y_val:.2f} │" + "".join(grid[ri]))

    lines.append("       └" + "─" * width)
    ticks = 5
    lbl_row = [" "] * width
    for ti in range(ticks):
        pos = int(ti / max(ticks - 1, 1) * (width - 1))
        val = f"{x_min + ti / max(ticks - 1, 1) * x_range:.2f}"
        for j, ch in enumerate(val):
            if pos + j < width:
                lbl_row[pos + j] = ch
    lines.append("        " + "".join(lbl_row))
    lines.append("        " + x_label.center(width))
    return lines


_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


def _visual_len(s: str) -> int:
    return len(_ANSI_RE.sub("", s))


def _side_by_side(charts: list[list[str]], sep: str = "  ") -> list[str]:
    
    if not charts:
        return []
    max_rows = max(len(c) for c in charts)
    widths = [max((_visual_len(r) for r in c), default=0) for c in charts]
    result = []
    for i in range(max_rows):
        parts = []
        for j, chart in enumerate(charts):
            line = chart[i] if i < len(chart) else ""
            pad = widths[j] - _visual_len(line)
            parts.append(line + " " * max(0, pad))
        result.append(sep.join(parts))
    return result


def print_training_terminal(
    history: list[dict],
    best_epoch: int | None,
    case_targets: np.ndarray | None = None,
    case_probs: np.ndarray | None = None,
    threshold: float = 0.5,
) -> None:
 
    enable_windows_ansi()
    R = "\033[0m"
    LINE = "═" * 72

    # ── Tabela de melhores métricas ───────────────────────────────────────
    print(f"\n{LINE}")
    print("  MELHORES MÉTRICAS POR ÉPOCA")
    print(LINE)

    aucs   = [h["val_case_metrics"].get("roc_auc",              float("nan")) for h in history]
    f1s    = [h["val_case_metrics"].get("f1",                   float("nan")) for h in history]
    senss  = [h["val_case_metrics"].get("sensitivity_recall",   float("nan")) for h in history]
    specs  = [h["val_case_metrics"].get("specificity",          float("nan")) for h in history]
    losses = [h["train_loss"] for h in history]

    def _best(vals: list[float], hi: bool = True) -> tuple[int, float]:
        fn = max if hi else min
        valid = [(i, v) for i, v in enumerate(vals) if np.isfinite(v)]
        if not valid:
            return 0, float("nan")
        return fn(valid, key=lambda x: x[1])

    best_auc_ep,  best_auc_v  = _best(aucs)
    best_f1_ep,   best_f1_v   = _best(f1s)
    best_sens_ep, best_sens_v = _best(senss)
    best_spec_ep, best_spec_v = _best(specs)
    best_loss_ep, best_loss_v = _best(losses, hi=False)

    def _fmt_row(name: str, val: float, ep: int, extra: str) -> None:
        col = "\033[92m" if val >= 0.85 else "\033[93m" if val >= 0.70 else "\033[91m"
        print(f"  {name:<20} {col}{val:.4f}{R}  ← época {ep + 1:>3}    {extra}")

    _fmt_row("Melhor AUC-ROC",
             best_auc_v, best_auc_ep,
             f"F1={f1s[best_auc_ep]:.3f}  Sens={senss[best_auc_ep]:.3f}  Spec={specs[best_auc_ep]:.3f}")
    _fmt_row("Melhor F1",
             best_f1_v, best_f1_ep,
             f"AUC={aucs[best_f1_ep]:.3f}  Sens={senss[best_f1_ep]:.3f}  Spec={specs[best_f1_ep]:.3f}")
    _fmt_row("Melhor Sensibilidade",
             best_sens_v, best_sens_ep,
             f"AUC={aucs[best_sens_ep]:.3f}  F1={f1s[best_sens_ep]:.3f}   Spec={specs[best_sens_ep]:.3f}")
    _fmt_row("Melhor Especificidade",
             best_spec_v, best_spec_ep,
             f"AUC={aucs[best_spec_ep]:.3f}  F1={f1s[best_spec_ep]:.3f}   Sens={senss[best_spec_ep]:.3f}")
    print(f"  {'Menor Loss':<20} \033[96m{best_loss_v:.4f}{R}  ← época {best_loss_ep + 1:>3}")

    if best_epoch is not None:
        ep = best_epoch - 1
        print()
        print(f"  \033[1m★  Época selecionada (checkpoint): {best_epoch}{R}")
        print(f"     AUC={aucs[ep]:.4f}  F1={f1s[ep]:.4f}  "
              f"Sens={senss[ep]:.4f}  Spec={specs[ep]:.4f}  Loss={losses[ep]:.4f}")

    # ── Tabela completa por época ─────────────────────────────────────────
    tr_aucs_h = [h.get("train_case_metrics", {}).get("roc_auc", float("nan")) for h in history]
    tr_f1s_h  = [h.get("train_case_metrics", {}).get("f1",      float("nan")) for h in history]
    has_tr_h  = any(np.isfinite(v) for v in tr_aucs_h)

    if has_tr_h:
        print(f"\n  {'Ép':>3}  {'Loss':>6}  {'trAUC':>6}  {'trF1':>6}  {'valAUC':>6}  {'valF1':>6}  {'Sens':>6}  {'Spec':>6}")
        print("  " + "─" * 62)
    else:
        print(f"\n  {'Ép':>3}  {'Loss':>6}  {'AUC':>6}  {'F1':>6}  {'Sens':>6}  {'Spec':>6}")
        print("  " + "─" * 45)

    for h, tr_auc, tr_f1 in zip(history, tr_aucs_h, tr_f1s_h):
        ep   = h["epoch"]
        m    = h["val_case_metrics"]
        marker = "  ★" if ep == best_epoch else ""
        auc_v  = m.get("roc_auc",            float("nan"))
        f1_v   = m.get("f1",                 float("nan"))
        sens_v = m.get("sensitivity_recall", float("nan"))
        spec_v = m.get("specificity",        float("nan"))
        col = "\033[92m" if auc_v >= 0.85 else "\033[93m" if auc_v >= 0.70 else ""
        if has_tr_h:
            tr_col = "\033[92m" if tr_auc >= 0.85 else "\033[93m" if tr_auc >= 0.70 else ""
            print(f"  {ep:>3}  {h['train_loss']:6.4f}  "
                  f"{tr_col}{tr_auc:6.4f}{R}  {tr_f1:6.4f}  "
                  f"{col}{auc_v:6.4f}{R}  {f1_v:6.4f}  {sens_v:6.4f}  {spec_v:6.4f}{marker}")
        else:
            print(f"  {ep:>3}  {h['train_loss']:6.4f}  "
                  f"{col}{auc_v:6.4f}{R}  {f1_v:6.4f}  {sens_v:6.4f}  {spec_v:6.4f}{marker}")

    # ── Curvas de treino — painéis lado a lado ────────────────────────────
    print(f"\n{LINE}")
    print("  PROGRESSO DE TREINAMENTO")
    print(LINE)

    markers_auc  = {best_auc_ep:  "★"} if best_auc_ep  is not None else {}
    markers_f1   = {best_f1_ep:   "★"} if best_f1_ep   is not None else {}
    markers_loss = {best_loss_ep: "★"} if best_loss_ep is not None else {}

    c_loss = _ascii_line_chart(losses, "Training Loss",  "\033[91m",
                               width=38, height=12, markers=markers_loss)
    c_auc  = _ascii_line_chart(aucs,   "ROC-AUC (val)", "\033[94m",
                               width=38, height=12, markers=markers_auc)
    c_f1   = _ascii_line_chart(f1s,    "F1 (val)",      "\033[92m",
                               width=38, height=12, markers=markers_f1)

    for line in _side_by_side([c_loss, c_auc, c_f1], sep="   "):
        print(line)

    if has_tr_h:
        print()
        c_tr_auc = _ascii_line_chart(tr_aucs_h, "ROC-AUC (treino)", "\033[91m", width=38, height=10)
        c_vl_auc = _ascii_line_chart(aucs,       "ROC-AUC (val)",   "\033[94m", width=38, height=10)
        c_tr_f1  = _ascii_line_chart(tr_f1s_h,   "F1 (treino)",     "\033[91m", width=38, height=10)
        c_vl_f1  = _ascii_line_chart(f1s,         "F1 (val)",       "\033[92m", width=38, height=10)
        print("  Treino (vermelho) vs Validação (azul/verde) por época:")
        for line in _side_by_side([c_tr_auc, c_vl_auc], sep="   "):
            print(line)
        print()
        for line in _side_by_side([c_tr_f1, c_vl_f1], sep="   "):
            print(line)

    print()

    c_sens = _ascii_line_chart(senss, "Sensibilidade (val)",   "\033[95m",
                               width=52, height=10)
    c_spec = _ascii_line_chart(specs, "Especificidade (val)",  "\033[96m",
                               width=52, height=10)

    for line in _side_by_side([c_sens, c_spec], sep="   "):
        print(line)

    # ── Curvas ROC e PR ───────────────────────────────────────────────────
    if case_targets is not None and case_probs is not None and len(np.unique(case_targets)) == 2:
        fpr, tpr, _ = roc_curve(case_targets, case_probs)
        roc_auc_v   = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(case_targets, case_probs)
        ap_v = average_precision_score(case_targets, case_probs)

        print(f"\n{LINE}")
        print("  CURVAS ROC E PRECISÃO-RECALL (CONJUNTO DE TESTE)")
        print(LINE + "\n")

        roc_lines = _ascii_scatter_chart(
            fpr.tolist(), tpr.tolist(),
            title=f"Curva ROC  (AUC = {roc_auc_v:.3f})",
            color="\033[94m", width=52, height=12,
            x_label="False Positive Rate", y_label="TPR",
            diagonal=True,
        )
        pr_lines = _ascii_scatter_chart(
            rec.tolist(), prec.tolist(),
            title=f"Precisão-Recall  (AP = {ap_v:.3f})",
            color="\033[92m", width=52, height=12,
            x_label="Recall", y_label="Precisão",
            diagonal=False,
        )

        max_h = max(len(roc_lines), len(pr_lines))
        roc_lines  += [""] * (max_h - len(roc_lines))
        pr_lines   += [""] * (max_h - len(pr_lines))
        for rl, pl in zip(roc_lines, pr_lines):
            rp = rl.ljust(72)
            pp = pl
            print(rp + "  " + pp)

    print("\n" + LINE + "\n")

def print_terminal_metrics(
    results: dict,
    output_dir: Path,
    split: str = "test",
    case_targets: np.ndarray | None = None,
    case_probs: np.ndarray | None = None,
) -> None:
 
    enable_windows_ansi()
    R = "\033[0m"
    LINE = "═" * 72

    # Tenta carregar probs do CSV se não foram passadas diretamente
    if case_targets is None or case_probs is None:
        csv_path = Path(output_dir) / f"{split}_case_predictions.csv"
        if csv_path.exists():
            try:
                _df = pd.read_csv(csv_path)
                case_targets = _df["target"].to_numpy()
                case_probs   = _df["probability_high_grade_or_cancer"].to_numpy()
            except Exception:
                pass

    primary = split if split in results else next(
        (s for s in ("test", "val") if s in results), None
    )
    if primary is None:
        return

    has_val  = "val"  in results and results["val"].get("case_level")
    has_test = "test" in results and results["test"].get("case_level")

    print(f"\n{LINE}")
    label = "TESTE" if primary == "test" else "VALIDAÇÃO"
    print(f"  AVALIAÇÃO FINAL — CONJUNTO DE {label}")
    print(LINE)

    METRICS = [
        ("AUC-ROC",       "roc_auc"),
        ("AUC-PR (AP)",   "average_precision"),
        ("F1 Score",      "f1"),
        ("Sensibilidade", "sensitivity_recall"),
        ("Especificidade","specificity"),
        ("Acurácia Bal.", "balanced_accuracy"),
        ("Acurácia",      "accuracy"),
    ]

    # ── Tabela comparativa val vs test ────────────────────────────────────
    if has_val and has_test:
        m_val  = results["val"]["case_level"]
        m_test = results["test"]["case_level"]
        ci     = results[primary].get("case_level_confidence_intervals", {})

        print(f"\n  {'Métrica':<20}  {'VAL':>6}   {'TESTE':>6}  {'Delta':>6}  IC 95% (teste)")
        print("  " + "─" * 66)
        for name, key in METRICS:
            v_v = m_val.get(key, float("nan"))
            v_t = m_test.get(key, float("nan"))
            if not (np.isfinite(v_v) or np.isfinite(v_t)):
                continue
            delta = v_t - v_v if (np.isfinite(v_v) and np.isfinite(v_t)) else float("nan")
            d_str  = f"{delta:+.3f}" if np.isfinite(delta) else "   n/a"
            d_col  = "\033[92m" if (np.isfinite(delta) and delta >= 0) else "\033[91m"
            t_col  = "\033[92m" if v_t >= 0.85 else "\033[93m" if v_t >= 0.70 else "\033[91m"
            vv_col = "\033[92m" if v_v >= 0.85 else "\033[93m" if v_v >= 0.70 else "\033[91m"
            ci_e   = ci.get(key, {})
            ci_str = f"[{ci_e['lower_95']:.3f}–{ci_e['upper_95']:.3f}]" if ci_e else ""
            print(f"  {name:<20}  {vv_col}{v_v:.3f}{R}   "
                  f"{t_col}{v_t:.3f}{R}  {d_col}{d_str}{R}  {ci_str}")
    else:
        m  = results[primary].get("case_level", {})
        ci = results[primary].get("case_level_confidence_intervals", {})
        BAR_W = 35
        print(f"\n  {'Métrica':<20} {'Valor':>6}  {'Barra':<{BAR_W}}  IC 95%")
        print("  " + "─" * (20 + 6 + BAR_W + 16))
        for name, key in METRICS:
            val = m.get(key, float("nan"))
            if not np.isfinite(val):
                continue
            bar = "█" * int(round(val * BAR_W)) + "░" * (BAR_W - int(round(val * BAR_W)))
            ci_e   = ci.get(key, {})
            ci_str = f"[{ci_e['lower_95']:.3f}–{ci_e['upper_95']:.3f}]" if ci_e else ""
            color  = "\033[92m" if val >= 0.85 else "\033[93m" if val >= 0.70 else "\033[91m"
            print(f"  {name:<20} {color}{val:.3f}{R}  {bar}  {ci_str}")

    # ── Matriz de confusão com bordas ─────────────────────────────────────
    m   = results[primary].get("case_level", {})
    tn  = m.get("tn", 0); fp = m.get("fp", 0)
    fn  = m.get("fn", 0); tp = m.get("tp", 0)
    tot = tn + fp + fn + tp
    if tot > 0:
        sens_v = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        spec_v = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
        ppv    = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        npv    = tn / (tn + fn) if (tn + fn) > 0 else float("nan")
        print(f"\n  Matriz de Confusão — {primary.upper()} — nível de caso (n={tot})")
        print("                         Predito")
        print("                   ┌────────────┬────────────┐")
        print("                   │  Negativo  │  Positivo  │")
        print("  ─────────────────┼────────────┼────────────┤")
        print(f"  Real  Negativo  │ \033[92m  TN={tn:<5d}\033[0m  │ \033[91m  FP={fp:<5d}\033[0m  │")
        print(f"        Positivo  │ \033[91m  FN={fn:<5d}\033[0m  │ \033[92m  TP={tp:<5d}\033[0m  │")
        print("                   └────────────┴────────────┘")
        s_col = "\033[92m" if sens_v >= 0.85 else "\033[93m" if sens_v >= 0.70 else "\033[91m"
        p_col = "\033[92m" if spec_v >= 0.85 else "\033[93m" if spec_v >= 0.70 else "\033[91m"
        print(f"  Sens={s_col}{sens_v:.3f}{R}  Spec={p_col}{spec_v:.3f}{R}  "
              f"PPV={ppv:.3f}  NPV={npv:.3f}")

    # ── Curvas ROC e PR em ASCII ──────────────────────────────────────────
    if (case_targets is not None and case_probs is not None
            and len(case_targets) > 0 and len(np.unique(case_targets)) == 2):
        fpr, tpr, _  = roc_curve(case_targets, case_probs)
        roc_auc_v    = auc(fpr, tpr)
        prec, rec, _ = precision_recall_curve(case_targets, case_probs)
        ap_v = average_precision_score(case_targets, case_probs)

        print(f"\n  {'─' * 68}")

        roc_lines = _ascii_scatter_chart(
            fpr.tolist(), tpr.tolist(),
            title=f"ROC — {primary.upper()}  (AUC={roc_auc_v:.3f})",
            color="\033[94m", width=50, height=12,
            x_label="FPR (1 - Especificidade)", diagonal=True,
        )
        pr_lines = _ascii_scatter_chart(
            rec.tolist(), prec.tolist(),
            title=f"Precisão-Recall — {primary.upper()}  (AP={ap_v:.3f})",
            color="\033[92m", width=50, height=12,
            x_label="Recall (Sensibilidade)", diagonal=False,
        )
        n = max(len(roc_lines), len(pr_lines))
        roc_lines += [""] * (n - len(roc_lines))
        pr_lines  += [""] * (n - len(pr_lines))
        for rl, pl in zip(roc_lines, pr_lines):
            print(rl.ljust(68) + "  " + pl)

    print("\n" + LINE + "\n")


# ---------------------------------------------------------------------------
# Geração de plots
# ---------------------------------------------------------------------------

def plot_calibration_curve(
    targets: np.ndarray,
    probabilities: np.ndarray,
    output_dir: Path,
    val_targets: np.ndarray | None = None,
    val_probs: np.ndarray | None = None,
) -> None:
    
    if len(np.unique(targets)) < 2:
        return
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Calibração perfeita")
    if val_targets is not None and val_probs is not None and len(np.unique(val_targets)) == 2:
        prob_true_v, prob_pred_v = sk_calibration_curve(val_targets, val_probs, n_bins=10, strategy="quantile")
        ax.plot(prob_pred_v, prob_true_v, marker="o", linewidth=2, color="tab:blue", label="Validação")
        ax.fill_between(prob_pred_v, prob_true_v, prob_pred_v, alpha=0.10, color="tab:blue")
    prob_true, prob_pred = sk_calibration_curve(targets, probabilities, n_bins=10, strategy="quantile")
    ax.plot(prob_pred, prob_true, marker="s", linewidth=2, color="tab:orange", label="Teste")
    ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.10, color="tab:orange")
    ax.set_xlabel("Probabilidade predita")
    ax.set_ylabel("Frequência observada")
    ax.set_title("Curva de Calibração (Reliability Diagram) — Validação vs Teste")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "calibration_curve.png", dpi=200)
    plt.show()
    plt.close()

    # versão interativa
    fig_go = go.Figure()
    fig_go.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                line=dict(dash="dash"), name="Perfeito"))
    if val_targets is not None and val_probs is not None and len(np.unique(val_targets)) == 2:
        fig_go.add_trace(go.Scatter(x=prob_pred_v.tolist(), y=prob_true_v.tolist(),
                                    mode="lines+markers", name="Validação"))
    fig_go.add_trace(go.Scatter(x=prob_pred.tolist(), y=prob_true.tolist(),
                                mode="lines+markers", name="Teste"))
    fig_go.update_layout(title="Curva de Calibração — Validação vs Teste",
                         xaxis_title="Probabilidade predita",
                         yaxis_title="Frequência observada")
    fig_go.write_html(output_dir / "calibration_curve.html")


def plot_threshold_sweep(
    targets: np.ndarray,
    probabilities: np.ndarray,
    best_threshold: float,
    output_dir: Path,
    val_targets: np.ndarray | None = None,
    val_probs: np.ndarray | None = None,
) -> None:
    
    if len(np.unique(targets)) < 2:
        return
    thresholds = np.arange(0.05, 0.951, 0.01)
    f1s, senss, specs = [], [], []
    for t in thresholds:
        m = binary_metrics(targets, probabilities, float(t))
        f1s.append(m["f1"])
        senss.append(m["sensitivity_recall"])
        specs.append(m["specificity"])

    has_val_sweep = val_targets is not None and val_probs is not None and len(np.unique(val_targets)) == 2
    if has_val_sweep:
        f1s_v, senss_v, specs_v = [], [], []
        for t in thresholds:
            m = binary_metrics(val_targets, val_probs, float(t))
            f1s_v.append(m["f1"])
            senss_v.append(m["sensitivity_recall"])
            specs_v.append(m["specificity"])
        fig, axes = plt.subplots(1, 2, figsize=(16, 5))
        for ax, (f1_data, sens_data, spec_data), title in zip(
            axes,
            [(f1s_v, senss_v, specs_v), (f1s, senss, specs)],
            ["Validação", "Teste"],
        ):
            ax.plot(thresholds, f1_data,   linewidth=2, label="F1")
            ax.plot(thresholds, sens_data, linewidth=2, label="Sensibilidade (Recall)")
            ax.plot(thresholds, spec_data, linewidth=2, label="Especificidade")
            ax.axvline(best_threshold, color="black", linestyle="--",
                       label=f"Threshold ótimo ({best_threshold:.2f})")
            ax.set_xlabel("Threshold"); ax.set_ylabel("Valor da métrica")
            ax.set_title(f"Métricas × Threshold — {title}")
            ax.legend(); ax.grid(alpha=0.3)
        plt.suptitle("Métricas × Threshold — Validação vs Teste", fontsize=14)
    else:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.plot(thresholds, f1s,   linewidth=2, label="F1")
        ax.plot(thresholds, senss, linewidth=2, label="Sensibilidade (Recall)")
        ax.plot(thresholds, specs, linewidth=2, label="Especificidade")
        ax.axvline(best_threshold, color="black", linestyle="--",
                   label=f"Threshold ótimo ({best_threshold:.2f})")
        ax.set_xlabel("Threshold"); ax.set_ylabel("Valor da métrica")
        ax.set_title("Métricas × Threshold — Teste")
        ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "threshold_sweep.png", dpi=200)
    plt.show()
    plt.close()

    # interativo
    fig_go = go.Figure()
    if has_val_sweep:
        fig_go.add_trace(go.Scatter(x=thresholds.tolist(), y=f1s_v,   mode="lines", name="F1 (Val)", line=dict(dash="dot")))
        fig_go.add_trace(go.Scatter(x=thresholds.tolist(), y=senss_v, mode="lines", name="Sens. (Val)", line=dict(dash="dot")))
        fig_go.add_trace(go.Scatter(x=thresholds.tolist(), y=specs_v, mode="lines", name="Spec. (Val)", line=dict(dash="dot")))
    fig_go.add_trace(go.Scatter(x=thresholds.tolist(), y=f1s,   mode="lines", name="F1 (Teste)"))
    fig_go.add_trace(go.Scatter(x=thresholds.tolist(), y=senss, mode="lines", name="Sensibilidade (Teste)"))
    fig_go.add_trace(go.Scatter(x=thresholds.tolist(), y=specs, mode="lines", name="Especificidade (Teste)"))
    fig_go.add_vline(x=best_threshold, line_dash="dash", annotation_text=f"Threshold={best_threshold:.2f}")
    fig_go.update_layout(title="Métricas × Threshold — Validação vs Teste",
                         xaxis_title="Threshold", yaxis_title="Métrica")
    fig_go.write_html(output_dir / "threshold_sweep.html")


def generate_plots(
    history: list[dict],
    case_targets: np.ndarray,
    case_probabilities: np.ndarray,
    output_dir: str | Path,
    threshold: float = 0.50,
    results: dict | None = None,
    val_targets: np.ndarray | None = None,
    val_probs: np.ndarray | None = None,
) -> None:
    output_dir = Path(output_dir)

    # --- Progresso de treino ---
    history_df = pd.DataFrame(history)
    losses     = history_df["train_loss"].tolist()
    val_aucs   = [h["val_case_metrics"].get("roc_auc", float("nan")) for h in history]
    val_f1s    = [h["val_case_metrics"].get("f1",      float("nan")) for h in history]
    val_senss  = [h["val_case_metrics"].get("sensitivity_recall", float("nan")) for h in history]
    val_specs  = [h["val_case_metrics"].get("specificity",        float("nan")) for h in history]
    tr_aucs    = [h.get("train_case_metrics", {}).get("roc_auc", float("nan")) for h in history]
    tr_f1s     = [h.get("train_case_metrics", {}).get("f1",      float("nan")) for h in history]
    tr_senss   = [h.get("train_case_metrics", {}).get("sensitivity_recall", float("nan")) for h in history]
    tr_specs   = [h.get("train_case_metrics", {}).get("specificity",        float("nan")) for h in history]
    epochs_ax  = range(1, len(history) + 1)
    has_tr     = any(np.isfinite(v) for v in tr_aucs)

    # painel 2×3: Loss | AUC | F1 / Sens | Spec | (vazio)
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # linha 0 — Loss
    axes[0, 0].plot(epochs_ax, losses, marker="o", linewidth=2, color="tab:red", label="Treino")
    axes[0, 0].set_title("Training Loss"); axes[0, 0].set_xlabel("Época"); axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()

    # linha 0 — AUC
    if has_tr:
        axes[0, 1].plot(epochs_ax, tr_aucs, marker="o", linewidth=2, color="tab:red",  label="Treino")
    axes[0, 1].plot(epochs_ax, val_aucs, marker="s", linewidth=2, color="tab:blue",  label="Validação")
    if results and "test" in results:
        _v = results["test"]["case_level"].get("roc_auc")
        if _v is not None and np.isfinite(_v):
            axes[0, 1].axhline(_v, color="tab:orange", linestyle="--", linewidth=2, label=f"Teste ({_v:.3f})")
    axes[0, 1].set_title("ROC-AUC por Época"); axes[0, 1].set_xlabel("Época"); axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()

    # linha 0 — F1
    if has_tr:
        axes[0, 2].plot(epochs_ax, tr_f1s, marker="o", linewidth=2, color="tab:red",   label="Treino")
    axes[0, 2].plot(epochs_ax, val_f1s, marker="^", linewidth=2, color="tab:green", label="Validação")
    if results and "test" in results:
        _v = results["test"]["case_level"].get("f1")
        if _v is not None and np.isfinite(_v):
            axes[0, 2].axhline(_v, color="tab:orange", linestyle="--", linewidth=2, label=f"Teste ({_v:.3f})")
    axes[0, 2].set_title("F1 por Época"); axes[0, 2].set_xlabel("Época"); axes[0, 2].grid(alpha=0.3)
    axes[0, 2].legend()

    # linha 1 — Sensibilidade
    if has_tr:
        axes[1, 0].plot(epochs_ax, tr_senss, marker="o", linewidth=2, color="tab:red",    label="Treino")
    axes[1, 0].plot(epochs_ax, val_senss, marker="D", linewidth=2, color="tab:purple", label="Validação")
    if results and "test" in results:
        _v = results["test"]["case_level"].get("sensitivity_recall")
        if _v is not None and np.isfinite(_v):
            axes[1, 0].axhline(_v, color="tab:orange", linestyle="--", linewidth=2, label=f"Teste ({_v:.3f})")
    axes[1, 0].set_title("Sensibilidade por Época"); axes[1, 0].set_xlabel("Época"); axes[1, 0].grid(alpha=0.3)
    axes[1, 0].legend()

    # linha 1 — Especificidade
    if has_tr:
        axes[1, 1].plot(epochs_ax, tr_specs, marker="o", linewidth=2, color="tab:red",  label="Treino")
    axes[1, 1].plot(epochs_ax, val_specs, marker="P", linewidth=2, color="tab:cyan", label="Validação")
    if results and "test" in results:
        _v = results["test"]["case_level"].get("specificity")
        if _v is not None and np.isfinite(_v):
            axes[1, 1].axhline(_v, color="tab:orange", linestyle="--", linewidth=2, label=f"Teste ({_v:.3f})")
    axes[1, 1].set_title("Especificidade por Época"); axes[1, 1].set_xlabel("Época"); axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()

    axes[1, 2].set_visible(False)

    plt.suptitle("Progresso de Treinamento — Treino / Validação / Teste (linha pontilhada)", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_dir / "training_progress.png", dpi=200)
    plt.show()
    plt.close()

    # interativo: progresso de treinamento (2×3 subplots)
    epochs_list = list(epochs_ax)
    fig_tp = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            "Training Loss", "ROC-AUC por Época", "F1 por Época",
            "Sensibilidade por Época", "Especificidade por Época", "",
        ],
        vertical_spacing=0.18,
        horizontal_spacing=0.08,
    )

    # --- Loss (1,1) ---
    fig_tp.add_trace(go.Scatter(
        x=epochs_list, y=losses, mode="lines+markers",
        name="Loss (Treino)", line=dict(color="red"),
    ), row=1, col=1)

    # --- AUC (1,2) ---
    if has_tr:
        fig_tp.add_trace(go.Scatter(
            x=epochs_list, y=tr_aucs, mode="lines+markers",
            name="AUC (Treino)", line=dict(color="red"),
            legendgroup="treino", showlegend=True,
        ), row=1, col=2)
    fig_tp.add_trace(go.Scatter(
        x=epochs_list, y=val_aucs, mode="lines+markers",
        name="AUC (Val)", line=dict(color="blue"),
        legendgroup="val", showlegend=True,
    ), row=1, col=2)
    if results and "test" in results:
        _v = results["test"]["case_level"].get("roc_auc")
        if _v is not None and np.isfinite(_v):
            fig_tp.add_hline(y=_v, line_dash="dash", line_color="orange",
                             annotation_text=f"Teste AUC={_v:.3f}",
                             annotation_position="top right", row=1, col=2)

    # --- F1 (1,3) ---
    if has_tr:
        fig_tp.add_trace(go.Scatter(
            x=epochs_list, y=tr_f1s, mode="lines+markers",
            name="F1 (Treino)", line=dict(color="red"),
            legendgroup="treino", showlegend=False,
        ), row=1, col=3)
    fig_tp.add_trace(go.Scatter(
        x=epochs_list, y=val_f1s, mode="lines+markers",
        name="F1 (Val)", line=dict(color="green"),
        legendgroup="val", showlegend=False,
    ), row=1, col=3)
    if results and "test" in results:
        _v = results["test"]["case_level"].get("f1")
        if _v is not None and np.isfinite(_v):
            fig_tp.add_hline(y=_v, line_dash="dash", line_color="orange",
                             annotation_text=f"Teste F1={_v:.3f}",
                             annotation_position="top right", row=1, col=3)

    # --- Sensibilidade (2,1) ---
    if has_tr:
        fig_tp.add_trace(go.Scatter(
            x=epochs_list, y=tr_senss, mode="lines+markers",
            name="Sens. (Treino)", line=dict(color="red"),
            legendgroup="treino", showlegend=False,
        ), row=2, col=1)
    fig_tp.add_trace(go.Scatter(
        x=epochs_list, y=val_senss, mode="lines+markers",
        name="Sens. (Val)", line=dict(color="purple"),
        legendgroup="val", showlegend=False,
    ), row=2, col=1)
    if results and "test" in results:
        _v = results["test"]["case_level"].get("sensitivity_recall")
        if _v is not None and np.isfinite(_v):
            fig_tp.add_hline(y=_v, line_dash="dash", line_color="orange",
                             annotation_text=f"Teste Sens={_v:.3f}",
                             annotation_position="top right", row=2, col=1)

    # --- Especificidade (2,2) ---
    if has_tr:
        fig_tp.add_trace(go.Scatter(
            x=epochs_list, y=tr_specs, mode="lines+markers",
            name="Spec. (Treino)", line=dict(color="red"),
            legendgroup="treino", showlegend=False,
        ), row=2, col=2)
    fig_tp.add_trace(go.Scatter(
        x=epochs_list, y=val_specs, mode="lines+markers",
        name="Spec. (Val)", line=dict(color="cyan"),
        legendgroup="val", showlegend=False,
    ), row=2, col=2)
    if results and "test" in results:
        _v = results["test"]["case_level"].get("specificity")
        if _v is not None and np.isfinite(_v):
            fig_tp.add_hline(y=_v, line_dash="dash", line_color="orange",
                             annotation_text=f"Teste Spec={_v:.3f}",
                             annotation_position="top right", row=2, col=2)

    fig_tp.update_layout(
        title="Progresso de Treinamento — Treino / Validação / Teste (linha pontilhada)",
        height=700,
        showlegend=True,
    )
    fig_tp.write_html(output_dir / "training_progress.html")

    if len(np.unique(case_targets)) < 2:
        return   # sem métricas de ranking sem ambas as classes

    has_val = (val_targets is not None and val_probs is not None
               and len(np.unique(val_targets)) == 2)

    # --- ROC ---
    fpr, tpr, _ = roc_curve(case_targets, case_probabilities)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Random")
    if has_val:
        fpr_v, tpr_v, _ = roc_curve(val_targets, val_probs)
        roc_auc_v = auc(fpr_v, tpr_v)
        ax.plot(fpr_v, tpr_v, linewidth=2, color="tab:blue", label=f"Validação (AUC = {roc_auc_v:.3f})")
    ax.plot(fpr, tpr, linewidth=2, color="tab:orange", label=f"Teste (AUC = {roc_auc:.3f})")
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Validação vs Teste"); ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=200)
    plt.show()
    plt.close()

    fig_roc = go.Figure([go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                                    line=dict(dash="dash"), name="Random")])
    if has_val:
        fig_roc.add_trace(go.Scatter(x=fpr_v.tolist(), y=tpr_v.tolist(), mode="lines",
                                     name=f"Validação (AUC={roc_auc_v:.3f})"))
    fig_roc.add_trace(go.Scatter(x=fpr.tolist(), y=tpr.tolist(), mode="lines",
                                 name=f"Teste (AUC={roc_auc:.3f})"))
    fig_roc.update_layout(title="ROC Curve — Validação vs Teste",
                          xaxis_title="FPR", yaxis_title="TPR")
    fig_roc.write_html(output_dir / "roc_curve.html")

    # --- Precision-Recall ---
    precision, recall, _ = precision_recall_curve(case_targets, case_probabilities)
    ap = average_precision_score(case_targets, case_probabilities)

    fig, ax = plt.subplots(figsize=(7, 6))
    if has_val:
        precision_v, recall_v, _ = precision_recall_curve(val_targets, val_probs)
        ap_v = average_precision_score(val_targets, val_probs)
        ax.plot(recall_v, precision_v, linewidth=2, color="tab:blue", label=f"Validação (AP = {ap_v:.3f})")
    ax.plot(recall, precision, linewidth=2, color="tab:orange", label=f"Teste (AP = {ap:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve — Validação vs Teste"); ax.legend(); ax.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall.png", dpi=200)
    plt.show()
    plt.close()

    fig_pr = go.Figure()
    if has_val:
        fig_pr.add_trace(go.Scatter(x=recall_v.tolist(), y=precision_v.tolist(), mode="lines",
                                    name=f"Validação (AP={ap_v:.3f})"))
    fig_pr.add_trace(go.Scatter(x=recall.tolist(), y=precision.tolist(), mode="lines",
                                name=f"Teste (AP={ap:.3f})"))
    fig_pr.update_layout(title="Precision-Recall Curve — Validação vs Teste",
                         xaxis_title="Recall", yaxis_title="Precision")
    fig_pr.write_html(output_dir / "precision_recall.html")

    # --- Confusion Matrix ---
    preds_bin = (case_probabilities >= threshold).astype(int)
    cm_test = confusion_matrix(case_targets, preds_bin)

    def _draw_cm(ax, cm, title):
        im = ax.imshow(cm, cmap="Blues")
        plt.colorbar(im, ax=ax)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Negativo", "Positivo"])
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Negativo", "Positivo"])
        ax.set_xlabel("Predito"); ax.set_ylabel("Real"); ax.set_title(title)

    if has_val:
        val_preds_bin = (val_probs >= threshold).astype(int)
        cm_val = confusion_matrix(val_targets, val_preds_bin)
        fig, axes_cm = plt.subplots(1, 2, figsize=(12, 5))
        _draw_cm(axes_cm[0], cm_val,  "Confusion Matrix — Validação")
        _draw_cm(axes_cm[1], cm_test, "Confusion Matrix — Teste")
        plt.suptitle("Confusion Matrix — Validação vs Teste", fontsize=14)
    else:
        fig, ax_cm = plt.subplots(figsize=(6, 5))
        _draw_cm(ax_cm, cm_test, "Confusion Matrix — Teste")
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=200)
    plt.show()
    plt.close()
    px.imshow(cm_test, text_auto=True, labels=dict(x="Predito", y="Real"),
              x=["Negativo", "Positivo"], y=["Negativo", "Positivo"],
              title="Confusion Matrix — Teste").write_html(output_dir / "confusion_matrix.html")

    # --- Calibração ---
    plot_calibration_curve(case_targets, case_probabilities, output_dir,
                           val_targets if has_val else None, val_probs if has_val else None)

    # --- Threshold sweep ---
    plot_threshold_sweep(case_targets, case_probabilities, threshold, output_dir,
                         val_targets if has_val else None, val_probs if has_val else None)

    # --- Dashboard de métricas: validação vs teste ---
    if results is not None and "val" in results and "test" in results:
        metric_names = ["Accuracy", "Balanced Acc", "Precision", "F1", "ROC-AUC", "AP"]
        keys = ["accuracy", "balanced_accuracy", "precision_ppv", "f1", "roc_auc", "average_precision"]
        val_vals  = [results["val"]["case_level"][k] for k in keys]
        test_vals = [results["test"]["case_level"][k] for k in keys]
        x = np.arange(len(metric_names))

        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(x - 0.2, val_vals, 0.4, label="Validação", color="steelblue")
        ax.bar(x + 0.2, test_vals, 0.4, label="Teste", color="coral")
        for xi, (v, t) in zip(x, zip(val_vals, test_vals)):
            ax.text(xi - 0.2, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
            ax.text(xi + 0.2, t + 0.01, f"{t:.2f}", ha="center", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels(metric_names, rotation=25)
        ax.set_ylim(0, 1.1); ax.legend(); ax.grid(alpha=0.3)
        ax.set_title("Dashboard de Métricas — Validação vs Teste")
        plt.tight_layout()
        plt.savefig(output_dir / "metrics_dashboard.png", dpi=200)
        plt.savefig('comparacao_acuracia.png', dpi=150, bbox_inches='tight')
        plt.show()
        plt.close()

        # interativo
        fig_bar = go.Figure([
            go.Bar(name="Validação", x=metric_names, y=val_vals),
            go.Bar(name="Teste",     x=metric_names, y=test_vals),
        ])
        fig_bar.update_layout(barmode="group", title="Métricas Val vs Teste",
                              yaxis=dict(range=[0, 1.1]))
        fig_bar.write_html(output_dir / "metrics_dashboard.html")


# ---------------------------------------------------------------------------
# Temperature scaling (calibração pós-treino)
# ---------------------------------------------------------------------------

def _calibrate_temperature(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> float:
    
    model.eval()
    logits_list, targets_list = [], []
    with torch.no_grad():
        for imgs, tgts, _ in val_loader:
            logits_list.append(model(imgs.to(device)).squeeze(1))
            targets_list.append(tgts.to(device))
    logits_all  = torch.cat(logits_list).detach()
    targets_all = torch.cat(targets_list).float().detach()

    temperature = nn.Parameter(torch.ones(1, device=device))
    optimizer   = torch.optim.LBFGS([temperature], lr=0.01, max_iter=100)

    def _step():
        optimizer.zero_grad()
        loss = nn.functional.binary_cross_entropy_with_logits(
            logits_all / temperature.clamp(min=1e-3), targets_all
        )
        loss.backward()
        return loss

    optimizer.step(_step)
    T = float(temperature.item())
    T = max(0.1, min(T, 10.0))
    print(f"  Temperature scaling: T = {T:.4f}")
    return T


# ---------------------------------------------------------------------------
# Treino principal
# ---------------------------------------------------------------------------

def train_model(
    manifest_path: str | Path = DEFAULT_MANIFEST,
    image_root: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    architecture: str = "efficientnet_b3",
    pretrained: bool = True,
    image_size: int = 384,
    batch_size: int = 16,
    epochs: int = 30,
    freeze_backbone_epochs: int = 5,
    patience: int = 20,
    learning_rate: float = 3e-4,
    weight_decay: float = 0.01,
    mixup_alpha: float = 0.4,
    loss_type: str = "focal",
    pos_weight_multiplier: float = 1.5,
    temperature_scaling: bool = True,
    fixed_threshold: float | None = None,
    bootstrap_iterations: int = 500,
    device_name: str = "auto",
    seed: int = 42,
) -> dict[str, Any]:
    set_seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    manifest = load_manifest(manifest_path)
    output_dir = ensure_dir(output_dir)
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    nw = 2 * max(n_gpus, 1) if n_gpus > 0 else 0
    loaders = create_dataloaders(manifest, image_root, image_size, batch_size, num_workers=nw)
    if "train" not in loaders or "val" not in loaders:
        raise ValueError("Manifesto deve conter splits train e val.")

    device = select_device(device_name)
    print(f"Dispositivo: {device} | Arquitetura: {architecture} | pretrained={pretrained}")
    if torch.cuda.is_available():
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
    if n_gpus > 1:
        print(f"  Aviso: nn.DataParallel incompatível com PyTorch/Python 3.14 nesta versão — treinando em GPU 0.")

    base_model = build_model(architecture, pretrained=pretrained, dropout=0.40).to(device)
    model = base_model

    # Peso de classe positiva para imbalance
    train_cases = manifest[manifest["split"] == "train"].drop_duplicates("patient_id")
    positives = int((train_cases["target"] == 1).sum())
    negatives = int((train_cases["target"] == 0).sum())
    positive_weight = (negatives / max(positives, 1)) * pos_weight_multiplier
    print(
        f"  Positivos (treino): {positives} | Negativos: {negatives} | "
        f"peso pos: {positive_weight:.2f} (mult={pos_weight_multiplier}x) | loss: {loss_type}"
    )

    pw_tensor = torch.tensor([positive_weight], device=device)
    if loss_type == "focal":
        criterion = FocalLoss(gamma=2.0, pos_weight=pw_tensor)
    else:
        criterion = LabelSmoothingBCE(smoothing=0.05, pos_weight=pw_tensor)

    # Differential learning rates: backbone 10× menor que cabeça
    if architecture == "resnet34":
        backbone_params = (list(base_model.conv1.parameters()) + list(base_model.layer1.parameters())
                           + list(base_model.layer2.parameters()) + list(base_model.layer3.parameters())
                           + list(base_model.layer4.parameters()))
        head_params = list(base_model.fc.parameters())
    elif architecture == "convnext_tiny":
        backbone_params = list(base_model.features.parameters())
        head_params = list(base_model.classifier.parameters())
    else:  # efficientnet_b3 / efficientnet_b4
        backbone_params = list(base_model.features.parameters())
        head_params = list(base_model.classifier.parameters())

    optimizer = AdamW(
        [
            {"params": backbone_params, "lr": learning_rate * 0.1},
            {"params": head_params,     "lr": learning_rate},
        ],
        weight_decay=weight_decay,
    )

    # CosineAnnealingWarmRestarts: T_0=10, T_mult=2 → reinicia em 10, 30, ...
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6,
    )

    # Loader de treino com transforms de avaliação (sem augmentação) para métricas por época
    train_eval_ds = CervixImageDataset(
        manifest[manifest["split"] == "train"].copy(),
        image_root=image_root,
        transform=build_transforms(image_size)["eval"],
    )
    train_eval_loader = DataLoader(
        train_eval_ds, batch_size=batch_size, shuffle=False,
        num_workers=nw, pin_memory=torch.cuda.is_available(),
    )

    best_score = -np.inf
    best_model_auc = float("nan")
    best_threshold = fixed_threshold if fixed_threshold is not None else 0.5
    best_state = copy.deepcopy(base_model.state_dict())
    epochs_without_improvement = 0
    best_epoch: int | None = None
    history: list[dict[str, Any]] = []
    started = time.time()

    for epoch in range(epochs):
        set_backbone_trainable(base_model, architecture, epoch >= freeze_backbone_epochs)

        train_loss = train_epoch(
            model, loaders["train"], criterion, optimizer, device,
            epoch, epochs, mixup_alpha=mixup_alpha,
        )

        pred = predict_loader(model, loaders["val"], device)
        _, c_tgt, c_prob = aggregate_case_predictions(
            pred["patient_ids"], pred["targets"], pred["probabilities"])
        if fixed_threshold is not None:
            opt_threshold, threshold_score = fixed_threshold, 0.0
        else:
            opt_threshold, threshold_score = optimize_threshold(c_tgt, c_prob)
        val_m = binary_metrics(c_tgt, c_prob, opt_threshold)

        # Métricas do conjunto de treino (eval mode, sem TTA, sem augmentação)
        pred_tr = predict_loader_simple(model, train_eval_loader, device)
        _, tr_c_tgt, tr_c_prob = aggregate_case_predictions(
            pred_tr["patient_ids"], pred_tr["targets"], pred_tr["probabilities"])
        train_m = binary_metrics(tr_c_tgt, tr_c_prob, opt_threshold)

        scheduler.step(epoch + 1)
        current_lr = optimizer.param_groups[-1]["lr"]

        print(
            f"Epoch {epoch + 1:03d}/{epochs} | "
            f"Loss={train_loss:.4f} | "
            f"trAUC={train_m['roc_auc']:.4f} trF1={train_m['f1']:.4f} | "
            f"valAUC={val_m['roc_auc']:.4f} valF1={val_m['f1']:.4f} | "
            f"Sens={val_m['sensitivity_recall']:.4f} Spec={val_m['specificity']:.4f} | "
            f"LR={current_lr:.2e}"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_case_metrics": train_m,
            "threshold": float(opt_threshold),
            "threshold_score": float(threshold_score),
            "val_case_metrics": val_m,
        })

        # Score balanceado: AUC + F1 + balanced_accuracy
        
        selection_score = (
            0.5 * val_m["roc_auc"]
            + 0.3 * val_m["f1"]
            + 0.2 * val_m["balanced_accuracy"]
        )

        if selection_score > best_score:
            best_score = selection_score
            best_model_auc = val_m["roc_auc"]
            best_threshold = opt_threshold
            best_state = copy.deepcopy(base_model.state_dict())
            epochs_without_improvement = 0
            best_epoch = epoch + 1
        else:
            epochs_without_improvement += 1   # FIX: era += 0 (early stopping nunca disparava)
            if epochs_without_improvement >= patience:
                print(f"Early stopping na época {epoch + 1}.")
                break

    base_model.load_state_dict(best_state)

    # Temperature scaling: calibração pós-treino no conjunto de validação
    temperature = 1.0
    if temperature_scaling:
        try:
            temperature = _calibrate_temperature(base_model, loaders["val"], device)
        except Exception as e:
            print(f"  Aviso: temperature scaling falhou ({e}), usando T=1.0")

    checkpoint = {
        "model_state_dict": base_model.state_dict(),
        "architecture": architecture,
        "dropout": 0.40,
        "image_size": image_size,
        "threshold": best_threshold,
        "temperature": temperature,
        "labels": ["negative_or_low_grade", "high_grade_or_cancer"],
        "target_definition": "IARC expert provisional impression: high-grade lesion or cancer",
    }
    checkpoint_path = output_dir / "best_model.pt"
    torch.save(checkpoint, checkpoint_path)
    write_json(output_dir / "training_history.json", history)

    results = evaluate_splits(
        model, loaders, device, best_threshold,
        output_dir, manifest, bootstrap_iterations, seed,
        temperature=temperature,
    )

    pred_test = predict_loader(model, loaders["test"], device, temperature=temperature)
    test_ids, test_targets, test_probs = aggregate_case_predictions(
        pred_test["patient_ids"], pred_test["targets"], pred_test["probabilities"])

    val_targets_plot: np.ndarray | None = None
    val_probs_plot: np.ndarray | None = None
    if "val" in loaders:
        pred_val = predict_loader(model, loaders["val"], device, temperature=temperature)
        _, val_targets_plot, val_probs_plot = aggregate_case_predictions(
            pred_val["patient_ids"], pred_val["targets"], pred_val["probabilities"])

    generate_plots(history, test_targets, test_probs, output_dir, best_threshold, results,
                   val_targets=val_targets_plot, val_probs=val_probs_plot)

    # GradCAM: painel de exemplares (TP/TN/FP/FN)
    exemplar_dir = ensure_dir(output_dir / "gradcam")
    generate_gradcam_exemplars(
        model=base_model,
        manifest=manifest,
        image_root=Path(image_root),
        image_size=image_size,
        threshold=best_threshold,
        output_dir=exemplar_dir,
        device=device,
        architecture=architecture,
        case_ids=test_ids,
        case_targets=test_targets,
        case_probabilities=test_probs,
    )

    # GradCAM: todas as imagens do dataset
    generate_gradcam_dataset(
        model=base_model,
        manifest=manifest,
        image_root=image_root,
        image_size=image_size,
        threshold=best_threshold,
        output_dir=exemplar_dir,
        device=device,
        architecture=architecture,
    )

    summary = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "elapsed_seconds": time.time() - started,
        "architecture": architecture,
        "positive_class_weight": positive_weight,
        "best_epoch": best_epoch,
        "best_auc": float(best_model_auc),
        "best_threshold": float(best_threshold),
        "best_selection_score": float(best_score),
        "results": results,
    }
    write_json(output_dir / "training_summary.json", summary)
    print_training_terminal(history, best_epoch, test_targets, test_probs, best_threshold)
    print_terminal_metrics(results, output_dir, split="test",
                           case_targets=test_targets, case_probs=test_probs)
    return summary


# ---------------------------------------------------------------------------
# Checkpoint e inferência
# ---------------------------------------------------------------------------

def load_checkpoint_model(
    checkpoint_path: str | Path,
    device_name: str = "auto",
) -> tuple[nn.Module, dict[str, Any], torch.device]:
    device = select_device(device_name)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_model(ckpt["architecture"], pretrained=False,
                        dropout=float(ckpt.get("dropout", 0.4)))
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device).eval()
    return model, ckpt, device


def evaluate_checkpoint(
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    manifest_path: str | Path = DEFAULT_MANIFEST,
    image_root: str | Path = DEFAULT_DATA_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    batch_size: int = 8,
    bootstrap_iterations: int = 500,
    device_name: str = "auto",
    seed: int = 42,
) -> dict[str, Any]:
    model, ckpt, device = load_checkpoint_model(checkpoint_path, device_name)
    manifest = load_manifest(manifest_path)
    loaders = create_dataloaders(
        manifest, image_root, int(ckpt["image_size"]), batch_size,
        num_workers=2 if torch.cuda.is_available() else 0,
    )
    results = evaluate_splits(
        model, loaders, device, float(ckpt.get("threshold", 0.50)),
        output_dir, manifest, bootstrap_iterations, seed,
        temperature=float(ckpt.get("temperature", 1.0)),
    )
    summary = {"checkpoint": str(checkpoint_path), "device": str(device), "results": results}
    write_json(output_dir / "checkpoint_evaluation_summary.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default))
    print_terminal_metrics(results, Path(output_dir), split="test")
    return summary


def predict_image(
    image_path: str | Path,
    checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
    device_name: str = "auto",
) -> dict[str, Any]:
    model, ckpt, device = load_checkpoint_model(checkpoint_path, device_name)
    threshold   = float(ckpt.get("threshold", 0.50))
    temperature = float(ckpt.get("temperature", 1.0))
    transform = build_transforms(int(ckpt["image_size"]))["eval"]
    with Image.open(image_path) as img:
        tensor = transform(img.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(tensor).squeeze()
        probability = float(torch.sigmoid(logit / max(temperature, 1e-3)).item())
    label = INDEX_TO_LABEL[int(probability >= threshold)]
    result = {
        "image_path": str(image_path),
        "probability_high_grade_or_cancer": probability,
        "threshold": threshold,
        "prediction": label,
        "checkpoint": str(checkpoint_path),
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


# ---------------------------------------------------------------------------
# Demo e CLI
# ---------------------------------------------------------------------------

def run_demo(zip_path: str | Path, data_dir: str | Path,
             output_dir: str | Path, seed: int) -> dict[str, Any]:
    print("=" * 72)
    print("DEMONSTRAÇÃO RÁPIDA — CERVIX VISUAL AI")
    print("=" * 72)
    prepare_iarc_dataset(zip_path, data_dir, seed)
    return train_model(
        manifest_path=Path(data_dir) / "manifest.csv",
        image_root=data_dir,
        output_dir=output_dir,
        architecture="efficientnet_b3",
        pretrained=True,
        image_size=384,
        batch_size=8,
        epochs=30,
        freeze_backbone_epochs=5,
        patience=15,
        bootstrap_iterations=500,
        device_name="auto",
        seed=seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pipeline IARC — imagens visuais do colo uterino."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("prepare")
    p.add_argument("--zip", dest="zip_path", default=str(DEFAULT_IARC_ZIP))
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("demo")
    p.add_argument("--zip", dest="zip_path", default=str(DEFAULT_IARC_ZIP))
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("train")
    p.add_argument("--manifest",         default=str(DEFAULT_MANIFEST))
    p.add_argument("--image-root",       default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output-dir",       default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--architecture",
                   choices=["resnet34", "efficientnet_b3", "efficientnet_b4", "convnext_tiny"],
                   default="efficientnet_b3")
    p.add_argument("--no-pretrained",    action="store_true")
    p.add_argument("--image-size",       type=int,   default=384)
    p.add_argument("--batch-size",       type=int,   default=8)
    p.add_argument("--epochs",           type=int,   default=30)
    p.add_argument("--freeze-epochs",    type=int,   default=5)
    p.add_argument("--patience",         type=int,   default=20)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--weight-decay",     type=float, default=0.01)
    p.add_argument("--mixup-alpha",      type=float, default=0.4)
    p.add_argument("--loss-type",        choices=["focal", "label_smoothing"], default="focal")
    p.add_argument("--pos-weight-mult",  type=float, default=1.5,
                   help="Multiplicador do peso da classe positiva (padrão 1.5×)")
    p.add_argument("--no-temp-scaling",  action="store_true",
                   help="Desativa temperature scaling pós-treino")
    p.add_argument("--threshold",        type=float, default=None,
                   help="Limiar fixo de decisão (ex: 0.35). Se omitido, otimiza no val.")
    p.add_argument("--bootstrap",        type=int,   default=500)
    p.add_argument("--device",           default="auto")
    p.add_argument("--seed",             type=int,   default=42)

    p = sub.add_parser("evaluate")
    p.add_argument("--checkpoint",  default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--manifest",    default=str(DEFAULT_MANIFEST))
    p.add_argument("--data-dir",    default=str(DEFAULT_DATA_DIR))
    p.add_argument("--output-dir",  default=str(DEFAULT_OUTPUT_DIR))
    p.add_argument("--batch-size",  type=int, default=8)
    p.add_argument("--bootstrap",   type=int, default=500)
    p.add_argument("--device",      default="auto")
    p.add_argument("--seed",        type=int, default=42)

    p = sub.add_parser("predict")
    p.add_argument("--image",      required=True)
    p.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--device",     default="auto")

    p = sub.add_parser("show", help="Exibe imagens ou GradCAM no terminal.")
    p.add_argument("--image",       help="Imagem a exibir (caminho)")
    p.add_argument("--checkpoint",  default=str(DEFAULT_CHECKPOINT))
    p.add_argument("--device",      default="auto")
    p.add_argument("--width",       type=int, default=80,
                   help="Largura em colunas (padrão 80)")
    p.add_argument("--samples",     action="store_true",
                   help="Mostrar amostras do dataset em vez de uma imagem")
    p.add_argument("--manifest",    default=str(DEFAULT_MANIFEST))
    p.add_argument("--data-dir",    default=str(DEFAULT_DATA_DIR))
    p.add_argument("--n-per-class", type=int, default=3)

    return parser


def main() -> None:
    if len(sys.argv) == 1:
        sys.argv.append("train")

    args = build_parser().parse_args()

    if args.command == "prepare":
        prepare_iarc_dataset(args.zip_path, args.data_dir, args.seed)
    elif args.command == "demo":
        run_demo(args.zip_path, args.data_dir, args.output_dir, args.seed)
    elif args.command == "train":
        train_model(
            manifest_path=args.manifest,
            image_root=args.image_root,
            output_dir=args.output_dir,
            architecture=args.architecture,
            pretrained=not args.no_pretrained,
            image_size=args.image_size,
            batch_size=args.batch_size,
            epochs=args.epochs,
            freeze_backbone_epochs=args.freeze_epochs,
            patience=args.patience,
            learning_rate=args.lr,
            weight_decay=args.weight_decay,
            mixup_alpha=args.mixup_alpha,
            loss_type=args.loss_type,
            pos_weight_multiplier=args.pos_weight_mult,
            temperature_scaling=not args.no_temp_scaling,
            fixed_threshold=args.threshold,
            bootstrap_iterations=args.bootstrap,
            device_name=args.device,
            seed=args.seed,
        )
    elif args.command == "evaluate":
        evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            manifest_path=args.manifest,
            image_root=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            bootstrap_iterations=args.bootstrap,
            device_name=args.device,
            seed=args.seed,
        )
    elif args.command == "predict":
        predict_image(args.image, args.checkpoint, args.device)

    elif args.command == "show":
        enable_windows_ansi()

        if args.samples:
            manifest = load_manifest(args.manifest)
            show_dataset_samples_terminal(
                manifest,
                image_root=args.data_dir,
                n_per_class=args.n_per_class,
                width_per_image=args.width // 4,
            )
        elif args.image:
            
            if Path(args.checkpoint).exists():
                model, ckpt, device = load_checkpoint_model(args.checkpoint, args.device)
                image_size = int(ckpt["image_size"])
                threshold  = float(ckpt.get("threshold", 0.5))
                image = Image.open(args.image).convert("RGB")
                w, h = image.size
                tensor = build_transforms(image_size)["eval"](image).unsqueeze(0).to(device)
                target_layer = _gradcam_target_layer(model, ckpt["architecture"])
                gradcam = GradCAM(model, target_layer)
                with torch.no_grad():
                    prob = float(torch.sigmoid(model(tensor).squeeze()).item())
                heatmap = gradcam.generate(tensor)
                gradcam.remove_hooks()
                heatmap = cv2.resize(heatmap, (w, h))
                show_gradcam_terminal(
                    image, heatmap, prob, threshold, width=args.width
                )
            else:
                show_image_terminal(args.image, width=args.width,
                                    title=Path(args.image).name)
        else:
            print("Use --image <caminho> ou --samples")


if __name__ == "__main__":
    main()
