import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image
import tifffile as tiff
from skimage.morphology import label as cc_label

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF
from tqdm import tqdm


# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    data_dir: str = "data/bbbc039"
    batch_size: int = 4
    num_workers: int = 2
    lr: float = 1e-3
    epochs: int = 3
    image_size: Tuple[int, int] = (512, 512)  # resize for speed
    limit_train: Optional[int] = 80          # keep small for MVP
    limit_val: Optional[int] = 20
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# Utilities
# -----------------------------
def read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def find_split_files(metadata_dir: str) -> Tuple[str, str]:
    """
    metadata.zip contains filenames for train/val/test. Exact names can vary
    slightly, so we search for common patterns.
    """
    candidates = os.listdir(metadata_dir)
    def pick(keys: List[str]) -> str:
        for name in candidates:
            low = name.lower()
            if any(k in low for k in keys) and low.endswith((".txt", ".csv")):
                return os.path.join(metadata_dir, name)
        raise FileNotFoundError(f"Could not find metadata file with keys={keys}. Found: {candidates}")

    train_path = pick(["train"])
    val_path = pick(["val", "valid"])
    return train_path, val_path


def decode_instance_png_to_labels(mask_png_path: str) -> np.ndarray:
    """
    BBBC039 masks are PNGs where each nucleus instance is a different color.
    BBBC's example decode: keep first channel then connected-components label. :contentReference[oaicite:4]{index=4}
    """
    gt = np.array(Image.open(mask_png_path))
    if gt.ndim == 3:
        gt = gt[:, :, 0]  # keep first channel (per BBBC example)
    gt = cc_label(gt)    # label connected components
    return gt.astype(np.int32)


def dice_iou_from_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6):
    """
    logits: (B,1,H,W), targets: (B,1,H,W) in {0,1}
    """
    probs = torch.sigmoid(logits)
    preds = (probs > 0.5).float()

    intersection = (preds * targets).sum(dim=(1, 2, 3))
    union = (preds + targets - preds * targets).sum(dim=(1, 2, 3))
    iou = (intersection + eps) / (union + eps)

    dice = (2 * intersection + eps) / (preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3)) + eps)
    return dice.mean().item(), iou.mean().item()


# -----------------------------
# Dataset
# -----------------------------
class BBBC039SemanticDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, names: List[str], image_size=(512, 512)):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.names = names
        self.image_size = image_size

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx: int):
        # Filenames in metadata are usually image filenames.
        img_name = self.names[idx]
        base = os.path.splitext(os.path.basename(img_name))[0]

        # Find the actual image file (BBBC039 images are usually .tif, but metadata may list .png names)
        img_candidates = [
            os.path.join(self.images_dir, base + ext)
            for ext in (".tif", ".tiff", ".png", ".jpg")
        ]
        img_path = next((p for p in img_candidates if os.path.exists(p)), None)
        if img_path is None:
            raise FileNotFoundError(
                f"Could not find image for base='{base}'. Tried: {img_candidates}"
            )

        # Masks are typically PNGs
        mask_candidates = [
            os.path.join(self.masks_dir, base + ext)
            for ext in (".png", ".tif", ".tiff")
        ]
        mask_path = next((p for p in mask_candidates if os.path.exists(p)), None)
        if mask_path is None:
            raise FileNotFoundError(
                f"Could not find mask for base='{base}'. Tried: {mask_candidates}"
            )

        # Load image TIFF (16-bit)
        img = tiff.imread(img_path).astype(np.float32)

        # Normalize to 0..1 (robust enough for MVP)
        img = img - img.min()
        denom = (img.max() - img.min()) + 1e-6
        img = img / denom

        # Resize
        img_pil = Image.fromarray((img * 255).astype(np.uint8))
        img_pil = img_pil.resize(self.image_size[::-1], resample=Image.BILINEAR)
        img = np.array(img_pil).astype(np.float32) / 255.0

        # Decode mask → labels → binary nucleus mask
        labels = decode_instance_png_to_labels(mask_path)
        m_pil = Image.fromarray((labels > 0).astype(np.uint8) * 255)
        m_pil = m_pil.resize(self.image_size[::-1], resample=Image.NEAREST)
        mask = (np.array(m_pil) > 0).astype(np.float32)

        # To tensors: (1,H,W)
        x = torch.from_numpy(img).unsqueeze(0)
        y = torch.from_numpy(mask).unsqueeze(0)

        return x, y


# -----------------------------
# Tiny U-Net
# -----------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class UNetSmall(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=32):
        super().__init__()
        self.d1 = DoubleConv(in_ch, base)
        self.p1 = nn.MaxPool2d(2)
        self.d2 = DoubleConv(base, base * 2)
        self.p2 = nn.MaxPool2d(2)
        self.d3 = DoubleConv(base * 2, base * 4)

        self.u2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.uconv2 = DoubleConv(base * 4, base * 2)
        self.u1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.uconv1 = DoubleConv(base * 2, base)

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        x1 = self.d1(x)
        x2 = self.d2(self.p1(x1))
        x3 = self.d3(self.p2(x2))

        x = self.u2(x3)
        x = self.uconv2(torch.cat([x, x2], dim=1))
        x = self.u1(x)
        x = self.uconv1(torch.cat([x, x1], dim=1))
        return self.out(x)


# -----------------------------
# Train / Eval
# -----------------------------
def train_one_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total = 0.0
    for x, y in tqdm(loader, desc="train", leave=False):
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        logits = model(x)
        loss = loss_fn(logits, y)
        loss.backward()
        opt.step()
        total += loss.item() * x.size(0)
    return total / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, loss_fn, device):
    model.eval()
    total = 0.0
    dices, ious = [], []
    for x, y in tqdm(loader, desc="val", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = loss_fn(logits, y)
        total += loss.item() * x.size(0)
        d, i = dice_iou_from_logits(logits, y)
        dices.append(d)
        ious.append(i)
    return total / len(loader.dataset), float(np.mean(dices)), float(np.mean(ious))


def main():
    cfg = Config()

    images_dir = os.path.join(cfg.data_dir, "images")
    masks_dir = os.path.join(cfg.data_dir, "masks")
    metadata_dir = os.path.join(cfg.data_dir, "metadata")

    train_list_path, val_list_path = find_split_files(metadata_dir)
    train_names = read_list(train_list_path)
    val_names = read_list(val_list_path)

    if cfg.limit_train:
        train_names = train_names[: cfg.limit_train]
    if cfg.limit_val:
        val_names = val_names[: cfg.limit_val]

    train_ds = BBBC039SemanticDataset(images_dir, masks_dir, train_names, image_size=cfg.image_size)
    val_ds = BBBC039SemanticDataset(images_dir, masks_dir, val_names, image_size=cfg.image_size)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = UNetSmall().to(cfg.device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    print(f"Device: {cfg.device}")
    for epoch in range(1, cfg.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, opt, loss_fn, cfg.device)
        va_loss, va_dice, va_iou = evaluate(model, val_loader, loss_fn, cfg.device)
        print(f"Epoch {epoch}/{cfg.epochs} | train_loss={tr_loss:.4f} | val_loss={va_loss:.4f} | dice={va_dice:.4f} | iou={va_iou:.4f}")

    os.makedirs("artifacts", exist_ok=True)
    torch.save(model.state_dict(), "artifacts/unet_bbbc039_semantic.pt")
    print("Saved: artifacts/unet_bbbc039_semantic.pt")


if __name__ == "__main__":
    main()