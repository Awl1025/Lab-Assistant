import os
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import tifffile as tiff
import torch
import torch.nn as nn


# Configuration

@dataclass
class Config:
    data_dir: str = "data/bbbc039"
    metadata_split: str = "val"  # "train" or "val"
    image_size: Tuple[int, int] = (512, 512)
    model_path: str = "artifacts/unet_bbbc039_semantic.pt"
    out_dir: str = "artifacts/overlays"
    max_images: int = 8
    threshold: float = 0.5
    overlay_alpha: float = 0.45  # 0..1
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# Helpers

def read_list(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def pick_split_file(metadata_dir: str, split: str) -> str:
    candidates = os.listdir(metadata_dir)
    split = split.lower()

    keys = [split]
    if split == "val":
        keys += ["valid", "validation"]

    for name in candidates:
        low = name.lower()
        if any(k in low for k in keys) and low.endswith((".txt", ".csv")):
            return os.path.join(metadata_dir, name)

    raise FileNotFoundError(f"Could not find metadata file for split='{split}'. Found: {candidates}")


def resolve_by_base(dir_path: str, base: str, exts: Tuple[str, ...]) -> str:
    base = os.path.splitext(os.path.basename(base))[0]
    tried = []
    for ext in exts:
        p = os.path.join(dir_path, base + ext)
        tried.append(p)
        if os.path.exists(p):
            return p
    raise FileNotFoundError(f"Could not resolve file for base='{base}' in '{dir_path}'. Tried: {tried}")


def load_grayscale_image(path: str) -> np.ndarray:
    if path.lower().endswith((".tif", ".tiff")):
        img = tiff.imread(path).astype(np.float32)
    else:
        img = np.array(Image.open(path)).astype(np.float32)
        if img.ndim == 3:
            img = img[:, :, 0]

    # Normalize to 0..1 (simple min/max)
    img = img - float(img.min())
    denom = float(img.max() - img.min()) + 1e-6
    img = img / denom
    return img


def resize_np_to_pil_gray(img01: np.ndarray, size_hw: Tuple[int, int]) -> Image.Image:
    h, w = size_hw
    pil = Image.fromarray((img01 * 255).astype(np.uint8), mode="L")
    return pil.resize((w, h), resample=Image.BILINEAR)


def pil_gray_to_tensor(pil: Image.Image) -> torch.Tensor:
    arr = np.array(pil).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)


def make_red_overlay(base_gray: Image.Image, mask01: np.ndarray, alpha: float) -> Image.Image:
    """
    base_gray: PIL L image resized to model size
    mask01: numpy (H,W) with {0,1}
    """
    base_rgb = base_gray.convert("RGB")

    # Build an RGBA red layer where alpha is proportional to mask
    h, w = mask01.shape
    a = (mask01 * int(255 * alpha)).astype(np.uint8)
    red = np.zeros((h, w, 4), dtype=np.uint8)
    red[..., 0] = 255  # R
    red[..., 1] = 0
    red[..., 2] = 0
    red[..., 3] = a

    red_rgba = Image.fromarray(red, mode="RGBA")
    out = Image.alpha_composite(base_rgb.convert("RGBA"), red_rgba).convert("RGB")

    return out


def mask_outline(mask01: np.ndarray) -> np.ndarray:
    """
    Simple outline: edge = mask - eroded(mask)
    Uses PIL min-filter as a cheap erosion.
    """
    m = Image.fromarray((mask01 * 255).astype(np.uint8), mode="L")
    eroded = m.filter(ImageFilter.MinFilter(3))
    edge = (np.array(m).astype(np.int16) - np.array(eroded).astype(np.int16)) > 0
    return edge.astype(np.uint8)


def draw_outline_rgb(img_rgb: Image.Image, edge01: np.ndarray) -> Image.Image:
    arr = np.array(img_rgb)
    # Outline in yellow for visibility
    arr[edge01 > 0] = np.array([255, 255, 0], dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


# Tiny U-Net

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


# Main

@torch.no_grad()
def main():
    cfg = Config()

    images_dir = os.path.join(cfg.data_dir, "images")
    masks_dir = os.path.join(cfg.data_dir, "masks")      # not required for overlay, kept for consistency
    metadata_dir = os.path.join(cfg.data_dir, "metadata")

    split_file = pick_split_file(metadata_dir, cfg.metadata_split)
    names = read_list(split_file)[: cfg.max_images]

    if not os.path.exists(cfg.model_path):
        raise FileNotFoundError(f"Model not found: {cfg.model_path}")

    os.makedirs(cfg.out_dir, exist_ok=True)

    model = UNetSmall().to(cfg.device)
    state = torch.load(cfg.model_path, map_location=cfg.device)
    model.load_state_dict(state)
    model.eval()

    print(f"Device: {cfg.device}")
    print(f"Split file: {split_file}")
    print(f"Saving overlays to: {cfg.out_dir}")

    for i, name in enumerate(names, start=1):
        base = os.path.splitext(os.path.basename(name))[0]

        # Resolve the true image file
        img_path = resolve_by_base(images_dir, base, (".tif", ".tiff", ".png", ".jpg"))

        img01 = load_grayscale_image(img_path)
        img_pil = resize_np_to_pil_gray(img01, cfg.image_size)
        x = pil_gray_to_tensor(img_pil).to(cfg.device)

        logits = model(x)
        probs = torch.sigmoid(logits)[0, 0].cpu().numpy()
        pred = (probs >= cfg.threshold).astype(np.uint8)

        overlay = make_red_overlay(img_pil, pred, cfg.overlay_alpha)

        # Optional outline for easier viewing
        edge = mask_outline(pred)
        overlay = draw_outline_rgb(overlay, edge)

        # Slight contrast boost for nicer visuals
        overlay = ImageEnhance.Contrast(overlay).enhance(1.15)

        out_path = os.path.join(cfg.out_dir, f"{i:02d}_{base}_overlay.png")
        overlay.save(out_path)

        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()