import os
import csv
import argparse
from typing import Optional, Tuple, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image


# -----------------------------
# I/O
# -----------------------------
def read_image_any(path: str) -> np.ndarray:
    """Load .npy or .png/.jpg as float32 in [0,1] (for png) or raw float for npy."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path).astype(np.float32)
        # If someone accidentally saved 0..255 as npy, normalize:
        mx = float(np.max(arr)) if arr.size else 0.0
        if mx > 1.5:
            arr = arr / 255.0
        return arr
    else:
        img = Image.open(path).convert("L")
        arr = np.array(img, dtype=np.float32)
        if arr.max() > 1.5:
            arr /= 255.0
        return arr


def save_png01(path: str, img01: np.ndarray) -> None:
    img01 = np.clip(img01, 0.0, 1.0)
    im = (img01 * 255.0).round().astype(np.uint8)
    Image.fromarray(im).save(path)


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def find_existing_path(base_dir: str, rel_or_abs: str) -> str:
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.join(base_dir, rel_or_abs)


def auto_find_keys(row: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Find manifest columns for noisy + sigma GT.
    Works with your manifest that has noisy_png/noisy_npy/sigma_npy etc.
    """
    keys = {k.strip().lower(): k for k in row.keys()}
    noisy_candidates = ["noisy_png", "noisy_npy", "noisy", "input", "image", "img", "x"]
    sigma_candidates = ["sigma_npy", "sigma", "sigma_map", "sigma_path", "gt", "target", "y", "label"]

    noisy_key = next((keys[c] for c in noisy_candidates if c in keys), None)
    sigma_key = next((keys[c] for c in sigma_candidates if c in keys), None)

    if noisy_key is None:
        for lk, orig in keys.items():
            if "noisy" in lk or "input" in lk:
                noisy_key = orig
                break
    if sigma_key is None:
        for lk, orig in keys.items():
            if "sigma" in lk:
                sigma_key = orig
                break

    return noisy_key, sigma_key


# -----------------------------
# Padding helpers
# -----------------------------
def pad_to_divisible(x: torch.Tensor, div: int = 8):
    """Pad (B,C,H,W) so H,W divisible by div."""
    _, _, H, W = x.shape
    pad_h = (div - H % div) % div
    pad_w = (div - W % div) % div

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    x_pad = F.pad(x, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
    return x_pad, (pad_top, pad_bottom, pad_left, pad_right)


def unpad(x: torch.Tensor, pad):
    pad_top, pad_bottom, pad_left, pad_right = pad
    if pad_top + pad_bottom + pad_left + pad_right == 0:
        return x
    return x[:, :, pad_top:x.shape[2]-pad_bottom, pad_left:x.shape[3]-pad_right]


# -----------------------------
# Homomorphic feature extraction
# -----------------------------
class GaussianBlur2D(nn.Module):
    def __init__(self, channels: int = 1, kernel_size: int = 31, sigma: float = 7.0):
        super().__init__()
        if kernel_size % 2 == 0:
            kernel_size += 1
        self.channels = channels
        self.kernel_size = kernel_size
        self.sigma = sigma

        kernel = self._make_gaussian_kernel(kernel_size, sigma)  # (k,k)
        weight = kernel[None, None, :, :].repeat(channels, 1, 1, 1)  # (C,1,k,k)
        self.register_buffer("weight", weight)
        self.padding = kernel_size // 2

    @staticmethod
    def _make_gaussian_kernel(ks: int, sigma: float) -> torch.Tensor:
        ax = torch.arange(ks, dtype=torch.float32) - (ks - 1) / 2.0
        xx, yy = torch.meshgrid(ax, ax, indexing="ij")
        kernel = torch.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
        kernel = kernel / kernel.sum()
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.conv2d(x, self.weight, padding=self.padding, groups=self.channels)


def homomorphic_features(I: torch.Tensor, blur: GaussianBlur2D, eps: float = 1e-6):
    mu = blur(I)
    r = torch.abs(I - mu) + eps
    h = torch.log(r)
    return mu, r, h


# -----------------------------
# Model (must match training)
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(8 if out_ch >= 8 else 1, out_ch),
            nn.SiLU(),
        )
    def forward(self, x):
        return self.net(x)


class SmallUNet(nn.Module):
    def __init__(self, in_ch: int = 1, base: int = 32):
        super().__init__()
        self.enc1 = ConvBlock(in_ch, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = ConvBlock(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = ConvBlock(base * 2, base * 4)

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = ConvBlock(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = ConvBlock(base * 2, base)

        self.out = nn.Conv2d(base, 1, 1)
        self.softplus = nn.Softplus(beta=1.0)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))

        d2 = self.up2(e3)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        sigma_hat = self.softplus(self.out(d1)) + 1e-8
        return sigma_hat


# -----------------------------
# Metrics
# -----------------------------
def log_mse(pred: np.ndarray, gt: np.ndarray, eps: float = 1e-8) -> float:
    return float(np.mean((np.log(pred + eps) - np.log(gt + eps)) ** 2))


# -----------------------------
# Inference + Eval
# -----------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", required=True, help="Path to sigmanet_best.pth or sigmanet_last.pth")
    ap.add_argument("--manifest", default=None, help="manifest.csv (recommended for GT eval)")
    ap.add_argument("--noisy_dir", default=None, help="folder of noisy images (png/npy) if no manifest")
    ap.add_argument("--out_dir", required=True, help="Base output folder")

    ap.add_argument("--base_dir", default=None, help="Base dir to resolve relative paths in manifest")
    ap.add_argument("--input_mode", choices=["h_only", "I_and_h"], default="h_only")
    ap.add_argument("--base", type=int, default=32)

    ap.add_argument("--blur_ksize", type=int, default=31)
    ap.add_argument("--blur_sigma", type=float, default=7.0)
    ap.add_argument("--div", type=int, default=8)

    ap.add_argument("--cpu", action="store_true")

    # output structure
    ap.add_argument("--pred_png_dirname", default="pred_sigma_png", help="subfolder name for predicted PNGs")
    ap.add_argument("--pred_npy_dirname", default="pred_sigma_npy", help="subfolder name for predicted NPYs")

    # evaluation
    ap.add_argument("--eval_gt", action="store_true",
                    help="If GT sigma exists (via manifest), compute metrics + print averages")
    ap.add_argument("--metrics_csv_name", default="metrics.csv")

    args = ap.parse_args()

    device = torch.device("cpu" if args.cpu or (not torch.cuda.is_available()) else "cuda")

    out_base = args.out_dir
    pred_png_dir = os.path.join(out_base, args.pred_png_dirname)
    pred_npy_dir = os.path.join(out_base, args.pred_npy_dirname)
    ensure_dir(out_base)
    ensure_dir(pred_png_dir)
    ensure_dir(pred_npy_dir)

    # Load ckpt
    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt

    in_ch = 1 if args.input_mode == "h_only" else 2
    model = SmallUNet(in_ch=in_ch, base=args.base).to(device)
    model.load_state_dict(state, strict=True)
    model.eval()

    blur = GaussianBlur2D(channels=1, kernel_size=args.blur_ksize, sigma=args.blur_sigma).to(device)

    # Build file list
    items: List[Tuple[str, Optional[str]]] = []
    if args.manifest is not None:
        base_dir = args.base_dir if args.base_dir is not None else os.path.dirname(args.manifest)
        with open(args.manifest, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                noisy_key, sigma_key = auto_find_keys(row)
                if noisy_key is None:
                    continue
                noisy_path = find_existing_path(base_dir, row[noisy_key].strip())
                sigma_path = None
                if sigma_key is not None:
                    sigma_path = find_existing_path(base_dir, row[sigma_key].strip())
                if os.path.exists(noisy_path):
                    items.append((noisy_path, sigma_path if (sigma_path and os.path.exists(sigma_path)) else None))
    elif args.noisy_dir is not None:
        exts = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".npy")
        for fn in sorted(os.listdir(args.noisy_dir)):
            if fn.lower().endswith(exts):
                items.append((os.path.join(args.noisy_dir, fn), None))
    else:
        raise SystemExit("Provide either --manifest or --noisy_dir")

    if len(items) == 0:
        raise SystemExit("No inputs found.")

    # Metrics
    metrics_rows = []
    metrics_csv_path = os.path.join(out_base, args.metrics_csv_name)

    for idx, (noisy_path, sigma_gt_path) in enumerate(items):
        I = read_image_any(noisy_path).astype(np.float32)
        I_t = torch.from_numpy(I)[None, None, ...].to(device)  # (1,1,H,W)

        I_pad, pad = pad_to_divisible(I_t, div=args.div)
        mu, r, h = homomorphic_features(I_pad, blur)
        x = h if args.input_mode == "h_only" else torch.cat([I_pad, h], dim=1)

        sigma_pred = model(x)
        sigma_pred = unpad(sigma_pred, pad)[0, 0].detach().cpu().numpy().astype(np.float32)

        stem = os.path.splitext(os.path.basename(noisy_path))[0]

        # Save NPY (true-valued prediction)
        np.save(os.path.join(pred_npy_dir, f"{stem}.npy"), sigma_pred)

        # Save PNG (visualization: min-max scaled per-image)
        sp = (sigma_pred - sigma_pred.min()) / (sigma_pred.max() - sigma_pred.min() + 1e-8)
        save_png01(os.path.join(pred_png_dir, f"{stem}.png"), sp)

        # Eval if GT exists and requested
        if args.eval_gt and (sigma_gt_path is not None):
            gt = read_image_any(sigma_gt_path).astype(np.float32)
            if gt.shape != sigma_pred.shape:
                metrics_rows.append([stem, noisy_path, sigma_gt_path, "shape_mismatch", "", "", ""])
            else:
                diff = sigma_pred - gt
                mae = float(np.mean(np.abs(diff)))
                rmse = float(np.sqrt(np.mean(diff**2)))
                lmse = log_mse(sigma_pred, gt)
                metrics_rows.append([stem, noisy_path, sigma_gt_path, "ok", mae, rmse, lmse])

        if (idx + 1) % 50 == 0:
            print(f"[INFO] processed {idx+1}/{len(items)}")

    # Write metrics + print averages
    if args.eval_gt:
        with open(metrics_csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "noisy_path", "sigma_gt_path", "status", "mae", "rmse", "logmse"])
            w.writerows(metrics_rows)
        print(f"[DONE] wrote metrics: {metrics_csv_path}")

        ok = [r for r in metrics_rows if len(r) >= 7 and r[3] == "ok"]
        if len(ok) == 0:
            print("[WARN] No valid GT comparisons (status=ok). Check your manifest sigma paths.")
        else:
            maes = np.array([float(r[4]) for r in ok], dtype=np.float64)
            rmses = np.array([float(r[5]) for r in ok], dtype=np.float64)
            lmses = np.array([float(r[6]) for r in ok], dtype=np.float64)

            print("===================================")
            print(f"Evaluated pairs : {len(ok)} / {len(items)}")
            print(f"MAE   mean/med  : {maes.mean():.6f} / {np.median(maes):.6f}")
            print(f"RMSE  mean/med  : {rmses.mean():.6f} / {np.median(rmses):.6f}")
            print(f"logMSE mean/med : {lmses.mean():.6f} / {np.median(lmses):.6f}")
            print(f"logMSE std      : {lmses.std():.6f}")
            print("===================================")

    print(f"[DONE] Saved predicted sigma maps:")
    print(f"  PNGs: {pred_png_dir}")
    print(f"  NPYs: {pred_npy_dir}")
    print(f"[DONE] outputs in: {out_base}")


if __name__ == "__main__":
    main()
