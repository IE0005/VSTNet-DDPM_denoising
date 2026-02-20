import os
import csv
import math
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from PIL import Image


# -----------------------------
# Utilities
# -----------------------------
def seed_everything(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_image_any(path: str) -> np.ndarray:
    """Load .npy or .png/.jpg as float32 in [0,1]."""
    ext = os.path.splitext(path)[1].lower()
    if ext == ".npy":
        arr = np.load(path).astype(np.float32)
        # try normalize if looks like 0..255
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


def ensure_dir(d: str):
    os.makedirs(d, exist_ok=True)


def find_existing_path(base_dir: str, rel_or_abs: str) -> str:
    """If path is absolute use it. Else join with base_dir."""
    if os.path.isabs(rel_or_abs):
        return rel_or_abs
    return os.path.join(base_dir, rel_or_abs)


def auto_find_keys(row: Dict[str, str]) -> Tuple[Optional[str], Optional[str]]:
    """
    Try to find columns for noisy and sigma paths in a manifest row.
    Supports many possible column names.
    """
    keys = {k.strip().lower(): k for k in row.keys()}

    noisy_candidates = [
        "noisy", "noisy_path", "noisy_png", "noisy_npy", "input", "input_path", "x", "image", "img", "i"
    ]
    sigma_candidates = [
        "sigma", "sigma_path", "sigma_npy", "sigma_map", "sigma_map_path", "y", "target", "gt", "label"
    ]

    noisy_key = None
    sigma_key = None

    for c in noisy_candidates:
        if c in keys:
            noisy_key = keys[c]
            break
    for c in sigma_candidates:
        if c in keys:
            sigma_key = keys[c]
            break

    # fallback: search by substring
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

def pad_to_divisible(x: torch.Tensor, div: int = 8):
    """
    Pad tensor (B,C,H,W) so H and W are divisible by `div`.
    Returns padded tensor and padding info for later unpadding.
    """
    _, _, H, W = x.shape
    pad_h = (div - H % div) % div
    pad_w = (div - W % div) % div

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    x_pad = F.pad(
        x,
        (pad_left, pad_right, pad_top, pad_bottom),
        mode="reflect"
    )
    return x_pad, (pad_top, pad_bottom, pad_left, pad_right)
def unpad(x: torch.Tensor, pad):
    pad_top, pad_bottom, pad_left, pad_right = pad
    if pad_top + pad_bottom + pad_left + pad_right == 0:
        return x
    return x[:, :, pad_top:x.shape[2]-pad_bottom,
                    pad_left:x.shape[3]-pad_right]
# -----------------------------
# Homomorphic feature extraction (fixed, physics-inspired)
# -----------------------------
class GaussianBlur2D(nn.Module):
    """Depthwise Gaussian blur using conv2d (no OpenCV)."""
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
        # x: (B,C,H,W)
        return F.conv2d(x, self.weight, padding=self.padding, groups=self.channels)


def homomorphic_features(
    I: torch.Tensor,
    blur: GaussianBlur2D,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    I: (B,1,H,W) noisy magnitude
    Returns:
      mu: (B,1,H,W) low-pass
      r:  (B,1,H,W) residual magnitude |I-mu| + eps
      h:  (B,1,H,W) log residual log(r)
    """
    mu = blur(I)
    r = torch.abs(I - mu) + eps
    h = torch.log(r)
    return mu, r, h



# -----------------------------
# SigmaNet model (CNN that learns bias correction + smoothing)
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
    """
    Lightweight U-Net-ish model.
    Input: h(x) (log residual) or [I, h]
    Output: sigma_hat(x) (positive via Softplus)
    """
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
# Dataset using manifest.csv (auto-detect columns)
# -----------------------------
class ManifestSigmaDataset(Dataset):
    def __init__(self, manifest_csv: str, base_dir: Optional[str] = None, input_mode: str = "h_only"):
        """
        input_mode:
          - "h_only": feed only h(x)=log(|I-mu|) to CNN  (recommended)
          - "I_and_h": concatenate [I, h] as 2 channels
        """
        self.manifest_csv = manifest_csv
        self.base_dir = base_dir if base_dir is not None else os.path.dirname(manifest_csv)
        self.input_mode = input_mode

        self.items: List[Tuple[str, str]] = []
        with open(manifest_csv, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                noisy_key, sigma_key = auto_find_keys(row)
                if noisy_key is None or sigma_key is None:
                    continue
                noisy_path = find_existing_path(self.base_dir, row[noisy_key].strip())
                sigma_path = find_existing_path(self.base_dir, row[sigma_key].strip())
                if os.path.exists(noisy_path) and os.path.exists(sigma_path):
                    self.items.append((noisy_path, sigma_path))

        if len(self.items) == 0:
            raise RuntimeError(
                "No (noisy, sigma) pairs found from manifest. "
                "Open manifest.csv and confirm it contains paths to noisy images and sigma maps."
            )

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        noisy_path, sigma_path = self.items[idx]
        I = read_image_any(noisy_path)   # (H,W) float32 [0,1]
        sigma = read_image_any(sigma_path)  # (H,W) float32

        # Ensure sigma is in same scale. If sigma map was saved in [0,1], ok.
        # If sigma map stored as float already, leave it.
        I_t = torch.from_numpy(I)[None, ...]       # (1,H,W)
        s_t = torch.from_numpy(sigma)[None, ...]   # (1,H,W)
        return I_t, s_t, os.path.basename(noisy_path)

class FolderSigmaDataset(Dataset):
    """
    Dataset that uses two folders:
      - noisy_dir: images or npy files for I(x)
      - sigma_dir: images or npy files for sigma(x)

    Matching is done by basename (without extension).
    Example:
      noisy_dir:  noisy_slice_001.npy
      sigma_dir:  noisy_slice_001_sigma.npy  (or same name)
    """
    def __init__(self, noisy_dir: str, sigma_dir: str, input_mode: str = "h_only"):
        self.noisy_dir = noisy_dir
        self.sigma_dir = sigma_dir
        self.input_mode = input_mode

        noisy_files = sorted([
            f for f in os.listdir(noisy_dir)
            if os.path.splitext(f)[1].lower() in [".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]
        ])
        sigma_files = sorted([
            f for f in os.listdir(sigma_dir)
            if os.path.splitext(f)[1].lower() in [".npy", ".png", ".jpg", ".jpeg", ".tif", ".tiff"]
        ])

        # Build a map from basename -> file for sigma_dir
        sigma_map = {}
        for f in sigma_files:
            base = os.path.splitext(f)[0]
            sigma_map[base] = f

        self.items: List[Tuple[str, str]] = []
        for nf in noisy_files:
            base = os.path.splitext(nf)[0]
            if base in sigma_map:
                noisy_path = os.path.join(noisy_dir, nf)
                sigma_path = os.path.join(sigma_dir, sigma_map[base])
                self.items.append((noisy_path, sigma_path))

        if len(self.items) == 0:
            raise RuntimeError(
                f"No matching (noisy, sigma) pairs found between {noisy_dir} and {sigma_dir}."
            )

        print(f"[INFO] FolderSigmaDataset: found {len(self.items)} pairs.")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        noisy_path, sigma_path = self.items[idx]
        I = read_image_any(noisy_path)      # (H,W)
        sigma = read_image_any(sigma_path)  # (H,W)

        I_t = torch.from_numpy(I)[None, ...]      # (1,H,W)
        s_t = torch.from_numpy(sigma)[None, ...]  # (1,H,W)
        return I_t, s_t, os.path.basename(noisy_path)


# -----------------------------
# Losses
# -----------------------------
def log_mse_loss(pred_sigma: torch.Tensor, gt_sigma: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return F.mse_loss(torch.log(pred_sigma + eps), torch.log(gt_sigma + eps))


def tv_loss(x: torch.Tensor) -> torch.Tensor:
    """Total variation (encourage smooth sigma)."""
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    return dx + dy


# -----------------------------
# Train / Eval
# -----------------------------
@torch.no_grad()
def evaluate(model, blur, loader, device, input_mode, out_dir, step_tag="val", max_viz=3):
    model.eval()
    losses = []
    maes = []
    rmses = []

    viz_done = 0
    for I, sigma_gt, name in loader:
        I = I.to(device)              # (B,1,H,W)
        sigma_gt = sigma_gt.to(device)
        # pad both identically
        I, pad = pad_to_divisible(I, div=8)
        sigma_gt, _ = pad_to_divisible(sigma_gt, div=8)

        mu, r, h = homomorphic_features(I, blur)
        if input_mode == "h_only":
            x = h
        else:
            x = torch.cat([I, h], dim=1)

        sigma_pred = model(x)
        # unpad before loss
        sigma_pred = unpad(sigma_pred, pad)
        sigma_gt   = unpad(sigma_gt, pad)

        loss = log_mse_loss(sigma_pred, sigma_gt)  # log-domain is stable
        losses.append(loss.item())

        diff = sigma_pred - sigma_gt
        maes.append(diff.abs().mean().item())
        rmses.append(torch.sqrt((diff**2).mean()).item())

        if viz_done < max_viz:
            # save a 3-panel viz
            b0 = 0
            fig = plt.figure(figsize=(12, 4))
            ax1 = plt.subplot(1, 3, 1)
            ax1.imshow(sigma_gt[b0, 0].detach().cpu().numpy(), cmap="hot")
            ax1.set_title("sigma_GT"); ax1.axis("off")

            ax2 = plt.subplot(1, 3, 2)
            ax2.imshow(sigma_pred[b0, 0].detach().cpu().numpy(), cmap="hot")
            ax2.set_title("sigma_pred"); ax2.axis("off")

            ax3 = plt.subplot(1, 3, 3)
            ax3.imshow((sigma_pred[b0, 0] - sigma_gt[b0, 0]).detach().cpu().numpy(), cmap="bwr")
            ax3.set_title("pred - GT"); ax3.axis("off")

            plt.tight_layout()
            out_path = os.path.join(out_dir, f"{step_tag}_viz_{viz_done}_{name[b0]}.png")
            plt.savefig(out_path, dpi=200)
            plt.close(fig)
            viz_done += 1

    return {
        "loss": float(np.mean(losses)),
        "mae": float(np.mean(maes)),
        "rmse": float(np.mean(rmses)),
    }


def train(args):
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    ensure_dir(args.out_dir)

    # Dataset
#    ds = ManifestSigmaDataset(args.manifest, base_dir=args.base_dir, input_mode=args.input_mode)

        # Dataset
    if args.manifest is not None:
        ds = ManifestSigmaDataset(args.manifest, base_dir=args.base_dir, input_mode=args.input_mode)
    else:
        if args.noisy_dir is None or args.sigma_dir is None:
            raise ValueError("Either --manifest OR both --noisy_dir and --sigma_dir must be provided.")
        ds = FolderSigmaDataset(args.noisy_dir, args.sigma_dir, input_mode=args.input_mode)

    n_val = max(1, int(len(ds) * args.val_split))
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

    # Fixed homomorphic LPF
    blur = GaussianBlur2D(channels=1, kernel_size=args.blur_ksize, sigma=args.blur_sigma).to(device)

    # Model
    in_ch = 1 if args.input_mode == "h_only" else 2
    model = SmallUNet(in_ch=in_ch, base=args.base).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp))

    best_val = float("inf")

    # Save training config
    with open(os.path.join(args.out_dir, "config.txt"), "w") as f:
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
        f.write(f"device: {device}\n")
        f.write(f"train_size: {n_train}, val_size: {n_val}\n")

    print(f"[INFO] device={device} train={n_train} val={n_val} total={len(ds)}")
    print(f"[INFO] input_mode={args.input_mode}, blur_ksize={args.blur_ksize}, blur_sigma={args.blur_sigma}")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = []


        for I, sigma_gt, _name in train_loader:
            I = I.to(device)
            sigma_gt = sigma_gt.to(device)
        
            # ?? PAD HERE (this was missing)
            I, pad = pad_to_divisible(I, div=8)
            sigma_gt, _ = pad_to_divisible(sigma_gt, div=8)
        
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and args.amp)):
                mu, r, h = homomorphic_features(I, blur)
                if args.input_mode == "h_only":
                    x = h
                else:
                    x = torch.cat([I, h], dim=1)
        
                sigma_pred = model(x)
        
                # ?? UNPAD BEFORE LOSS
                sigma_pred = unpad(sigma_pred, pad)
                sigma_gt   = unpad(sigma_gt, pad)
        
                loss_main = log_mse_loss(sigma_pred, sigma_gt)
                loss_smooth = tv_loss(sigma_pred) if args.tv_weight > 0 else torch.tensor(0.0, device=device)
                loss = loss_main + args.tv_weight * loss_smooth
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running.append(loss.item())
            global_step += 1

        train_loss = float(np.mean(running))


        # Evaluate
        metrics = evaluate(model, blur, val_loader, device, args.input_mode, args.out_dir,
                           step_tag=f"epoch{epoch:03d}", max_viz=args.max_viz)

        msg = (f"Epoch {epoch:03d}/{args.epochs} | "
               f"train_loss={train_loss:.6f} | "
               f"val_loss={metrics['loss']:.6f} | val_mae={metrics['mae']:.6f} | val_rmse={metrics['rmse']:.6f}")
        print(msg)

        # Save last
        ckpt_last = {
            "epoch": epoch,
            "model": model.state_dict(),
            "opt": opt.state_dict(),
            "args": vars(args),
        }
        torch.save(ckpt_last, os.path.join(args.out_dir, "sigmanet_last.pth"))

        # Save best
        if metrics["loss"] < best_val:
            best_val = metrics["loss"]
            torch.save(ckpt_last, os.path.join(args.out_dir, "sigmanet_best.pth"))
            print(f"[INFO] saved best checkpoint (val_loss={best_val:.6f})")

    print("[DONE] Training complete.")
    print(f"Best val_loss: {best_val:.6f}")
    print(f"Checkpoints saved in: {args.out_dir}")


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default=None,
                    help="Path to manifest.csv (optional if using --noisy_dir/--sigma_dir)")
    ap.add_argument("--base_dir", default=None,
                    help="Base dir to resolve relative paths in manifest (default: manifest folder)")
    ap.add_argument("--noisy_dir", default=None,
                    help="Folder with noisy images or npy files")
    ap.add_argument("--sigma_dir", default=None,
                    help="Folder with sigma maps (same basename as noisy)")
    ap.add_argument("--out_dir", required=True, help="Output folder for checkpoints/plots")


    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--val_split", type=float, default=0.1)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)

    # Homomorphic LPF params
    ap.add_argument("--blur_ksize", type=int, default=31, help="Gaussian kernel size (odd)")
    ap.add_argument("--blur_sigma", type=float, default=7.0, help="Gaussian sigma for low-pass (mu)")

    # Model params
    ap.add_argument("--base", type=int, default=32, help="Base channels for UNet")
    ap.add_argument("--input_mode", choices=["h_only", "I_and_h"], default="h_only",
                    help="Feed only h=log(|I-mu|) or concatenate [I,h]")

    # Regularization
    ap.add_argument("--tv_weight", type=float, default=0.05, help="TV smoothness weight on sigma_pred (0 to disable)")
    ap.add_argument("--max_viz", type=int, default=3, help="How many val examples to save per epoch")

    # System
    ap.add_argument("--amp", action="store_true", help="Use mixed precision on GPU")
    ap.add_argument("--cpu", action="store_true", help="Force CPU")

    args = ap.parse_args()
    train(args)


if __name__ == "__main__":
    main()
