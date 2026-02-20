
import os
import glob
import csv
import argparse
from pathlib import Path
from typing import Tuple
import numpy as np
from PIL import Image

# Required for GaussianBlur + diagnostics
import cv2


# ---------------------------
# I/O helpers
# ---------------------------
def load_image(path: str, grayscale: bool = True) -> np.ndarray:
    img = Image.open(path)
    if grayscale:
        img = img.convert("L")
    arr = np.array(img, dtype=np.float32)

    # Normalize to [0,1]
    if arr.max() > 1.0:
        arr = arr / 255.0
    arr = np.clip(arr, 0.0, 1.0)
    return arr


def save_png01(path: str, img01: np.ndarray) -> None:
    img01 = np.clip(img01, 0.0, 1.0)
    im = (img01 * 255.0).round().astype(np.uint8)
    Image.fromarray(im).save(path)


# ---------------------------
# Sigma-map base-field generators
#   Return "base" in [0,1]
# ---------------------------
def make_base_field(H: int, W: int,
                    mode: str,
                    blur_sigma: float,
                    rng: np.random.Generator) -> np.ndarray:
    """
    Returns base(x) in [0,1].
    We'll scale it FOI-style afterwards:
        sigma_map = sigma_base * (0.5 + base)
    """
    mode = mode.lower()

    if mode == "radial":
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
        rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
        rr = rr / (rr.max() + 1e-8)
        base = rr  # [0,1]

    elif mode == "random_smooth":
        base = rng.normal(0.0, 1.0, size=(H, W)).astype(np.float32)
        base = cv2.GaussianBlur(base, (0, 0), blur_sigma, borderType=cv2.BORDER_REFLECT)
        base = (base - base.min()) / (base.max() - base.min() + 1e-8)

    elif mode == "piecewise":
        base = np.zeros((H, W), dtype=np.float32)
        yy, xx = np.mgrid[0:H, 0:W]
        n_blobs = int(rng.integers(3, 7))
        for _ in range(n_blobs):
            y0 = int(rng.integers(0, H))
            x0 = int(rng.integers(0, W))
            ry = int(rng.integers(max(8, H // 8), max(9, H // 3)))
            rx = int(rng.integers(max(8, W // 8), max(9, W // 3)))
            val = float(rng.random())  # [0,1]
            mask = (((yy - y0) / max(ry, 1)) ** 2 + ((xx - x0) / max(rx, 1)) ** 2) <= 1.0
            base[mask] = val

        base = cv2.GaussianBlur(base, (0, 0), max(1.0, blur_sigma / 2.0), borderType=cv2.BORDER_REFLECT)
        base = (base - base.min()) / (base.max() - base.min() + 1e-8)

    else:
        raise ValueError(f"Unknown sigma_map_mode: {mode}")

    return base.astype(np.float32)


# ---------------------------
# FOI-style sigma scaling
# ---------------------------
def make_sigma_map_foi(clean_mag01: np.ndarray,
                       base_field01: np.ndarray,
                       percentNoise: float) -> Tuple[np.ndarray, float]:
    """
    FOI MATLAB equivalent:
        sigma_base = percentNoise/100 * max(clean)
        sigma_map  = sigma_base * (0.5 + base_field), where base_field in [0,1]

    Returns:
        sigma_map, sigma_base
    """
    sigma_base = (percentNoise / 100.0) * float(clean_mag01.max())
    sigma_map = sigma_base * (0.5 + base_field01)  # in [0.5, 1.5]*sigma_base
    return sigma_map.astype(np.float32), float(sigma_base)


# ---------------------------
# Non-stationary Rician corruption (matches MATLAB exactly)
# ---------------------------
def add_nonstat_rician(clean_mag01: np.ndarray,
                       sigma_map: np.ndarray,
                       rng: np.random.Generator) -> np.ndarray:
    """
    MATLAB:
      n1 = sigma_map .* randn(size(nu));
      n2 = sigma_map .* randn(size(nu));
      z  = sqrt( (nu + n1).^2 + n2.^2 );
    """
    n1 = rng.normal(0.0, 1.0, size=clean_mag01.shape).astype(np.float32) * sigma_map
    n2 = rng.normal(0.0, 1.0, size=clean_mag01.shape).astype(np.float32) * sigma_map
    noisy = np.sqrt((clean_mag01 + n1) ** 2 + (n2) ** 2).astype(np.float32)
    return noisy


# ---------------------------
# Diagnostics (optional)
# ---------------------------
def local_variance(img01: np.ndarray, blur_sigma: float = 7.0) -> np.ndarray:
    img01 = img01.astype(np.float32)
    mu = cv2.GaussianBlur(img01, (0, 0), blur_sigma, borderType=cv2.BORDER_REFLECT)
    mu2 = cv2.GaussianBlur(img01 ** 2, (0, 0), blur_sigma, borderType=cv2.BORDER_REFLECT)
    return (mu2 - mu ** 2).astype(np.float32)


def corrcoef(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1)
    b = b.reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    denom = (np.sqrt((a * a).mean()) * np.sqrt((b * b).mean()) + 1e-12)
    return float((a * b).mean() / denom)


# ---------------------------
# Main
# ---------------------------
def main():
    p = argparse.ArgumentParser(
        description="Create FOI-style synthetic non-stationary Rician dataset and save sigma GT maps."
    )
    p.add_argument("--in_dir", type=str, required=True,
                   help="Folder with clean magnitude images (png/jpg/tif).")
    p.add_argument("--out_dir", type=str, required=True, help="Output folder.")
    p.add_argument("--exts", type=str, default=".png,.jpg,.jpeg,.tif,.tiff,.bmp",
                   help="Comma-separated extensions to read.")
    p.add_argument("--n_aug", type=int, default=1,
                   help="How many noisy realizations per input image (fixed to 1 for same-name files).")
    p.add_argument("--sigma_map_mode", type=str, default="radial",
                   choices=["random_smooth", "radial", "piecewise"],
                   help="Spatial pattern of sigma (FOI paper often uses smooth/radial).")
    p.add_argument("--sigma_blur", type=float, default=25.0,
                   help="Smoothness of base field for random_smooth/piecewise (Gaussian sigma).")
    p.add_argument("--percentNoise", type=float, default=11.0,
                   help="FOI-style: sigma_base = percentNoise/100 * max(clean).")
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducibility.")
    p.add_argument("--save_png", action="store_true",
                   help="Also save noisy/sigma preview PNGs.")
    p.add_argument("--save_clean_npy", action="store_true",
                   help="Also save clean as .npy (float32).")
    p.add_argument("--check", action="store_true",
                   help="Run sanity checks (corr between local var and sigma^2).")
    p.add_argument("--check_every", type=int, default=50,
                   help="How often to print check results.")

    args = p.parse_args()

    # Enforce n_aug = 1 to avoid overwriting same-name files
    if args.n_aug != 1:
        raise ValueError(
            f"This script is configured so that all output files share the same name as the input. "
            f"To avoid overwriting, it enforces --n_aug 1. You passed n_aug={args.n_aug}."
        )

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)

    noisy_dir = out_dir / "noisy_npy"
    sigma_dir = out_dir / "sigma_npy"
    clean_dir = out_dir / "clean_npy"
    noisy_png_dir = out_dir / "noisy_png"
    sigma_png_dir = out_dir / "sigma_png"

    noisy_dir.mkdir(parents=True, exist_ok=True)
    sigma_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.save_clean_npy:
        clean_dir.mkdir(parents=True, exist_ok=True)
    if args.save_png:
        noisy_png_dir.mkdir(parents=True, exist_ok=True)
        sigma_png_dir.mkdir(parents=True, exist_ok=True)

    exts = [e.strip().lower() for e in args.exts.split(",") if e.strip()]
    paths = []
    for e in exts:
        paths.extend(glob.glob(str(in_dir / f"*{e}")))
    paths = sorted(paths)
    if not paths:
        raise SystemExit(f"No images found in {in_dir} with exts {exts}")

    manifest_path = out_dir / "manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "id", "source_image", "clean_npy", "noisy_npy", "sigma_npy",
            "noisy_png", "sigma_png",
            "percentNoise", "sigma_base", "sigma_map_mode", "sigma_blur", "seed_used"
        ])

        sample_id = 0
        for i, src_path in enumerate(paths):
            clean = load_image(src_path, grayscale=True)  # [0,1]
            H, W = clean.shape
            stem = Path(src_path).stem  # base name, e.g., "t1_slice_001"

            clean_npy_path = ""
            if args.save_clean_npy:
                # SAME NAME as original (no suffix), only .npy extension
                clean_npy_path = str(clean_dir / f"{stem}.npy")
                np.save(clean_npy_path, clean.astype(np.float32))

            # Only one augmentation (n_aug = 1)
            for a in range(args.n_aug):
                seed_used = int(args.seed + 100000 * i + a)
                local_rng = np.random.default_rng(seed_used)

                base = make_base_field(
                    H, W,
                    mode=args.sigma_map_mode,
                    blur_sigma=args.sigma_blur,
                    rng=local_rng
                )

                sigma, sigma_base = make_sigma_map_foi(
                    clean_mag01=clean,
                    base_field01=base,
                    percentNoise=args.percentNoise
                )

                noisy = add_nonstat_rician(clean, sigma, rng=local_rng)

                # id and filenames use ONLY the original stem
                sid = stem

                noisy_npy = noisy_dir / f"{sid}.npy"
                sigma_npy = sigma_dir / f"{sid}.npy"
                np.save(noisy_npy, noisy.astype(np.float32))
                np.save(sigma_npy, sigma.astype(np.float32))

                noisy_png = ""
                sigma_png = ""
                if args.save_png:
                    noisy_png = str(noisy_png_dir / f"{sid}.png")
                    sigma_png = str(sigma_png_dir / f"{sid}.png")
                    save_png01(noisy_png, noisy)

                    # sigma preview only: scale to [0,1] for viewing
                    sigma_prev = (sigma - sigma.min()) / (sigma.max() - sigma.min() + 1e-8)
                    save_png01(sigma_png, sigma_prev)

                if args.check and (sample_id % max(1, args.check_every) == 0):
                    lv = local_variance(noisy, blur_sigma=7.0)
                    c = corrcoef(lv, sigma ** 2)
                    print(f"[check] {sid}: corr(local_var(noisy), sigma^2) = {c:.3f}")

                writer.writerow([
                    sid,
                    str(src_path),
                    clean_npy_path,
                    str(noisy_npy),
                    str(sigma_npy),
                    noisy_png,
                    sigma_png,
                    args.percentNoise,
                    sigma_base,
                    args.sigma_map_mode,
                    args.sigma_blur,
                    seed_used
                ])
                sample_id += 1

    print(f"Done.\nSaved dataset to: {out_dir}\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
