# infer_vstnet_fixed.py

import os, glob, argparse, csv
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


# -----------------------------
# Utils
# -----------------------------
def seed_everything(seed: int = 0):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def load_gray_png(path: str) -> np.ndarray:
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr

def load_npy(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32)

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def save_npy(path: str, x: np.ndarray):
    np.save(path, x.astype(np.float32))

def save_png_01(path: str, x01: np.ndarray):
    """Assumes x01 already in [0,1]"""
    x01 = np.clip(x01, 0.0, 1.0)
    im = (x01 * 255.0).astype(np.uint8)
    Image.fromarray(im).save(path)


# -----------------------------
# Fixed LPF: Gaussian blur
# -----------------------------
def gaussian_kernel_2d(kernel_size: int, sigma: float, device, dtype):
    ax = torch.arange(kernel_size, device=device, dtype=dtype) - (kernel_size - 1) / 2.0
    xx, yy = torch.meshgrid(ax, ax, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel

class FixedGaussianBlur(nn.Module):
    def __init__(self, kernel_size: int = 21, sigma: float = 3.0):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma
        self.register_buffer("_w", torch.empty(0), persistent=False)

    def forward(self, x):
        if self._w.numel() == 0 or self._w.device != x.device or self._w.dtype != x.dtype:
            k = gaussian_kernel_2d(self.kernel_size, self.sigma, x.device, x.dtype)
            w = k.view(1, 1, self.kernel_size, self.kernel_size)
            w = w.repeat(x.shape[1], 1, 1, 1)  # [C,1,ks,ks]
            self._w = w

        pad = self.kernel_size // 2
        x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
        y = F.conv2d(x_pad, self._w, groups=x.shape[1])
        return y


# -----------------------------
# CNN blocks (same as training)
# -----------------------------
class ConvBlock(nn.Module):
    def __init__(self, cin, cout, k=3, s=1, p=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cout, k, s, p),
            nn.GroupNorm(num_groups=min(8, cout), num_channels=cout),
            nn.SiLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)

class SimpleUNetLike(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, base: int = 32):
        super().__init__()
        self.enc1 = nn.Sequential(ConvBlock(in_ch, base), ConvBlock(base, base))
        self.down1 = nn.Conv2d(base, base * 2, 4, 2, 1)  # /2
        self.enc2 = nn.Sequential(ConvBlock(base * 2, base * 2), ConvBlock(base * 2, base * 2))

        self.down2 = nn.Conv2d(base * 2, base * 4, 4, 2, 1)  # /4
        self.bot = nn.Sequential(ConvBlock(base * 4, base * 4), ConvBlock(base * 4, base * 4))

        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 4, 2, 1)
        self.dec2 = nn.Sequential(ConvBlock(base * 4, base * 2), ConvBlock(base * 2, base * 2))

        self.up1 = nn.ConvTranspose2d(base * 2, base, 4, 2, 1)
        self.dec1 = nn.Sequential(ConvBlock(base * 2, base), ConvBlock(base, base))

        self.out = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2in = self.down1(e1)
        e2 = self.enc2(e2in)
        b_in = self.down2(e2)
        b = self.bot(b_in)

        u2 = self.up2(b)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))
        u1 = self.up1(d2)

        if u1.shape[-2:] != e1.shape[-2:]:
            u1 = F.interpolate(u1, size=e1.shape[-2:], mode="bilinear", align_corners=False)

        d1 = self.dec1(torch.cat([u1, e1], dim=1))
        return self.out(d1)


# -----------------------------
# VSTNet (same as training)
# -----------------------------
class VSTNet(nn.Module):
    def __init__(self, use_snr_proxy: bool = False, blur_ks: int = 21, blur_sigma: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.use_snr_proxy = use_snr_proxy
        self.eps = eps

        in_ch_A = 2 + (1 if use_snr_proxy else 0)
        self.cnnA = SimpleUNetLike(in_ch_A, out_ch=2, base=32)
        self.lpf = FixedGaussianBlur(kernel_size=blur_ks, sigma=blur_sigma)

    def forward(self, I: torch.Tensor, sigma0: torch.Tensor) -> dict:
        eps = self.eps
        sigma0_safe = sigma0.clamp_min(eps)

        if self.use_snr_proxy:
            snr_proxy = self.lpf(I) / sigma0_safe
            A_in = torch.cat([I, sigma0_safe, snr_proxy], dim=1)
        else:
            A_in = torch.cat([I, sigma0_safe], dim=1)

        ab_map = self.cnnA(A_in)                         # [B,2,H,W]
        ab = ab_map.mean(dim=(-1, -2), keepdim=True)     # [B,2,1,1]
        a, b = ab[:, :1], ab[:, 1:]

        u1 = F.softplus(a) + 1e-3
        u2 = F.softplus(b)

        frac = (I * I) / (sigma0_safe * sigma0_safe)
        inside = (u1 * u1) * frac - u2
        inside = inside.clamp_min(0.0)
        I_tilde = sigma0_safe * torch.sqrt(inside + eps)

        mu_hat = torch.zeros_like(I_tilde)
        return {"u1": u1, "u2": u2, "I_tilde": I_tilde, "mu_hat": mu_hat}


# -----------------------------
# Dataset (pad fix included)
# -----------------------------
class RealFolderPairDataset(Dataset):
    def __init__(
        self,
        noisy_dir: str,
        sigma_dir: str,
        sigma_kind: str = "npy",
        sigma0_is_variance: bool = False,
        pad_multiple: int = 8,
        noisy_exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp",".npy"),
    ):
        super().__init__()
        self.noisy_dir = noisy_dir
        self.sigma_dir = sigma_dir
        self.sigma_kind = sigma_kind
        self.sigma0_is_variance = sigma0_is_variance
        self.pad_multiple = pad_multiple

        noisy_paths = []
        for e in noisy_exts:
            noisy_paths.extend(glob.glob(os.path.join(noisy_dir, f"*{e}")))
        noisy_paths = sorted(noisy_paths)
        if not noisy_paths:
            raise ValueError(f"No noisy images found in {noisy_dir}")
        noisy_map = {os.path.splitext(os.path.basename(p))[0]: p for p in noisy_paths}

        if sigma_kind == "npy":
            sigma_paths = sorted(glob.glob(os.path.join(sigma_dir, "*.npy")))
        else:
            sigma_paths = []
            for e in noisy_exts:
                sigma_paths.extend(glob.glob(os.path.join(sigma_dir, f"*{e}")))
            sigma_paths = sorted(sigma_paths)
        if not sigma_paths:
            raise ValueError(f"No sigma files found in {sigma_dir} (sigma_kind={sigma_kind})")
        sigma_map = {os.path.splitext(os.path.basename(p))[0]: p for p in sigma_paths}

        keys = sorted(set(noisy_map.keys()) & set(sigma_map.keys()))
        if not keys:
            raise ValueError("No matching basenames between noisy_dir and sigma_dir.")
        self.items = [(k, noisy_map[k], sigma_map[k]) for k in keys]

        if len(keys) != len(noisy_map) or len(keys) != len(sigma_map):
            print(f"[WARN] noisy files: {len(noisy_map)}, sigma files: {len(sigma_map)}, matched: {len(keys)}")

    @staticmethod
    def pad_to_multiple(x: torch.Tensor, multiple: int):
        _, H, W = x.shape
        pad_h = (multiple - H % multiple) % multiple
        pad_w = (multiple - W % multiple) % multiple

        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left

        x4 = x.unsqueeze(0)  # [1,1,H,W]
        x4 = F.pad(x4, (pad_left, pad_right, pad_top, pad_bottom), mode="reflect")
        x = x4.squeeze(0)
        return x, (pad_left, pad_right, pad_top, pad_bottom)

    @staticmethod
    def unpad(x: torch.Tensor, pad4):
        pl, pr, pt, pb = pad4
        if (pl + pr + pt + pb) == 0:
            return x
        return x[..., pt:x.shape[-2]-pb, pl:x.shape[-1]-pr]

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        key, noisy_path, sigma_path = self.items[idx]

        if noisy_path.lower().endswith(".npy"):
            I = load_npy(noisy_path)
        else:
            I = load_gray_png(noisy_path)

        if self.sigma_kind == "npy":
            s = load_npy(sigma_path)
        else:
            s = load_gray_png(sigma_path)

        if self.sigma0_is_variance:
            s = np.sqrt(np.maximum(s, 1e-12)).astype(np.float32)

        if s.shape != I.shape:
            raise ValueError(f"Shape mismatch for '{key}': I {I.shape} vs sigma0 {s.shape}")

        I_t = torch.from_numpy(I)[None, ...]  # [1,H,W]
        s_t = torch.from_numpy(s)[None, ...]

        I_t, pad = self.pad_to_multiple(I_t, self.pad_multiple)
        s_t, _   = self.pad_to_multiple(s_t, self.pad_multiple)

        # ? IMPORTANT: return pad as Tensor([pl,pr,pt,pb]) so batching is stable
        pad_t = torch.tensor(pad, dtype=torch.int64)  # [4]
        return key, I_t, s_t, pad_t


# -----------------------------
# Inference
# -----------------------------
@torch.no_grad()
def run_infer(args):
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model = VSTNet(
        use_snr_proxy=bool(args.use_snr_proxy),
        blur_ks=args.blur_ks,
        blur_sigma=args.blur_sigma,
        eps=1e-6,
    ).to(device)

    ckpt = torch.load(args.ckpt, map_location="cpu")
    sd = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(sd, strict=True)
    model.eval()

    ds = RealFolderPairDataset(
        noisy_dir=args.noisy_dir,
        sigma_dir=args.sigma_dir,
        sigma_kind=args.sigma_kind,
        sigma0_is_variance=bool(args.sigma0_is_variance),
        pad_multiple=args.pad_multiple,
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    out_root = args.out_dir
    ensure_dir(out_root)

    # ? DDPM outputs
    out_it_npy = os.path.join(out_root, "I_tilde_npy"); ensure_dir(out_it_npy)  # RAW float32
    out_it_png = os.path.join(out_root, "I_tilde_png"); ensure_dir(out_it_png)  # clamp(0,1) png

    # optional extra outputs (raw)
    out_hp_npy = os.path.join(out_root, "Ihp_npy"); ensure_dir(out_hp_npy)
    out_z_npy  = os.path.join(out_root, "Z_npy");   ensure_dir(out_z_npy)

    # u1/u2 csv
    csv_path = os.path.join(out_root, "u1_u2.csv")
    fcsv = open(csv_path, "w", newline="")
    w = csv.writer(fcsv)
    w.writerow(["key", "u1", "u2"])

    for bi, batch in enumerate(loader):
        keys, I, sigma0, pads = batch
        I = I.to(device)
        sigma0 = sigma0.to(device)

        out = model(I, sigma0)
        u1 = out["u1"].detach().cpu().view(-1).numpy()
        u2 = out["u2"].detach().cpu().view(-1).numpy()

        I_tilde = out["I_tilde"]
        I_hp = I_tilde - model.lpf(I_tilde)
        Z = I_hp / (sigma0.clamp_min(1e-6) + 1e-6)

        for j in range(I.shape[0]):
            key = keys[j]

            pad4 = tuple(int(x) for x in pads[j].cpu().tolist())  # (pl,pr,pt,pb)

            It = RealFolderPairDataset.unpad(I_tilde[j:j+1], pad4)[0, 0].detach().cpu().numpy()
            Ih = RealFolderPairDataset.unpad(I_hp[j:j+1],    pad4)[0, 0].detach().cpu().numpy()
            Zj = RealFolderPairDataset.unpad(Z[j:j+1],       pad4)[0, 0].detach().cpu().numpy()

            # ? RAW: best for DDPM
            save_npy(os.path.join(out_it_npy, f"{key}.npy"), It)

            # ? PNG like training visualization: clamp(0,1) (NO per-image min-max)
            save_png_01(os.path.join(out_it_png, f"{key}.png"), np.clip(It, 0.0, 1.0))

            # optional raw saves
            if args.save_extra == 1:
                save_npy(os.path.join(out_hp_npy, f"{key}.npy"), Ih)
                save_npy(os.path.join(out_z_npy,  f"{key}.npy"), Zj)

            w.writerow([key, float(u1[j]), float(u2[j])])

        if (bi % max(1, args.log_every)) == 0:
            print(f"[{bi:05d}/{len(loader):05d}] example key={keys[0]} u1={u1[0]:.4f} u2={u2[0]:.4f}")

    fcsv.close()
    print("Done. Saved to:", out_root)
    print("I_tilde raw npy:", out_it_npy)
    print("I_tilde png clamp:", out_it_png)
    print("u1/u2 csv:", csv_path)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--noisy_dir", type=str, required=True)
    ap.add_argument("--sigma_dir", type=str, required=True)
    ap.add_argument("--sigma_kind", type=str, default="npy", choices=["npy", "png"])
    ap.add_argument("--sigma0_is_variance", type=int, default=0)
    ap.add_argument("--pad_multiple", type=int, default=8)

    ap.add_argument("--out_dir", type=str, required=True)

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)

    # must match how your model was configured at training
    ap.add_argument("--use_snr_proxy", type=int, default=0)
    ap.add_argument("--blur_ks", type=int, default=21)
    ap.add_argument("--blur_sigma", type=float, default=3.0)

    # optional extra saves
    ap.add_argument("--save_extra", type=int, default=0, help="1 to save Ihp_npy and Z_npy too")

    args = ap.parse_args()
    seed_everything(args.seed)
    run_infer(args)

if __name__ == "__main__":
    main()

