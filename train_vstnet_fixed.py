# train_vstnet_fixed.py
import os, math, glob, argparse
from dataclasses import dataclass

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
    """Load grayscale image to float32 in [0,1]."""
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def load_npy(path: str) -> np.ndarray:
    return np.load(path).astype(np.float32)


def save_image_grid(path, imgs: torch.Tensor, nrow=4):
    """
    imgs: [B,1,H,W] (roughly in [0,1] for visualization)
    """
    from torchvision.utils import make_grid
    grid = make_grid(imgs.clamp(0, 1), nrow=nrow, padding=2)
    nd = (grid.permute(1, 2, 0).detach().cpu().numpy() * 255.0).astype(np.uint8)
    Image.fromarray(nd).save(path)


def norm01(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-image normalize to [0,1] for visualization."""
    x = x.clone()
    b = x.shape[0]
    x = x.view(b, -1)
    mn = x.min(dim=1, keepdim=True).values
    mx = x.max(dim=1, keepdim=True).values
    x = (x - mn) / (mx - mn + eps)
    return x.view(b, 1, -1, 1).reshape_as(x.view(b, -1, 1, 1))  # dummy, overwritten below


def norm01_like(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Per-image normalize to [0,1] for visualization (keeps shape)."""
    b = x.shape[0]
    flat = x.view(b, -1)
    mn = flat.min(dim=1, keepdim=True).values
    mx = flat.max(dim=1, keepdim=True).values
    out = (flat - mn) / (mx - mn + eps)
    return out.view_as(x)


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
# CNN blocks
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
# Moments: var, skew, excess kurt
# -----------------------------
def masked_moments(x: torch.Tensor, mask: torch.Tensor, eps=1e-8):
    x = x.reshape(-1)
    m = mask.reshape(-1)
    wsum = m.sum().clamp_min(1.0)

    mu = (x * m).sum() / wsum
    xc = x - mu
    var = (m * (xc**2)).sum() / wsum
    std = torch.sqrt(var.clamp_min(eps))

    skew = (m * (xc**3)).sum() / (wsum * (std**3 + eps))
    kurt = (m * (xc**4)).sum() / (wsum * (std**4 + eps))
    exkurt = kurt - 3.0
    return mu, var, skew, exkurt


# -----------------------------
# Core model (GLOBAL u1,u2)
# -----------------------------
class VSTNet(nn.Module):
    def __init__(self, use_snr_proxy: bool = False, blur_ks: int = 21, blur_sigma: float = 3.0, eps: float = 1e-6):
        super().__init__()
        self.use_snr_proxy = use_snr_proxy
        self.eps = eps

        in_ch_A = 2 + (1 if use_snr_proxy else 0)

        # cnnA produces 2-channel map; we global-average -> 2 scalars per image
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

        u1 = F.softplus(a) + 1e-3                        # [B,1,1,1]
        u2 = F.softplus(b)                               # [B,1,1,1]

        frac = (I * I) / (sigma0_safe * sigma0_safe)
        inside = (u1 * u1) * frac - u2
        inside = inside.clamp_min(0.0)
        I_tilde = sigma0_safe * torch.sqrt(inside + eps)

        mu_hat = torch.zeros_like(I_tilde)

        return {"u1": u1, "u2": u2, "I_tilde": I_tilde, "mu_hat": mu_hat}


# -----------------------------
# Real folder dataset with padding
# -----------------------------
class RealFolderPairDataset(Dataset):
    def __init__(
        self,
        noisy_dir: str,
        sigma_dir: str,
        sigma_kind: str = "npy",
        sigma0_is_variance: bool = False,
        mask_dir: str = "",
        pad_multiple: int = 8,
        noisy_exts=(".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"),
    ):
        super().__init__()
        self.noisy_dir = noisy_dir
        self.sigma_dir = sigma_dir
        self.sigma_kind = sigma_kind
        self.sigma0_is_variance = sigma0_is_variance
        self.mask_dir = mask_dir
        self.pad_multiple = pad_multiple

        assert os.path.isdir(noisy_dir), f"noisy_dir not found: {noisy_dir}"
        assert os.path.isdir(sigma_dir), f"sigma_dir not found: {sigma_dir}"
        if mask_dir:
            assert os.path.isdir(mask_dir), f"mask_dir not found: {mask_dir}"

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
            raise ValueError(
                "No matching basenames between noisy_dir and sigma_dir.\n"
                f"Example noisy key: {next(iter(noisy_map.keys()))}\n"
                f"Example sigma key: {next(iter(sigma_map.keys()))}"
            )
        self.pairs = [(noisy_map[k], sigma_map[k], k) for k in keys]

        if len(keys) != len(noisy_map) or len(keys) != len(sigma_map):
            print(f"[WARN] noisy files: {len(noisy_map)}, sigma files: {len(sigma_map)}, matched: {len(keys)}")

    @staticmethod
    def pad_to_multiple(x: torch.Tensor, multiple: int):
        # x: [1,H,W]
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

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        noisy_path, sigma_path, key = self.pairs[idx]

        I = load_gray_png(noisy_path)
        s = load_npy(sigma_path) if self.sigma_kind == "npy" else load_gray_png(sigma_path)

        if self.sigma0_is_variance:
            s = np.sqrt(np.maximum(s, 1e-12)).astype(np.float32)

        if s.shape != I.shape:
            raise ValueError(f"Shape mismatch for '{key}': I {I.shape} vs sigma0 {s.shape}")

        I_t = torch.from_numpy(I)[None, ...]  # [1,H,W]
        s_t = torch.from_numpy(s)[None, ...]

        if self.mask_dir:
            mp = os.path.join(self.mask_dir, key + ".png")
            if not os.path.exists(mp):
                raise FileNotFoundError(f"Mask not found for {key}: {mp}")
            m = load_gray_png(mp)
            mask = torch.from_numpy((m > 0.5).astype(np.float32))[None, ...]
        else:
            # default: background-ish
            mask = (I_t < 0.08).float()

        I_t, _ = self.pad_to_multiple(I_t, self.pad_multiple)
        s_t, _ = self.pad_to_multiple(s_t, self.pad_multiple)
        mask, _ = self.pad_to_multiple(mask, self.pad_multiple)

        return I_t, s_t, mask


# -----------------------------
# Synthetic dataset
# -----------------------------
class SyntheticRicianDataset(Dataset):
    def __init__(self, n=2000, H=128, W=128, seed=0):
        super().__init__()
        self.n = n
        self.H, self.W = H, W
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        H, W = self.H, self.W
        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)

        cx, cy = self.rng.uniform(0.2 * W, 0.8 * W), self.rng.uniform(0.2 * H, 0.8 * H)
        rad = self.rng.uniform(0.15 * min(H, W), 0.35 * min(H, W))
        A = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * (rad**2))).astype(np.float32)
        A = 0.9 * A

        gx = np.linspace(0.02, 0.08, W, dtype=np.float32)[None, :].repeat(H, axis=0)
        cy2 = self.rng.uniform(0.2 * H, 0.8 * H)
        blob = np.exp(-((yy - cy2) ** 2) / (2 * (0.22 * H) ** 2)).astype(np.float32)
        sigma = gx * (0.6 + 0.6 * blob)

        n1 = self.rng.normal(0, 1, (H, W)).astype(np.float32) * sigma
        n2 = self.rng.normal(0, 1, (H, W)).astype(np.float32) * sigma
        I = np.sqrt((A + n1) ** 2 + (n2) ** 2).astype(np.float32)

        mask = (A < 0.05).astype(np.float32)
        return torch.from_numpy(I)[None, ...], torch.from_numpy(sigma)[None, ...], torch.from_numpy(mask)[None, ...]


# -----------------------------
# Train config
# -----------------------------
@dataclass
class TrainConfig:
    device: str = "cuda"
    lr: float = 2e-4
    batch_size: int = 8
    epochs: int = 20
    use_snr_proxy: bool = False
    lambda_var: float = 0.998
    lambda_skew: float = 0.001
    lambda_kurt: float = 0.001
    lambda_mean: float = 0.1
    log_every: int = 50
    save_every: int = 200


def train_one_epoch(model, loader, opt, cfg: TrainConfig, epoch: int, out_dir: str, log_f=None):
    model.train()
    step = 0

    for (I, sigma0, _mask_ds) in loader:
        I = I.to(cfg.device)
        sigma0 = sigma0.to(cfg.device)

        # ---- Quantile gradient mask (your version) ----
        dx = F.pad(I[:, :, :, 1:] - I[:, :, :, :-1], (0, 1, 0, 0))
        dy = F.pad(I[:, :, 1:, :] - I[:, :, :-1, :], (0, 0, 0, 1))
        g = torch.sqrt(dx * dx + dy * dy + 1e-12)

        #tg = torch.quantile(g.flatten(1), 0.1, dim=1).view(-1, 1, 1, 1)
        #gmask = (g < tg).float()

        #ti = torch.quantile(I.flatten(1), 0.08, dim=1).view(-1, 1, 1, 1)
        #imask = (I > ti).float()
        # --- stricter gradient mask: keep only the flattest 10% regions ---
        tg = torch.quantile(g.flatten(1), 0.10, dim=1).view(-1, 1, 1, 1)
        gmask = (g < tg).float()
        
        # --- stricter intensity mask: exclude dark background AND very bright structure ---
        ti_low  = torch.quantile(I.flatten(1), 0.05, dim=1).view(-1, 1, 1, 1)
        ti_high = torch.quantile(I.flatten(1), 0.995, dim=1).view(-1, 1, 1, 1)
        imask = ((I > ti_low) & (I < ti_high)).float()

        mask = gmask * imask
        mask_frac = mask.mean().item()

        out = model(I, sigma0)

        # Paper-style stabilized RV:
        # Z = (I_tilde - LPF(I_tilde)) / (sigma0 + eps)
        I_tilde = out["I_tilde"]
        I_hp = I_tilde - model.lpf(I_tilde)
        Z = I_hp / (sigma0.clamp_min(1e-6) + 1e-6)

        mu, var, skew, exkurt = masked_moments(Z, mask)

        # Piechowiak Eq.(8) + optional mean
        loss = cfg.lambda_var * (1.0 - var).pow(2) \
             + cfg.lambda_skew * skew.pow(2) \
             + cfg.lambda_kurt * exkurt.pow(2)

        if cfg.lambda_mean > 0:
            loss = loss + cfg.lambda_mean * mu.pow(2)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()

        # stats for logging
        m = mask > 0.5
        if m.any():
            Zm = Z[m]
            rm = Zm.mean()
            rstd = Zm.std(unbiased=False)
        else:
            rm = Z.mean()
            rstd = Z.std(unbiased=False)

        if step % cfg.log_every == 0:
            u1v = out["u1"].detach().mean().item()
            u2v = out["u2"].detach().mean().item()
            msg = (
                f"[E{epoch:03d} S{step:05d}] "
                f"loss={loss.item():.6f} "
                f"var={var.item():.4f} skew={skew.item():.4f} exkurt={exkurt.item():.4f} "
                f"mu={mu.item():.4f} rm={rm.item():.4f} rstd={rstd.item():.4f} "
                f"u1={u1v:.4f} u2={u2v:.4f} "
                f"mask={mask_frac:.3f}"
            )
            print(msg)
            if log_f is not None:
                log_f.write(msg + "\n")

        if step % cfg.save_every == 0:
            os.makedirs(out_dir, exist_ok=True)
            with torch.no_grad():
                # normalize for visualization
                I_show = I[:4].clamp(0, 1)
                It_show = I_tilde[:4].clamp(0, 1)
                Ihp_show = norm01_like(I_hp[:4].abs())
                Z_show = norm01_like(Z[:4].abs())
                s_show = norm01_like(sigma0[:4])

                vis = torch.cat([I_show, It_show, Ihp_show, Z_show, s_show], dim=0)
                save_image_grid(os.path.join(out_dir, f"debug_e{epoch:03d}_s{step:05d}.png"), vis, nrow=4)

        step += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="./runs_vstnet")
    ap.add_argument("--seed", type=int, default=0)

    # data
    ap.add_argument("--synthetic", type=int, default=1)
    ap.add_argument("--noisy_dir", type=str, default="")
    ap.add_argument("--sigma_dir", type=str, default="")
    ap.add_argument("--mask_dir", type=str, default="")
    ap.add_argument("--sigma_kind", type=str, default="npy", choices=["npy", "png"])
    ap.add_argument("--sigma0_is_variance", type=int, default=0)
    ap.add_argument("--pad_multiple", type=int, default=8)

    # model
    ap.add_argument("--use_snr_proxy", type=int, default=0)
    ap.add_argument("--blur_ks", type=int, default=21)
    ap.add_argument("--blur_sigma", type=float, default=3.0)

    # train
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=200)

    # loss weights
    ap.add_argument("--lambda_var", type=float, default=0.998)
    ap.add_argument("--lambda_skew", type=float, default=0.001)
    ap.add_argument("--lambda_kurt", type=float, default=0.001)
    ap.add_argument("--lambda_mean", type=float, default=0.1)

    args = ap.parse_args()

    seed_everything(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    log_path = os.path.join(args.out_dir, "train_log.txt")
    log_f = open(log_path, "a", buffering=1)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    log_f.write(f"Using device: {device}\n")

    # dataset
    if args.synthetic == 1:
        ds = SyntheticRicianDataset(n=2000, H=128, W=128, seed=args.seed)
    else:
        if (not args.noisy_dir) or (not args.sigma_dir):
            raise ValueError("For --synthetic 0, you must set --noisy_dir and --sigma_dir")
        ds = RealFolderPairDataset(
            noisy_dir=args.noisy_dir,
            sigma_dir=args.sigma_dir,
            sigma_kind=args.sigma_kind,
            sigma0_is_variance=bool(args.sigma0_is_variance),
            mask_dir=args.mask_dir,
            pad_multiple=args.pad_multiple,
        )

    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    cfg = TrainConfig(
        device=str(device),
        lr=args.lr,
        batch_size=args.batch_size,
        epochs=args.epochs,
        use_snr_proxy=bool(args.use_snr_proxy),
        lambda_var=args.lambda_var,
        lambda_skew=args.lambda_skew,
        lambda_kurt=args.lambda_kurt,
        lambda_mean=args.lambda_mean,
        log_every=args.log_every,
        save_every=args.save_every,
    )

    model = VSTNet(
        use_snr_proxy=cfg.use_snr_proxy,
        blur_ks=args.blur_ks,
        blur_sigma=args.blur_sigma,
        eps=1e-6,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    try:
        for e in range(cfg.epochs):
            train_one_epoch(model, loader, opt, cfg, epoch=e, out_dir=args.out_dir, log_f=log_f)

            ckpt = {
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "epoch": e,
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(args.out_dir, f"ckpt_epoch_{e:03d}.pt"))
            msg = f"Saved: ckpt_epoch_{e:03d}.pt"
            print(msg)
            log_f.write(msg + "\n")
    finally:
        log_f.close()


if __name__ == "__main__":
    main()
