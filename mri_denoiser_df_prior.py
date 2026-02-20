import math
from pathlib import Path
from random import random
from functools import partial
from multiprocessing import cpu_count
from collections import namedtuple
import numpy as np

import os
import glob

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList
from torch.utils.data import Dataset as TorchDataset, DataLoader

from torchvision import transforms as T, utils
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm

# Optional deps (comment out if not installed)
try:
    from ema_pytorch import EMA
    HAS_EMA = True
except Exception:
    HAS_EMA = False

try:
    from accelerate import Accelerator
    HAS_ACCELERATE = True
except Exception:
    HAS_ACCELERATE = False

# If you have lucidrains' attend kernels; else we fall back to vanilla attention below
try:
    from denoising_diffusion_pytorch.attend import Attend
    HAS_ATTEND = True
except Exception:
    HAS_ATTEND = False

# --------------------
# Utilities & helpers
# --------------------

# --- keep original resolution: pad to multiple of 8, then unpad back ---
def pad_to_multiple(tensor, multiple=8):
    # tensor: [B, C, H, W]
    _, _, h, w = tensor.shape
    pad_h = (multiple - (h % multiple)) % multiple
    pad_w = (multiple - (w % multiple)) % multiple
    if pad_h == 0 and pad_w == 0:
        return tensor, (0, 0, 0, 0)
    # F.pad order: (left, right, top, bottom)
    pad = (0, pad_w, 0, pad_h)
    return F.pad(tensor, pad, mode='reflect'), pad


def unpad_to_original(tensor, pad):
    # tensor: [B, C, H, W]; pad = (left, right, top, bottom)
    l, r, t, b = pad
    if (l, r, t, b) == (0, 0, 0, 0):
        return tensor
    return tensor[..., t:tensor.shape[-2] - b if b > 0 else None,
                  l:tensor.shape[-1] - r if r > 0 else None]


ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])


def exists(x):
    return x is not None


def default(val, d):
    return val if exists(val) else (d() if callable(d) else d)


def cast_tuple(t, length=1):
    return t if isinstance(t, tuple) else ((t,) * length)


def divisible_by(numer, denom):
    return (numer % denom) == 0


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


def convert_image_to_fn(img_type, image):
    if img_type and getattr(image, "mode", None) != img_type:
        return image.convert(img_type)
    return image

# --------------------
# RMSNorm
# --------------------


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.g * self.scale

# --------------------
# Positional embeddings
# --------------------


class SinusoidalPosEmb(Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        self.dim = dim
        self.theta = theta

    def forward(self, x):
        device = x.device
        half = self.dim // 2
        emb = math.log(self.theta) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(Module):
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half = dim // 2
        self.weights = nn.Parameter(torch.randn(half), requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return torch.cat((x, fouriered), dim=-1)

# --------------------
# Blocks
# --------------------


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1)
    )


def Downsample(dim, dim_out=None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1=2, p2=2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )


class Block(Module):
    def __init__(self, dim, dim_out, dropout=0.):
        super().__init__()
        self.proj = nn.Conv2d(dim, dim_out, 3, padding=1)
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return self.dropout(x)


class ResnetBlock(Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None, dropout=0.):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, dropout=dropout)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)


class LinearAttention(Module):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden = dim_head * heads
        self.norm = RMSNorm(dim)
        self.mem_kv = nn.Parameter(torch.randn(2, heads, dim_head, num_mem_kv))
        self.to_qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.to_out = nn.Sequential(nn.Conv2d(hidden, dim, 1), RMSNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads), qkv)
        mk, mv = map(lambda t: repeat(t, 'h c n -> b h c n', b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-1), ((mk, k), (mv, v)))
        q = q.softmax(dim=-2) * self.scale
        k = k.softmax(dim=-1)
        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)
        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out, 'b h c (x y) -> b (h c) x y', h=self.heads, x=h, y=w)
        return self.to_out(out)


class VanillaAttend(Module):
    def __init__(self):
        super().__init__()

    def forward(self, q, k, v):
        scale = q.shape[-1] ** -0.5
        attn = (q * scale) @ k.transpose(-1, -2)
        attn = attn.softmax(dim=-1)
        return attn @ v


class Attention(Module):
    def __init__(self, dim, heads=4, dim_head=32, num_mem_kv=4, flash=False):
        super().__init__()
        self.heads = heads
        hidden = dim_head * heads
        self.norm = RMSNorm(dim)
        self.attend = Attend(flash=flash) if HAS_ATTEND else VanillaAttend()
        self.mem_kv = nn.Parameter(torch.randn(2, heads, num_mem_kv, dim_head))
        self.to_qkv = nn.Conv2d(dim, hidden * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h (x y) c', h=self.heads), qkv)
        mk, mv = map(lambda t: repeat(t, 'h n d -> b h n d', b=b), self.mem_kv)
        k, v = map(partial(torch.cat, dim=-2), ((mk, k), (mv, v)))
        out = self.attend(q, k, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)

# --------------------
# U-Net Backbone (unconditional)
# --------------------


class Unet(Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=1,                 # MRI is usually single-channel
        self_condition=False,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
        sinusoidal_pos_emb_theta=10000,
        dropout=0.,
        attn_dim_head=32,
        attn_heads=4,
        full_attn=None,             # full attention only for inner most layer by default
        flash_attn=False
    ):
        super().__init__()
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 4
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim, theta=sinusoidal_pos_emb_theta)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        if not full_attn:
            full_attn = (*((False,) * (len(dim_mults) - 1)), True)

        num_stages = len(dim_mults)
        full_attn = cast_tuple(full_attn, num_stages)
        attn_heads = cast_tuple(attn_heads, num_stages)
        attn_dim_head = cast_tuple(attn_dim_head, num_stages)
        assert len(full_attn) == len(dim_mults)

        FullAttention = partial(Attention, flash=flash_attn)
        resnet_block = partial(ResnetBlock, time_emb_dim=time_dim, dropout=dropout)

        self.downs = ModuleList([])
        self.ups = ModuleList([])
        num_resolutions = len(in_out)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(
            zip(in_out, full_attn, attn_heads, attn_dim_head)
        ):
            is_last = ind >= (num_resolutions - 1)
            attn_klass = FullAttention if layer_full_attn else LinearAttention

            self.downs.append(ModuleList([
                resnet_block(dim_in, dim_in),
                resnet_block(dim_in, dim_in),
                attn_klass(dim_in, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(dim_in, dim_out, 3, padding=1)
            ]))

        mid_dim = dims[-1]
        self.mid_block1 = resnet_block(mid_dim, mid_dim)
        self.mid_attn = FullAttention(mid_dim, heads=attn_heads[-1], dim_head=attn_dim_head[-1])
        self.mid_block2 = resnet_block(mid_dim, mid_dim)

        for ind, ((dim_in, dim_out), layer_full_attn, layer_attn_heads, layer_attn_dim_head) in enumerate(
            zip(*map(reversed, (in_out, full_attn, attn_heads, attn_dim_head)))
        ):
            is_last = ind == (len(in_out) - 1)
            attn_klass = FullAttention if layer_full_attn else LinearAttention
            self.ups.append(ModuleList([
                resnet_block(dim_out + dim_in, dim_out),
                resnet_block(dim_out + dim_in, dim_out),
                attn_klass(dim_out, dim_head=layer_attn_dim_head, heads=layer_attn_heads),
                Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(dim_out, dim_in, 3, padding=1)
            ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = resnet_block(init_dim * 2, init_dim)
        self.final_conv = nn.Conv2d(init_dim, self.out_dim, 1)

    @property
    def downsample_factor(self):
        return 2 ** (len(self.downs) - 1)

    def forward(self, x, time, x_self_cond=None):
        assert all([divisible_by(d, self.downsample_factor) for d in x.shape[-2:]]), \
            f'input dims {x.shape[-2:]} must be divisible by {self.downsample_factor}'

        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()
        t = self.time_mlp(time)

        h = []
        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)
            x = block2(x, t)
            x = attn(x) + x
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x) + x
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x) + x
            x = upsample(x)

        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, t)
        return self.final_conv(x)

# --------------------
# Schedules & helpers
# --------------------


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-((t * (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def _match_sigma_to_t(alphas_cumprod, sigma2):
    # choose t such that 1 - alpha_bar_t ≈ sigma^2
    target = 1.0 - sigma2
    diffs = (alphas_cumprod - target).abs()
    t = int(torch.argmin(diffs).item())
    return max(0, min(t, len(alphas_cumprod) - 1))


@torch.no_grad()
def sigma_noise_from_highpass_mad(I_tilde, k=5, eps=1e-8):
    blur = F.avg_pool2d(I_tilde, kernel_size=k, stride=1, padding=k//2)
    r = I_tilde - blur
    # robust per-image MAD over residual
    B = I_tilde.size(0)
    rf = r.view(B, -1)
    med = rf.median(dim=1, keepdim=True).values
    mad = (rf - med).abs().median(dim=1, keepdim=True).values
    sigma = 1.4826 * mad
    return sigma.view(B,1,1,1).clamp(min=eps)

# --------------------
# GaussianDiffusion (pure prior, noise-MSE training + PET-style fidelity at inference)
# --------------------


class GaussianDiffusion(Module):
    def __init__(
        self,
        model,
        *,
        image_size,
        timesteps=1000,
        sampling_timesteps=None,
        objective='pred_noise',        # pure noise prediction
        beta_schedule='sigmoid',
        schedule_fn_kwargs=dict(),
        ddim_sampling_eta=0.,
        auto_normalize=True,
        offset_noise_strength=0.,
        min_snr_loss_weight=False,
        min_snr_gamma=5,
        immiscible=False
    ):
        super().__init__()
        assert objective in {'pred_noise', 'pred_x0', 'pred_v'}
        assert model.channels == model.out_dim, "Model out_dim must match channels"

        self.model = model
        self.channels = model.channels
        self.self_condition = model.self_condition
        self.objective = objective

        # image size handling (free-size mode supported)
        if isinstance(image_size, int):
            if image_size > 0:
                image_size = (image_size, image_size)
            else:
                image_size = None
        elif isinstance(image_size, (tuple, list)):
            image_size = tuple(image_size)
        else:
            image_size = None
        self.image_size = image_size

        # beta schedule
        if beta_schedule == 'linear':
            beta_schedule_fn = linear_beta_schedule
        elif beta_schedule == 'cosine':
            beta_schedule_fn = cosine_beta_schedule
        elif beta_schedule == 'sigmoid':
            beta_schedule_fn = sigmoid_beta_schedule
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        betas = beta_schedule_fn(timesteps, **schedule_fn_kwargs)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)

        self.sampling_timesteps = default(sampling_timesteps, timesteps)
        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        rb = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        rb('betas', betas)
        rb('alphas_cumprod', alphas_cumprod)
        rb('alphas_cumprod_prev', alphas_cumprod_prev)

        rb('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        rb('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        rb('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        rb('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        rb('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        rb('posterior_variance', posterior_variance)
        rb('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        rb('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        rb('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.immiscible = immiscible
        self.offset_noise_strength = offset_noise_strength

        # loss weighting (min-SNR trick)
        snr = alphas_cumprod / (1 - alphas_cumprod)
        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == 'pred_noise':
            rb('loss_weight', maybe_clipped_snr / snr)
        elif objective == 'pred_x0':
            rb('loss_weight', maybe_clipped_snr)
        elif objective == 'pred_v':
            rb('loss_weight', maybe_clipped_snr / (snr + 1))

        self.normalize = normalize_to_neg_one_to_one if auto_normalize else (lambda x: x)
        self.unnormalize = unnormalize_to_zero_to_one if auto_normalize else (lambda x: x)

    @property
    def device(self):
        return self.betas.device

    # ---- prediction transforms ----

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * x_start
        )

    def predict_start_from_v(self, x_t, t, v):
        return (
            extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self, x, t, x_self_cond=None, clip_x_start=False, rederive_pred_noise=False):
        model_output = self.model(x, t, x_self_cond)
        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else (lambda z: z)

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)
            if clip_x_start and rederive_pred_noise:
                pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_x0':
            x_start = maybe_clip(model_output)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        return ModelPrediction(pred_noise, x_start)
    # inside class GaussianDiffusion
    def _pick_t_from_sigma(self, sigma_d, k=1.0):
        device = sigma_d.device
        abar = self.alphas_cumprod.to(device)              # [T]
        sigma_t = torch.sqrt((1.0 - abar).clamp(min=0.0))  # [T]
        tgt = (k * sigma_d).view(-1).max()
        return int(torch.argmin((sigma_t - tgt).abs()).item())


    @torch.inference_mode()
    def ddim_step(self, img, time, time_next, x_self_cond=None):
        """
        Standard DDIM step (prior-only).
        We will add fidelity *after* this when doing inverse-problem sampling.
        """
        batch = img.shape[0]
        time_cond = torch.full((batch,), time, device=self.device, dtype=torch.long)
        preds = self.model_predictions(img, time_cond, x_self_cond, clip_x_start=True, rederive_pred_noise=True)
        x_start, pred_noise = preds.pred_x_start, preds.pred_noise
        if time_next < 0:
            return x_start
        alpha = self.alphas_cumprod[time]
        alpha_next = self.alphas_cumprod[time_next]
        sigma = self.ddim_sampling_eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
        c = (1 - alpha_next - sigma ** 2).sqrt()
        noise = torch.randn_like(img)
        img = x_start * alpha_next.sqrt() + c * pred_noise + sigma * noise
        return img

    # ---- q_sample for training (pure DDPM prior) ----

    @torch.no_grad()
    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
            extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    # ---- training loss: pure noise MSE (prior only) ----

    def p_losses(self, x_start, t, noise=None, offset_noise_strength=None):
        """
        Pure DDPM prior training:
            - Take a clean x_start ~ p_data
            - Sample t
            - Create x_t = q(x_t | x_0, t)
            - Predict noise and minimize ||eps_pred - eps||^2 (with min-SNR weighting)
        """
        b, c, h, w = x_start.shape
        device = x_start.device

        noise = default(noise, lambda: torch.randn_like(x_start))
        offset_noise_strength = default(offset_noise_strength, self.offset_noise_strength)

        if offset_noise_strength > 0.:
            offset_noise = torch.randn(x_start.shape[:2], device=device)
            noise = noise + offset_noise_strength * rearrange(offset_noise, 'b c -> b c 1 1')

        # forward diffusion
        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # optional self-conditioning
        x_self_cond = None
        if self.self_condition and random() < 0.5:
            with torch.no_grad():
                x_self_cond = self.model_predictions(x, t).pred_x_start
                x_self_cond.detach_()

        model_out = self.model(x, t, x_self_cond)

        if self.objective == 'pred_noise':
            target = noise
            pred = model_out
        elif self.objective == 'pred_x0':
            # not used in your new setup, but left for completeness
            target = x_start
            pred = model_out
        else:  # pred_v
            v = self.predict_v(x_start=x_start, t=t, noise=noise)
            target = v
            pred = model_out

        loss = F.mse_loss(pred, target, reduction='none')
        loss = reduce(loss, 'b ... -> b', 'mean')

        # min-SNR weighting
        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, img, *args, **kwargs):
        """
        Training entrypoint.

        - If img is a tensor: clean-only dataset, we treat img as x_start.
        - If img is (noisy, clean): we IGNORE noisy and use clean as x_start,
          because this is a pure prior model p(x), not a supervised denoiser.
        """
        if isinstance(img, (tuple, list)) and len(img) == 2:
            _, x_clean = img
            x_start = x_clean
        else:
            x_start = img

        b, c, h, w = x_start.shape
        device = x_start.device

        factor = self.model.downsample_factor
        assert divisible_by(h, factor) and divisible_by(w, factor), \
            f"input dims {(h, w)} must be divisible by {factor}"

        # time step across full range [0, T-1]
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        # normalize to [-1,1] and compute loss
        x_start = self.normalize(x_start)
        return self.p_losses(x_start, t, *args, **kwargs)

    # ---- inverse-problem style denoising (posterior sampling) ----

    @torch.no_grad()
    def one_shot_denoise(self, noisy_01, sigma_data=None, lambda_df=0.0):
        """
        Single-step posterior approximation:

        1) Treat noisy measurement y in [0,1], normalize to [-1,1].
        2) Pick a timestep t that roughly matches sigma_data (data noise).
        3) Use the prior to predict x0.
        4) Apply a PET-style data fidelity correction:

               x_hat <- x_hat + λ_df * (σ_t^2 / σ_data^2) * (y_norm - x_hat)

        If sigma_data is None, we just pick a mid-timestep.
        """
        x_noisy_norm = self.normalize(noisy_01.to(self.device))  # y
        B = x_noisy_norm.size(0)

        # choose timestep
        if sigma_data is None:
            t_idx = int(1.0 * (self.num_timesteps - 1))
        else:
            t_idx = _match_sigma_to_t(self.alphas_cumprod, float(sigma_data) ** 2)

        t = torch.full((B,), t_idx, device=self.device, dtype=torch.long)

        # prior prediction
        preds = self.model_predictions(x_noisy_norm, t, clip_x_start=True, rederive_pred_noise=True)
        x_hat = preds.pred_x_start  # prior-only denoised

        # data fidelity correction in normalized space
        if (lambda_df > 0.0) and (sigma_data is not None):
            alpha_bar_t = self.alphas_cumprod[t_idx]
            sigma_t2 = float(1.0 - alpha_bar_t.item())
            sigma_d2 = float(sigma_data ** 2)
            factor = lambda_df * (sigma_t2 / (sigma_d2 + 1e-8))
            x_hat = x_hat + factor * (x_noisy_norm - x_hat)

        return self.unnormalize(x_hat)

#    
    @torch.no_grad()
    def few_step_denoise(self, I_tilde_01, steps=20, eta=0.0, lambda_df=0.0, k_sigma=5):
        """
        DDIM inference on a single VST image batch with data fidelity at each step.

        I_tilde_01: [B,1,H,W] tensor in [0,1] (your saved VST images as png)
                    (If your VST images are not in [0,1], DO NOT use auto_normalize=True this way.)
        steps: number of DDIM steps
        eta: DDIM eta
        lambda_df: data fidelity strength (0 disables)
        k_sigma: kernel size for sigma_d estimation (high-pass MAD)

        Returns: denoised image in [0,1], same shape.
        """

        self.ddim_sampling_eta = eta

        # ---- 0) y = measurement in normalized space [-1,1] ----
        y = self.normalize(I_tilde_01.to(self.device))   # y_norm
        x = y.clone()
        t_start = int(0.6 * (self.num_timesteps - 1))

        # integer schedule from t_start -> 0 with `steps` hops
        times = torch.linspace(t_start, 0, steps + 1, device=self.device).long()

        # ---- 3) DDIM loop with fidelity each step ----
        for i in range(steps):
            t = int(times[i].item())
            t_next = int(times[i + 1].item())
            # prior DDIM step (x_t -> x_{t_next})
            x = self.ddim_step(x, t, t_next, x_self_cond=None)
            # fidelity
            if lambda_df > 0.0:
                # apply: x <- x +  (y - x)
                x = x - (lambda_df) * (x - y) #This is for Brainweb dataset

        return self.unnormalize(x)
   

# --------------------
# Datasets
# --------------------


class ImageFolderDataset(TorchDataset):
    """
    Simple image-folder dataset: loads all images from a directory (non-recursive).
    """
    def __init__(self, folder, image_size=None, convert_image_to='L', center_crop=True):
        super().__init__()
        self.folder = folder
        self.paths = []
        exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif','*.npy', '*.tiff')
        for e in exts:
            self.paths.extend(sorted(glob.glob(os.path.join(folder, e))))

        if len(self.paths) == 0:
            raise FileNotFoundError(f'No images found in {folder}')

        transforms = []
        if center_crop and image_size is not None:
            size = image_size if isinstance(image_size, int) else image_size[0]
            transforms.append(T.CenterCrop(size))
        transforms.append(T.ToTensor())
        self.transform = T.Compose(transforms)
        self.convert_image_to = convert_image_to

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path)
        img = convert_image_to_fn(self.convert_image_to, img)
        tensor = self.transform(img)  # [C,H,W] in [0,1]
        return tensor


class PairedMRIDataset(TorchDataset):
    """
    Paired dataset: (noisy_folder, clean_folder) with same filenames.
    """
    def __init__(self, noisy_folder, clean_folder, image_size=None, convert_image_to='L'):
        super().__init__()
        self.noisy_folder = noisy_folder
        self.clean_folder = clean_folder
        self.paths = []

        exts = ('*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff')
        noisy_paths = []
        for e in exts:
            noisy_paths.extend(sorted(glob.glob(os.path.join(noisy_folder, e))))
        if len(noisy_paths) == 0:
            raise FileNotFoundError(f'No images found in noisy folder: {noisy_folder}')

        self.paths = noisy_paths
        transforms = []
        if image_size is not None:
            size = image_size if isinstance(image_size, int) else image_size[0]
            transforms.append(T.CenterCrop(size))
        transforms.append(T.ToTensor())
        self.transform = T.Compose(transforms)
        self.convert_image_to = convert_image_to

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        noisy_path = self.paths[idx]
        fname = os.path.basename(noisy_path)
        clean_path = os.path.join(self.clean_folder, fname)
        if not os.path.exists(clean_path):
            raise FileNotFoundError(f'Missing clean file for {fname}')

        noisy_img = Image.open(noisy_path)
        noisy_img = convert_image_to_fn(self.convert_image_to, noisy_img)
        noisy_tensor = self.transform(noisy_img)

        clean_img = Image.open(clean_path)
        clean_img = convert_image_to_fn(self.convert_image_to, clean_img)
        clean_tensor = self.transform(clean_img)

        return noisy_tensor, clean_tensor


# --------------------
# Trainer (pure prior DDPM)
# --------------------


class Trainer:
    def __init__(
        self,
        diffusion_model: GaussianDiffusion,
        folder,                         # str for clean-only OR (noisy_folder, clean_folder) for pairs
        *,
        train_batch_size=8,
        gradient_accumulate_every=1,
        train_lr=2e-4,
        train_num_steps=100000,
        ema_update_every=10,
        ema_decay=0.995,
        save_and_sample_every=1000,
        num_samples=16,
        results_folder='./results_mri_denoise',
        amp=False,
        convert_image_to='L',
        center_crop=True,
    ):
        super().__init__()
        self.model = diffusion_model
        self.channels = diffusion_model.channels
        self.image_size = diffusion_model.image_size
        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps
        self.max_grad_norm = 1.0

        # Accelerator or fallback
        if HAS_ACCELERATE:
            self.accelerator = Accelerator(mixed_precision='fp16' if amp else 'no')
            self.device = self.accelerator.device
        else:
            class _Dummy:
                def __init__(self):
                    self.is_main_process = True

                def prepare(self, *objs):
                    return objs

                def backward(self, loss):
                    loss.backward()

                @property
                def device(self):
                    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            self.accelerator = _Dummy()
            self.device = self.accelerator.device

        # Dataset: clean-only or paired; in paired case we use ONLY clean for prior
        if isinstance(folder, (tuple, list)) and len(folder) == 2:
            noisy_folder, clean_folder = folder
            ds = PairedMRIDataset(noisy_folder, clean_folder, self.image_size, convert_image_to=convert_image_to)
        else:
            ds = ImageFolderDataset(folder, self.image_size, convert_image_to=convert_image_to, center_crop=center_crop)

        assert len(ds) >= 20, 'Need at least 20 images to train something meaningful.'

        dl = DataLoader(ds, batch_size=train_batch_size, shuffle=True,
                        pin_memory=True, num_workers=min(8, cpu_count()))
        self.model, dl = self.accelerator.prepare(self.model, dl)
        self.dl = self._cycle(dl)

        self.opt = torch.optim.Adam(self.model.parameters(), lr=train_lr, betas=(0.9, 0.99))
        self.model, self.opt, dl = self.accelerator.prepare(self.model, self.opt, dl)
        self.dl = self._cycle(dl)

        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device) \
            if HAS_EMA and getattr(self.accelerator, 'is_main_process', True) else None

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every
        self.step = 0
        self.best_loss = float('inf')

    def _cycle(self, dl):
        while True:
            for data in dl:
                yield data

    def save(self, tag='latest'):
        if not getattr(self.accelerator, 'is_main_process', True):
            return
        data = {
            'step': self.step,
            'model': self.accelerator.get_state_dict(self.model),
            'opt': self.opt.state_dict(),
        }
        if self.ema is not None:
            data['ema'] = self.ema.state_dict()
        torch.save(data, str(self.results_folder / f'model-{tag}.pt'))

    def load(self, path_or_tag='latest'):
        device = self.device
        path = path_or_tag if Path(path_or_tag).exists() else (self.results_folder / f'model-{path_or_tag}.pt')
        data = torch.load(str(path), map_location=device, weights_only=False)
        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])
        self.step = data.get('step', 0)
        self.opt.load_state_dict(data['opt'])
        if self.ema is not None and 'ema' in data:
            self.ema.load_state_dict(data['ema'])

    def train(self):
        with tqdm(initial=self.step, total=self.train_num_steps,
                  disable=not getattr(self.accelerator, 'is_main_process', True)) as pbar:
            while self.step < self.train_num_steps:
                self.model.train()
                total_loss = 0.0

                for _ in range(self.gradient_accumulate_every):
                    batch = next(self.dl)

                    if isinstance(batch, (tuple, list)) and len(batch) == 2:
                        noisy, clean = batch
                        clean = clean.to(self.device)
                    else:
                        clean = batch.to(self.device)

                    # pad to multiple of 8
                    clean, _ = pad_to_multiple(clean, multiple=8)

                    # pure prior loss
                    loss = self.model(clean)
                    loss = loss / self.gradient_accumulate_every
                    total_loss += loss.item()
                    self.accelerator.backward(loss)

                # optimizer update after gradient accumulation
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.opt.step()
                self.opt.zero_grad()

                self.step += 1
                if self.ema is not None:
                    self.ema.update()

                # ---- compute avg loss for this step ----
                avg_loss = total_loss / self.gradient_accumulate_every

                # ---- update best model only (no images) ----
                if avg_loss < self.best_loss and getattr(self.accelerator, 'is_main_process', True):
                    self.best_loss = avg_loss
                    self.save('best')

                pbar.set_description(f'loss: {avg_loss:.4f}')
                pbar.update(1)

        print('Training complete.')
        # optional: also save final model as "latest"
        if getattr(self.accelerator, 'is_main_process', True):
            self.save('latest')

# --------------------
# Simple CLI / Example
# --------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MRI Denoiser (Prior DDPM + Inference Fidelity)")
    parser.add_argument("--mode", type=str, default="train", choices=["train", "denoise_one", "denoise_few"])
    parser.add_argument("--image_size", type=int, default=256)

    # training data
    parser.add_argument("--train_clean", type=str, help="Folder of clean MRIs (for prior training)")
    parser.add_argument("--train_noisy", type=str, help="(Optional) Folder of noisy MRIs; if given with clean, only clean is used for prior.")
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--train_steps", type=int, default=10000)
    parser.add_argument("--save_dir", type=str, default="./results_mri_denoise")

    # diffusion config
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--sampling_timesteps", type=int, default=50)

    # inference config
    parser.add_argument("--weights", type=str, default="", help="Path to weights for denoising mode")
    parser.add_argument("--noisy_image", type=str, default="", help="Path to a single noisy MRI for inference")
    parser.add_argument("--out_image", type=str, default="./denoised.png")
    parser.add_argument("--few_steps", type=int, default=20)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--noisy_folder", type=str, help="Folder containing noisy MRI images for inference")
    parser.add_argument("--out_folder", type=str, help="Folder to save denoised output images")

    # posterior / fidelity parameters
    parser.add_argument("--lambda_df", type=float, default=0.0,
                        help="Data fidelity strength in inference (0 = prior-only, >0 = posterior-ish)")
    parser.add_argument("--sigma_data", type=float, default=None,
                        help="Estimated measurement noise std (in [0,1] scale). If None, use mid-level t.")

    # misc
    parser.add_argument("--resume", action="store_true", help="Resume training from model-latest.pt in save_dir if it exists")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    net = Unet(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        learned_sinusoidal_cond=False
    )

    img_size_arg = (args.image_size, args.image_size) if (args.image_size and args.image_size > 0) else None
    diff = GaussianDiffusion(
        model=net,
        image_size=img_size_arg,
        timesteps=args.timesteps,
        sampling_timesteps=args.sampling_timesteps,
        objective='pred_noise',           # pure noise objective
        beta_schedule='sigmoid',
        auto_normalize=True
    ).to(device)

    if args.mode == "train":
        if args.train_noisy and args.train_clean:
            folder = (args.train_noisy, args.train_clean)
        elif args.train_clean:
            folder = args.train_clean
        else:
            raise ValueError("Provide --train_clean (for prior) or both --train_noisy and --train_clean (paired; clean used for prior).")

        trainer = Trainer(
            diffusion_model=diff,
            folder=folder,
            train_batch_size=args.train_batch_size,
            train_lr=2e-4,
            train_num_steps=args.train_steps,
            results_folder=args.save_dir,
        )
        if args.resume:
            trainer.load("best")
            print(f"Resuming training from step {trainer.step}")

        trainer.train()

    else:
        assert args.weights and (args.noisy_image or args.noisy_folder), \
            "Provide --weights and either --noisy_image (with --out_image) or --noisy_folder (with --out_folder)"

        # ---- load weights (your checkpoints saved by Trainer.save use accelerator state dict) ----
        ckpt = torch.load(args.weights, map_location=device, weights_only=False)
        state = ckpt.get('model', ckpt)  # support either raw state_dict or trainer checkpoint dict
        missing = diff.load_state_dict(state, strict=False)
        print("Loaded with strict=False. Missing / unexpected keys:", missing)
        diff.eval()

        to_tensor = T.ToTensor()

        def _denoise_tensor(I_tilde_01):
            """
            I_tilde_01: [1,1,H,W] in [0,1]
            returns: [1,1,H,W] in [0,1]
            """
            with torch.no_grad():
                if args.mode == "denoise_one":
                    # keep your one_shot if you want, but it needs sigma_data; here we just run few-step
                    return diff.few_step_denoise(
                        I_tilde_01,
                        steps=1,
                        eta=args.eta,
                        lambda_df=args.lambda_df,
                        k_sigma=5
                    )
                else:
                    return diff.few_step_denoise(
                        I_tilde_01,
                        steps=args.few_steps,
                        eta=args.eta,
                        lambda_df=args.lambda_df,
                        k_sigma=5
                    )   

        def _load_to_tensor(path):
            path = str(path)
        
            if path.lower().endswith(".npy"):
                arr = np.load(path)  # could be (H,W) or (1,H,W)
                if arr.ndim == 2:
                    arr = arr[None, ...]  # (1,H,W)
                elif arr.ndim == 3 and arr.shape[0] != 1:
                    # if it's (H,W,1) or something else, adjust if needed
                    # simplest: take first channel
                    arr = arr[:1, ...]
                arr = arr.astype(np.float32)
        
                # ensure values are in [0,1]
                mn, mx = float(arr.min()), float(arr.max())
                if mx > 1.0 or mn < 0.0:
                    arr = (arr - mn) / (mx - mn + 1e-8)
        
                x = torch.from_numpy(arr).unsqueeze(0).to(device)  # (1,1,H,W)
        
            else:
                img = Image.open(path).convert("L")
                x = to_tensor(img).unsqueeze(0).to(device)  # (1,1,H,W) in [0,1]
        
            # pad to multiple of 8
            x_padded, pad = pad_to_multiple(x, multiple=8)
        
            # denoise
            x_denoised_padded = _denoise_tensor(x_padded).clamp(0, 1).cpu()
        
            # unpad
            x_denoised = unpad_to_original(x_denoised_padded, pad)
            return x_denoised

        if args.noisy_folder:
            assert args.out_folder, "When using --noisy_folder, also provide --out_folder"
            Path(args.out_folder).mkdir(parents=True, exist_ok=True)

            exts = ('*.png', '*.jpg', '*.jpeg', '*.tif', '*.bmp','*.npy')
            img_paths = []
            for e in exts:
                img_paths.extend(sorted(glob.glob(os.path.join(args.noisy_folder, e))))
            if not img_paths:
                raise FileNotFoundError(f"No images found in folder: {args.noisy_folder}")


        for p in img_paths:
            x_denoised = _load_to_tensor(p)
            base = os.path.splitext(os.path.basename(p))[0]
            out_path = os.path.join(args.out_folder, base + ".png")
            utils.save_image(x_denoised, out_path, nrow=1)
            print(f"Saved: {out_path}")


        else:
            assert args.noisy_image and args.out_image, \
                "When using --noisy_image, also provide --out_image"
            x_denoised = _load_to_tensor(args.noisy_image)
            utils.save_image(x_denoised, args.out_image, nrow=1)
            print(f"Saved denoised image to: {args.out_image}")

