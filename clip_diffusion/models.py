import torch
import clip
import math
import os
import gc
from urllib import request
from pathlib import Path
from tqdm import tqdm
from torch import nn
from dataclasses import dataclass
from functools import partial
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from clip_diffusion.utils.dir_utils import MODEL_PATH
from clip_diffusion.utils.functional import clear_gpu_cache

# 下載網址
_GUIDED_DIFFUSION_MODEL_URL = "https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt"
_SECONDARY_MODEL_URL = "https://the-eye.eu/public/AI/models/v-diffusion/secondary_model_imagenet_2.pth"
_LATENT_DIFFUSION_MODEL_URL = "https://huggingface.co/multimodalart/compvis-latent-diffusion-text2img-large/resolve/main/txt2img-f8-large-jack000-finetuned-fp16.ckpt"
_REAL_ESRGAN_MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"

# 模型名稱
_GUIDED_DIFFUSION_MODEL_NAME = "512x512_diffusion_uncond_finetune_008100.pt"
_SECONDARY_MODEL_NAME = "secondary_model_imagenet_2.pth"
_LATENT_DIFFUSION_MODEL_NAME = "txt2img-f8-large-jack000-finetuned-fp16.ckpt"
_REAL_ESRGAN_MODEL_NAME = "RealESRGAN_x4plus.pth"


# 參考並修改自：https://github.com/lucidrains/DALLE-pytorch/blob/d355100061911b13e1f1c22de8c2b5deb44e65f8/dalle_pytorch/vae.py
def _download_model(url, model_name):
    """
    下載模型並儲存，回傳儲存位置
    """

    download_target = Path(os.path.join(MODEL_PATH, model_name))
    download_target_tmp = download_target.with_suffix(".tmp")

    if os.path.exists(download_target):
        if not os.path.isfile(download_target):
            raise RuntimeError(f"{download_target} exists and is not a regular file")
        else:
            return str(download_target)

    with request.urlopen(url) as source, open(download_target_tmp, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80) as loop:
            while True:
                buffer = source.read(4096)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))

    os.rename(download_target_tmp, download_target)
    return str(download_target)


def _to_eval_and_freeze_layers(model, half=False, device=None):
    """
    將model換成eval模式並凍結所有layer
    """

    if half:
        model.half()

    model.eval().requires_grad_(False).to(device)


def load_clip_models_and_preprocessings(chosen_models, device=None):
    """
    選擇並載入要使用的Clip模型和preprocess function
    """

    models = []
    preprocessings = []

    for model_name in chosen_models:
        model, preprocess = clip.load(model_name, device=device)
        _to_eval_and_freeze_layers(model, False, device)
        models.append(model)
        preprocessings.append(preprocess)

    clear_gpu_cache()

    return models, preprocessings


def load_guided_diffusion_model(steps=200, use_checkpoint=True, use_fp16=True, device=None):
    """
    載入guided diffusion model和diffusion
    """

    model_config = model_and_diffusion_defaults()
    model_config.update(
        {
            "attention_resolutions": "32, 16, 8",
            "class_cond": False,
            "diffusion_steps": (
                (1000 // steps) * steps if steps < 1000 else steps
            ),  # 如果steps小於1000，就將diffusion_steps補正到接近1000以配合cutout
            "rescale_timesteps": True,
            "timestep_respacing": f"ddim{steps}",  # 調整diffusion的timestep數量(使用DDIM sample)
            "image_size": 512,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 256,
            "num_head_channels": 64,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "use_checkpoint": use_checkpoint,  # 使用gradient checkpoint
            "use_fp16": use_fp16,
            "use_scale_shift_norm": True,
        }
    )
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(
        torch.load(
            _download_model(_GUIDED_DIFFUSION_MODEL_URL, _GUIDED_DIFFUSION_MODEL_NAME),
            map_location="cpu",
        )
    )
    _to_eval_and_freeze_layers(model, False, device)

    for name, param in model.named_parameters():
        if "qkv" in name or "norm" in name or "proj" in name:
            param.requires_grad_()

    if use_fp16:
        model.convert_to_fp16()

    clear_gpu_cache()

    return model, diffusion


# 作者：Katherine Crowson(https://github.com/crowsonkb)
def append_dims(x, n):
    return x[(Ellipsis, *(None,) * (n - x.ndim))]


def expand_to_planes(x, shape):
    return append_dims(x, len(shape)).repeat([1, 1, *shape[2:]])


def alpha_sigma_to_t(alpha, sigma):
    return torch.atan2(sigma, alpha) * 2 / math.pi


def t_to_alpha_sigma(t):
    return torch.cos(t * math.pi / 2), torch.sin(t * math.pi / 2)


@dataclass
class DiffusionOutput:
    v: torch.Tensor
    pred: torch.Tensor
    eps: torch.Tensor


class ConvBlock(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.ReLU(inplace=True),
        )


class SkipBlock(nn.Module):
    def __init__(self, main, skip=None):
        super().__init__()
        self.main = nn.Sequential(*main)
        self.skip = skip if skip else nn.Identity()

    def forward(self, input):
        return torch.cat([self.main(input), self.skip(input)], dim=1)


class FourierFeatures(nn.Module):
    def __init__(self, in_features, out_features, std=1.0):
        super().__init__()
        assert out_features % 2 == 0
        self.weight = nn.Parameter(torch.randn([out_features // 2, in_features]) * std)

    def forward(self, input):
        f = 2 * math.pi * input @ self.weight.T
        return torch.cat([f.cos(), f.sin()], dim=-1)


class SecondaryDiffusionImageNet(nn.Module):
    """
    用來為Clip清除中繼的diffusion image
    """

    def __init__(self):
        super().__init__()
        c = 64  # The base channel count

        self.timestep_embed = FourierFeatures(1, 16)

        self.net = nn.Sequential(
            ConvBlock(3 + 16, c),
            ConvBlock(c, c),
            SkipBlock(
                [
                    nn.AvgPool2d(2),
                    ConvBlock(c, c * 2),
                    ConvBlock(c * 2, c * 2),
                    SkipBlock(
                        [
                            nn.AvgPool2d(2),
                            ConvBlock(c * 2, c * 4),
                            ConvBlock(c * 4, c * 4),
                            SkipBlock(
                                [
                                    nn.AvgPool2d(2),
                                    ConvBlock(c * 4, c * 8),
                                    ConvBlock(c * 8, c * 4),
                                    nn.Upsample(
                                        scale_factor=2,
                                        mode="bilinear",
                                        align_corners=False,
                                    ),
                                ]
                            ),
                            ConvBlock(c * 8, c * 4),
                            ConvBlock(c * 4, c * 2),
                            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                        ]
                    ),
                    ConvBlock(c * 4, c * 2),
                    ConvBlock(c * 2, c),
                    nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                ]
            ),
            ConvBlock(c * 2, c),
            nn.Conv2d(c, 3, 3, padding=1),
        )

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        v = self.net(torch.cat([input, timestep_embed], dim=1))
        alphas, sigmas = map(partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)


class SecondaryDiffusionImageNet2(nn.Module):
    def __init__(self):
        super().__init__()
        c = 64  # The base channel count
        cs = [c, c * 2, c * 2, c * 4, c * 4, c * 8]

        self.timestep_embed = FourierFeatures(1, 16)
        self.down = nn.AvgPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

        self.net = nn.Sequential(
            ConvBlock(3 + 16, cs[0]),
            ConvBlock(cs[0], cs[0]),
            SkipBlock(
                [
                    self.down,
                    ConvBlock(cs[0], cs[1]),
                    ConvBlock(cs[1], cs[1]),
                    SkipBlock(
                        [
                            self.down,
                            ConvBlock(cs[1], cs[2]),
                            ConvBlock(cs[2], cs[2]),
                            SkipBlock(
                                [
                                    self.down,
                                    ConvBlock(cs[2], cs[3]),
                                    ConvBlock(cs[3], cs[3]),
                                    SkipBlock(
                                        [
                                            self.down,
                                            ConvBlock(cs[3], cs[4]),
                                            ConvBlock(cs[4], cs[4]),
                                            SkipBlock(
                                                [
                                                    self.down,
                                                    ConvBlock(cs[4], cs[5]),
                                                    ConvBlock(cs[5], cs[5]),
                                                    ConvBlock(cs[5], cs[5]),
                                                    ConvBlock(cs[5], cs[4]),
                                                    self.up,
                                                ]
                                            ),
                                            ConvBlock(cs[4] * 2, cs[4]),
                                            ConvBlock(cs[4], cs[3]),
                                            self.up,
                                        ]
                                    ),
                                    ConvBlock(cs[3] * 2, cs[3]),
                                    ConvBlock(cs[3], cs[2]),
                                    self.up,
                                ]
                            ),
                            ConvBlock(cs[2] * 2, cs[2]),
                            ConvBlock(cs[2], cs[1]),
                            self.up,
                        ]
                    ),
                    ConvBlock(cs[1] * 2, cs[1]),
                    ConvBlock(cs[1], cs[0]),
                    self.up,
                ]
            ),
            ConvBlock(cs[0] * 2, cs[0]),
            nn.Conv2d(cs[0], 3, 3, padding=1),
        )

    def forward(self, input, t):
        timestep_embed = expand_to_planes(self.timestep_embed(t[:, None]), input.shape)
        v = self.net(torch.cat([input, timestep_embed], dim=1))
        alphas, sigmas = map(partial(append_dims, n=v.ndim), t_to_alpha_sigma(t))
        pred = input * alphas - v * sigmas
        eps = input * sigmas + v * alphas
        return DiffusionOutput(v, pred, eps)


def load_secondary_model(device=None):
    """
    載入secondary model
    """

    model = SecondaryDiffusionImageNet2()
    model.load_state_dict(
        torch.load(
            _download_model(_SECONDARY_MODEL_URL, _SECONDARY_MODEL_NAME),
            map_location="cpu",
        )
    )
    _to_eval_and_freeze_layers(model, False, device)

    clear_gpu_cache()

    return model


# 參考並修改自：https://huggingface.co/spaces/multimodalart/latentdiffusion/blob/main/app.py
def load_latent_diffusion_model(device=None):
    """
    載入latent diffusion模型
    """

    model_config = OmegaConf.load("./latent-diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml")

    model = instantiate_from_config(model_config.model)
    model.load_state_dict(
        torch.load(
            _download_model(_LATENT_DIFFUSION_MODEL_URL, _LATENT_DIFFUSION_MODEL_NAME),
            map_location="cpu",
        ),
        strict=False,
    )
    _to_eval_and_freeze_layers(model, True, device)

    clear_gpu_cache()

    return model


def load_real_esrgan_upsampler(device=None):
    """
    載入real-esrgan模型
    """

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path=_download_model(_REAL_ESRGAN_MODEL_URL, _REAL_ESRGAN_MODEL_NAME),
        model=model,
        half=True,
        device=device,
    )
    return upsampler
