import torch
import clip
import math
import gc
from torch import nn
from dataclasses import dataclass
from functools import partial
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from bsrgan.models import RRDBNet
from clip_diffusion.config import config
from clip_diffusion.download_utils import (
    DIFFUSION_MODEL_URL,
    SECONDARY_MODEL_URL,
    LATENT_DIFFUSION_MODEL_REPO,
    BSRGAN_MODEL_URL,
    download,
)


def load_clip_models_and_preprocessings(chosen_models):
    """
    選擇並載入要使用的Clip模型和preprocess function
    """

    clip_models = {}
    preprocessings = {}

    for model_name in chosen_models:
        model, preprocess = clip.load(model_name, config.device)
        clip_models[model_name] = model.eval().requires_grad_(False)
        preprocessings[model_name] = preprocess

    gc.collect()
    torch.cuda.empty_cache()

    return clip_models, preprocessings


def load_guided_diffusion_model(steps=200, use_checkpoint=True):
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
            ),  # 如果steps小於1000，就將diffusion_steps補正到接近1000
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
            "use_fp16": True,
            "use_scale_shift_norm": True,
        }
    )
    model, diffusion = create_model_and_diffusion(**model_config)
    model.load_state_dict(
        torch.load(
            download(DIFFUSION_MODEL_URL, config.diffusion_model_name),
            map_location="cpu",
        )
    )
    model.eval().requires_grad_(False).to(config.device)

    for name, param in model.named_parameters():
        if "qkv" in name or "norm" in name or "proj" in name:
            param.requires_grad_()

    if model_config["use_fp16"]:
        model.convert_to_fp16()

    gc.collect()
    torch.cuda.empty_cache()

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
                            nn.Upsample(
                                scale_factor=2, mode="bilinear", align_corners=False
                            ),
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


def load_secondary_model():
    """
    載入secondary model
    """

    model = SecondaryDiffusionImageNet2()
    model.load_state_dict(
        torch.load(
            download(SECONDARY_MODEL_URL, config.secondary_model_name),
            map_location="cpu",
        )
    )
    model.eval().requires_grad_(False).to(config.device)

    gc.collect()
    torch.cuda.empty_cache()

    return model


# 參考並修改自：https://huggingface.co/spaces/multimodalart/latentdiffusion/blob/main/app.py
def load_latent_diffusion_model():
    """
    載入latent diffusion模型
    """

    model_config = OmegaConf.load(
        "./latent-diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml"
    )

    model = instantiate_from_config(model_config.model)
    model.load_state_dict(
        torch.load(
            download(
                "",
                config.latent_diffusion_model_name,
                download_from_huggingface=True,
                repo=LATENT_DIFFUSION_MODEL_REPO,
            ),
            map_location="cpu",
        ),
        strict=False,
    )
    model.half().eval().requires_grad_(False).to(config.device)

    gc.collect()
    torch.cuda.empty_cache()

    return model


def load_bsrgan_model():
    """
    載入bsrgan模型
    """

    model = RRDBNet(in_nc=3, out_nc=3, nf=64, nb=23, gc=32, sf=4)
    model.load_state_dict(
        torch.load(
            download(BSRGAN_MODEL_URL, config.bsrgan_model_name), map_location="cpu"
        ),
        strict=True,
    )
    model.eval().requires_grad_(False).to(config.device)

    gc.collect()
    torch.cuda.empty_cache()

    return model
