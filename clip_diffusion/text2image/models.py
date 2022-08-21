import torch
import math
import os
from urllib import request
from pathlib import Path
from tqdm import tqdm
from torch import nn
from dataclasses import dataclass
from functools import partial
from clip_diffusion.utils.dir_utils import MODEL_PATH
from clip_diffusion.utils.functional import clear_gpu_cache

# 下載網址
_GUIDED_DIFFUSION_MODEL_URL = "https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt"
_LATENT_DIFFUSION_MODEL_URL = "https://huggingface.co/multimodalart/compvis-latent-diffusion-text2img-large/resolve/main/txt2img-f8-large-jack000-finetuned-fp16.ckpt"
_REAL_ESRGAN_X4_MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
_REAL_ESRGAN_X2_MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
_AESTHETIC_PREDICTOR_URLS = {
    "ViT-B/32": "https://github.com/crowsonkb/simulacra-aesthetic-models/raw/master/models/sac_public_2022_06_29_vit_b_32_linear.pth",
    "ViT-B/16": "https://github.com/crowsonkb/simulacra-aesthetic-models/raw/master/models/sac_public_2022_06_29_vit_b_16_linear.pth",
    "ViT-L/14": "https://github.com/christophschuhmann/improved-aesthetic-predictor/raw/main/sac%2Blogos%2Bava1-l14-linearMSE.pth",
}

# 模型名稱
_GUIDED_DIFFUSION_MODEL_NAME = "512x512_diffusion_uncond_finetune_008100.pt"
_LATENT_DIFFUSION_MODEL_NAME = "txt2img-f8-large-jack000-finetuned-fp16.ckpt"
_REAL_ESRGAN_X4_MODEL_NAME = "RealESRGAN_x4plus.pth"
_REAL_ESRGAN_X2_MODEL_NAME = "RealESRGAN_x2plus.pth"
_AESTHETIC_PREDICTOR_NAMES = {
    "ViT-B/32": "sac_public_2022_06_29_vit_b_32_linear.pth",
    "ViT-B/16": "sac_public_2022_06_29_vit_b_16_linear.pth",
    "ViT-L/14": "sac+logos+ava1-l14-linearMSE.pth",
}

# Clip與對應維度
_CLIP_DIMS = {
    "ViT-B/32": 512,
    "ViT-B/16": 512,
    "ViT-L/14": 768,
}


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

    opener = request.build_opener()
    opener.addheaders = [("User-Agent", "Mozilla/5.0")]

    with opener.open(url) as source, open(download_target_tmp, "wb") as output:
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


def load_clip_models(chosen_models, device=None):
    """
    選擇並載入要使用的Clip模型
    """

    import clip

    models = {}

    for model_name in chosen_models:
        model, _ = clip.load(model_name, device=device)
        _to_eval_and_freeze_layers(model, False, device)
        models[model_name] = model

    clear_gpu_cache()

    return models


def load_guided_diffusion_model(custom_model_path=None, steps=200, use_checkpoint=True, use_fp16=True, device=None):
    """
    載入guided diffusion model和diffusion
    """

    from guided_diffusion.script_util import (
        model_and_diffusion_defaults,
        create_model_and_diffusion,
    )

    model_config = model_and_diffusion_defaults()

    if not custom_model_path:
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
    else:
        assert custom_model_path is not None, "need to specify custom_model_path"
        # 由於自己訓練的模型是調整過參數的，所以要額外處理
        model_config.update(
            {
                "attention_resolutions": "16",
                "class_cond": False,
                "diffusion_steps": ((1000 // steps) * steps if steps < 1000 else steps),
                "rescale_timesteps": True,
                "timestep_respacing": f"ddim{steps}",
                "image_size": 256,
                "learn_sigma": True,
                "noise_schedule": "linear",
                "num_channels": 128,
                "num_heads": 1,
                "num_res_blocks": 2,
                "use_checkpoint": use_checkpoint,
                "use_fp16": use_fp16,
                "use_scale_shift_norm": False,
            }
        )
        model, diffusion = create_model_and_diffusion(**model_config)
        model.load_state_dict(torch.load(custom_model_path, map_location="cpu"))

    _to_eval_and_freeze_layers(model, False, device)

    for name, param in model.named_parameters():
        if "qkv" in name or "norm" in name or "proj" in name:
            param.requires_grad_()

    if use_fp16:
        model.convert_to_fp16()

    clear_gpu_cache()

    return model, diffusion


def load_latent_diffusion_model(device=None):
    """
    載入latent diffusion模型
    """

    from ldm.util import instantiate_from_config
    from omegaconf import OmegaConf

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


def load_real_esrgan_upsampler(scale=4, device=None):
    """
    載入real-esrgan模型
    """

    assert scale in (2, 4), "scale can only be 2 or 4"

    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer

    if scale == 2:
        model_path = _download_model(_REAL_ESRGAN_X2_MODEL_URL, _REAL_ESRGAN_X2_MODEL_NAME)
    else:
        model_path = _download_model(_REAL_ESRGAN_X4_MODEL_URL, _REAL_ESRGAN_X4_MODEL_NAME)

    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=scale)
    upsampler = RealESRGANer(
        scale=4,
        model_path=model_path,
        model=model,
        half=True,
        device=device,
    )
    return upsampler


def load_sentence_transformer(model_name, device=None):
    """
    載入指定的sentence transformer
    """

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    _to_eval_and_freeze_layers(model, False, device)
    clear_gpu_cache()
    return model


class LinearAestheticPredictor(nn.Module):
    """
    在Clip之上加一層linear
    """

    def __init__(self, input_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, input):
        return self.linear(input)


# 修改自：https://github.com/christophschuhmann/improved-aesthetic-predictor
class MLPAestheticPredictor(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 1024),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            # nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            # nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            # nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, input):
        return self.layers(input)


def load_aesthetic_predictors(predictor_names, device=None):
    """
    載入指定的aesthetic predictor
    """

    predictors = {}

    for predictor_name in predictor_names:
        input_dim = _CLIP_DIMS[predictor_name]
        if input_dim == 768:
            model = MLPAestheticPredictor(input_dim)
        else:
            model = LinearAestheticPredictor(input_dim)

        model.load_state_dict(
            torch.load(
                _download_model(_AESTHETIC_PREDICTOR_URLS[predictor_name], _AESTHETIC_PREDICTOR_NAMES[predictor_name]),
                map_location="cpu",
            )
        )
        _to_eval_and_freeze_layers(model, False, device)
        predictors[predictor_name] = model
        clear_gpu_cache()

    return predictors
