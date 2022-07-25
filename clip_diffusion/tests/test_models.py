import torch
from torch import nn
from realesrgan.utils import RealESRGANer
from clip_diffusion.text2image.config import config
from clip_diffusion.text2image.models import (
    load_clip_models_and_preprocessings,
    load_guided_diffusion_model,
    load_secondary_model,
    load_latent_diffusion_model,
    load_real_esrgan_upsampler,
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def test_clip_models():
    """
    測試clip_models是否正常載入
    """

    models, _ = load_clip_models_and_preprocessings(config.chosen_clip_models, device)
    for model in models:
        assert isinstance(model, nn.Module), "failed to loading clip model"


def test_guided_diffusion_model():
    """
    測試guided_diffusion_model是否正常載入
    """

    model, _ = load_guided_diffusion_model(steps=200, device=device)
    assert isinstance(model, nn.Module), "failed to loading guided diffusion model"


def test_secondary_model():
    """
    測試secondary_model是否正常載入
    """

    model = load_secondary_model(device)
    assert isinstance(model, nn.Module), "failed to loading secondary model"


def test_latent_diffusion_model():
    """
    測試latent_diffusion_model是否正常載入
    """

    model = load_latent_diffusion_model(device)
    assert isinstance(model, nn.Module), "failed to loading latent diffusion model"


def test_real_esrgan_upsampler():
    """
    測試real_esrgan_upsampler是否正常載入
    """

    upsampler = load_real_esrgan_upsampler(device)
    assert isinstance(upsampler, RealESRGANer), "failed to loading real-esrgan upsampler"
