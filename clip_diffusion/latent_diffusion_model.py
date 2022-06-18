import torch
from ldm.util import instantiate_from_config
from omegaconf import OmegaConf
from .config import config
from .download_utils import LATENT_DIFFUSION_MODEL_URL, download

# 參考並修改自：https://huggingface.co/spaces/multimodalart/latentdiffusion/blob/main/app.py


def load_latent_diffusion_model():
    """
    載入latent diffusion模型
    """

    model_config = OmegaConf.load(
        "./latent-diffusion/configs/latent-diffusion/txt2img-1p4B-eval.yaml"
    )
    checkpoint = torch.load(
        download(LATENT_DIFFUSION_MODEL_URL, config.latent_diffusion_model_name),
        map_location="cpu",
    )
    model = instantiate_from_config(model_config.model)
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    # 轉fp16
    model.half().to(config.device)
    model.eval()

    return model


latent_diffusion_model = load_latent_diffusion_model()
