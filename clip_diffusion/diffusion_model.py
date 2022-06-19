import torch
import gc
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from clip_diffusion.config import config
from clip_diffusion.download_utils import DIFFUSION_MODEL_URL, download


def load_model_and_diffusion():
    """
    載入diffusion和model
    """

    model_config = model_and_diffusion_defaults()
    model_config.update(
        {
            "attention_resolutions": "32, 16, 8",
            "class_cond": False,
            "diffusion_steps": config.diffusion_steps,
            "rescale_timesteps": True,
            "timestep_respacing": config.timestep_respacing,
            "image_size": 512,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 256,
            "num_head_channels": 64,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "use_checkpoint": config.use_checkpoint,
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
