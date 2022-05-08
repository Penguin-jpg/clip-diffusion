from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
)
from .config import timestep_respacing, use_checkpoint, steps


def load_model_and_diffusion():
    model_config = model_and_diffusion_defaults()
    model_config.update(
        {
            "attention_resolutions": "32, 16, 8",
            "class_cond": False,
            "diffusion_steps": (1000 // steps) * steps if steps < 1000 else steps,
            "rescale_timesteps": True,
            "timestep_respacing": timestep_respacing,
            "image_size": 512,
            "learn_sigma": True,
            "noise_schedule": "linear",
            "num_channels": 256,
            "num_head_channels": 64,
            "num_res_blocks": 2,
            "resblock_updown": True,
            "use_checkpoint": use_checkpoint,
            "use_fp16": True,
            "use_scale_shift_norm": True,
        }
    )

    model, diffusion = create_model_and_diffusion(**model_config)
    return model, diffusion
