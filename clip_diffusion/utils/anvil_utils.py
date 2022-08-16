import anvil.server
import os
from clip_diffusion.text2image.config import Config
from clip_diffusion.utils.functional import random_seed
from clip_diffusion.text2image.prompt import Prompt
from clip_diffusion.utils.image_utils import image_to_blob_media
from clip_diffusion.utils.dir_utils import OUTPUT_PATH


@anvil.server.callable
def get_seed():
    """
    將種子傳給anvil
    """

    return str(random_seed())  # 以字串回傳避免anvil產生overflow


@anvil.server.callable
def change_settings(width, height, use_secondary_model):
    """
    修改Config設定
    """

    Config.change(width=width, height=height, use_secondary_model=use_secondary_model)


@anvil.server.callabe
def get_random_prompt(prompt_type):
    """
    回傳隨機的prompt給anvil
    """

    return Prompt.random_prompt(prompt_type)


@anvil.server.callable
def get_chosen_image(choice):
    """
    回傳選中的latent diffusion生成圖片
    """

    image_path = os.path.join(OUTPUT_PATH, "latent", "sr", f"latent_{choice}.png")
    return image_to_blob_media("image/png", image_path)
