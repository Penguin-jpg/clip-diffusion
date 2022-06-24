import re
import clip
import torch
import numpy as np
import random
from transformers import pipeline
from opencc import OpenCC
from PIL import Image
from torchvision.transforms import functional as TF
from clip_diffusion.config import config
from clip_diffusion.prompt_utils import parse_prompt
from clip_diffusion.image_utils import get_image_from_bytes
from clip_diffusion.perlin_utils import regen_perlin_no_expand

translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-zh-en",
    tokenizer="Helsinki-NLP/opus-mt-zh-en",
)  # 中翻英

converter = OpenCC("tw2sp.json")  # 繁體轉簡體


def contains_zh(prompt):
    """
    檢查是否包含中文
    """
    if re.search(r"[\u4e00-\u9FFF]", prompt):
        return True
    return False


def translate_zh_to_en(prompts):
    """
    將中文翻譯成英文
    """

    # 先轉簡體，以符合模型輸入
    for index, prompt in enumerate(prompts):
        # 如果包含中文
        if contains_zh(prompt):
            prompt = converter.convert(prompt)
            # 翻譯成英文
            result = translator(prompt)[0]
            # 更新prompt
            prompts[index] = result["translation_text"]

    return prompts


def set_seed(seed):
    """
    設定種子
    """

    if seed:
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True


def get_embedding_and_weights(prompts, clip_models):
    """
    取得prompt的embedding及weight
    """

    model_stats = []

    for clip_model in clip_models:
        model_stat = {
            "clip_model": None,
            "target_embeds": [],
            "make_cutouts": None,
            "weights": [],
        }
        model_stat["clip_model"] = clip_model

        for prompt in prompts:
            text, weight = parse_prompt(prompt)  # 取得text及weight
            text = clip_model.encode_text(
                clip.tokenize(prompt).to(config.device)
            ).float()

            if config.fuzzy_prompt:
                for _ in range(25):
                    model_stat["target_embeds"].append(
                        (text + torch.randn(text.shape).cuda() * config.rand_mag).clamp(
                            0, 1
                        )
                    )
                    model_stat["weights"].append(weight)
            else:
                model_stat["target_embeds"].append(text)
                model_stat["weights"].append(weight)

        model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
        model_stat["weights"] = torch.tensor(
            model_stat["weights"], device=config.device
        )

        if model_stat["weights"].sum().abs() < 1e-3:
            raise RuntimeError("The weights must not sum to 0.")

        model_stat["weights"] /= model_stat["weights"].sum().abs()
        model_stats.append(model_stat)

    return model_stats


def create_init_noise(init_image=None, use_perlin=True, perlin_mode="mixed"):
    """
    建立初始雜訊，init_image或perlin noise只能擇一
    """

    init_noise = None  # 初始雜訊

    # 如果初始圖片不為空
    if init_image is not None:
        init_noise = get_image_from_bytes(init_image.get_bytes()).convert(
            "RGB"
        )  # 透過anvil傳來的圖片的bytes開啟圖片
        init_noise = init_noise.resize((config.side_x, config.side_y), Image.LANCZOS)
        init_noise = (
            TF.to_tensor(init_noise).to(config.device).unsqueeze(0).mul(2).sub(1)
        )
    elif use_perlin:  # 使用perlin noise
        init_noise = regen_perlin_no_expand(perlin_mode)

    return init_noise
