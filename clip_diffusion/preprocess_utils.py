import re
import clip
import torch
import numpy as np
import random
from transformers import pipeline
from opencc import OpenCC
from clip_diffusion.config import config
from clip_diffusion.prompt_utils import parse_prompt
from clip_diffusion.clip_utils import clip_models

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


def translate_zh_to_en(text_prompts):
    """
    將中文翻譯成英文
    """

    # 先轉簡體，以符合模型輸入
    for index, prompt in enumerate(text_prompts):
        # 如果包含中文
        if contains_zh(prompt):
            prompt = converter.convert(prompt)
            # 翻譯成英文
            result = translator(prompt)[0]
            # 更新prompt
            text_prompts[index] = result["translation_text"]

    return text_prompts


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


def get_embedding_and_weights(text_prompts):
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

        for prompt in text_prompts:
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
