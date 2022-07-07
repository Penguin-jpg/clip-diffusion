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
from clip_diffusion.utils.prompt_utils import parse_prompt
from clip_diffusion.utils.image_utils import get_image_from_bytes
from clip_diffusion.utils.perlin_utils import regen_perlin_no_expand

_translator = pipeline(
    "translation",
    model="Helsinki-NLP/opus-mt-zh-en",
    tokenizer="Helsinki-NLP/opus-mt-zh-en",
)  # 用來中翻英
_converter = OpenCC("tw2sp.json")  # 繁體轉簡體
_STYLES = {
    "奇幻 / 科幻": [
        "James Gurney",
        "Greg Rutkowski",
        "Pascal Blanche",
        "Jakub Rozalski",
        "Federico Pelat",
    ],
    "注重光影": ["James Paick", "Fitz Henry Lane"],
    "立體效果": ["unreal engine"],
    "風景畫": ["Julian Falat", "Isaac Levitan", "John Constable", "Ivan Shishkin"],
    "印象派": ["Van Gogh", "Monet"],
    "早晨": ["morning"],
    "傍晚": ["evening"],
    "夕陽": ["sunset"],
}  # 風格標籤


def _contains_zh(prompt):
    """
    檢查是否包含中文
    """

    if re.search(r"[\u4e00-\u9FFF]", prompt):
        return True
    return False


def _translate_zh_to_en(prompts):
    """
    將中文翻譯成英文
    """

    # 先轉簡體，以符合模型輸入
    for index, prompt in enumerate(prompts):
        # 如果包含中文
        if _contains_zh(prompt):
            prompt = _converter.convert(prompt)
            # 翻譯成英文
            result = _translator(prompt)[0]
            # 更新prompts
            prompts[index] = result["translation_text"]

    return prompts


def _append_styles_to_prompts(prompts, styles=["自訂"]):
    """
    根據對應的風格加上風格標籤
    """

    for index, prompt in enumerate(prompts):
        # 如果使用者想要自訂就不補上風格標籤就直接跳出
        if "自訂" in styles:
            break

        # 從prompts中暫時移除句點
        if prompt[-1] == ".":
            append_period = True
            prompt = prompt[:-1]
        else:
            append_period = False

        # 一律加入artstation
        prompt += ", artstation"

        # 從選定的風格中挑出一個選項
        for style in styles:
            prompt += f", {_STYLES[style][random.randint(0, len(_STYLES[style])-1)]}"

        # 需要時補回句點
        if append_period:
            prompt += "."

        # 更新prompts
        prompts[index] = prompt

    return prompts


def prompts_preprocessing(prompts, styles=["自訂"]):
    """
    對prompts做需要的前處理: 1. 中翻英 2. prompt engineering
    """

    # 先中翻英
    prompts = _translate_zh_to_en(prompts)
    # 根據選擇的風格做prompt engineering
    prompts = _append_styles_to_prompts(prompts, styles)

    return prompts


def set_seed():
    """
    設定種子
    """

    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True


def get_embeddings_and_weights(prompts, clip_models, device=None):
    """
    取得prompt的embedding及weight
    """

    clip_model_stats = []

    for clip_model_name, clip_model in clip_models.items():
        clip_model_stat = {
            "clip_model_name": clip_model_name,  # 對應的Clip model名稱
            "text_embeddings": [],  # text的embedding
            "make_cutouts": None,  # 後續用來儲存cutout
            "text_weights": [],  # text對應的權重
        }
        for prompt in prompts:
            text, weight = parse_prompt(prompt)  # 取得text及weight
            text = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()

            clip_model_stat["text_embeddings"].append(text)
            clip_model_stat["text_weights"].append(weight)

        clip_model_stat["text_embeddings"] = torch.cat(clip_model_stat["text_embeddings"])
        clip_model_stat["text_weights"] = torch.tensor(clip_model_stat["text_weights"], device=device)

        # 權重和不可為0
        if clip_model_stat["text_weights"].sum().abs() < 1e-3:
            raise RuntimeError("The text_weights must not sum to 0.")

        # 正規化
        clip_model_stat["text_weights"] /= clip_model_stat["text_weights"].sum().abs()
        clip_model_stats.append(clip_model_stat)

    return clip_model_stats


def create_init_noise(init_image=None, use_perlin=False, perlin_mode="mixed", device=None):
    """
    建立初始雜訊(init_image或perlin noise只能擇一)
    """

    init_noise = None  # 初始雜訊

    # 如果初始圖片不為空
    if init_image is not None:
        init_noise = get_image_from_bytes(init_image.get_bytes()).convert("RGB")  # 透過anvil傳來的圖片的bytes開啟圖片
        init_noise = init_noise.resize((config.width, config.height), Image.LANCZOS)
        init_noise = TF.to_tensor(init_noise).to(device).unsqueeze(0).mul(2).sub(1)
    elif use_perlin:  # 使用perlin noise
        init_noise = regen_perlin_no_expand(perlin_mode, device)

    return init_noise
