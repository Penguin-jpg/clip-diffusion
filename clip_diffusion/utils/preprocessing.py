import re
import clip
import torch
import numpy as np
import random
from transformers import pipeline
from opencc import OpenCC
from PIL import Image
from clip_diffusion.config import config
from clip_diffusion.utils.image_utils import get_image_from_bytes, image_to_tensor, normalize_image_neg_one_to_one
from clip_diffusion.utils.perlin import regen_perlin_no_expand

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


def _append_styles_to_prompts(prompts, styles=[]):
    """
    根據對應的風格加上風格標籤
    """

    # 如果使用者有選擇風格才做
    if styles:
        for index, prompt in enumerate(prompts):
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


def prompts_preprocessing(prompts, styles=[]):
    """
    對prompts做需要的前處理: 1. 中翻英 2. prompt engineering
    """

    # 先中翻英
    prompts = _translate_zh_to_en(prompts)
    # 根據選擇的風格做prompt engineering
    prompts = _append_styles_to_prompts(prompts, styles)

    return prompts


# 參考並修改自：https://colab.research.google.com/drive/12a_Wrfi2_gwwAuN3VvMTwVMz9TfqctNj
def _parse_prompt(prompt):
    """
    解析prompt(分離文字與權重)
    """

    parsed = prompt.split(":", 1)  # prompt的格式為"文字:權重"，所以透過":"進行切割
    parsed = parsed + [1.0] if len(parsed) == 1 else parsed  # 如果原本未標示權重，就補上權重1
    return parsed[0], parsed[1]  # 回傳文字與權重


def set_seed():
    """
    設定種子
    """

    np.random.seed(config.seed)
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    torch.backends.cudnn.deterministic = True  # 確保每次卷積演算法是固定的


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
            text, weight = _parse_prompt(prompt)  # 取得text及weight
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


def create_init_noise(init_image=None, resize=True, use_perlin=False, perlin_mode="mixed", device=None):
    """
    建立初始雜訊(init_image或perlin noise只能擇一)
    """

    init_noise = None  # 初始雜訊

    # 如果初始圖片不為空
    if init_image is not None:
        image = get_image_from_bytes(init_image.get_bytes()).convert("RGB")  # 將anvil傳來的image bytes轉成Pillow Image
        if resize:
            image = image.resize((config.width, config.height), Image.LANCZOS)  # resize
        image_tensor = image_to_tensor(image, device).unsqueeze(0)  # 轉tensor
        init_noise = normalize_image_neg_one_to_one(image_tensor)  # 將範圍normalize到[-1, 1]
    elif use_perlin:  # 使用perlin noise
        init_noise = regen_perlin_no_expand(perlin_mode, device)

    return init_noise


def preprocess_mask_image(mask_image, width, height, device=None):
    """
    latent diffusion的mask_image前處理
    """

    mask = get_image_from_bytes(mask_image.get_bytes()).convert("1")  # 轉成二值化(黑白)圖片
    mask = mask.resize((width, height), Image.LANCZOS)  # resize
    mask_tensor = image_to_tensor(mask, device)  # 轉tensor
    return mask_tensor
