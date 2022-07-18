import re
import torch
import random
from transformers import pipeline
from opencc import OpenCC
from PIL import Image
from clip_diffusion.utils.functional import tokenize, get_text_embedding
from clip_diffusion.utils.image_utils import get_image_from_bytes, image_to_tensor, normalize_image_neg_one_to_one
from clip_diffusion.utils.perlin import generate_perlin_noise

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
            text_embedding = get_text_embedding(clip_model, tokenize(text, device))  # 取得text embedding
            clip_model_stat["text_embeddings"].append(text_embedding)
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


def create_init_noise(init_image=None, resize_shape=None, use_perlin=False, perlin_mode="mixed", device=None):
    """
    建立初始雜訊(init_image或perlin noise只能擇一)
    """

    init_noise = None  # 初始雜訊

    # 如果初始圖片不為空
    if init_image is not None:
        image = get_image_from_bytes(init_image.get_bytes()).convert("RGB")  # 將anvil傳來的image bytes轉成Pillow Image
        image = image.resize(resize_shape, Image.LANCZOS)  # 調整圖片大小
        image_tensor = image_to_tensor(image, device).unsqueeze(0)  # 轉tensor並擴增一個batch_size維度
        init_noise = normalize_image_neg_one_to_one(image_tensor)  # 將範圍normalize到[-1, 1]
    elif use_perlin:  # 使用perlin noise
        init_noise = generate_perlin_noise(perlin_mode, device)

    return init_noise


def create_mask_tensor(mask_image, resize_shape, device=None):
    """
    建立latent diffusion的mask tensor
    """

    mask = get_image_from_bytes(mask_image.get_bytes())
    # 建立一個白色的背景(因為anvil傳來的圖片會去背，如果直接二值化會導致全部變成黑色)
    background = Image.new("RGB", mask.size, "WHITE")
    background.paste(mask, box=(0, 0), mask=mask)  # 將mask貼到background上
    mask = background.convert("1")  # 將background轉黑白圖片
    mask = mask.resize(resize_shape, Image.LANCZOS)  # 調整圖片大小
    mask_tensor = image_to_tensor(mask, device).unsqueeze(0)  # 轉tensor並擴增一個batch_size維度
    return mask_tensor
