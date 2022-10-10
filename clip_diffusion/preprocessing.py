import torch
from PIL import Image
from clip_diffusion.utils.functional import tokenize, embed_text
from clip_diffusion.utils.image_utils import (
    get_image_from_bytes,
    image_to_tensor,
    normalize_image_neg_one_to_one,
)


def get_text_embeddings_and_text_weights(prompt, clip_models, device=None):
    """取得prompt的embedding及weight"""
    text_embeddings_and_weights = {}
    for clip_model_name, clip_model in clip_models.items():
        # text的embedding和權重
        text_embeddings_and_weights[clip_model_name] = {}
        text_embeddings_and_weights[clip_model_name]["embeddings"] = embed_text(
            clip_model, tokenize([prompt.text], device)
        )
        text_embeddings_and_weights[clip_model_name]["weights"] = torch.tensor(prompt.weight, device=device)
        # 權重和不可為0
        if text_embeddings_and_weights[clip_model_name]["weights"].item() < 1e-3:
            raise RuntimeError("The text_weights must not sum to 0.")
    return text_embeddings_and_weights


def create_init_image_tensor(init_image, resize_shape, device=None):
    """建立初始圖片tensor"""
    init_image_tensor = None
    # 如果初始圖片不為空
    if init_image is not None:
        if isinstance(init_image, str):
            # 如果是字串就直接開啟
            image = Image.open(init_image).convert("RGB")
        else:
            # 將anvil傳來的image bytes轉成Pillow Image
            image = get_image_from_bytes(init_image.get_bytes()).convert("RGB")
        image = image.resize(resize_shape, Image.LANCZOS)  # 調整圖片大小
        image_tensor = image_to_tensor(image, device).unsqueeze(0)  # 轉tensor並擴增一個batch_size維度
        init_image_tensor = normalize_image_neg_one_to_one(image_tensor)  # 將範圍normalize到[-1, 1]
    return init_image_tensor


def create_mask_tensor(mask_image, resize_shape, device=None):
    """建立mask tensor"""
    mask_tensor = None
    if mask_image is not None:
        if isinstance(mask_image, str):
            mask = Image.open(mask_image)
        else:
            mask = get_image_from_bytes(mask_image.get_bytes())
        # 建立一個白色的背景(因為anvil傳來的圖片會去背，如果直接二值化會導致全部變成黑色)
        background = Image.new("RGB", mask.size, "WHITE")
        background.paste(mask, box=(0, 0), mask=mask)  # 將mask貼到background上
        mask = background.convert("1")  # 將background轉黑白圖片
        mask = mask.resize(resize_shape, Image.LANCZOS)  # 調整圖片大小
        mask_tensor = image_to_tensor(mask, device).unsqueeze(0)  # 轉tensor並擴增一個batch_size維度
    return mask_tensor
