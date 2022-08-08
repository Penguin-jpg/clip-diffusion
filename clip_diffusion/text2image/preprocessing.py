import torch
from PIL import Image
from clip_diffusion.utils.functional import tokenize, embed_text
from clip_diffusion.utils.image_utils import get_image_from_bytes, image_to_tensor, normalize_image_neg_one_to_one
from clip_diffusion.text2image.perlin import generate_perlin_noise


def get_text_embeddings_and_text_weights(prompt, clip_models, device=None):
    """
    取得prompt的embedding及weight
    """

    text_embeddings = []  # text的embedding
    text_weights = []  # text對應的權重

    for clip_model in clip_models:
        text_embedding = embed_text(clip_model, tokenize([prompt.text], device))
        text_weight = torch.tensor(prompt.weight, device=device)
        text_embeddings.append(text_embedding)
        text_weights.append(text_weight)

        # 權重和不可為0
        if text_weight.item() < 1e-3:
            raise RuntimeError("The text_weights must not sum to 0.")

    return text_embeddings, text_weights


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
