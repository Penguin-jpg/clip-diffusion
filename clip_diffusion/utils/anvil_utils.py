import pandas as pd
import anvil.server
import os
from clip_diffusion.text2image.config import Config
from clip_diffusion.text2image.embedding_index import load_faiss_index, get_topk_results
from clip_diffusion.utils.dir_utils import CSV_PATH, INDEX_PATH, OUTPUT_PATH
from clip_diffusion.utils.functional import random_seed, to_clip_image, embed_image
from clip_diffusion.text2image.prompt import Prompt
from clip_diffusion.utils.image_utils import image_to_blob_media, get_image_from_bytes

styles_df = pd.read_csv(os.path.join(CSV_PATH, "styles.csv"))
media_df = pd.read_csv(os.path.join(CSV_PATH, "media.csv"))
keywords_index = load_faiss_index(os.path.abspath(os.path.join(INDEX_PATH, "embeddings.index")))
styles_indices = {
    clip_model_name: load_faiss_index(os.path.join(INDEX_PATH, f"{clip_model_name.replace('/', '_')}_style_embeddings.index"))
    for clip_model_name in Config.chosen_clip_models
}
media_indices = {
    clip_model_name: load_faiss_index(os.path.join(INDEX_PATH, f"{clip_model_name.replace('/', '_')}_media_embeddings.index"))
    for clip_model_name in Config.chosen_clip_models
}


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


@anvil.server.callable
def analyze_image(image):
    """
    分析圖片風格與媒介並回傳
    """

    global clip_models

    image = get_image_from_bytes(image.get_bytes())
    image = to_clip_image(image, Config.device)
    results = {}

    for clip_model_name, clip_model in clip_models.items():
        image_embedding = embed_image(clip_model, image, False, True)
        style_similarities, style_indices = get_topk_results(styles_indices[clip_model_name], image_embedding, 5)
        media_similarities, media_indices = get_topk_results(media_indices[clip_model_name], image_embedding, 5)
        results[clip_model_name] = {
            "style_similarities": style_similarities[0].tolist(),
            "style_indices": [styles_df.iloc[index] for index in style_indices[0].tolist()],
            "media_similarities": media_similarities[0].tolist(),
            "media_indices": [media_df.iloc[index] for index in media_indices[0].tolist()],
        }

    return results
