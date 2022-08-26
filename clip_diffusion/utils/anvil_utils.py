import pandas as pd
import anvil.server
import os
from clip_diffusion.config import Config
from clip_diffusion.utils.embedding_index import load_faiss_index, get_topk_results
from clip_diffusion.utils.dir_utils import CSV_PATH, INDEX_PATH, OUTPUT_PATH
from clip_diffusion.utils.functional import random_seed, to_clip_image, embed_image
from clip_diffusion.prompt import Prompt
from clip_diffusion.utils.image_utils import image_to_blob_media, get_image_from_bytes
from clip_diffusion.sample import clip_models

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
def change_settings(width, height, clip_guidance_scale, LPIPS_scale, aesthetic_scale):
    """
    修改Config設定
    """

    Config.change(
        width=width,
        height=height,
        clip_guidance_scale=clip_guidance_scale,
        LPIPS_scale=LPIPS_scale,
        aesthetic_scale=aesthetic_scale,
    )


@anvil.server.callable
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
    分析圖片風格與媒介並回傳相似度前三高的
    """

    image = get_image_from_bytes(image.get_bytes())
    image = to_clip_image(image, Config.device)
    results = {}

    for clip_model_name in ("ViT-B/16", "ViT-L/14"):
        image_embedding = embed_image(clip_models[clip_model_name], image, False, True)
        s_similarities, s_indices = get_topk_results(styles_indices[clip_model_name], image_embedding, 3)
        m_similarities, m_indices = get_topk_results(media_indices[clip_model_name], image_embedding, 3)
        results[clip_model_name] = {
            "styles": [(f"{value:.2%}", styles_df.iloc[index]["style"]) for value, index in zip(s_similarities[0], s_indices[0])],
            "media": [(f"{value:.2%}", media_df.iloc[index]["medium"]) for value, index in zip(m_similarities[0], m_indices[0])],
        }

    return results
