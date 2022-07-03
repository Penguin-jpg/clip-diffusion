import os
import json
from IPython.display import Image, display
from clip_retrieval.clip_client import ClipClient, Modality
from clip_diffusion.utils.dir_utils import make_dir


def _show_result(result):
    id, caption, url, similarity = (
        result["id"],
        result["caption"],
        result["url"],
        result["similarity"],
    )
    print(f"id: {id}")
    print(f"caption: {caption}")
    print(f"url: {url}")
    print(f"similarity: {similarity}")
    display(Image(url=url, unconfined=True))


def _results_to_json(results ,output_path):
    """
    將query的結果存成json
    """

    if output_path:
        dir_path = os.path.dirname(output_path) # 取出output_path中的資料夾名稱
        # 如果output_path包含資料夾路徑
        if dir_path != "":
            make_dir(dir_path)

        with open(output_path, "w") as file:
            json.dump(results, file)
    else:
        print("path cannot be empty")


def create_clip_client(
    backend_url="https://knn5.laion.ai/knn-service",
    indice_name="laion5B",
    aesthetic_score=9,
    aesthetic_weight=0.5,
    modality=Modality.IMAGE,
    num_images=500,
):
    """
    建立Clip retrieval client
    """

    return ClipClient(
        url=backend_url,
        indice_name=indice_name,
        aesthetic_score=aesthetic_score,
        aesthetic_weight=aesthetic_weight,
        modality=modality,
        num_images=num_images,
    )


def query_by_text(client, text, show_first_result=True, to_json=False, json_file_path=None):
    """
    透過文字進行query
    """

    results = client.query(text=text)

    if show_first_result:
        _show_result(results[0])

    if to_json:
        _results_to_json(results, json_file_path)

    return results


def query_by_image(client, image_url, show_first_result=True, to_json=False, json_file_path=None):
    """
    透過圖片進行query
    """

    results = client.query(image=image_url)

    if show_first_result:
        _show_result(results[0])

    if to_json:
        _results_to_json(results, json_file_path)

    return results