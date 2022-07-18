import os
import torch
import clip
import requests
import io
from clip_retrieval.clip_client import ClipClient, Modality
from clip_diffusion.utils.dir_utils import make_dir
from clip_diffusion.utils.functional import to_clip_image, tokenize, get_text_embedding, get_image_embedding

_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
_model, _preprocess = clip.load("ViT-L/14", device=_device)
_model.eval().requires_grad_(False)


def _results_to_json(results, output_path):
    """
    將query的結果存成json
    """

    if output_path:
        dir_path = os.path.dirname(output_path)  # 取出output_path中的資料夾名稱
        # 如果output_path包含資料夾路徑
        if dir_path != "":
            make_dir(dir_path)

        with open(output_path, "w") as file:
            import json

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


# 參考並修改自：https://colab.research.google.com/drive/1V66mUeJbXrTuQITvJunvnWVn96FEbSI3#scrollTo=YHOj78Yvx8jP
def _fetch_image(url):
    if not url.startswith("http://") or not url.startswith("https://"):
        print("not a valid url")
        return
    else:
        request = requests.get(url)
        request.raise_for_status()
        output = io.BytesIO()
        output.write(request.content)
        output.seek(0)
        return output


def _get_embedding(text=None, image_url=None):
    """
    根據輸入決定要使用的embedding
    """

    assert text or image_url, "need to specify text or image_url"

    if text:
        return get_text_embedding(_model, tokenize(text, _device), divided_by_norm=True)[0].tolist()
    else:
        image = _fetch_image(image_url)
        return get_image_embedding(
            _model,
            to_clip_image(_preprocess, image, _device),
            use_normalize=False,
            divided_by_norm=True,
        )[0].tolist()


def get_query_results(
    client,
    text=None,
    image_url=None,
    use_embedding=False,
    num_results=500,
    to_json=False,
    output_path=None,
):
    """
    透過文字或圖片進行query
    """

    assert num_results >= 0, "number of results cannot be zero"

    if use_embedding:
        results = client.query(embedding_input=_get_embedding(text, image_url))
    else:
        results = client.query(text=text, image=image_url)

    if num_results > len(results):
        print("excceeds max number of results! automatically shorten to match max length")
    else:
        results = results[:num_results]

    if to_json:
        _results_to_json(results, output_path)

    return results


def combine_results(results_1, results_2, num_results=1000, to_json=False, output_path=None):
    """
    將兩個results結合
    """

    if num_results < 0:
        print("number of results cannot be zero")
        return

    new_results = results_1 + results_2

    if num_results > len(new_results):
        print("excceeds max number of results! automatically shorten to match max length")
    else:
        new_results = new_results[:num_results]

    if to_json:
        _results_to_json(new_results, output_path)

    return new_results
