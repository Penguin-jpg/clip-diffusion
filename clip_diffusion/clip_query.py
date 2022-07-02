from IPython.display import Image, display
from clip_retrieval.clip_client import ClipClient, Modality


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


def query_by_text(client, text, show_first_result=True):
    """
    透過文字進行query
    """

    results = client.query(text=text)

    if show_first_result:
        _show_result(results[0])

    return results


def query_by_image(client, image_url, show_first_result=True):
    """
    透過圖片進行query
    """

    results = client.query(image=image_url)

    if show_first_result:
        _show_result(results[0])

    return results
