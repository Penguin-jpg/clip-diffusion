import os
from IPython.display import Image, display
from clip_retrieval.clip_client import ClipClient, Modality
from img2dataset import download
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
    ).query


def get_queries(
    client,
    text=None,
    image=None,
    num_results=100,
    show_first_result=True,
    to_json=False,
    json_file_path=None,
):
    """
    透過文字或圖片進行query
    """

    if num_results < 0:
        print("number of results cannot be zero")
        return

    results = client.query(text=text, image=image)

    if num_results > len(results):
        print(
            "excceeds max number of results! automatically shorten to match max length"
        )
    else:
        results = results[:num_results]

    if show_first_result:
        _show_result(results[0])

    if to_json:
        _results_to_json(results, json_file_path)

    return results


def download_images_from_urls(
    url_file_path,
    output_dir,
    num_processes=1,
    num_threads=256,
    image_size=256,
    resize_mode="border",
    encode_format="png",
    encode_quality=0,
    input_format="json",
    output_format="files",
    num_samples_per_shard=10000,
    timeout=10,
    num_retires=0,
    distributor="multiprocessing",
):
    """
    透過指定的url_file下載圖片
    url_file_path: 儲存要下載的url的檔案
    output_dir: 下載圖片的儲存位置
    num_processes: 處理的process數量
    num_threads: 處理的thread數量
    image_size: 下載的圖片會resize到這個大小
    resize_mode: resize的方式(no, border, keep_ratio, center_crop)
    encode_format: 輸出的圖片格式(jpg, png, webp)
    encode_quality: encode的品質，範圍從0~100(當使用png時應為0~9)，越高圖像壓縮越多
    input_format: url_file_path的格式(txt, csv, tsv.gz, json, parquet)
    output_format: 下載的圖片要如何儲存(files, webdataset, parquet, tfrecord,dummy)
    num_samples_per_shard: 每個subfolder最多能存幾張圖片
    timeout: 下載最多等幾秒
    num_retires: 當timeout時重試的次數
    distributor: 分散下載的方式(multiprocessing, pyspark)
    """

    make_dir(output_dir, remove_old=True)

    # 下載圖片
    download(
        url_list=url_file_path,
        output_folder=output_dir,
        processes_count=num_processes,
        thread_count=num_threads,
        image_size=image_size,
        resize_mode=resize_mode,
        encode_format=encode_format,
        encode_quality=encode_quality,
        input_format=input_format,
        output_format=output_format,
        number_sample_per_shard=num_samples_per_shard,
        timeout=timeout,
        retries=num_retires,
        distributor=distributor,
    )
