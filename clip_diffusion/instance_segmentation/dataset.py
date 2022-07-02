import os
from img2dataset import download
from mmdet.datasets import build_dataset
from labelme2coco import get_coco_from_labelme_folder, save_json
from clip_diffusion.utils.dir_utils import make_dir


def images_to_dataset(
    url_list,
    output_dir,
    process_count=1,
    thread_count=256,
    image_size=256,
    output_format="files",
    input_format="json",
    number_sample_per_shard=1000,
    distributor="multiprocessing",
):
    """
    將圖片轉為dataset
    """

    make_dir(output_dir, remove_old=True)

    # 下載圖片
    download(
        url_list=url_list,
        output_folder=output_dir,
        processes_count=process_count,
        thread_count=thread_count,
        image_size=image_size,
        output_format=output_format,
        input_format=input_format,
        number_sample_per_shard=number_sample_per_shard,
        distributor=distributor,
    )


def convert_dataset_to_coco_format(dataset_paths, output_dir):
    """
    將img2dataset轉為coco dataset格式
    """

    train_coco = get_coco_from_labelme_folder(dataset_path["train"])

    for split, dataset_path in dataset_paths.items():
        coco_format = get_coco_from_labelme_folder(
            dataset_path, coco_category_list=train_coco.json_categories
        )
        save_json(
            data=coco_format.json, save_path=os.path.join(output_dir, f"{split}.json")
        )


def build_datasets_from_config(config):
    """
    建立dataset
    """

    return [build_dataset(config.data["train"])]
