import os
from img2dataset import download
from mmdet import __version__
from mmcv.utils import get_git_hash
from mmdet.datasets import build_dataset, build_dataloader
from labelme2coco import get_coco_from_labelme_folder, save_json
from clip_diffusion.utils.dir_utils import make_dir


def images_to_dataset(
    url_list,
    output_dir,
    num_processes=1,
    num_threads=256,
    image_size=256,
    output_format="files",
    input_format="json",
    num_samples_per_shard=1000,
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
        processes_count=num_processes,
        thread_count=num_threads,
        image_size=image_size,
        output_format=output_format,
        input_format=input_format,
        number_sample_per_shard=num_samples_per_shard,
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


def build_datasets(config, split="train"):
    """
    建立dataset
    """

    if split == "train":
        datasets = [build_dataset(config.data.train)]
        config.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7], CLASSES=datasets[0].CLASSES
        )
        return datasets
    elif split == "test":
        return build_dataset(config.data.test)


def build_test_dataloader(config, dataset):
    """
    建立dataloader(目前只有test才需要用，train已經包含在train_detector內)
    """

    return build_dataloader(dataset, **config.data.test_dataloader)
