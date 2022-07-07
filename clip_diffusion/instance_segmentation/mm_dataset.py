from mmdet import __version__
from mmcv.utils import get_git_hash
from mmdet.datasets import build_dataset, build_dataloader


def build_mm_dataset(config, split="train"):
    """
    建立dataset
    """

    if split == "train":
        train_datasets = [build_dataset(config.data.train)]
        config.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=train_datasets[0].CLASSES,
        )
        return train_datasets
    elif split == "test":
        test_dataset = build_dataset(config.data.test)
        return test_dataset
    else:
        print('split should be "train" or "test"')


def build_mm_dataloader(config, datasets, split="train"):
    """
    建立dataloader
    """

    if split == "train":
        train_dataloader_config = config.data.get("train_dataloader", {})
        if not train_dataloader_config:
            train_dataloader_config = dict(
                samples_per_gpu=2,
                workers_per_gpu=2,
                num_gpus=len(config.gpu_ids),
                dist=False,
                seed=config.seed,
                runner_type=config.runner["type"],
                persistent_workers=False,
            )

        train_dataloaders = [
            build_dataloader(dataset, **train_dataloader_config) for dataset in datasets
        ]
        return train_dataloaders
    elif split == "test":
        test_dataloader_config = config.data.get("test_dataloader", {})
        if not test_dataloader_config:
            test_dataloader_config = dict(
                samples_per_gpu=1, workers_per_gpu=2, dist=False, shuffle=False
            )

        test_dataloader = build_dataloader(datasets, **test_dataloader_config)
        return test_dataloader
    else:
        print('split should be "train" or "test"')
