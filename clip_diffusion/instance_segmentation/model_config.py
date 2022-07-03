import torch
import os
import wandb
from mmcv import Config, mkdir_or_exist
from mmdet.apis import set_random_seed


def _load_config_from_file(config_path):
    """
    從檔案讀取模型config
    """

    return Config.fromfile(config_path)


def _show_config(config):
    """
    顯示config資訊
    """

    print(f"config:\n{config.pretty_text}")


def setup_config(
    config_path,
    seed=42,
    dataset_type="COCODataset",
    annotation_paths={"train": "datasets/train/train.json", "val": "datasets/val/val.json", "test": "datasets/test/test.json"},
    classes=(),
    pretrained_path=None,
    save_dir="checkpoints",
    num_gpus=1,
    log_interval=10,
    evaluation_interval=1,
    checkpoint_save_interval=12,
    wandb_init_kwargs={},
):
    """
    config_path: config檔案路徑
    seed: 亂數種子
    dataset_type: 資料集格式
    annotation_path: annotation檔案路徑(型態為dict; 格式為: dataset_name/{split}/annotation_file_name)
    classes: 資料集的class(型態為tuple)
    pretrained_path: 預訓練模型路徑
    save_dir: 儲存checkpoint的資料夾
    num_gpus: gpu數量
    log_interval: 多少iteration更新一次訊息
    evaluation_interval: 幾個epoch做一次evaluation
    checkpoint_save_interval: 多少個epoch儲存一次checkpoint
    wandb_init_kwargs: wandb的初始化參數
    """

    config = _load_config_from_file(config_path)
    config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    config.dataset_type = dataset_type

    for split, annotation_path in annotation_paths.items():
        config.data[split].ann_file = annotation_path
        config.data[split].img_prefix = os.path.dirname(annotation_path) # 取出訓練圖片的prefix
        config.data[split].classes = classes

    config.model.mask_head.num_classes = len(classes)
    config.load_from = pretrained_path

    config.work_dir = save_dir
    mkdir_or_exist(os.path.abspath(config.work_dir))

    config.optimizer.lr /= 8 // num_gpus  # config預設是用8個gpu執行，所以lr要等比例縮減
    config.log_config.interval = log_interval
    config.evaluation.interval = evaluation_interval
    config.checkpoint_config.interval = checkpoint_save_interval

    config.seed = seed
    set_random_seed(seed, deterministic=True)

    config.gpu_ids = range(num_gpus)

    wandb.login()  # 登入wandb

    if not wandb_init_kwargs:
        print("wandb_init_kwargs cannot be empty")
        return

    config.log_config.hooks = [
        dict(type="TextLoggerHook"),
        dict(
            type="WandbLoggerHook", init_kwargs=wandb_init_kwargs, interval=log_interval
        ),
    ]
    _show_config(config)

    return config
