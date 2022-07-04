import os
from mmcv import Config, mkdir_or_exist
from mmdet.apis import init_random_seed, set_random_seed
from mmdet.utils import replace_cfg_vals, setup_multi_processes, get_device
from mmdet.datasets import replace_ImageToTensor


def _load_config(config_path):
    """
    從檔案讀取模型config
    """

    config = Config.fromfile(config_path)
    config = replace_cfg_vals(config)
    setup_multi_processes(config)
    return config


def show_config(config):
    """
    顯示config資訊
    """

    print(config.pretty_text)


def setup_train_config(
    config_path,
    seed=None,
    deterministic=True,
    dataset_type="COCODataset",
    annotation_paths={
        "train": "datasets/train/train.json",
        "val": "datasets/val/val.json",
    },
    classes=(),
    resume_from=None,
    pretrained_path=None,
    num_epochs=36,
    use_fp16=False,
    loss_scale="dynamic",
    save_dir="checkpoints",
    num_gpus=1,
    log_interval=10,
    evaluation_interval=1,
    checkpoint_save_interval=12,
    use_wandb=True,
    wandb_init_kwargs={},
):
    """
    建立訓練用的config
    config_path: config檔案路徑
    seed: 亂數種子
    deterministic: 是否要開啟cudnn deterministic和bench deterministic
    dataset_type: 資料集格式
    annotation_path: annotation檔案路徑(型態為dict; 格式為: dataset_name/{split}/annotation_file_name)
    classes: 資料集的class(型態為tuple)
    resume_from: 從指定的checkpoint繼續訓練
    pretrained_path: 預訓練模型路徑
    num_epochs: 訓練的epoch數量
    use_fp16: 是否要使用fp16混合精準度
    loss_scale: 混合精準度的loss放大倍率(dynamic或浮點數)
    save_dir: 儲存checkpoint的資料夾
    num_gpus: gpu數量
    log_interval: 多少iteration更新一次訊息
    evaluation_interval: 幾個epoch做一次evaluation
    checkpoint_save_interval: 多少個epoch儲存一次checkpoint
    use_wandb: 是否要使用wandb的logger
    wandb_init_kwargs: wandb的初始化參數
    """

    config = _load_config(config_path)  # 建立config

    config.device = get_device()
    config.dataset_type = dataset_type

    for split, annotation_path in annotation_paths.items():
        config.data[split].ann_file = annotation_path
        config.data[split].img_prefix = os.path.dirname(
            annotation_path
        )  # 取出訓練圖片的prefix
        config.data[split].classes = classes

    config.resume_from = resume_from
    config.load_from = pretrained_path

    config.runner = dict(type="EpochBasedRunner", max_epochs=num_epochs)

    if use_fp16:
        config.fp16 = dict(loss_scale=loss_scale)

    config.work_dir = save_dir
    mkdir_or_exist(os.path.abspath(config.work_dir))

    config.optimizer.lr /= 8 // num_gpus  # config預設是用8個gpu執行，所以lr要等比例縮減
    config.log_config.interval = log_interval
    config.evaluation.interval = evaluation_interval
    config.checkpoint_config.interval = checkpoint_save_interval

    config.seed = init_random_seed(seed)
    set_random_seed(seed, deterministic=deterministic)

    config.gpu_ids = range(num_gpus)

    config.log_config.hooks = [
        dict(type="TextLoggerHook"),
    ]

    if use_wandb:
        import wandb

        wandb.login()  # 登入wandb

        if not wandb_init_kwargs:
            print("wandb_init_kwargs cannot be empty")
            return

        config.log_config.hooks.append(
            dict(
                type="WandbLoggerHook",
                init_kwargs=wandb_init_kwargs,
                interval=log_interval,
            ),
        )

    show_config(config)  # 顯示目前的config

    return config


def setup_test_config(
    config_path,
    annotation_path="datasets/test/test.json",
    classes=(),
    samples_per_gpu=1,
    workers_per_gpu=2,
    shuffle=False,
    use_fp16=False,
    loss_scale="dynamic",
    save_dir="tests",
    num_gpus=1,
):
    """
    建立測試用的config
    samples_per_gpu: 每個gpu的batch size
    worker_per_gpu: 每個gpu的worker數
    shuffle: 是否要對資料集shuffle
    """

    config = _load_config(config_path)  # 建立config

    config.data.test.ann_file = annotation_path
    config.data.test.img_prefix = os.path.dirname(annotation_path)
    config.data.test.classes = classes
    config.data.test.test_mode = True
    config.model.train_cfg = None
    config.data.test_dataloader = dict(
        samples_per_gpu=samples_per_gpu,
        workers_per_gpu=workers_per_gpu,
        dist=False,
        shuffle=shuffle,
    )
    config.device = get_device()

    if use_fp16:
        config.fp16 = dict(loss_scale=loss_scale)

    config.work_dir = save_dir
    mkdir_or_exist(os.path.abspath(config.work_dir))

    config.gpu_ids = range(num_gpus)

    # 將不需要的部分設為None
    if "pretrained" in config.model:
        config.model.pretrained = None
    elif "init_config" in config.model.backbone:
        config.model.backbone.init_config = None

    if config.model.get("neck"):
        if isinstance(config.model.neck, list):
            for neck_config in config.model.neck:
                if neck_config.get("rfp_backbone"):
                    if neck_config.rfp_backbone.get("pretrained"):
                        neck_config.rfp_backbone.pretrained = None
        elif config.model.neck.get("rfp_backbone"):
            if config.model.neck.rfp_backbone.get("pretrained"):
                config.model.neck.rfp_backbone.pretrained = None

    if config.data.test_dataloader.get("samples_per_gpu", 1) > 1:
        # 將test_pipeline的ImageToTensor替換為DefaultFormatBundle
        config.data.test.pipeline = replace_ImageToTensor(config.data.test.pipeline)

    show_config(config)

    return config
