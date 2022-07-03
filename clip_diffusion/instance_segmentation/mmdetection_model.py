from mmdet.models import build_detector
from mmdet.apis import (
    train_detector,
    inference_detector,
    show_result_pyplot,
    single_gpu_test,
)
from mmdet.utils import build_dp
from mmcv.runner import load_checkpoint


def train(config, datasets, do_eval=True):
    """
    使用mmdetection訓練
    """

    model = build_detector(config.model)
    model.CLASSES = datasets[0].CLASSES  # 修改model的參數

    train_detector(model, datasets, config, distributed=False, validate=do_eval)  # 訓練


def test(
    config,
    checkpoint,
    dataset,
    data_loader,
    show_result=True,
    output_dir=None,
    score_threshold=0.3,
):
    """
    測試模型的效果
    """

    model = build_detector(config.model, test_cfg=config.get("test_cfg"))
    model.CLASSES = dataset.CLASSES

    model = build_dp(model, config.device, device_ids=config.gpu_ids)
    checkpoint = load_checkpoint(
        model, checkpoint, map_location="cpu"
    )  # 載入model checkpoint

    single_gpu_test(
        model,
        data_loader,
        show=show_result,
        out_dir=output_dir,
        show_score_thr=score_threshold,
    )  # 單GPU測試


def inference(model, image_path, score_threshold=0.3):
    """
    對單張圖片做inference
    """

    # 測試模型
    result = inference_detector(model, image_path)
    # 將結果視覺化
    show_result_pyplot(model, image_path, result, score_thr=score_threshold)
