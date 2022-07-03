import gc
import torch
from mmdet.models import build_detector
from mmdet.apis import (
    train_detector,
    init_detector,
    inference_detector,
    show_result_pyplot,
    single_gpu_test,
)
from mmdet.utils import build_dp


def train(config, datasets, do_eval=True):
    """
    使用mmdetection訓練
    """

    model = build_detector(config.model)
    model.CLASSES = datasets[0].CLASSES  # 修改model的參數

    train_detector(model, datasets, config, distributed=False, validate=do_eval)  # 訓練

    gc.collect()
    torch.cuda.empty_cache()


def test(
    config,
    checkpoint,
    dataset,
    dataloader,
    show_results=True,
    output_dir=None,
    score_threshold=0.3,
):
    """
    測試模型的效果(會遇到OOM，似乎是mmdetection的問題)
    """

    model = init_detector(config, checkpoint, device=config.device)
    model.CLASSES = dataset.CLASSES

    model = build_dp(model, config.device, device_ids=config.gpu_ids)

    single_gpu_test(
        model,
        dataloader,
        show=show_results,
        out_dir=output_dir,
        show_score_thr=score_threshold,
    )  # 單GPU測試

    gc.collect()
    torch.cuda.empty_cache()


def inference(model, image_path, score_threshold=0.3):
    """
    對單張圖片做inference
    """

    # 測試模型
    result = inference_detector(model, image_path)
    # 將結果視覺化
    show_result_pyplot(model, image_path, result, score_thr=score_threshold)

    gc.collect()
    torch.cuda.empty_cache()
