import gc
import torch
import mmcv
from mmcv.runner import wrap_fp16_model, load_checkpoint
from mmdet.models import build_detector
from mmdet.apis import (
    train_detector,
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
    output_dir="test_results",
    score_threshold=0.3,
):
    """
    測試模型的效果
    """

    model = build_detector(config, test_cfg=config.get("test_cfg"))
    model.CLASSES = dataset.CLASSES

    if config.get("fp16", None):
        wrap_fp16_model(model)

    _ = load_checkpoint(model, checkpoint, map_location="cpu")

    model = build_dp(model, config.device, device_ids=config.gpu_ids)

    single_gpu_test(
        model,
        dataloader,
        show=False,
        out_dir=output_dir,
        show_score_thr=score_threshold,
    )  # 單GPU測試

    gc.collect()
    torch.cuda.empty_cache()


def inference(model, image_path, output_path="inference_results", score_threshold=0.3):
    """
    對單張圖片做inference
    """

    # 讀取圖片
    image = mmcv.imread(image_path)

    # 模型推論
    result = inference_detector(model, image)

    # 將結果視覺化(無法顯示，原因不明)
    # show_result_pyplot(model, image_path, result, score_thr=score_threshold)

    # 將推論結果存到output_path
    model.show_result(image, result, out_file=output_path)

    gc.collect()
    torch.cuda.empty_cache()
