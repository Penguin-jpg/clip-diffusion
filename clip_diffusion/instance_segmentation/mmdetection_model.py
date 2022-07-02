import mmcv
from mmdet.models import build_detector
from mmdet.apis import train_detector, inference_detector, show_result_pyplot

def load_model(config):
    """
    載入mmdetection模型
    """

    model = build_detector(config.model)
    return model

def train(model, config, datasets, distributed=False, validate=True):
    """
    使用mmdetection訓練
    """

    model.CLASSES = datasets[0].CLASSES  # 修改model的參數
    train_detector(model, datasets, config, distributed=distributed, validate=validate) # 訓練


def test(model, config, image_path, score_threshold=0.3):
    """
    測試模型的效果
    """

    image = mmcv.imread(image_path)

    model.config = config
    result = inference_detector(model, image)
    show_result_pyplot(model, image, result, score_thr=score_threshold)
