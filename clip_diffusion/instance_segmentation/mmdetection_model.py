from mmdet.apis import init_detector
from mmdet.apis import train_detector, inference_detector, show_result_pyplot

def load_model(config, checkpoint=None):
    """
    載入mmdetection模型
    """

    # 如果checkpoint為None，則model會使用隨機初始化的權重
    model = init_detector(config, checkpoint, device="cuda:0")
    return model

def train(model, config, datasets, distributed=False, validate=True):
    """
    使用mmdetection訓練
    """

    model.CLASSES = datasets[0].CLASSES  # 修改model的參數
    train_detector(model, datasets, config, distributed=distributed, validate=validate) # 訓練


def test(model, image_path, score_threshold=0.3):
    """
    測試模型的效果
    """

    # 轉為eval模式
    model.eval()
    # 測試模型
    result = inference_detector(model, image_path)
    # 將結果視覺化
    show_result_pyplot(model, image_path, result, score_thr=score_threshold)
