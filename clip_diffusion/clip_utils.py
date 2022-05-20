import clip
from .config import config

chosen_models = {
    "ViT-B/32": True,
    "ViT-B/16": True,
    "ViT-L/14": False,
    "RN50": True,
    "RN50x4": True,
    "RN50x16": False,
    "RN50x64": False,
    "RN101": False,
}

clip_models = []

# 初始載入
for model_name, selected in chosen_models.items():
    if selected:
        # 取[0]代表只取Clip模型(不取後續的compose)
        clip_models.append(
            clip.load(model_name, config.device)[0].eval().requires_grad_(False)
        )


def choose_clip_models(choices):
    """
    選擇並載入要使用的Clip模型
    """

    # 將chosen_models宣告為全域變數
    global chosen_models

    # 載入選擇的Clip模型
    for model_name, selected in choices.items():
        if selected and not chosen_models[model_name]:
            # 取[0]代表只取Clip模型(不取後續的compose)
            clip_models.append(
                clip.load(model_name, config.device)[0].eval().requires_grad_(False)
            )

    # 更新
    chosen_models = choices
