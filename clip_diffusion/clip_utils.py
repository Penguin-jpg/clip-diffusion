import clip
from clip_diffusion.config import config

clip_models = []


def load_clip_models(chosen_models):
    """
    選擇並載入要使用的Clip模型
    """

    # 如果還沒載入模型
    if not clip_models:
        for model_name, selected in chosen_models.items():
            if selected:
                # 取[0]代表只取Clip模型(不取後續的compose)
                clip_models.append(
                    clip.load(model_name, config.device)[0].eval().requires_grad_(False)
                )
    else:
        print("clip models already loaded")
