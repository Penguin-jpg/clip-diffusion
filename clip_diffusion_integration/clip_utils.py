import clip
from .config import device

chosen_models = {
    "ViT-B/32": True,
    "ViT-B/16": True,
    "ViT-L/14": False,
    "RN50": True,
    "RN50x4": False,
    "RN50x16": False,
    "RN50x64": False,
    "RN101": False,
}

clip_models = []

for model_name, selected in chosen_models.items():
    if selected:
        # 取[0]代表只取Clip模型(不取後續的compose)
        clip_models.append(
            clip.load(model_name, device)[0].eval().requires_grad_(False)
        )
