import clip
import numpy as np
import random
import gc
import torch
import anvil.server
from torchvision import transforms as T
from tqdm.notebook import trange
from IPython import display

# Clip用到的normalize
CLIP_NORMALIZE = T.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])


def get_num_model_parameters(model, grad=True):
    """
    取得模型參數量
    """

    if grad:
        return sum(param.numel() for param in model.paramarameters() if param.requires_grad)
    return sum(param.numel() for param in model.paramarameters())


def tokenize(text, device=None):
    """
    將text tokenize成clip需要的格式
    """

    if isinstance(text, str):
        text = [text]

    return clip.tokenize(text).to(device)


def to_clip_image(preprocess, image, device=None):
    """
    預設的Clip圖片處理方式
    """

    return preprocess(image).unsqueeze(0).to(device)


def get_text_embedding(clip_model, text, divided_by_norm=False):
    """
    取得text embedding
    """

    text_embedding = clip_model.encode_text(text).float()

    # 不考慮維度，只保留特徵
    if divided_by_norm:
        text_embedding /= text_embedding.norm(dim=-1, keepdim=True)

    return text_embedding


def get_image_embedding(clip_model, image, use_normalize=True, divided_by_norm=False):
    """
    取得image embedding
    """

    if use_normalize:
        image = CLIP_NORMALIZE(image)

    image_embedding = clip_model.encode_image(image).float()

    # 不考慮維度，只保留特徵
    if divided_by_norm:
        image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

    return image_embedding


def set_seed(seed):
    """
    設定種子
    """

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # 確保每次卷積演算法是固定的


def get_sample_function(diffusion, mode="ddim"):
    """
    根據mode回傳對應的sample function
    """

    assert mode in ("ddim", "plms"), "unsupported diffusion sample mode"

    if mode == "ddim":
        return diffusion.ddim_sample_loop_progressive
    else:
        return diffusion.plms_sample_loop_progressive


def get_sampler(latent_diffusion_model, mode="ddim"):
    """
    根據mode回傳對應的sampler
    """

    assert mode in ("ddim", "plms"), "unsupported sampler mode"

    from ldm.models.diffusion.ddim import DDIMSampler
    from ldm.models.diffusion.plms import PLMSSampler

    if mode == "ddim":
        return DDIMSampler(latent_diffusion_model)
    else:
        return PLMSSampler(latent_diffusion_model)


def clear_gpu_cache():
    """
    清除vram的cache
    """

    gc.collect()
    torch.cuda.empty_cache()


class ProgressBar:
    def __init__(self, length, description):
        self.length = length
        self.description = description
        self._progress_bar = self._get_progress_bar()

    def _get_progress_bar(self):
        """
        建立progress bar
        """
        progress_bar = trange(self.length, desc=self.description)
        return progress_bar

    def update_progress(self, value):
        """
        更新progress bar進度
        """

        self._progress_bar.n = value
        self._progress_bar.refresh()


def set_display_widget(widget):
    """
    設定ipython顯示的widget
    """

    display.display(widget)


def display_image(image_path=None, url=None, unconfined=False):
    """
    在ipython顯示圖片
    """

    assert image_path is not None or url is not None, "need to specify image_path or url"

    display.display(display.Image(filename=image_path, url=url, unconfined=unconfined))


def clear_output(widget=None, wait=False):
    """
    清除ipython顯示的內容
    """

    if widget is not None:
        widget.clear_output(wait=wait)

    display.clear_output(wait=wait)


def store_task_state(key, value):
    """
    將key與value存到anvil的task_state中
    """

    anvil.server.task_state[key] = value