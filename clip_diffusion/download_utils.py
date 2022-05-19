import os
import hashlib
from urllib import request
from pathlib import Path
from tqdm import tqdm
from .config import config
from .dir_utils import model_path, diffusion_model_path, secondary_model_path

# 參考並修改自：
# 1. https://github.com/lucidrains/DALLE-pytorch/blob/d355100061911b13e1f1c22de8c2b5deb44e65f8/dalle_pytorch/vae.py
# 2. https://colab.research.google.com/github/alembics/disco-diffusion/blob/main/Disco_Diffusion.ipynb

# 下載網址
model_512_link = "https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt"
secondary_model_link = (
    "https://the-eye.eu/public/AI/models/v-diffusion/secondary_model_imagenet_2.pth"
)

# 檢查用的SHA
model_512_SHA = "9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648"
secondary_model_SHA = "983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a"


def does_SHA_match(model_name):
    """
    檢查SHA是否吻合
    """

    if model_name == config.diffusion_model_name:
        check_path = diffusion_model_path
        model_SHA = model_512_SHA
    else:
        check_path = secondary_model_path
        model_SHA = secondary_model_SHA

    with open(check_path, "rb") as f:
        bytes = f.read()
        hash = hashlib.sha256(bytes).hexdigest()

    if hash == model_SHA:
        print(f"{model_name} SHA matches")
        return True
    else:
        print(f"{model_name} SHA mismatches")
        return False


def download(url, model_name):
    """
    下載模型並儲存，回傳儲存位置
    """

    download_target = Path(os.path.join(model_path, model_name))
    download_target_tmp = download_target.with_suffix(".tmp")

    if os.path.exists(download_target):
        if not os.path.isfile(download_target):
            raise RuntimeError(f"{download_target} exists and is not a regular file")
        else:
            return str(download_target)

    with request.urlopen(url) as source, open(download_target_tmp, "wb") as output:
        with tqdm(total=int(source.info().get("Content-Length")), ncols=80) as loop:
            while True:
                buffer = source.read(4096)
                if not buffer:
                    break
                output.write(buffer)
                loop.update(len(buffer))

    os.rename(download_target_tmp, download_target)

    if does_SHA_match(model_name):
        return str(download_target)
