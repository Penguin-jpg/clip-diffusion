import os
from urllib import request
from pathlib import Path
from tqdm import tqdm
from clip_diffusion.dir_utils import MODEL_PATH

# 參考並修改自：https://github.com/lucidrains/DALLE-pytorch/blob/d355100061911b13e1f1c22de8c2b5deb44e65f8/dalle_pytorch/vae.py

# 下載網址
DIFFUSION_MODEL_URL = "https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt"
SECONDARY_MODEL_URL = (
    "https://the-eye.eu/public/AI/models/v-diffusion/secondary_model_imagenet_2.pth"
)
LATENT_DIFFUSION_MODEL_URL = "https://huggingface.co/multimodalart/compvis-latent-diffusion-text2img-large/resolve/main/txt2img-f8-large-jack000-finetuned-fp16.ckpt"
BSRGAN_MODEL_URL = "https://github.com/cszn/KAIR/releases/download/v1.0/BSRGAN.pth"


def download(url, model_name):
    """
    下載模型並儲存，回傳儲存位置
    """

    download_target = Path(os.path.join(MODEL_PATH, model_name))
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
    return str(download_target)
