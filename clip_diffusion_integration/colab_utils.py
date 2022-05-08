import subprocess
import os
import hashlib
from .config import (
    model_path,
    diffusion_model_path,
    secondary_model_path,
)

# 如果直接用!wget都抓不到模型，不知道為什麼
def wget(url, output_dir):
    """
    wget功能
    """
    res = subprocess.run(
        ["wget", url, "-P", f"{output_dir}"], stdout=subprocess.PIPE
    ).stdout.decode("utf-8")
    print(res)


def download_models(
    diffusion_model_name, use_secondary_model, check_model_SHA=True, fallback=False
):
    """
    下載所需的模型
    """
    model_512_SHA = "9c111ab89e214862b76e1fa6a1b3f1d329b1a88281885943d2cdbe357ad57648"
    model_secondary_SHA = (
        "983e3de6f95c88c81b2ca7ebb2c217933be1973b1ff058776b970f901584613a"
    )
    model_512_link = "https://v-diffusion.s3.us-west-2.amazonaws.com/512x512_diffusion_uncond_finetune_008100.pt"
    model_secondary_link = (
        "https://v-diffusion.s3.us-west-2.amazonaws.com/secondary_model_imagenet_2.pth"
    )
    model_512_link_fb = "https://huggingface.co/lowlevelware/512x512_diffusion_unconditional_ImageNet/resolve/main/512x512_diffusion_uncond_finetune_008100.pt"
    model_secondary_link_fb = (
        "https://the-eye.eu/public/AI/models/v-diffusion/secondary_model_imagenet_2.pth"
    )

    if fallback:
        model_512_link = model_512_link_fb
        model_secondary_link = model_secondary_link_fb

    # 下載diffusion model
    if os.path.exists(diffusion_model_path) and check_model_SHA:
        print("Checking 512 Diffusion File")
        with open(diffusion_model_path, "rb") as f:
            bytes = f.read()
            hash = hashlib.sha256(bytes).hexdigest()
        if hash == model_512_SHA:
            print("512 Model SHA matches")
            if not os.path.exists(diffusion_model_path):
                print("First URL Failed using FallBack")
                download_models(diffusion_model_name, use_secondary_model, True)
        else:
            print("512 Model SHA doesn't match, redownloading...")
            wget(model_512_link, diffusion_model_path)
            if not os.path.exists(diffusion_model_path):
                print("First URL Failed using FallBack")
                download_models(diffusion_model_name, use_secondary_model, True)
    elif os.path.exists(diffusion_model_path) and not check_model_SHA:
        print(
            "512 Model already downloaded, check check_model_SHA if the file is corrupt"
        )
    else:
        wget(model_512_link, model_path)

    # 下載secondary diffusion model v2
    if use_secondary_model == True:
        if os.path.exists(secondary_model_path) and check_model_SHA:
            print("Checking Secondary Diffusion File")
            with open(secondary_model_path, "rb") as f:
                bytes = f.read()
                hash = hashlib.sha256(bytes).hexdigest()
            if hash == model_secondary_SHA:
                print("Secondary Model SHA matches")
            else:
                print("Secondary Model SHA doesn't match, redownloading...")
                wget(model_secondary_link, model_path)
                if not os.path.exists(secondary_model_path):
                    print("First URL Failed using FallBack")
                    download_models(diffusion_model_name, use_secondary_model, True)
        elif os.path.exists(secondary_model_path) and not check_model_SHA:
            print(
                "Secondary Model already downloaded, check check_model_SHA if the file is corrupt"
            )
        else:
            wget(model_secondary_link, model_path)
            if not os.path.exists(secondary_model_path):
                print("First URL Failed using FallBack")
                download_models(diffusion_model_name, use_secondary_model, True)
