import os
import logging
import torch
import gc
from SCUNet.utils import utils_model
from SCUNet.utils import utils_image as util
from SCUNet.models.network_scunet import SCUNet as net
from .dir_utils import out_dir_path, model_path, make_dir
from download_utils import download, denoise_model_url
from .config import config


def load_denoise_model(model_name="scunet_color_real_psnr.pth"):
    """
    載入去噪模型
    """

    # 模型路徑
    denoise_model_path = os.path.join(model_path, model_name)

    # 載入模型
    model = net(in_nc=3, config=[4, 4, 4, 4, 4, 4, 4], dim=64)
    model.load_state_dict(
        torch.load(
            download(denoise_model_url, config.denoise_model_name, False),
            map_location="cpu",
        ),
        strict=True,
    )
    model.requires_grad_(False).eval().to(config.device)

    gc.collect()
    torch.cuda.empty_cache()

    return model


def denoise_real_image(
    model,
    image_path=f"{out_dir_path}/diffusion",
    result_path=f"{out_dir_path}/diffusion/denoised",
):
    """
    去除圖片雜訊(真實圖片專用)
    model: 使用的model
    image_path: 雜訊圖片的路徑
    result_path: 去除雜訊後的圖片存放路徑
    """

    make_dir(result_path)

    logger_name = "denoised_logger"
    utils_logger.logger_info(
        logger_name, log_path=os.path.join(result_path, logger_name + ".log")
    )
    logger = logging.getLogger(logger_name)

    logger.info(f"model_name:{os.path.basename(model_path)}")
    logger.info(image_path)
    image_paths = util.get_image_paths(image_path)

    for index, image_path in enumerate(image_paths):

        # ------------------------------------
        # (1) noise_image
        # ------------------------------------
        image_name = os.path.basename(image_path)
        logger.info(f"{index + 1:->4d}--> {image_name:>10s}")

        noise_image = util.imread_uint(image_path, n_channels=3)
        noise_image = util.uint2tensor4(noise_image)
        noise_image = noise_image.to(config.device)

        # ------------------------------------
        # (2) result_image
        # ------------------------------------
        result_image = model(noise_image)
        result_image = util.tensor2uint(result_image)

        # ------------------------------------
        # save results
        # ------------------------------------
        util.imsave(result_image, os.path.join(result_path, image_name + ".png"))
