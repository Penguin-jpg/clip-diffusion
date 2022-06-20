import os.path
import logging
import torch
import gc
import bsrgan.utils_image as util
from bsrgan.utils_logger import logger_info
from clip_diffusion.config import config
from clip_diffusion.dir_utils import make_dir, remove_old_files


def super_resolution(model, batch_folder):
    """
    將圖片解析度放大4倍
    """

    logger_info("blind_sr_log", log_path="blind_sr_log.log")
    logger = logging.getLogger("blind_sr_log")
    result_path = f"{batch_folder}/sr"  # 存放sr結果的路徑
    make_dir(result_path)
    remove_old_files(result_path)  # 移除舊的圖片

    # 對batch_folder內的每張圖片做sr
    for index, image in enumerate(util.get_image_paths(batch_folder)):
        # 取得圖片名稱和副檔名
        image_name, ext = os.path.splitext(os.path.basename(image))
        logger.info(f"{index:4d} --> {image_name + ext:<s}")

        # 原圖轉tensor
        original_image = util.imread_uint(image, n_channels=3)
        original_image = util.uint2tensor4(original_image).to(config.device)

        # 進行sr
        result_image = model(original_image)
        result_image = util.tensor2uint(result_image)

        # 儲存圖片
        util.imsave(
            result_image,
            os.path.join(result_path, f"{image_name}_sr.{ext}"),
        )
        del original_image  # 刪除以釋放記憶體

        gc.collect()
        torch.cuda.empty_cache()

    print("sr finished!")
