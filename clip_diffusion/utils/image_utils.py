import os
import io
import pyimgur
import logging
import torch
import gc
from glob import glob
from PIL import Image
from anvil import BlobMedia
from bsrgan.utils_image import (
    get_image_paths,
    imread_uint,
    uint2tensor4,
    tensor2uint,
    imsave,
)
from bsrgan.utils_logger import logger_info
from clip_diffusion.utils.dir_utils import make_dir

_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
CLIENT_ID = "9bc11312c2c8b9a"
imgur = pyimgur.Imgur(CLIENT_ID)


def upload_png(image_path):
    """
    將生成過程的png上傳至imgur並回傳該png的url
    """

    image = imgur.upload_image(image_path, title=f"{os.path.basename(image_path)}")  # 上傳至imgur
    return image.link  # 回傳url


def upload_gif(
    batch_folder,
    current_batch,
    display_rate=30,
    gif_duration=500,
    append_last_timestep=False,
):
    """
    用生成過程的圖片建成gif，上傳至imgur並回傳該gif的url
    """

    # 選出目前batch的所有圖片
    images_glob = sorted(glob(os.path.join(batch_folder, f"guided_{current_batch}*.png")))

    images = []  # 儲存要找的圖片
    for index, image_path in enumerate(images_glob):
        # 按照更新速率找出需要的圖片
        if index % display_rate == 0:
            images.append(Image.open(image_path))

    # 如果diffusion_steps無法被display_rate整除，就要手動補上最後一個timestep的圖片
    if append_last_timestep:
        images.append(Image.open(images_glob[-1]))

    filename = os.path.join(batch_folder, f"diffusion_{current_batch}.gif")

    # 儲存成gif
    images[0].save(
        fp=filename,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=gif_duration,
        loop=0,
    )

    # 將生成過程的gif上傳至Imgur
    gif_image = imgur.upload_image(filename, title=f"diffusion_{current_batch}.gif")

    return gif_image.link  # 回傳url


def get_image_from_bytes(image_bytes):
    """
    透過io.BytesIO讀取圖片的bytes再轉成Image
    """
    return Image.open(io.BytesIO(image_bytes))


def _image_to_bytes(image_path):
    """
    將指定圖片轉為bytes
    """

    image = Image.open(image_path)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def image_to_blob_media(content_type, image_path):
    """
    將指定圖片轉為anvil的BlobMedia
    """

    return BlobMedia(content_type, _image_to_bytes(image_path))


def images_to_grid_image(batch_folder, images, num_rows, num_cols):
    """
    將圖片變成grid格式
    """

    # 檢查row*col數量是否等於圖片數量
    if len(images) != num_rows * num_cols:
        print("num_rows * num_cols should equal to num_images")
        return

    width, height = images[0].size  # 取出一張的寬高

    grid_image = Image.new("RGB", size=(num_cols * width, num_rows * height))  # 建立一個空的grid image

    for index, image in enumerate(images):
        grid_image.paste(image, box=(index % num_cols * width, index // num_cols * height))  # 將圖片貼到grid image對應的位置

    filename = os.path.join(batch_folder, "grid_image.png")
    grid_image.save(filename)

    return upload_png(filename)


def super_resolution(model, batch_folder, exception_paths=[]):
    """
    將圖片解析度放大4倍
    """

    logger_info("blind_sr_log", log_path="blind_sr_log.log")
    logger = logging.getLogger("blind_sr_log")
    result_path = f"{batch_folder}/sr"  # 存放sr結果的路徑
    make_dir(result_path, remove_old=True)

    # 對batch_folder內的每張圖片做sr
    for index, image_path in enumerate(get_image_paths(batch_folder)):
        # 如果圖片路徑不是例外路徑(不想做sr的圖片)
        if image_path not in exception_paths:
            # 取得圖片名稱和副檔名
            image_name = os.path.basename(image_path)
            logger.info(f"{index:4d} --> {image_name:<s}")

            # 原圖轉tensor
            original_image = imread_uint(image_path, n_channels=3)
            original_image = uint2tensor4(original_image).to(_device)

            # 進行sr
            result_image = model(original_image)
            result_image = tensor2uint(result_image)

            # 儲存圖片
            imsave(
                result_image,
                os.path.join(result_path, image_name),
            )
            del original_image  # 刪除以釋放記憶體

            gc.collect()
            torch.cuda.empty_cache()
