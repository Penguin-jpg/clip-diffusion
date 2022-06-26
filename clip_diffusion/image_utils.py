import os
import io
import pyimgur
from glob import glob
from PIL import Image
from anvil import BlobMedia

# 參考並修改自：https://github.com/afiaka87/clip-guided-diffusion/blob/a631a06b51ac5c6636136fab27833c68862eaa24/cgd/script_util.py

CLIENT_ID = "9bc11312c2c8b9a"
imgur = pyimgur.Imgur(CLIENT_ID)


def upload_png(image_path):
    """
    將生成過程的png上傳至imgur並回傳該png的url
    """

    image = imgur.upload_image(
        image_path, title=f"{os.path.basename(image_path)}"
    )  # 上傳至imgur
    return image.link  # 回傳url


def upload_gif(batch_folder, display_rate=30, append_last_timestep=False):
    """
    用生成過程的圖片建成gif，上傳至imgur並回傳該gif的url
    """

    # 找出batch_folder下所有的png
    images_glob = sorted(glob(os.path.join(batch_folder, "*.png")))

    images = []  # 儲存要找的圖片
    for index, image_path in enumerate(images_glob):
        # 按照更新速率找出需要的圖片
        if index % display_rate == 0:
            images.append(Image.open(image_path))

    # 如果diffusion_steps無法被display_rate整除，就要手動補上最後一個timestep的圖片
    if append_last_timestep:
        images.append(Image.open(images_glob[-1]))

    filename = os.path.join(batch_folder, "diffusion.gif")

    # 儲存成gif
    images[0].save(
        fp=filename,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=300,
        loop=0,
    )

    # 將生成過程的gif上傳至Imgur
    gif_image = imgur.upload_image(filename, title="diffusion.gif")

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
