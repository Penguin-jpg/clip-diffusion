from multiprocessing.connection import Client
import os
import glob
from PIL import Image
import pyimgur

# 參考並修改自：https://github.com/afiaka87/clip-guided-diffusion/blob/a631a06b51ac5c6636136fab27833c68862eaa24/cgd/script_util.py

CLIENT_ID = "9bc11312c2c8b9a"
imgur = pyimgur.Imgur(CLIENT_ID)


def create_gif(text_prompts, image_path, batch_name):
    """
    用生成過程的圖片建成gif
    """

    file_name = "_".join(text_prompts)
    # 找出image_path下所有的png
    images_glob = os.path.join(image_path, "*.png")
    # 開啟所有的圖片
    images = [Image.open(image) for image in sorted(glob.glob(images_glob))]
    # 檔名
    gif_name = f"{image_path}/{batch_name}_{file_name}.gif"

    # 儲存gif
    images[0].save(
        fp=gif_name,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=200,
        loop=0,
    )

    image = imgur.upload_image(
        gif_name, title=f"{batch_name}_{file_name}"
    )  # 將gif上傳至Imgur
    return image.link  # 回傳url
