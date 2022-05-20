import os
import glob
from PIL import Image
import pyimgur

# 參考並修改自：https://github.com/afiaka87/clip-guided-diffusion/blob/a631a06b51ac5c6636136fab27833c68862eaa24/cgd/script_util.py

CLIENT_ID = "9bc11312c2c8b9a"
imgur = pyimgur.Imgur(CLIENT_ID)


def create_gif(image_path, batch_name):
    """
    用生成過程的圖片建成gif，並回傳最後一個timestep的png及生成過程的gif
    """

    # 找出image_path下所有的png
    images_glob = os.path.join(image_path, "*.png")
    # 開啟所有的圖片
    images = [Image.open(image) for image in sorted(glob.glob(images_glob))]

    # 儲存gif
    images[0].save(
        fp=f"{image_path}/{batch_name}.gif",
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=200,
        loop=0,
    )

    # png_image = imgur.upload_image(
    #     f"{image_path}/{batch_name}.png", title=f"{batch_name}.png"
    # )  # 將最後一個timestep的png上傳至Imgur
    # gif_image = imgur.upload_image(
    #     f"{image_path}/{batch_name}.gif", title=f"{batch_name}.gif"
    # )  # 將gif上傳至Imgur

    # return png_image.link, gif_image.link  # 回傳url
