import os
import pyimgur
from glob import glob
from PIL import Image

# 參考並修改自：https://github.com/afiaka87/clip-guided-diffusion/blob/a631a06b51ac5c6636136fab27833c68862eaa24/cgd/script_util.py

CLIENT_ID = "9bc11312c2c8b9a"
imgur = pyimgur.Imgur(CLIENT_ID)


def upload_png(image_path, batch_name):
    """
    將生成過程的png上傳至imgur並回傳該png的url
    """

    # 找出image_path下所有的png
    images_glob = sorted(glob(os.path.join(image_path, "*.png")))
    # 上傳至imgur
    image = imgur.upload_image(
        images_glob[-1], title=f"{batch_name}_{len(images_glob)}.png"
    )
    return image.link  # 回傳url


def upload_gif(image_path, batch_name):
    """
    用生成過程的圖片建成gif，上傳至imgur並回傳該gif的url
    """

    # 找出image_path下所有的png
    images_glob = sorted(glob(os.path.join(image_path, "*.png")))
    # 開啟所有的圖片
    images = [Image.open(image) for image in images_glob]

    # 儲存成gif
    images[0].save(
        fp=f"{image_path}/{batch_name}.gif",
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=200,
        loop=0,
    )

    # 將最後一個timestep的png上傳至Imgur
    # png_image = imgur.upload_image(images_glob[-1], title=f"{batch_name}.png")
    # 將生成過程的gif上傳至Imgur
    gif_image = imgur.upload_image(
        f"{image_path}/{batch_name}.gif", title=f"{batch_name}.gif"
    )

    return gif_image.link  # 回傳url
