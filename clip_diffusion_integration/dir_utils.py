import os
from .config import steps_per_checkpoint, intermediates_in_subfolder


def make_dir(dir_path):
    """
    建立資料夾
    """
    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
    else:
        print(f"{dir_path} already exists")


init_dir_path = "./init_images"  # 儲存init_image的資料夾
out_dir_path = "./out_images"  # 儲存輸出圖片的資料夾

make_dir(init_dir_path)
make_dir(out_dir_path)
