import os
import shutil


def remove_old_dirs_and_files(dir_path):
    """
    移除指定路徑下的所有資料夾及檔案
    """

    if os.path.exists(dir_path):
        # 移除所有舊檔案
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            # 如果是檔案就直接刪除
            if os.path.isfile(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):  # 如果是資料夾就用rmtree
                shutil.rmtree(file_path)
    else:
        print(f"{dir_path} does not exist")


def make_dir(dir_path, remove_old=True):
    """
    建立資料夾
    """

    if not os.path.exists(dir_path):
        os.mkdir(dir_path)
        if remove_old:
            remove_old_dirs_and_files(dir_path)
    else:
        print(f"{dir_path} already exists")


init_dir_path = "./init_images"  # 儲存init_image的資料夾
out_dir_path = "./out_images"  # 儲存輸出圖片的資料夾
model_path = "./models"  # 儲存model的資料夾
diffusion_model_path = (
    f"{model_path}/512x512_diffusion_uncond_finetune_008100.pt"  # diffusion model路徑
)
secondary_model_path = (
    f"{model_path}/secondary_model_imagenet_2.pth"  # secondary model路徑
)
