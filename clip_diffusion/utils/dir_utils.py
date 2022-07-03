import os
import shutil


OUTPUT_PATH = "output_images"  # 儲存輸出圖片的資料夾
MODEL_PATH = "models"  # 儲存model的資料夾


def _remove_old_dirs_and_files(dir_path):
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


def make_dir(dir_path, remove_old=False):
    """
    建立資料夾
    """

    if dir_path:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            print(f"{dir_path} already exists")

            if remove_old:
                _remove_old_dirs_and_files(dir_path)
    else:
        print("path cannot be empty")