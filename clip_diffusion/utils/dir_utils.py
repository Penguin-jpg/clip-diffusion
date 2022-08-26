import os
import shutil
from glob import glob


OUTPUT_PATH = "output_images"  # 儲存輸出圖片的資料夾路徑
MODEL_PATH = "models"  # 儲存model的資料夾路徑
ASSET_PATH = os.path.join("clip-diffusion", "assets")  # 儲存assets的資料夾路徑
CSV_PATH = os.path.join("clip-diffusion", "data", "csv")  # 儲存csv的資料夾路徑
INDEX_PATH = os.path.join("clip-diffusion", "data", "indices")  # 儲存index的資料夾路徑


def _remove_old_dirs_and_files(dir_path):
    """移除指定路徑下的所有資料夾及檔案"""
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
    """建立資料夾"""
    if dir_path:
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        else:
            if remove_old:
                _remove_old_dirs_and_files(dir_path)
            else:
                print(f"{dir_path} already exists")
    else:
        print("path cannot be empty")


def get_file_paths(dir_path, pattern, sort_paths=True):
    """找出dir_path下符合pattern的檔案路徑"""
    # 搜尋的pattern
    targets = os.path.join(dir_path, pattern)
    # 找出符合pattern的圖片路徑
    file_paths = glob(targets)
    # 如果要排序
    if sort_paths:
        file_paths = sorted(file_paths)
    return file_paths
