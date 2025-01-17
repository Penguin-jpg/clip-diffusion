import os
import io
import pyimgur
import cv2
import firebase_admin
import datetime
from firebase_admin import storage
from PIL import Image
from torchvision.transforms import functional as TF
from anvil import BlobMedia
from clip_diffusion.utils.dir_utils import make_dir, get_file_paths
from clip_diffusion.utils.functional import clear_gpu_cache

_IMAGE_EXTENSIONS = ("jpg", "jpeg", "png", "gif")
# imgur
imgur = pyimgur.Imgur(os.environ.get("IMGUR_CLIENT_ID"))
# firebase憑證
credential = firebase_admin.credentials.Certificate(os.environ.get("FIREBASE_CREDENTIAL_PATH"))
# 初始化firebase app
firebase_admin.initialize_app(credential, {"storageBucket": os.environ.get("FIREBASE_STORAGE_URL")})
# 建立storage bucket
bucket = storage.bucket()


def image_to_tensor(pillow_image_or_ndarray, device=None):
    """將Pillow Image或numpy ndarray轉成tensor"""
    return TF.to_tensor(pillow_image_or_ndarray).to(device)


def tensor_to_pillow_image(image_tensor):
    """將image_tensor轉為Pillow Image"""
    return TF.to_pil_image(image_tensor)


def normalize_image_neg_one_to_one(image_tensor):
    """將image_tensor的元素範圍從[0, 1]normalize到[-1, 1](因為DDPM預期輸入的image tensor的元素範圍是-1 ~ 1)"""
    return image_tensor.mul(2).sub(1)


def denormalize_image_zero_to_one(image_tensor):
    """將image_tensor的元素範圍從[-1, 1]denormalize回[0, 1]"""
    return image_tensor.add(1).div(2)


def create_gif(
    batch_folder,
    batch_index,
    gif_duration=500,
):
    """建立一張gif"""
    assert batch_folder is not None, "batch_folder cannot be None"

    # 選出目前batch的所有圖片路徑
    image_paths = get_file_paths(batch_folder, f"guided_{batch_index}*.png")
    images = []  # 儲存要找的圖片
    slice = len(image_paths) // 5  # 固定切成5份
    for index, image_path in enumerate(image_paths):
        # 按照更新速率找出需要的圖片
        if index % slice == 0:
            images.append(Image.open(image_path))
    # 手動補上最後一個timestep的圖片
    images.append(Image.open(image_paths[-1]))
    image_path = os.path.join(batch_folder, f"diffusion_{batch_index}.gif")
    # 儲存成gif
    images[0].save(
        fp=image_path,
        format="GIF",
        save_all=True,
        append_images=images[1:],
        duration=gif_duration,
        loop=False,
    )
    return image_path


def _upload_to_firebase(image_path, **kwargs):
    """上傳至firebase，kwargs負責圖片token到期時間"""
    blob = bucket.blob(image_path)
    blob.upload_from_filename(image_path)
    # 如果不為空
    if kwargs:
        return blob.generate_signed_url(datetime.timedelta(**kwargs))
    else:
        # 預設存在10分鐘
        return blob.generate_signed_url(datetime.timedelta(minutes=10))


def upload_image(image_path, use_firebase=True, extension="png", **kwargs):
    """將靜態圖片上傳至網站並回傳該圖片的url，kwargs負責圖片token到期時間"""
    assert extension.lower() in _IMAGE_EXTENSIONS, "not a valid image extension"

    if use_firebase:
        return _upload_to_firebase(image_path, **kwargs)
    else:
        image = imgur.upload_image(image_path, title=f"{os.path.basename(image_path)}")
        return image.link  # 回傳url


def get_image_from_bytes(image_bytes):
    """透過io.BytesIO讀取圖片的bytes再轉成Image"""
    return Image.open(io.BytesIO(image_bytes))


def _image_to_bytes(image_path):
    """將指定圖片轉為bytes"""
    image = Image.open(image_path)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def image_to_blob_media(content_type, image_path):
    """將指定圖片轉為anvil的BlobMedia"""
    return BlobMedia(content_type, _image_to_bytes(image_path))


def images_to_grid_image(batch_folder, images, num_rows, num_cols):
    """將圖片變成grid格式"""
    # 檢查row*col數量是否等於圖片數量
    assert len(images) == num_rows * num_cols, "num_rows * num_cols should equal to num_images"

    width, height = images[0].size  # 取出一張的寬高
    grid_image = Image.new("RGB", size=(num_cols * width, num_rows * height))  # 建立一個空的grid image
    for index, image in enumerate(images):
        grid_image.paste(
            image, box=(index % num_cols * width, index // num_cols * height)
        )  # 將圖片貼到grid image對應的位置
    image_path = os.path.join(batch_folder, "grid_image.png")
    grid_image.save(image_path)
    return upload_image(image_path, "png")


def super_resolution(upsampler, batch_folder, exception_paths=[]):
    """將圖片解析度放大4倍"""
    result_path = os.path.join(batch_folder, "sr")  # 存放sr結果的路徑
    make_dir(result_path, remove_old=True)
    # 找出batch_folder下的所有png圖片路徑
    image_paths = get_file_paths(batch_folder, "*.png")

    # 對batch_folder內的每張圖片做sr
    for index, image_path in enumerate(image_paths):
        # 如果圖片路徑不是例外路徑(不想做sr的圖片)
        if image_path not in exception_paths:
            # 取得圖片名稱和副檔名
            image_name = os.path.basename(image_path)
            # 印出目前的圖片名稱
            print(f"image {index}: {image_name}")
            # 讀取圖片
            image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            # 將圖片解析度放大4倍
            output_image, _ = upsampler.enhance(image, outscale=4)
            # 寫出圖片
            image_path = os.path.join(result_path, image_name)
            cv2.imwrite(image_path, output_image)
            clear_gpu_cache()
