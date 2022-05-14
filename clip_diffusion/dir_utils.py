import os


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
model_path = "./models"  # 儲存model的資料夾
diffusion_model_path = (
    f"{model_path}/512x512_diffusion_uncond_finetune_008100.pt"  # diffusion model路徑
)
secondary_model_path = (
    f"{model_path}/secondary_model_imagenet_2.pth"  # secondary model路徑
)

make_dir(init_dir_path)
make_dir(out_dir_path)
make_dir(model_path)