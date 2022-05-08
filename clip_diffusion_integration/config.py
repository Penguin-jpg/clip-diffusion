import torch
import random
import json
import lpips
import math

INT_MAX = 2 ** 32


def get_seed():
    random.seed()
    return random.randint(0, INT_MAX)


width = 0  # 生成圖片的寬度
height = 0  # 　生成圖片的高度
# resize用的x, y(一定要是64的倍數)
side_x = (width / 64) * 64
side_y = (height / 64) * 64
perlin_mode = "mixed"  # 使用的perlin模式
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU或CPU
batch_size = 1  # 要生成幾張圖片
skip_augs = False  # 是否不做圖片的augmentation
seed = get_seed()  # 亂數種子
text_prompts = [
    "A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation.",
    "yellow color scheme",
]  # 要生成的東西(可以將不同特徵分開寫)
image_prompts = []  # 輔助生成的圖片
randomize_class = True  # imagenet的class是否要每個iteration都隨機改變
clip_denoised = False  # clip是否要區分有噪音和沒有噪音的圖片
fuzzy_prompt = False  # 是否要加入multiple noisy prompts到prompt losses內
rand_mag = 0.05  # 控制隨機噪音的強度
init_image = None  # 初始化圖片(能幫助生成成果)
perlin_init = False  # 是否要使用隨機的perlin噪音
use_secondary_model = True  # 是否要使用secondary model輔助生成結果
timestep_respacing = "ddim100"  # 減少timestep的數量
diffusion_steps = 1000  # diffusion要跑的step數
use_checkpoint = True  # 是否要使用model checkpoint
steps = 250  # 每個iteration要跑的step數
model_path = "models"  # 模型存放路徑
secondary_model_path = f"{model_path}/secondary_model_imagenet_2.pth"
clip_guidance_scale = 5000  # clip引導的強度(生成圖片要多接近prompt)
tv_scale = 0  # 控制最後輸出的平滑程度
range_scale = 150  # 控制允許超出多遠的RGB值
sat_scale = 0  # 控制允許多少飽和
cutn = 16  # 要從圖片中做幾次crop
cutn_batches = 1  # 從cut的batch中累加Clip的梯度
cut_overview = [35] * 400 + [5] * 600  # 前400/1000個steps會做40個cut；後600/1000個steps會做20個cut
cut_innercut = [5] * 400 + [35] * 600
cut_ic_pow = 1
cut_icgray_p = [0.2] * 400 + [0] * 900
init_scale = 0  # 增強init_image的效果
skip_timesteps = 0  # 控制要跳過的step數(從第幾個step開始)
eta = 1.0  # DDIM用的超參數
clamp_grad = True  # 是否在cond_fn中要使用adaptive的Clip梯度
clamp_max = 0.05  # 限制的最大梯度
# diffusion_model = "512x512_diffusion_uncond_finetune_008100"  # 使用的diffusion model
setting_name = "my_setting"  # 設定資料的名稱
lpips_model = lpips.LPIPS(net="vgg").to(device)  # LPIPS model
num_batches = 100  # 希望denoising diffuison生成幾張靜態圖片
intermediate_saves = [200, 225, 245]  # 分別在哪些step的圖片要存起來
intermediates_in_subfolder = True  # 是否要將圖片存在"partials"資料夾內
steps_per_checkpoint = (
    math.floor((steps - skip_timesteps - 1) // (intermediate_saves + 1))
    if not isinstance(intermediate_saves, list)
    else None
)  # 每個checkpoint隔多少個step
display_rate = 50  # 多少個step要更新顯示的圖片一次

# 確保有大於0
if steps_per_checkpoint:
    steps_per_checkpoint = steps_per_checkpoint if steps_per_checkpoint > 0 else 1


def save_settings():
    """
    儲存設定資訊
    """
    setting_list = {
        "text_prompts": text_prompts,
        "image_prompts": image_prompts,
        "clip_guidance_scale": clip_guidance_scale,
        "tv_scale": tv_scale,
        "range_scale": range_scale,
        "sat_scale": sat_scale,
        # 'cutn': cutn,
        "cutn_batches": cutn_batches,
        "init_image": init_image,
        "init_scale": init_scale,
        "skip_timesteps": skip_timesteps,
        "perlin_init": perlin_init,
        "perlin_mode": perlin_mode,
        "skip_augs": skip_augs,
        "randomize_class": randomize_class,
        "clip_denoised": clip_denoised,
        "clamp_grad": clamp_grad,
        "clamp_max": clamp_max,
        "seed": seed,
        "fuzzy_prompt": fuzzy_prompt,
        "rand_mag": rand_mag,
        "eta": eta,
        "width": width,
        "height": height,
        "use_secondary_model": use_secondary_model,
        "steps": steps,
        "diffusion_steps": diffusion_steps,
    }

    with open(f"{setting_name}.txt", "w+") as file:
        json.dump(setting_list, file, ensure_ascii=False, indent=4)
