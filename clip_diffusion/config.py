import torch
import random
import json

INT_MAX = 2 ** 32


def get_seed():
    """
    生成種子
    """
    random.seed()
    return random.randint(0, INT_MAX)


# 圖片長寬相關
width = 1280  # 生成圖片的寬度
height = 768  # 　生成圖片的高度
# resize用的x, y(一定要是64的倍數)
side_x = (width // 64) * 64
side_y = (height // 64) * 64

# device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU或CPU

# cutout相關
cutn = 16  # 要從圖片中做幾次crop
cutn_batches = 1  # 從cut的batch中累加Clip的梯度
cut_overview = [35] * 400 + [5] * 600  # 前400/1000個steps會做40個cut；後600/1000個steps會做20個cut
cut_innercut = [5] * 400 + [35] * 600
cut_ic_pow = 1
cut_icgray_p = [0.2] * 400 + [0] * 900

# model相關
steps = 250  # 每個iteration要跑的step數
timestep_respacing = f"ddim{steps}"  # 調整diffusion的timestep數量
diffusion_steps = (
    (1000 // steps) * steps if steps < 1000 else steps
)  # diffusion要跑的step數
use_checkpoint = True  # 是否要使用model checkpoint
use_secondary_model = True  # 是否要使用secondary model輔助生成結果
diffusion_model_name = (
    "512x512_diffusion_uncond_finetune_008100.pt"  # 使用的diffusion model checkpoint
)
secondary_model_name = "secondary_model_imagenet_2.pth"  # 使用的secondary model checkpoint

# Clip相關
clip_guidance_scale = 5000  # clip引導的強度(生成圖片要多接近prompt)
clip_denoised = False  # clip是否要區分有噪音和沒有噪音的圖片
clamp_grad = True  # 是否在cond_fn中要使用adaptive的Clip梯度
clamp_max = 0.05  # 限制的最大梯度

# loss相關
tv_scale = 0  # 控制最後輸出的平滑程度
range_scale = 150  # 控制允許超出多遠的RGB值
sat_scale = 0  # 控制允許多少飽和

# 生成相關
seed = get_seed()  # 亂數種子
batch_size = 1  # 一次要sample的數量
num_batches = 1  # 要生成的圖片數量
skip_augs = False  # 是否不做圖片的augmentation
randomize_class = True  # imagenet的class是否要每個iteration都隨機改變(製造unconditional的效果)
fuzzy_prompt = False  # 是否要加入multiple noisy prompts到prompt losses內
rand_mag = 0.05  # 控制隨機噪音的強度
init_scale = 0  # 增強init_image的效果
skip_timesteps = 0  # 控制要跳過的step數(從第幾個step開始)
eta = 1.0  # DDIM用的超參數
settings_name = "default_settings"  # 設定資料的名稱
display_rate = 25  # 多少個step要更新顯示的圖片一次
intermediate_saves = [
    display_rate * i for i in range(steps // display_rate + 1)
]  # 分別在哪些step的圖片要存起來(要+1才能包含最後一個step)


def save_settings():
    """
    儲存設定資訊
    """
    setting_list = {
        "clip_guidance_scale": clip_guidance_scale,
        "tv_scale": tv_scale,
        "range_scale": range_scale,
        "sat_scale": sat_scale,
        "cutn_batches": cutn_batches,
        "init_scale": init_scale,
        "skip_timesteps": skip_timesteps,
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

    with open(f"{settings_name}.txt", "w+") as file:
        json.dump(setting_list, file, ensure_ascii=False, indent=4)


def adjust_config(
    adjust_width=1280,
    adjust_height=768,
    adjust_steps=250,
    adjust_clip_guidance_scale=5000,
    adjust_clip_denoised=False,
    adjust_clamp_grad=True,
    adjust_clamp_max=0.05,
    adjust_tv_scale=0,
    adjust_range_scale=150,
    adjust_sat_scale=0,
    adjust_batch_size=1,
    adjust_num_batches=1,
    adjust_fuzzy_prompt=False,
    adjust_rand_mag=0.05,
    adjust_skip_timesteps=0,
    adjust_eta=1.0,
    adjust_settings_name="default_settings",
    adjust_display_rate=25,
):
    """
    調整設定
    """

    width = adjust_width
    height = adjust_height
    side_x = (width // 64) * 64
    side_y = (height // 64) * 64
    steps = adjust_steps
    timestep_respacing = f"ddim{steps}"
    diffusion_steps = (1000 // steps) * steps if steps < 1000 else steps
    clip_guidance_scale = adjust_clip_guidance_scale
    clip_denoised = adjust_clip_denoised
    clamp_grad = adjust_clamp_grad
    clamp_max = adjust_clamp_max
    tv_scale = adjust_tv_scale
    range_scale = adjust_range_scale
    sat_scale = adjust_sat_scale
    batch_size = adjust_batch_size
    num_batches = adjust_num_batches
    fuzzy_prompt = adjust_fuzzy_prompt
    rand_mag = adjust_rand_mag
    skip_timesteps = adjust_skip_timesteps
    eta = adjust_eta
    settings_name = adjust_settings_name
    display_rate = adjust_display_rate
    intermediate_saves = [display_rate * i for i in range(steps // display_rate + 1)]
