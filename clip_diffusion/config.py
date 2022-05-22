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


class Config:
    """
    將設定統整在一個class
    """

    def __init__(self):
        """
        初始化預設值
        """

        # 圖片長寬相關
        self.width = 896  # 生成圖片的寬度
        self.height = 768  # 　生成圖片的高度
        # resize用的x, y(一定要是64的倍數)
        self.side_x = (self.width // 64) * 64
        self.side_y = (self.height // 64) * 64

        # device
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )  # GPU或CPU

        # cutout相關
        self.cutn = 16  # 要從圖片中做幾次crop
        self.cutn_batches = 4  # 從cut的batch中累加Clip的梯度
        self.cut_overview = [35] * 400 + [
            5
        ] * 600  # 前400/1000個steps會做40個cut；後600/1000個steps會做20個cut
        self.cut_innercut = [5] * 400 + [35] * 600
        self.cut_ic_pow = 1
        self.cut_icgray_p = [0.2] * 400 + [0] * 900

        # model相關
        self.steps = 250  # 每個iteration要跑的step數
        self.timestep_respacing = f"ddim{self.steps}"  # 調整diffusion的timestep數量
        self.diffusion_steps = (
            (1000 // self.steps) * self.steps if self.steps < 1000 else self.steps
        )  # diffusion要跑的step數
        self.use_checkpoint = True  # 是否要使用model checkpoint
        self.use_secondary_model = True  # 是否要使用secondary model輔助生成結果
        self.diffusion_model_name = "512x512_diffusion_uncond_finetune_008100.pt"  # 使用的diffusion model checkpoint
        self.secondary_model_name = (
            "secondary_model_imagenet_2.pth"  # 使用的secondary model checkpoint
        )

        # Clip相關
        self.clip_guidance_scale = 5000  # clip引導的強度(生成圖片要多接近prompt)
        self.clip_denoised = False  # clip是否要區分有噪音和沒有噪音的圖片
        self.clamp_grad = True  # 是否在cond_fn中要使用adaptive的Clip梯度
        self.clamp_max = 0.05  # 限制的最大梯度

        # loss相關
        self.tv_scale = 0  # 控制最後輸出的平滑程度
        self.range_scale = 150  # 控制允許超出多遠的RGB值
        self.sat_scale = 0  # 控制允許多少飽和

        # 生成相關
        self.seed = get_seed()  # 亂數種子
        self.batch_size = 1  # 一次要sample的數量
        self.num_batches = 1  # 要生成的圖片數量
        self.skip_augs = False  # 是否不做圖片的augmentation
        self.randomize_class = (
            True  # imagenet的class是否要每個iteration都隨機改變(製造unconditional的效果)
        )
        self.fuzzy_prompt = False  # 是否要加入multiple noisy prompts到prompt losses內
        self.rand_mag = 0.05  # 控制隨機噪音的強度
        self.init_scale = 1000  # 增強init_image的效果
        self.skip_timesteps = (
            10  # 控制要跳過的step數(從第幾個step開始)，當使用init_image時時最好調整為原先 Step 的 0~50%
        )
        self.eta = 0.8  # 調整每個timestep混入的噪音量(0：無噪音；1.0：最多噪音)
        self.settings_name = "default_settings"  # 設定資料的名稱
        self.display_rate = 25  # 多少個step要更新顯示的圖片一次
        self.intermediate_saves = [
            self.display_rate * i for i in range(self.steps // self.display_rate + 1)
        ]  # 分別在哪些step的圖片要存起來(要+1才能包含最後一個step)

    def adjust_settings(
        self,
        width=1024,
        height=768,
        steps=250,
        clip_guidance_scale=5000,
        clip_denoised=False,
        clamp_grad=True,
        clamp_max=0.05,
        tv_scale=0,
        range_scale=150,
        sat_scale=0,
        batch_size=1,
        num_batches=1,
        skip_augs=False,
        randomize_class=True,
        fuzzy_prompt=False,
        rand_mag=0.05,
        init_scale=1000,
        skip_timesteps=10,
        eta=0.8,
        settings_name="default_settings",
        display_rate=25,
    ):
        """
        調整設定
        """

        self.width = width
        self.height = height
        self.side_x = (self.width // 64) * 64
        self.side_y = (self.height // 64) * 64
        self.steps = steps
        self.timestep_respacing = f"ddim{self.steps}"
        self.diffusion_steps = (
            (1000 // self.steps) * self.steps if self.steps < 1000 else self.steps
        )
        self.clip_guidance_scale = clip_guidance_scale
        self.clip_denoised = clip_denoised
        self.clamp_grad = clamp_grad
        self.clamp_max = clamp_max
        self.tv_scale = tv_scale
        self.range_scale = range_scale
        self.sat_scale = sat_scale
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.skip_augs = skip_augs
        self.randomize_class = randomize_class
        self.fuzzy_prompt = fuzzy_prompt
        self.rand_mag = rand_mag
        self.init_scale = init_scale
        self.skip_timesteps = skip_timesteps
        self.eta = eta
        self.settings_name = settings_name
        self.display_rate = display_rate
        self.intermediate_saves = [
            self.display_rate * i for i in range(self.steps // self.display_rate + 1)
        ]

    def save_settings(self):
        """
        儲存設定資訊
        """

        setting_list = {
            "clip_guidance_scale": self.clip_guidance_scale,
            "tv_scale": self.tv_scale,
            "range_scale": self.range_scale,
            "sat_scale": self.sat_scale,
            "cutn_batches": self.cutn_batches,
            "init_scale": self.init_scale,
            "skip_timesteps": self.skip_timesteps,
            "skip_augs": self.skip_augs,
            "randomize_class": self.randomize_class,
            "clip_denoised": self.clip_denoised,
            "clamp_grad": self.clamp_grad,
            "clamp_max": self.clamp_max,
            "seed": self.seed,
            "fuzzy_prompt": self.fuzzy_prompt,
            "rand_mag": self.rand_mag,
            "init_scale": self.init_scale,
            "eta": self.eta,
            "width": self.width,
            "height": self.height,
            "use_secondary_model": self.use_secondary_model,
            "steps": self.steps,
            "diffusion_steps": self.diffusion_steps,
        }

        with open(f"{self.settings_name}.txt", "w+") as file:
            json.dump(setting_list, file, ensure_ascii=False, indent=4)


config = Config()
