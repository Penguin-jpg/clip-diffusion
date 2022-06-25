import torch
import random

INT_MAX = 2**32


class _Config:
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
        self.use_secondary_model = (
            True  # 是否要使用secondary model(如果關閉的話則會用原本的diffusion model進行清除)
        )
        self.chosen_clip_models = ["ViT-B/16", "ViT-B/32", "RN50"]

        # Clip相關
        self.clip_denoised = False  # clip是否要區分有噪音和沒有噪音的圖片
        self.clamp_grad = True  # 是否在cond_fn中要使用adaptive的Clip梯度
        self.clamp_max = 0.05  # 限制的最大梯度

        # loss相關
        self.tv_scale = 0  # 控制最後輸出的平滑程度
        self.range_scale = 150  # 控制允許超出多遠的RGB值
        self.sat_scale = 0  # 控制允許多少飽和

        # 生成相關
        self.seed = self.get_seed()  # 亂數種子
        self.batch_size = 1  # 一次要sample的數量
        self.num_batches = 1  # 要生成的圖片數量
        self.skip_augs = False  # 是否不做圖片的augmentation
        self.fuzzy_prompt = False  # 是否要加入multiple noisy prompts到prompt losses內
        self.rand_mag = 0.05  # 控制隨機噪音的強度

    def get_seed(self):
        """
        生成種子
        """
        random.seed()
        return random.randint(0, INT_MAX)

    def adjust_settings(
        self,
        width=896,
        height=768,
        use_secondary_model=True,
        chosen_clip_models=["ViT-B/16", "ViT-B/32", "RN50"],
        clip_denoised=False,
        clamp_grad=True,
        clamp_max=0.05,
        tv_scale=0,
        range_scale=150,
        sat_scale=0,
        batch_size=1,
        num_batches=1,
        skip_augs=False,
        fuzzy_prompt=False,
        rand_mag=0.05,
    ):
        """
        調整設定
        """

        self.width = width
        self.height = height
        self.side_x = (self.width // 64) * 64
        self.side_y = (self.height // 64) * 64
        self.use_secondary_model = use_secondary_model
        self.chosen_clip_models = chosen_clip_models
        self.clip_denoised = clip_denoised
        self.clamp_grad = clamp_grad
        self.clamp_max = clamp_max
        self.tv_scale = tv_scale
        self.range_scale = range_scale
        self.sat_scale = sat_scale
        self.batch_size = batch_size
        self.num_batches = num_batches
        self.skip_augs = skip_augs
        self.fuzzy_prompt = fuzzy_prompt
        self.rand_mag = rand_mag


config = _Config()
