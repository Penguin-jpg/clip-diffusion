import random

INT_MAX = 2**32


class _Config:
    """
    將生成設定統整在這個class
    """

    def __init__(self):
        """
        初始化預設值
        """

        # 圖片長寬相關
        self.width = 960  # 生成圖片的寬度
        self.height = 768  # 　生成圖片的高度
        # resize用的x, y(一定要是64的倍數)
        self.side_x = (self.width // 64) * 64
        self.side_y = (self.height // 64) * 64

        # cutout相關
        self.num_cutout_batches = 4  # 要做的cutout次數
        self.overview_cut_schedule = [12] * 400 + [
            4
        ] * 600  # 前400/1000個diffusion steps會做12個cut；後600/1000個steps會做4個cut
        self.inner_cut_schedule = [4] * 400 + [12] * 600
        self.inner_cut_size_pow = 1  # 控制生成圖片的景物豐富度(越高會有越多物件)
        self.cut_gray_portion_schedule = [0.2] * 400 + [0] * 600  # 控制多少百分比的cut要取出做灰階化

        # model相關
        self.use_secondary_model = (
            True  # 是否要使用secondary model(如果關閉的話則會用原本的diffusion model進行清除)
        )
        self.chosen_clip_models = ["ViT-B/32", "ViT-B/16", "RN50", "RN50x4"]

        # Clip相關
        self.clip_denoised = False  # clip是否要區分有噪音和沒有噪音的圖片
        self.clamp_grad = True  # 限制cond_fn中的梯度大小(避免產生一些極端生成結果)
        self.clamp_max = 0.05  # 限制的最大梯度

        # loss相關
        self.tv_scale = 0  # 控制最後輸出的平滑程度
        self.range_scale = 150  # 控制允許超出多遠的RGB值
        self.sat_scale = 0  # 控制允許多少飽和

        # 生成相關
        self.seed = None  # 亂數種子(會由anvil客戶端做第一次的初始化)
        self.batch_size = 1  # 一次要sample的數量
        self.num_batches = 1  # 要生成的圖片數量
        self.use_augmentation = True  # 是否要做圖片的augmentation

    def get_seed(self):
        """
        生成種子
        """
        random.seed()
        return random.randint(0, INT_MAX)

    def adjust_settings(
        self,
        width=960,
        height=768,
        cutn_batches=4,
        cut_overview=[12] * 400 + [4] * 600,
        cut_innercut=[4] * 400 + [12] * 600,
        cut_ic_pow=1,
        cut_icgray_p=[0.2] * 400 + [0] * 600,
        use_secondary_model=True,
        chosen_clip_models=["ViT-B/32", "ViT-B/16", "RN50", "RN50x4"],
        clip_denoised=False,
        clamp_grad=True,
        clamp_max=0.05,
        tv_scale=0,
        range_scale=150,
        sat_scale=0,
        batch_size=1,
        num_batches=1,
        use_augmentation=True,
    ):
        """
        調整設定
        """

        self.width = width
        self.height = height
        self.num_cutout_batches = cutn_batches
        self.overview_cut_schedule = cut_overview
        self.inner_cut_schedule = cut_innercut
        self.inner_cut_size_pow = cut_ic_pow
        self.cut_gray_portion_schedule = cut_icgray_p
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
        self.use_augmentation = use_augmentation


config = _Config()
