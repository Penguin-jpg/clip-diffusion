import random

_INT_MAX = 2**32


class Config:
    """
    儲存全域設定
    """

    def __init__(self):
        """
        初始化預設值
        """

        # 圖片長寬相關(一定要是64的倍數)
        self.width = 960  # 生成圖片的寬度
        self.height = 768  # 　生成圖片的高度

        # cutout相關
        self.num_cutout_batches = 4  # 要做的cutout次數
        # overview cutout的schedule，以1000當作基準，前200/1000個step會做14次cutout中間200/1000個step會做12次cutout，以此類推(建議一開始高，隨著過程逐漸降低)
        self.overview_cut_schedule = self.create_schedule(values=(14, 12, 4, 0), steps=(200, 200, 400, 200))
        # inner cutout的schedule(建議一開始低，隨著過程逐漸升高)
        self.inner_cut_schedule = self.create_schedule(values=(2, 4, 2, 12), steps=(200, 200, 400, 200))
        # 控制inner cutout大小的schedule(越高會讓inner cutout圖片大小越接近Clip的解析度)
        self.inner_cut_size_power_schedule = self.create_schedule(values=(5,), steps=(1000,))
        # 控制多少百分比的cut要取出做灰階化(建議剛開始高，隨著過程逐漸降低)
        self.cut_gray_portion_schedule = self.create_schedule(values=(0.7, 0.6, 0.45, 0.3, 0), steps=(100, 100, 100, 100, 600))

        # model相關
        self.use_secondary_model = True  # 是否要使用secondary model(如果關閉的話則會用原本的diffusion model進行清除)
        self.chosen_clip_models = ("ViT-B/32", "ViT-B/16", "ViT-L/14", "RN101")  # 要選擇的Clip模型

        # 梯度相關
        self.clamp_max = 0.05  # 限制的最大梯度(越高顏色會越明亮、增加對比與細節，但同時會提高出現極端結果的可能)

        # loss相關
        self.tv_scale = 0  # 控制最後輸出的平滑程度
        self.range_scale = 150  # 控制允許超出多遠的RGB值
        self.sat_scale = 0  # 控制允許多少飽和

        # 生成相關
        self.seed = self.get_seed()  # 亂數種子
        self.use_augmentations = True  # 是否要做圖片的augmentation

    def get_seed(self):
        """
        生成種子
        """
        random.seed()
        return random.randint(0, _INT_MAX)

    def adjust_settings(
        self,
        width=960,
        height=768,
        num_cutout_batches=4,
        use_secondary_model=True,
        chosen_clip_models=("ViT-B/32", "ViT-B/16", "ViT-L/14", "RN101"),
        clamp_max=0.05,
        tv_scale=0,
        range_scale=150,
        sat_scale=0,
        use_augmentations=True,
    ):
        """
        調整設定
        """

        self.width = (width // 64) * 64  # 調整成64的倍數
        self.height = (height // 64) * 64
        self.num_cutout_batches = num_cutout_batches
        self.use_secondary_model = use_secondary_model
        self.chosen_clip_models = chosen_clip_models
        self.clamp_max = clamp_max
        self.tv_scale = tv_scale
        self.range_scale = range_scale
        self.sat_scale = sat_scale
        self.use_augmentations = use_augmentations

    def create_schedule(self, values, steps):
        """
        建立schedule:
        (values[0],) * steps[0] + (values[1],) * steps[1]...
        """

        assert len(values) == len(steps), "length of values and steps must be the same"

        schedule = ()

        for value, num_steps in zip(values, steps):
            schedule += (value,) * num_steps

        return schedule


config = Config()
