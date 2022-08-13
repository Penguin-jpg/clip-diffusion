import torch


def create_schedule(values, steps):
    """
    建立schedule:
    (values[0],) * steps[0] + (values[1],) * steps[1]...
    """

    assert len(values) == len(steps), "length of values and steps must be the same"

    schedule = ()

    for value, num_steps in zip(values, steps):
        schedule += (value,) * num_steps

    return schedule


class Config:
    """
    儲存全域設定
    """

    # device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 圖片長寬相關(一定要是64的倍數)
    width = 768  # 生成圖片的寬度
    height = 512  # 　生成圖片的高度

    # cutout相關
    num_cutout_batches = 4  # 要做的cutout次數
    # overview cutout的schedule，以1000當作基準，前200/1000個step會做14次cutout中間200/1000個step會做12次cutout，以此類推(建議一開始高，隨著過程逐漸降低)
    num_overview_cuts_schedule = create_schedule(values=(14, 12, 4, 0), steps=(200, 200, 400, 200))
    # inner cutout的schedule(建議一開始低，隨著過程逐漸升高)
    num_inner_cuts_schedule = create_schedule(values=(2, 4, 2, 12), steps=(200, 200, 400, 200))
    # 控制inner cutout大小的schedule(越高會讓inner cutout圖片大小越接近Clip的解析度)
    inner_cut_size_power_schedule = create_schedule(values=(5,), steps=(1000,))
    # 控制多少百分比的cut要取出做灰階化(建議剛開始高，隨著過程逐漸降低)
    cut_gray_portion_schedule = create_schedule(values=(0.7, 0.6, 0.45, 0.3, 0), steps=(100, 100, 100, 100, 600))

    # model相關
    use_secondary_model = True  # 是否要使用secondary model(如果關閉的話則會用原本的diffusion model進行清除)
    chosen_clip_models = ("ViT-B/32", "ViT-B/16", "ViT-L/14", "RN101")  # 要選擇的Clip模型

    # 梯度相關
    clamp_max = 0.05  # 限制的最大梯度(越高顏色會越明亮、增加對比與細節，但同時會提高出現極端結果的可能)

    # loss相關
    tv_scale = 0  # 控制最後輸出的平滑程度
    range_scale = 150  # 控制允許超出多遠的RGB值
    sat_scale = 0  # 控制允許多少飽和
    aesthetic_scale = 0  # 調整aesthetic loss影響程度

    @classmethod
    def change(
        cls,
        width=768,
        height=512,
        num_cutout_batches=4,
        use_secondary_model=True,
        chosen_clip_models=("ViT-B/32", "ViT-B/16", "ViT-L/14", "RN101"),
        clamp_max=0.05,
        tv_scale=0,
        range_scale=150,
        sat_scale=0,
        aesthetic_scale=0,
    ):
        """
        調整設定
        """

        cls.width = (width // 64) * 64  # 調整成64的倍數
        cls.height = (height // 64) * 64
        cls.num_cutout_batches = num_cutout_batches
        cls.use_secondary_model = use_secondary_model
        cls.chosen_clip_models = chosen_clip_models
        cls.clamp_max = clamp_max
        cls.tv_scale = tv_scale
        cls.range_scale = range_scale
        cls.sat_scale = sat_scale
        cls.aesthetic_scale = aesthetic_scale
