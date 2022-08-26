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
    chosen_clip_models = ("ViT-B/32", "ViT-B/16", "ViT-L/14", "RN101")  # 要選擇的Clip模型
    chosen_predictors = ("ViT-B/32", "ViT-B/16", "ViT-L/14")  # 要選擇的aesthetic predictor

    # 梯度相關
    grad_threshold = 0.05  # 限制的最大與最小的梯度(越高顏色會越明亮、增加對比與細節，但同時會提高出現極端結果的可能)

    # loss相關
    clip_guidance_scale = 8000  # clip引導的強度(生成圖片要多接近prompt)
    LPIPS_scale = 1000  # 調整perceptual loss影響程度
    aesthetic_scale = 0  # 調整aesthetic loss影響程度

    @classmethod
    def update(
        cls,
        width=768,
        height=512,
        num_cutout_batches=4,
        chosen_clip_models=("ViT-B/32", "ViT-B/16", "ViT-L/14", "RN101"),
        chosen_predictors=("ViT-B/32", "ViT-B/16", "ViT-L/14"),
        grad_threshold=0.05,
        clip_guidance_scale=8000,
        LPIPS_scale=1000,
        aesthetic_scale=0,
    ):
        """
        更新設定
        """

        cls.width = (width // 64) * 64  # 調整成64的倍數
        cls.height = (height // 64) * 64
        cls.num_cutout_batches = num_cutout_batches
        cls.chosen_clip_models = chosen_clip_models
        cls.chosen_predictors = chosen_predictors
        cls.grad_threshold = grad_threshold
        cls.clip_guidance_scale = clip_guidance_scale
        cls.LPIPS_scale = LPIPS_scale
        cls.aesthetic_scale = aesthetic_scale
