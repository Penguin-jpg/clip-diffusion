from torch.nn import functional as F
from clip_diffusion.utils.functional import L2_norm

# 來源：https://colab.research.google.com/drive/1QBsaDAZv8np29FPbvjffbE1eytoJcsgA#scrollTo=YHOj78Yvx8jP


def square_spherical_distance_loss(x, y):
    """計算spherical distance loss"""

    # 求x, y最後一個維度的L2 norm(歐基里德距離)
    x = L2_norm(x, dim=-1)
    y = L2_norm(y, dim=-1)
    # 套入公式
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


# 公式：https://blog.csdn.net/qq_34622844/article/details/88846411
def total_variational_loss(input):
    """計算L2 total variation loss，用來消除雜訊"""

    # 把寬、高往外pad一格
    input = F.pad(input, (0, 1, 0, 1), "replicate")
    # x2-x1, x3-x2, ...
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    # y2-y1, y3-y2, ...
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff.pow(2) + y_diff.pow(2)).mean([1, 2, 3])


def rgb_range_loss(input):
    """控制RGB允許的範圍"""

    # 由於輸入模型內的圖像必須在-1到1之間，但計算過程中是有可能超出範圍的，
    # 透過計算這個loss來告訴模型把值拉回正常範圍
    return (input - input.clamp(min=-1, max=1)).pow(2).mean([1, 2, 3])


def aesthetic_loss(predictor, input):
    """計算aesthetic score作為loss"""

    score_sum = 0

    for i in range(input.shape[0]):
        score_sum += predictor(input[i].unsqueeze(0)).item()

    return score_sum / input.shape[0]
