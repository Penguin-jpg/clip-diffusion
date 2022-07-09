from torch.nn import functional as F

# 來源：https://colab.research.google.com/drive/1QBsaDAZv8np29FPbvjffbE1eytoJcsgA#scrollTo=YHOj78Yvx8jP


def L2_norm(input, dim=-1):
    """
    對input的dim做L2 norm
    """

    return F.normalize(input, dim=dim)


def spherical_dist_loss(x, y):
    """計算大圓距離"""

    # 求x, y最後一個維度的L2 norm(歐基里德距離)
    x = L2_norm(x, dim=-1)
    y = L2_norm(y, dim=-1)
    # 套入公式(作者有稍微對原式做修改)
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


def tv_loss(input):
    """計算L2 total variation loss，用來做總變差去躁"""

    input = F.pad(input, (0, 1, 0, 1), "replicate")
    x_diff = input[..., :-1, 1:] - input[..., :-1, :-1]
    y_diff = input[..., 1:, :-1] - input[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3])


def range_loss(input):
    """控制RGB允許的範圍"""

    return (input - input.clamp(-1, 1)).pow(2).mean([1, 2, 3])
