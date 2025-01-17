from torch.nn import functional as F
from pytorch_msssim import MS_SSIM
from clip_diffusion.utils.functional import L2_norm
from clip_diffusion.utils.image_utils import denormalize_image_zero_to_one


_ms_ssim_loss = MS_SSIM(win_size=11, win_sigma=1.5, data_range=1, size_average=True, channel=3)


def square_spherical_distance_loss(x, y):
    """計算spherical distance loss"""
    # 求x, y最後一個維度的L2 norm(歐基里德距離)
    x = L2_norm(x, dim=-1)
    y = L2_norm(y, dim=-1)
    # 套入公式
    return (x - y).norm(dim=-1).div(2).arcsin().pow(2).mul(2)


# 公式：https://blog.csdn.net/hanlin_tan/article/details/52448803
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


def LPIPS_loss(LPIPS_model, input, image):
    """計算input與image之間的perceptual loss"""
    return LPIPS_model(input, image)


def aesthetic_loss(predictor, input):
    """計算aesthetic score作為loss"""
    return predictor(L2_norm(input, dim=-1)).mean()


def structural_dissimilarity_loss(input, image):
    """計算input與image之間的structural_dissimilarity(透過MS SSIM)作為loss"""
    # 先將input及imagedenormalize回到[0, 1]
    input = denormalize_image_zero_to_one(input)
    image = denormalize_image_zero_to_one(image)
    # 使用1-ssim_loss以計算結構相異性(越小越好)
    return 1.0 - _ms_ssim_loss(input, image)
