import torch
import math
from torch.nn import functional as F

# 來源：https://colab.research.google.com/drive/1go6YwMFe5MX6XM9tv-cnQiSTU50N9EeT?fbclid=IwAR30ZqxIJG0-2wDukRydFA3jU5OpLHrlC_Sg1iRXqmoTkEhaJtHdRi6H7AI#scrollTo=EXMSuW2EQWsd


def conditional_sin(x):
    """
    條件式的sin，當x不為0時就填入sin(pi/x)/pi*x，否則就填入1
    """
    return torch.where(x != 0, torch.sin(math.pi * x) / (math.pi * x), x.new_ones([]))


def lanczos(x, a):
    """
    計算lanczos kernel
    """
    # 條件：-a < x < a且x!=0
    condition = torch.logical_and(-a < x, x < a)
    # 代入公式
    out = torch.where(
        condition, conditional_sin(x) * conditional_sin(x / a), x.new_zeros([])
    )
    return out / out.sum()


def ramp(ratio, width):
    """
    不確定這是什麼
    """
    n = math.ceil(width / ratio + 1)
    out = torch.empty([n])
    cur = 0
    for i in range(out.shape[0]):
        out[i] = cur
        cur += ratio
    return torch.cat([-out[1:].flip([0]), out])[1:-1]


def resample(input, size, align_corners=True):
    """
    lanczos resample
    """
    n, c, h, w = input.shape
    dh, dw = size

    input = input.reshape([n * c, 1, h, w])

    if dh < h:
        kernel_h = lanczos(ramp(dh / h, 2), 2).to(input.device, input.dtype)
        pad_h = (kernel_h.shape[0] - 1) // 2
        input = F.pad(input, (0, 0, pad_h, pad_h), "reflect")
        input = F.conv2d(input, kernel_h[None, None, :, None])

    if dw < w:
        kernel_w = lanczos(ramp(dw / w, 2), 2).to(input.device, input.dtype)
        pad_w = (kernel_w.shape[0] - 1) // 2
        input = F.pad(input, (pad_w, pad_w, 0, 0), "reflect")
        input = F.conv2d(input, kernel_w[None, None, None, :])

    input = input.reshape([n, c, h, w])
    return F.interpolate(input, size, mode="bicubic", align_corners=align_corners)
