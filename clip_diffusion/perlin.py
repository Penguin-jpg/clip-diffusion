import torch
from torchvision.transforms import functional as TF
from PIL import ImageOps
from clip_diffusion.config import config
from clip_diffusion.utils.image_utils import normalize_image_neg_one_to_one, image_to_tensor, tensor_to_pillow_image

# 維持disco diffusion所採用的二維perlin noise
# 參考並修改自
# 1. https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869
# 2. disco diffusion


def _fade_power_3(t):
    """
    三次方淡出公式，在一階微分時代入t=0和t=1皆為0
    """
    return 3 * t**2 - 2 * t**3


def _fade_power_5(t):
    """ "
    五次方淡出公式，在一階和二階微分時代入t=0和t=1皆為0
    """
    return 6 * t**5 - 15 * t**4 + 10 * t**3


def _perlin(power, width, height, scale=10, device=None):
    """
    計算二維的perlin noise
    """

    assert power in (3, 5), "power must be 3 or 5"

    if power == 5:
        fade = _fade_power_5
    else:
        fade = _fade_power_3

    # 隨機展生的梯度
    gx, gy = torch.randn(2, width + 1, height + 1, 1, 1, device=device)
    # 二維平面上的4個向量(噪音圖的4個頂點)，這些向量帶有高度
    xs = torch.linspace(0, 1, scale + 1)[:-1, None].to(device)
    ys = torch.linspace(0, 1, scale + 1)[None, :-1].to(device)
    # joint fade function(x, y皆參與影響)
    # 越接近1影響力要越大時就用x(或y)
    # 越接近0影響力越大時就用1-x(或1-y)
    wx = 1 - fade(xs)
    wy = 1 - fade(ys)
    # 對4個頂點計算noise(wx,wy計算noise；gx,gy計算高度)
    dots = 0
    dots += wx * wy * (gx[:-1, :-1] * xs + gy[:-1, :-1] * ys)
    dots += (1 - wx) * wy * (-gx[1:, :-1] * (1 - xs) + gy[1:, :-1] * ys)
    dots += wx * (1 - wy) * (gx[:-1, 1:] * xs - gy[:-1, 1:] * (1 - ys))
    dots += (1 - wx) * (1 - wy) * (-gx[1:, 1:] * (1 - xs) - gy[1:, 1:] * (1 - ys))
    return dots.permute(0, 2, 1, 3).contiguous().view(width * scale, height * scale)


def _perlin_ms(octaves, width, height, grayscale, device=None):
    """
    透過疊加增進perlin noise的隨機性
    """

    out_array = [0.5] if grayscale else [0.5, 0.5, 0.5]
    for i in range(1 if grayscale else 3):
        scale = 2 ** len(octaves)
        oct_width = width
        oct_height = height
        # 對每個生成的square都要做
        for oct in octaves:
            # 0~1之間的值，決定東西會多快減小
            p = _perlin(5, oct_width, oct_height, scale, device)
            out_array[i] += p * oct
            scale //= 2
            # 以每次兩倍的頻率上升
            oct_width *= 2
            oct_height *= 2
    return torch.cat(out_array)


def _perlin_noise_wrapper(octaves=[1, 1, 1, 1], width=2, height=2, grayscale=True, device=None):
    """
    整合上方的function
    """

    out = _perlin_ms(octaves, width, height, grayscale, device)

    if grayscale:  # 灰階
        out = TF.resize(size=(config.height, config.width), img=out.unsqueeze(0))
        out = tensor_to_pillow_image(out.clamp(0, 1)).convert("RGB")
    else:  # 有顏色
        out = out.reshape(-1, 3, out.shape[0] // 3, out.shape[1])
        out = TF.resize(size=(config.height, config.width), img=out)
        out = tensor_to_pillow_image(out.clamp(0, 1).squeeze())

    out = ImageOps.autocontrast(out)
    return out


def generate_perlin_noise(perlin_mode, device=None):
    """
    生成perlin noise
    """

    if perlin_mode == "color":
        init = _perlin_noise_wrapper([1.5**-i * 0.5 for i in range(12)], 1, 1, False, device)
        init2 = _perlin_noise_wrapper([1.5**-i * 0.5 for i in range(8)], 4, 4, False, device)
    elif perlin_mode == "gray":
        init = _perlin_noise_wrapper([1.5**-i * 0.5 for i in range(12)], 1, 1, True, device)
        init2 = _perlin_noise_wrapper([1.5**-i * 0.5 for i in range(8)], 4, 4, True, device)
    else:
        init = _perlin_noise_wrapper([1.5**-i * 0.5 for i in range(12)], 1, 1, False, device)
        init2 = _perlin_noise_wrapper([1.5**-i * 0.5 for i in range(8)], 4, 4, True, device)

    init = image_to_tensor(init).add(image_to_tensor(init2)).div(2).to(device).unsqueeze(0)
    init = normalize_image_neg_one_to_one(init)
    del init2
    return init
