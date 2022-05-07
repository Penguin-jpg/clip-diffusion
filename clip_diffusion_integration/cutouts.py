import torch
from torch import nn
from torchvision import transforms
from ResizeRight.resize_right import resize
from .resample_utils import resample
from .config import skip_augs
from resize_right import resize

# 作者：Katherine Crowson(https://github.com/crowsonkb)
class MakeCutouts(nn.Module):
    def __init__(self, cut_size, cutn, skip_augs=False):
        super().__init__()
        self.cut_size = cut_size
        self.cutn = cutn
        self.skip_augs = skip_augs
        self.augs = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                transforms.RandomPerspective(distortion_scale=0.4, p=0.7),
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                transforms.RandomGrayscale(p=0.15),
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                # transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ]
        )

    def forward(self, input):
        input = transforms.Pad(input.shape[2] // 4, fill=0)(input)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)

        cutouts = []
        for ch in range(self.cutn):
            if ch > self.cutn - self.cutn // 4:
                cutout = input.clone()
            else:
                size = int(
                    max_size
                    * torch.zeros(
                        1,
                    )
                    .normal_(mean=0.8, std=0.3)
                    .clip(float(self.cut_size / max_size), 1.0)
                )
                offsetx = torch.randint(0, abs(sideX - size + 1), ())
                offsety = torch.randint(0, abs(sideY - size + 1), ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]

            if not self.skip_augs:
                cutout = self.augs(cutout)
            cutouts.append(resample(cutout, (self.cut_size, self.cut_size)))
            del cutout

        cutouts = torch.cat(cutouts, dim=0)
        return cutouts


# for dango cutouts
cutout_debug = False
padargs = {}

# 作者： Dango233
class MakeCutoutsDango(nn.Module):
    def __init__(
        self, cut_size, Overview=4, InnerCrop=0, IC_Size_Pow=0.5, IC_Grey_P=0.2
    ):
        super().__init__()
        self.cut_size = cut_size
        self.Overview = Overview
        self.InnerCrop = InnerCrop
        self.IC_Size_Pow = IC_Size_Pow
        self.IC_Grey_P = IC_Grey_P
        self.augs = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                transforms.RandomAffine(
                    degrees=10,
                    translate=(0.05, 0.05),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ),
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                transforms.RandomGrayscale(p=0.1),
                transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
                ),
            ]
        )

    def forward(self, input):
        cutouts = []
        gray = transforms.Grayscale(3)
        sideY, sideX = input.shape[2:4]
        max_size = min(sideX, sideY)
        min_size = min(sideX, sideY, self.cut_size)
        l_size = max(sideX, sideY)
        output_shape = [1, 3, self.cut_size, self.cut_size]
        output_shape_2 = [1, 3, self.cut_size + 2, self.cut_size + 2]
        pad_input = F.pad(
            input,
            (
                (sideY - max_size) // 2,
                (sideY - max_size) // 2,
                (sideX - max_size) // 2,
                (sideX - max_size) // 2,
            ),
            **padargs
        )
        cutout = resize(pad_input, out_shape=output_shape)

        if self.Overview > 0:
            if self.Overview <= 4:
                if self.Overview >= 1:
                    cutouts.append(cutout)
                if self.Overview >= 2:
                    cutouts.append(gray(cutout))
                if self.Overview >= 3:
                    cutouts.append(transforms.functional.hflip(cutout))
                if self.Overview == 4:
                    cutouts.append(gray(transforms.functional.hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.Overview):
                    cutouts.append(cutout)

            if cutout_debug:
                transforms.functional.to_pil_image(
                    cutouts[0].add(1).div(2).clamp(0, 1).squeeze(0)
                ).save("/content/cutout_overview.jpg", quality=99)

        if self.InnerCrop > 0:
            for i in range(self.InnerCrop):
                size = int(
                    torch.rand([]) ** self.IC_Size_Pow * (max_size - min_size)
                    + min_size
                )
                offsetx = torch.randint(0, sideX - size + 1, ())
                offsety = torch.randint(0, sideY - size + 1, ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                if i <= int(self.IC_Grey_P * self.InnerCrop):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)
            if cutout_debug:
                transforms.functional.to_pil_image(
                    cutouts[-1].add(1).div(2).clamp(0, 1).squeeze(0)
                ).save("/content/cutout_InnerCrop.jpg", quality=99)
        cutouts = torch.cat(cutouts)
        if skip_augs is not True:
            cutouts = self.augs(cutouts)
        return cutouts
