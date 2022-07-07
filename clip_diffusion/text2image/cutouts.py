import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from resize_right import resize

# 參考並修改自：disco diffusion
# 作者： Dango233(https://github.com/Dango233)
class MakeCutouts(nn.Module):
    """
    在Clip引導時取出圖片(cutout)做處理，讓引導效果更好
    overview cut: 取出整張圖片，有助於整體線條
    inner cut: 只取出圖片中指定大小的部分(crop)，有助於細節調整
    """

    def __init__(
        self,
        cut_size,
        overview,
        inner_cut,
        inner_cut_size_pow,
        cut_gray_portion,
        use_augmentations,
    ):
        super().__init__()
        self.cut_size = cut_size  # 要取的inner cut圖片大小(對應Clip模型的input resolution)
        self.overview = overview  # 要做的overview cut次數
        self.inner_cut = inner_cut  # 要做的inner cut次數
        self.inner_cut_size_pow = inner_cut_size_pow  # inner cut size的指數
        self.cut_gray_portion = cut_gray_portion  # 要做灰階化的cut比例
        self.use_augmentations = use_augmentations  # 是否要對cutout圖片使用augmentations
        self._augmentations = T.Compose(
            [
                T.RandomHorizontalFlip(p=0.5),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomAffine(
                    degrees=10,
                    translate=(0.05, 0.05),
                    interpolation=T.InterpolationMode.BILINEAR,
                ),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.RandomGrayscale(p=0.1),
                T.Lambda(lambda x: x + torch.randn_like(x) * 0.01),
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
            ]
        )  # 要做的image augmentation

    def forward(self, input):
        cutouts = []
        gray = T.Grayscale(3)
        side_y, side_x = input.shape[2:4]
        max_size = min(side_x, side_y)
        min_size = min(side_x, side_y, self.cut_size)
        output_shape = [1, 3, self.cut_size, self.cut_size]
        pad_input = F.pad(
            input,
            ((side_y - max_size) // 2, (side_y - max_size) // 2, (side_x - max_size) // 2, (side_x - max_size) // 2),
        )
        cutout = resize(pad_input, out_shape=output_shape)

        # 做overview cut
        if self.overview > 0:
            if self.overview <= 4:
                if self.overview >= 1:
                    cutouts.append(cutout)
                if self.overview >= 2:
                    cutouts.append(gray(cutout))
                if self.overview >= 3:
                    cutouts.append(TF.hflip(cutout))
                if self.overview == 4:
                    cutouts.append(gray(TF.hflip(cutout)))
            else:
                cutout = resize(pad_input, out_shape=output_shape)
                for _ in range(self.overview):
                    cutouts.append(cutout)

        # 做inner cut
        if self.inner_cut > 0:
            for i in range(self.inner_cut):
                size = int(torch.rand([]) ** self.inner_cut_size_pow * (max_size - min_size) + min_size)
                offsetx = torch.randint(0, side_x - size + 1, ())
                offsety = torch.randint(0, side_y - size + 1, ())
                cutout = input[:, :, offsety : offsety + size, offsetx : offsetx + size]
                if i <= int(self.cut_gray_portion * self.inner_cut):
                    cutout = gray(cutout)
                cutout = resize(cutout, out_shape=output_shape)
                cutouts.append(cutout)

        cutouts = torch.cat(cutouts)

        # 對cutout的圖片做augmentation
        if self.use_augmentations:
            cutouts = self._augmentations(cutouts)

        return cutouts
