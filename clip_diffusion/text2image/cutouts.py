import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision.transforms import functional as TF
from resize_right import resize
from clip_diffusion.utils.image_utils import unnormalize_image_zero_to_one
from clip_diffusion.utils.functional import embed_image

# 作者： Dango233(https://github.com/Dango233)
class Cutouts(nn.Module):
    """
    在Clip引導時取出圖片(cutout)做處理，讓引導效果更好
    overview cut: 取出整張圖片，有助於整體線條
    inner cut: 只取出圖片中指定大小的部分(crop)，有助於細節調整
    """

    def __init__(
        self,
        cut_size,
        num_overview_cuts,
        num_inner_cuts,
        inner_cut_size_power,
        cut_gray_portion,
    ):
        super().__init__()
        self.cut_size = cut_size  # 要取的inner cut圖片大小(對應Clip模型的input resolution)
        self.num_overview_cuts = num_overview_cuts  # 要做的overview cut次數
        self.num_inner_cuts = num_inner_cuts  # 要做的inner cut次數
        self.inner_cut_size_power = inner_cut_size_power  # inner cut size的指數
        self.cut_gray_portion = cut_gray_portion  # 要做灰階化的cut比例
        self.augmentations = T.Compose(
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
        cutout_images = []
        gray = T.Grayscale(3)
        height, width = input.shape[2:4]
        shorter_side = min(width, height)
        min_size = min(width, height, self.cut_size)
        output_shape = [1, 3, self.cut_size, self.cut_size]
        pad_input = F.pad(
            input,
            (
                (height - shorter_side) // 2,
                (height - shorter_side) // 2,
                (width - shorter_side) // 2,
                (width - shorter_side) // 2,
            ),
        )  # pad到與最長的邊等大，假設768*512就是pad到(1, 3, 768, 768)
        # 將整張圖片重新resize成(1, 3, cut_size, cut_size)，用於overview cut
        cut_size_input = resize(pad_input, out_shape=output_shape)

        # 做overview cut
        if self.num_overview_cuts > 0:
            if self.num_overview_cuts <= 4:
                if self.num_overview_cuts >= 1:
                    cutout_images.append(cut_size_input)
                if self.num_overview_cuts >= 2:
                    cutout_images.append(gray(cut_size_input))
                if self.num_overview_cuts >= 3:
                    cutout_images.append(TF.hflip(cut_size_input))
                if self.num_overview_cuts == 4:
                    cutout_images.append(gray(TF.hflip(cut_size_input)))
            else:
                for _ in range(self.num_overview_cuts):
                    cutout_images.append(cut_size_input)

        # 做inner cut
        if self.num_inner_cuts > 0:
            for i in range(self.num_inner_cuts):
                innert_cut_size = int(torch.rand([]) ** self.inner_cut_size_power * (shorter_side - min_size) + min_size)
                width_offset = torch.randint(
                    0, width - innert_cut_size + 1, ()
                )  # 從0~(width - inner_cut_size +1)中隨機選一個數字當cutout寬度的offset
                height_offset = torch.randint(
                    0, height - innert_cut_size + 1, ()
                )  # 從0~(height - innert_cut_size +1)中隨機選一個數字當cutout高度的offset

                # 取input高度介於[y_offset, y_offset + size)；寬度介於[x_offset, x_offset + size)的區域，用於inner cut
                inner_cut_image = input[
                    :, :, height_offset : height_offset + innert_cut_size, width_offset : width_offset + innert_cut_size
                ]

                if i <= int(self.cut_gray_portion * self.num_inner_cuts):
                    inner_cut_image = gray(inner_cut_image)

                inner_cut_image = resize(inner_cut_image, out_shape=output_shape)  # 重新resize成(1, 3, cut_size, cut_size)
                cutout_images.append(inner_cut_image)

        # 將所有cutout圖片相接
        cutout_images = torch.cat(cutout_images)  # shape=(num_cuts, num_channels, cut_size, cut_size)

        # 對cutout的圖片做augmentation
        cutout_images = self.augmentations(cutout_images)

        return cutout_images


def make_cutouts(
    clip_model,
    input,
    cut_size,
    num_overview_cuts,
    num_inner_cuts,
    inner_cut_size_power,
    cut_gray_portion,
):
    """
    對目前生成圖片做cutout
    """

    cutouts = Cutouts(
        cut_size=cut_size,
        num_overview_cuts=num_overview_cuts,
        num_inner_cuts=num_inner_cuts,
        inner_cut_size_power=inner_cut_size_power,
        cut_gray_portion=cut_gray_portion,
    )
    cutout_images = cutouts(unnormalize_image_zero_to_one(input))
    return embed_image(clip_model, cutout_images, use_clip_normalize=True)
