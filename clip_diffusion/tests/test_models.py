import unittest
import torch
from torch import nn
from realesrgan.utils import RealESRGANer
from clip_diffusion.config import config
from clip_diffusion.models import (
    load_clip_models_and_preprocessings,
    load_guided_diffusion_model,
    load_secondary_model,
    load_latent_diffusion_model,
    load_real_esrgan_upsampler,
)


class TestModels(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_clip_models(self):
        """
        測試clip_models是否正常載入
        """

        print("testing clip models")

        models, _ = load_clip_models_and_preprocessings(config.chosen_clip_models, self.device)
        for model in models:
            self.assertIsInstance(model, nn.Module)

    def test_guided_diffusion_model(self):
        """
        測試guided_diffusion_model是否正常載入
        """

        print("testing guided diffusion model")

        model, diffusion = load_guided_diffusion_model(steps=200, device=self.device)
        self.assertIsInstance(model, nn.Module)

    def test_secondary_model(self):
        """
        測試secondary_model是否正常載入
        """

        print("testing secondary model")

        model = load_secondary_model(self.device)
        self.assertIsInstance(model, nn.Module)

    def test_latent_diffusion_model(self):
        """
        測試latent_diffusion_model是否正常載入
        """

        print("testing latent diffusion model")

        model = load_latent_diffusion_model(self.device)
        self.assertIsInstance(model, nn.Module)

    def test_real_esrgan_upsampler(self):
        """
        測試real_esrgan_upsampler是否正常載入
        """

        print("testing real esrgan upsampler")

        upsampler = load_real_esrgan_upsampler(self.device)
        self.assertIsInstance(upsampler, RealESRGANer)


if __name__ == "__main__":
    unittest.main()
