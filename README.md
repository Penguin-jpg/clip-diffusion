## Clip Diffusion

### Dependency For Generation
```
git clone https://github.com/openai/CLIP
git clone https://github.com/crowsonkb/guided-diffusion
git clone https://github.com/Penguin-jpg/ResizeRight.git
git clone https://github.com/CompVis/latent-diffusion.git
git clone https://github.com/CompVis/taming-transformers.git
git clone https://github.com/Penguin-jpg/BSRGAN.git
pip install -e ./CLIP
pip install -e ./guided-diffusion
pip install -e ./resize-right
pip install -e ./latent-diffusion
pip install -e ./taming-transformers
pip install -e ./BSRGAN
pip install -r requirements/generation.txt
```

### Dependency For Dataset Images
```
pip install -r requirements/dataset_images.txt
```

### Dependency For Instance Segmentation
```
pip install -r requirements/instance_segmentation.txt
git clone https://github.com/Penguin-jpg/mmdetection.git
pip install -e ./mmdetection
# 如果需要手動標註資料才需額外安裝
# pip install labelme
# 如果要訓練 Instaboost 模型才需額外安裝
# pip install instaboostfast
```

### 簡介
- 使用 CLIP 引導 Guided Diffusion 進行 text-to-image generation，並且
  融合 Latent Diffusion 生成初始圖片
- 主要參考並改寫、整合自以下作者、repository 或 colab notebook：
  1. [Disco Diffusion](https://github.com/alembics/disco-diffusion)
  2. [Katherine Crowson](https://github.com/crowsonkb)
  3. [dalle-pytorch](https://github.com/lucidrains/DALLE-pytorch)
  4. [clip-guided-diffusion](https://github.com/afiaka87/clip-guided-diffusion)
  5. [CLIP Guided Diffusion HQ 512x512 Uncond.ipynb](https://colab.research.google.com/drive/1QBsaDAZv8np29FPbvjffbE1eytoJcsgA)
  6. [perlin-noise](https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869)
  7. [VQGAN+CLIP](https://colab.research.google.com/drive/1go6YwMFe5MX6XM9tv-cnQiSTU50N9EeT?fbclid=IwAR30ZqxIJG0-2wDukRydFA3jU5OpLHrlC_Sg1iRXqmoTkEhaJtHdRi6H7AI)
  8. [latent diffusion huggingface](https://huggingface.co/spaces/multimodalart/latentdiffusion)
- 使用 [anvil](https://anvil.works/) 撰寫網頁前端及伺服器功能