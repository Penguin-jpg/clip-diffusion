## Clip Diffusion

### Dependency
```
git clone https://github.com/openai/CLIP
git clone https://github.com/crowsonkb/guided-diffusion
git clone https://github.com/assafshocher/ResizeRight.git
git clone https://github.com/CompVis/latent-diffusion.git
git clone https://github.com/CompVis/taming-transformers
git clone https://github.com/Penguin-jpg/BSRGAN.git
pip install -e ./CLIP
pip install -e ./guided-diffusion
pip install -e ./latent-diffusion
pip install -e ./taming-transformers
pip install -e ./BSRGAN
pip install -r requirements.txt
```

### 簡介
- 這個專題使用了 CLIP 引導 Diffusion 進行 text-to-image generation
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
- 因為 anvil 不能序列化任意物件，為了配合，程式碼的相依性會比較高