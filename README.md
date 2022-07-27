## Clip Diffusion

### Dependency For Generation
```
git clone https://github.com/crowsonkb/guided-diffusion.git
git clone https://github.com/CompVis/latent-diffusion.git
git clone https://github.com/Penguin-jpg/taming-transformers.git
pip install -e ./guided-diffusion
pip install -e ./latent-diffusion
pip install -e ./taming-transformers
pip install -r requirements/generation.txt
```

### Dependency For CLIP Query
```
pip install -r requirements/clip_query.txt
```

### 簡介
- 使用 CLIP 引導 Guided Diffusion 進行 text-to-image generation，並且融合 Latent Diffusion 生成初始圖片以獲得更棒的結果
- codebase 大量參考並修改自以下作者的 github gist、repository 和 colab notebook
  1. [Disco Diffusion](https://github.com/alembics/disco-diffusion)
  2. [Katherine Crowson](https://github.com/crowsonkb)
  3. [perlin noise pytorch 實作](https://gist.github.com/adefossez/0646dbe9ed4005480a2407c62aac8869)
  4. [dalle-pytorch](https://github.com/lucidrains/DALLE-pytorch)
  5. [clip-guided-diffusion](https://github.com/afiaka87/clip-guided-diffusion)
  6. [CLIP Guided Diffusion HQ 512x512 Uncond.ipynb](https://colab.research.google.com/drive/1QBsaDAZv8np29FPbvjffbE1eytoJcsgA)
  7. [VQGAN+CLIP](https://colab.research.google.com/drive/1go6YwMFe5MX6XM9tv-cnQiSTU50N9EeT?fbclid=IwAR30ZqxIJG0-2wDukRydFA3jU5OpLHrlC_Sg1iRXqmoTkEhaJtHdRi6H7AI)
  8.  [latent diffusion 使用方式](https://huggingface.co/spaces/multimodalart/latentdiffusion)
- 使用 [anvil](https://anvil.works/) 撰寫網頁前端
- 使用 colab notebook 當作伺服器
- 關鍵字資料來源：https://docs.google.com/spreadsheets/d/1j7zaDi_PkndizQ2pL8B_yMcwfKUdE6tSMhL31bYtJNs/edit#gid=0