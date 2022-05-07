import torch
import random

INT_MAX = 2 ** 32


def get_seed():
    random.seed()
    return random.randint(0, INT_MAX)


width = 0  # 生成圖片的寬度
height = 0  # 　生成圖片的高度
# resize用的x, y(一定要是64的倍數)
side_x = (width / 64) * 64
side_y = (height / 64) * 64
perlin_mode = "mixed"  # 使用的perlin模式
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # GPU或CPU
batch_size = 1  # 要生成幾張圖片
skip_augs = False  # 是否不做圖片的augmentation
seed = get_seed()  # 亂數種子
text_prompts = [
    "A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, Trending on artstation.",
    "yellow color scheme",
]  # 要生成的東西(可以將不同特徵分開寫)
randomize_class = True  # imagenet的class是否要每個iteration都隨機改變
clip_denoised = False  # clip是否要區分有噪音和沒有噪音的圖片
fuzzy_prompt = False  # 是否要加入multiple noisy prompts到prompt losses內
rand_mag = 0.05  # 控制隨機噪音的強度
init_image = None  # 初始化圖片(能幫助生成成果)
perlin_init = False  # 是否要使用隨機的perlin噪音
use_secondary_model = True  # 是否要使用secondary model輔助生成結果
timestep_respacing = "ddim100"  # 減少timestep的數量
diffusion_steps = 1000  # diffusion要跑的step數
use_checkpoint = True  # 是否要使用model checkpoint
steps = 250  # 每個iteration要跑的step數
model_path = "models"  # 模型存放路徑
