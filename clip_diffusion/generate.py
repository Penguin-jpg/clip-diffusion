import torch
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import random
import numpy as np
import clip
import matplotlib.pyplot as plt
import os
import gc
import lpips
from PIL import Image
from tqdm.notebook import tqdm
from ipywidgets import Output
from IPython import display
from datetime import datetime
from glob import glob
from .config import config
from .prompt_utils import parse_prompt
from .perlin_utils import regen_perlin, regen_perlin_no_expand
from .clip_utils import *
from .secondary_model import *
from .diffusion_model import load_model_and_diffusion
from .cutouts import MakeCutouts, MakeCutoutsDango
from .loss import *
from .dir_utils import *
from .image_utils import *

# 參考並修改自：disco diffusion

lpips_model = lpips.LPIPS(net="vgg").to(config.device)  # LPIPS model
normalize = T.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
)


def set_seed(seed):
    """
    設定種子
    """

    if seed:
        np.random.seed(config.seed)
        random.seed(config.seed)
        torch.manual_seed(config.seed)
        torch.cuda.manual_seed_all(config.seed)
        torch.backends.cudnn.deterministic = True


def get_embedding_and_weights(text_prompts):
    """
    取得prompt的embedding及weight
    """

    # target_embeds, weights = [], []

    model_stats = []

    for clip_model in clip_models:
        model_stat = {
            "clip_model": None,
            "target_embeds": [],
            "make_cutouts": None,
            "weights": [],
        }
        model_stat["clip_model"] = clip_model

        for prompt in text_prompts:
            text, weight = parse_prompt(prompt)  # 取得text及weight
            text = clip_model.encode_text(
                clip.tokenize(prompt).to(config.device)
            ).float()

            if config.fuzzy_prompt:
                for i in range(25):
                    model_stat["target_embeds"].append(
                        (text + torch.randn(text.shape).cuda() * config.rand_mag).clamp(
                            0, 1
                        )
                    )
                    model_stat["weights"].append(weight)
            else:
                model_stat["target_embeds"].append(text)
                model_stat["weights"].append(weight)

        model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
        model_stat["weights"] = torch.tensor(
            model_stat["weights"], device=config.device
        )

        if model_stat["weights"].sum().abs() < 1e-3:
            raise RuntimeError("The weights must not sum to 0.")

        model_stat["weights"] /= model_stat["weights"].sum().abs()
        model_stats.append(model_stat)

    return model_stats


def generate(
    text_prompts=[
        "A beautiful painting of a singular lighthouse, shining its light across a tumultuous sea of blood by greg rutkowski and thomas kinkade, trending on artstation.",
    ],
    init_image=None,
    use_perlin=False,
    perlin_mode="mixed",
    batch_name="diffusion",
    chosen_clip_models=chosen_models,
):
    """
    生成圖片
    text_prompts: 要生成的東西(第一個item寫敘述，後續的item寫特徵)
    init_image: 模型會參考該圖片生成初始噪聲
    use_perlin: 是否要使用perlin noise
    perlin_mode: 使用的perlin noise模式
    batch_name: 本次生成的名稱
    chosen_clip_models: 選擇要使用的Clip模型
    """

    model, diffusion = load_model_and_diffusion()
    batch_folder = f"{out_dir_path}/{batch_name}"  # 儲存設定的資料夾
    make_dir(batch_folder)
    batch_num = len(glob(batch_folder + "/*.txt"))

    while os.path.isfile(
        f"{batch_folder}/{batch_name}({batch_num})_settings.txt"
    ) or os.path.isfile(f"{batch_folder}/{batch_name}-{batch_num}_settings.txt"):
        batch_num += 1

    # 載入選擇的Clip模型
    choose_clip_models(chosen_clip_models)

    loss_values = []

    # 設定種子
    set_seed(config.seed)

    # 取得prompt的embedding及weight
    model_stats = get_embedding_and_weights(text_prompts)

    init = None  # init_image或perlin noise只能擇一

    # 如果初始圖片不為空
    if init_image is not None:
        # 透過anvil傳來的image_file的bytes開啟圖片
        init = get_image_from_bytes(init_image.get_bytes()).convert("RGB")
        init = init.resize((config.side_x, config.side_y), Image.LANCZOS)
        init = TF.to_tensor(init).to(config.device).unsqueeze(0).mul(2).sub(1)

    # 使用perlin noise
    if use_perlin:
        init = regen_perlin_no_expand(perlin_mode)

    cur_t = None

    # 透過clip引導guided diffusion
    def cond_fn(x, t, y=None):
        with torch.enable_grad():
            x_is_NaN = False
            x = x.detach().requires_grad_()
            n = x.shape[0]

            if config.use_secondary_model:
                alpha = torch.tensor(
                    diffusion.sqrt_alphas_cumprod[cur_t],
                    device=config.device,
                    dtype=torch.float32,
                )
                sigma = torch.tensor(
                    diffusion.sqrt_one_minus_alphas_cumprod[cur_t],
                    device=config.device,
                    dtype=torch.float32,
                )
                cosine_t = alpha_sigma_to_t(alpha, sigma)
                out = secondary_model(x, cosine_t[None].repeat([n])).pred
                fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                x_in = out * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)
            else:
                my_t = torch.ones([n], device=config.device, dtype=torch.long) * cur_t
                out = diffusion.p_mean_variance(
                    model, x, my_t, clip_denoised=False, model_kwargs={"y": y}
                )
                fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                x_in = out["pred_xstart"] * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)

            for model_stat in model_stats:
                for i in range(config.cutn_batches):
                    t_int = (
                        int(t.item()) + 1
                    )  # errors on last step without +1, need to find source
                    input_resolution = model_stat["clip_model"].visual.input_resolution
                    cuts = MakeCutoutsDango(
                        input_resolution,
                        Overview=config.cut_overview[1000 - t_int],
                        InnerCrop=config.cut_innercut[1000 - t_int],
                        IC_Size_Pow=config.cut_ic_pow,
                        IC_Grey_P=config.cut_icgray_p[1000 - t_int],
                    )
                    clip_in = normalize(cuts(x_in.add(1).div(2)))
                    image_embeds = (
                        model_stat["clip_model"].encode_image(clip_in).float()
                    )
                    dists = spherical_dist_loss(
                        image_embeds.unsqueeze(1),
                        model_stat["target_embeds"].unsqueeze(0),
                    )
                    dists = dists.view(
                        [
                            config.cut_overview[1000 - t_int]
                            + config.cut_innercut[1000 - t_int],
                            n,
                            -1,
                        ]
                    )
                    losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                    loss_values.append(
                        losses.sum().item()
                    )  # log loss, probably shouldn't do per cutn_batch
                    x_in_grad += (
                        torch.autograd.grad(
                            losses.sum() * config.clip_guidance_scale, x_in
                        )[0]
                        / config.cutn_batches
                    )
            tv_losses = tv_loss(x_in)

            if config.use_secondary_model:
                range_losses = range_loss(out)
            else:
                range_losses = range_loss(out["pred_xstart"])

            sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
            loss = (
                tv_losses.sum() * config.tv_scale
                + range_losses.sum() * config.range_scale
                + sat_losses.sum() * config.sat_scale
            )

            # numpy array, tensor的判斷式使用is not None
            if init is not None and config.init_scale:
                init_losses = lpips_model(x_in, init)
                loss = loss + init_losses.sum() * config.init_scale
            x_in_grad += torch.autograd.grad(loss, x_in)[0]

            if not torch.isnan(x_in_grad).any():
                grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
            else:
                x_is_NaN = True
                grad = torch.zeros_like(x)

        if config.clamp_grad and not x_is_NaN:
            magnitude = grad.square().mean().sqrt()
            return (
                grad
                * magnitude.clamp(min=-config.clamp_max, max=config.clamp_max)
                / magnitude
            )  # min=-0.02,

        return grad

    image_display = Output()

    for i in range(config.num_batches):
        display.clear_output(wait=True)
        batchBar = tqdm(range(config.num_batches), desc="Batches")
        batchBar.n = i
        batchBar.refresh()
        print("")
        display.display(image_display)
        gc.collect()
        torch.cuda.empty_cache()
        cur_t = diffusion.num_timesteps - config.skip_timesteps - 1
        total_steps = cur_t

        if use_perlin:
            init = regen_perlin(perlin_mode)

        # 使用DDIM進行sample
        samples = diffusion.ddim_sample_loop_progressive(
            model,
            (config.batch_size, 3, config.side_y, config.side_x),
            clip_denoised=config.clip_denoised,
            model_kwargs={},
            cond_fn=cond_fn,
            progress=True,
            skip_timesteps=config.skip_timesteps,
            init_image=init,
            randomize_class=config.randomize_class,
            eta=config.eta,
        )

        for j, sample in enumerate(samples):
            cur_t -= 1
            intermediate_step = False
            if j in config.intermediate_saves:
                intermediate_step = True

            with image_display:
                if j % config.display_rate == 0 or cur_t == -1 or intermediate_step:
                    for k, image in enumerate(sample["pred_xstart"]):
                        # current_time = datetime.now().strftime("%y%m%d-%H%M%S_%f")
                        # percent = math.ceil(j / total_steps * 100)

                        if config.num_batches > 0:
                            if cur_t == -1:
                                filename = f"{batch_name}({batch_num})_{i:04}.png"
                            else:
                                filename = (
                                    f"{batch_name}({batch_num})_{i:04}-{j:03}.png"
                                )
                        image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                        image.save("progress.png")

                        if j % config.display_rate == 0 or cur_t == -1:
                            display.clear_output(wait=True)
                            display.display(display.Image("progress.png"))

                        if j in config.intermediate_saves:
                            image.save(f"{batch_folder}/{filename}")

                        if cur_t == -1:
                            if i == 0:
                                config.save_settings()
                            image.save(f"{batch_folder}/{filename}")
                            display.clear_output()

        plt.plot(np.array(loss_values), "r")

        gc.collect()
        torch.cuda.empty_cache()
        return [
            upload_png(os.path.join(batch_folder, filename)),
            upload_gif(batch_folder, batch_name),
        ]  # 回傳最後一個timestep的png及生成過程的gif
