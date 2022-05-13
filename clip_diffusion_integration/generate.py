import torch
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import random
import numpy as np
import clip
import matplotlib.pyplot as plt
import os
import gc
from tqdm.notebook import tqdm
from ipywidgets import Output
from IPython import display
from datetime import datetime
from glob import glob
from .config import *
from .prompt_utils import fetch, parse_prompt
from .perlin_utils import regen_perlin, regen_perlin_no_expand
from .clip_utils import clip_models
from .secondary_model import *
from .diffusion_model import model_config, load_model_and_diffusion
from .cutouts import MakeCutoutsDango
from .loss import *
from .dir_utils import *
from .gif_utils import create_gif

# 參考並修改自：disco diffusion

normalize = T.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
)


def get_embedding_and_weights():
    """
    取得prompt的embedding及weight
    """

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
            text = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()

            if fuzzy_prompt:
                for i in range(25):
                    model_stat["target_embeds"].append(
                        (text + torch.randn(text.shape).cuda() * rand_mag).clamp(0, 1)
                    )
                    model_stat["weights"].append(weight)
            else:
                model_stat["target_embeds"].append(text)
                model_stat["weights"].append(weight)

        model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
        model_stat["weights"] = torch.tensor(model_stat["weights"], device=device)

        if model_stat["weights"].sum().abs() < 1e-3:
            raise RuntimeError("The weights must not sum to 0.")

        model_stat["weights"] /= model_stat["weights"].sum().abs()
        model_stats.append(model_stat)

    return model_stats


def generate(batch_name="diffusion", partial_folder="images/partial"):
    """
    生成圖片
    batch_name: 本次生成的名稱
    partial_folder: 儲存過程圖片的資料夾
    """

    model, diffusion = load_model_and_diffusion()
    batch_folder = f"{out_dir_path}/{batch_name}"  # 儲存設定的資料夾
    make_dir(batch_folder)
    batch_num = len(glob(batch_folder + "/*.txt"))

    if steps_per_checkpoint != 0:
        partial_folder = f"{batch_folder}/partials"
        make_dir(partial_folder)

    while os.path.isfile(
        f"{batch_folder}/{batch_name}({batch_num})_settings.txt"
    ) or os.path.isfile(f"{batch_folder}/{batch_name}-{batch_num}_settings.txt"):
        batch_num += 1

    loss_values = []

    if seed:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    # target_embeds, weights = [], []

    # 取得prompt的embedding及weight
    model_stats = get_embedding_and_weights()

    init = None

    if perlin_init:
        init = regen_perlin_no_expand()

    cur_t = None

    # 透過clip引導guided diffusion
    def cond_fn(x, t, y=None):
        with torch.enable_grad():
            x_is_NaN = False
            x = x.detach().requires_grad_()
            n = x.shape[0]

            if use_secondary_model:
                alpha = torch.tensor(
                    diffusion.sqrt_alphas_cumprod[cur_t],
                    device=device,
                    dtype=torch.float32,
                )
                sigma = torch.tensor(
                    diffusion.sqrt_one_minus_alphas_cumprod[cur_t],
                    device=device,
                    dtype=torch.float32,
                )
                cosine_t = alpha_sigma_to_t(alpha, sigma)
                out = secondary_model(x, cosine_t[None].repeat([n])).pred
                fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                x_in = out * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)
            else:
                my_t = torch.ones([n], device=device, dtype=torch.long) * cur_t
                out = diffusion.p_mean_variance(
                    model, x, my_t, clip_denoised=False, model_kwargs={"y": y}
                )
                fac = diffusion.sqrt_one_minus_alphas_cumprod[cur_t]
                x_in = out["pred_xstart"] * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)

            for model_stat in model_stats:
                for i in range(cutn_batches):
                    t_int = (
                        int(t.item()) + 1
                    )  # errors on last step without +1, need to find source
                    input_resolution = model_stat["clip_model"].visual.input_resolution
                    cuts = MakeCutoutsDango(
                        input_resolution,
                        Overview=cut_overview[1000 - t_int],
                        InnerCrop=cut_innercut[1000 - t_int],
                        IC_Size_Pow=cut_ic_pow,
                        IC_Grey_P=cut_icgray_p[1000 - t_int],
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
                        [cut_overview[1000 - t_int] + cut_innercut[1000 - t_int], n, -1]
                    )
                    losses = dists.mul(model_stat["weights"]).sum(2).mean(0)
                    loss_values.append(
                        losses.sum().item()
                    )  # log loss, probably shouldn't do per cutn_batch
                    x_in_grad += (
                        torch.autograd.grad(losses.sum() * clip_guidance_scale, x_in)[0]
                        / cutn_batches
                    )
            tv_losses = tv_loss(x_in)

            if use_secondary_model:
                range_losses = range_loss(out)
            else:
                range_losses = range_loss(out["pred_xstart"])

            sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
            loss = (
                tv_losses.sum() * tv_scale
                + range_losses.sum() * range_scale
                + sat_losses.sum() * sat_scale
            )

            # numpy array, tensor的判斷式需要寫完整
            if init is not None and init_scale:
                init_losses = lpips_model(x_in, init)
                loss = loss + init_losses.sum() * init_scale
            x_in_grad += torch.autograd.grad(loss, x_in)[0]

            if not torch.isnan(x_in_grad).any():
                grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
            else:
                # print("NaN'd")
                x_is_NaN = True
                grad = torch.zeros_like(x)

        if clamp_grad and not x_is_NaN:
            magnitude = grad.square().mean().sqrt()
            return (
                grad * magnitude.clamp(min=-clamp_max, max=clamp_max) / magnitude
            )  # min=-0.02,

        return grad

    # 使用DDIM
    if model_config["timestep_respacing"].startswith("ddim"):
        sample_fn = diffusion.ddim_sample_loop_progressive
    # else:
    #     sample_fn = diffusion.p_sample_loop_progressive

    image_display = Output()

    for i in range(num_batches):
        display.clear_output(wait=True)
        batchBar = tqdm(range(num_batches), desc="Batches")
        batchBar.n = i
        batchBar.refresh()
        print("")
        display.display(image_display)
        gc.collect()
        torch.cuda.empty_cache()
        cur_t = diffusion.num_timesteps - skip_timesteps - 1
        total_steps = cur_t

        if perlin_init:
            init = regen_perlin()

        if model_config["timestep_respacing"].startswith("ddim"):
            samples = sample_fn(
                model,
                (batch_size, 3, side_y, side_x),
                clip_denoised=clip_denoised,
                model_kwargs={},
                cond_fn=cond_fn,
                progress=True,
                skip_timesteps=skip_timesteps,
                init_image=init,
                randomize_class=randomize_class,
                eta=eta,
            )
        # else:
        #     samples = sample_fn(
        #         model,
        #         (batch_size, 3, side_y, side_x),
        #         clip_denoised=clip_denoised,
        #         model_kwargs={},
        #         cond_fn=cond_fn,
        #         progress=True,
        #         skip_timesteps=skip_timesteps,
        #         init_image=init,
        #         randomize_class=randomize_class,
        #     )

        for j, sample in enumerate(samples):
            cur_t -= 1
            intermediate_step = False
            if steps_per_checkpoint:
                if j % steps_per_checkpoint == 0 and j > 0:
                    intermediate_step = True
            elif j in intermediate_saves:
                intermediate_step = True

            with image_display:
                if j % display_rate == 0 or cur_t == -1 or intermediate_step:
                    for k, image in enumerate(sample["pred_xstart"]):
                        current_time = datetime.now().strftime("%y%m%d-%H%M%S_%f")
                        percent = math.ceil(j / total_steps * 100)
                        if num_batches > 0:
                            # if intermediates are saved to the subfolder, don't append a step or percentage to the name
                            if cur_t == -1:
                                filename = f"{batch_name}({batch_num})_{i:04}.png"
                            else:
                                # If we're working with percentages, append it
                                if steps_per_checkpoint:
                                    filename = f"{batch_name}({batch_num})_{i:04}-{percent:02}%.png"
                                # Or else, iIf we're working with specific steps, append those
                                else:
                                    filename = (
                                        f"{batch_name}({batch_num})_{i:04}-{j:03}.png"
                                    )
                        image = TF.to_pil_image(image.add(1).div(2).clamp(0, 1))
                        image.save("progress.png")
                        if j % display_rate == 0 or cur_t == -1:
                            display.clear_output(wait=True)
                            display.display(display.Image("progress.png"))
                        if steps_per_checkpoint:
                            if j % steps_per_checkpoint == 0 and j > 0:
                                image.save(f"{partial_folder}/{filename}")
                                # image.save(f"{batch_folder}/{filename}")
                        else:
                            if j in intermediate_saves:
                                image.save(f"{partial_folder}/{filename}")
                                # image.save(f"{batch_folder}/{filename}")
                        if cur_t == -1:
                            if i == 0:
                                save_settings()
                            image.save(f"{batch_folder}/{filename}")
                            display.clear_output()

        create_gif(partial_folder, batch_name)  # 建立一張gif
        plt.plot(np.array(loss_values), "r")

        gc.collect()
        torch.cuda.empty_cache()
