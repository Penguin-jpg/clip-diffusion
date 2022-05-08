import torch
from torch import nn
from torchvision import transforms
import random
import numpy as np
import clip
import matplotlib.pyplot as plt
import tqdm
import gc
from PIL import Image
from ipywidgets import Output
from IPython import display
from datetime import datetime
from .config import *
from .prompt_utils import fetch, parse_prompt
from .perlin_utils import regen_perlin, regen_perlin_no_expand
from .clip_utils import clip_models
from .secondary_model import *
from .diffusion_model import model, diffusion
from .cutouts import MakeCutoutsDango
from .loss import *

# 參考並修改自：disco diffusion

normalize = transforms.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
)


def generate(
    batch_name="diffusion", partial_foler="images/partial", batch_folder="images/batch"
):
    """
    生成圖片
    batch_name: 本次生成的名稱
    partial_folder: 儲存過程圖片的資料夾
    batch_folder: 儲存最後圖片的資料夾
    """
    loss_values = []

    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    target_embeds, weights = [], []

    model_stats = []
    for clip_model in clip_models:
        model_stat = {
            "clip_model": None,
            "target_embeds": [],
            "make_cutouts": None,
            "weights": [],
        }
        model_stat["clip_model"] = clip_model
        # model_stat["make_cutouts"] = MakeCutouts(clip_model.visual.input_resolution, cutn, skip_augs=skip_augs)

        for prompt in text_prompts:
            txt, weight = parse_prompt(prompt)
            txt = clip_model.encode_text(clip.tokenize(prompt).to(device)).float()

            if fuzzy_prompt:
                for i in range(25):
                    model_stat["target_embeds"].append(
                        (txt + torch.randn(txt.shape).cuda() * rand_mag).clamp(0, 1)
                    )
                    model_stat["weights"].append(weight)
            else:
                model_stat["target_embeds"].append(txt)
                model_stat["weights"].append(weight)

        # for prompt in image_prompts:
        #     path, weight = parse_prompt(prompt)
        #     img = Image.open(fetch(path)).convert('RGB')
        #     img = transforms.functional.resize(img, min(side_x, side_y, *img.size), T.InterpolationMode.LANCZOS)
        #     batch = model_stat["make_cutouts"](transforms.functional.to_tensor(img).to(device).unsqueeze(0).mul(2).sub(1))
        #     embed = clip_model.encode_image(normalize(batch)).float()
        #     if fuzzy_prompt:
        #         for i in range(25):
        #             model_stat["target_embeds"].append((embed + torch.randn(embed.shape).cuda() * rand_mag).clamp(0,1))
        #             weights.extend([weight / cutn] * cutn)
        #     else:
        #         model_stat["target_embeds"].append(embed)
        #         model_stat["weights"].extend([weight / cutn] * cutn)

        # if image_prompt:
        #     model_stat["make_cutouts"] = MakeCutouts(clip_model.visual.input_resolution, cutn, skip_augs=skip_augs)
        #     for prompt in image_prompt:
        #         path, weight = parse_prompt(prompt)
        #         img = Image.open(fetch(path)).convert('RGB')
        #         img = transforms.functional.resize(img, min(side_x, side_y, *img.size), T.InterpolationMode.LANCZOS)
        #         batch = model_stat["make_cutouts"](transforms.functional.to_tensor(img).to(device).unsqueeze(0).mul(2).sub(1))
        #         embed = clip_model.encode_image(normalize(batch)).float()
        #         if fuzzy_prompt:
        #             for i in range(25):
        #                 model_stat["target_embeds"].append((embed + torch.randn(embed.shape).cuda() * rand_mag).clamp(0,1))
        #                 weights.extend([weight / cutn] * cutn)
        #         else:
        #             model_stat["target_embeds"].append(embed)
        #             model_stat["weights"].extend([weight / cutn] * cutn)

        model_stat["target_embeds"] = torch.cat(model_stat["target_embeds"])
        model_stat["weights"] = torch.tensor(model_stat["weights"], device=device)
        if model_stat["weights"].sum().abs() < 1e-3:
            raise RuntimeError("The weights must not sum to 0.")
        model_stat["weights"] /= model_stat["weights"].sum().abs()
        model_stats.append(model_stat)

    init = None
    if init_image is not None:
        init = Image.open(fetch(init_image)).convert("RGB")
        init = init.resize((side_x, side_y), Image.LANCZOS)
        init = (
            transforms.functional.to_tensor(init).to(device).unsqueeze(0).mul(2).sub(1)
        )

    if perlin_init:
        init = regen_perlin_no_expand()

    cur_t = None

    # 透過clip引導guided diffusion
    def cond_fn(x, t, y=None):
        with torch.enable_grad():
            x_is_NaN = False
            x = x.detach().requires_grad_()
            n = x.shape[0]
            if use_secondary_model is True:
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
                    # when using SLIP Base model the dimensions need to be hard coded to avoid AttributeError: 'VisionTransformer' object has no attribute 'input_resolution'
                    try:
                        input_resolution = model_stat[
                            "clip_model"
                        ].visual.input_resolution
                    except:
                        input_resolution = 224

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
            if use_secondary_model is True:
                range_losses = range_loss(out)
            else:
                range_losses = range_loss(out["pred_xstart"])
            sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
            loss = (
                tv_losses.sum() * tv_scale
                + range_losses.sum() * range_scale
                + sat_losses.sum() * sat_scale
            )
            if init is not None and init_scale:
                init_losses = lpips_model(x_in, init)
                loss = loss + init_losses.sum() * init_scale
            x_in_grad += torch.autograd.grad(loss, x_in)[0]
            if torch.isnan(x_in_grad).any() == False:
                grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
            else:
                # print("NaN'd")
                x_is_NaN = True
                grad = torch.zeros_like(x)
        if clamp_grad and x_is_NaN == False:
            magnitude = grad.square().mean().sqrt()
            return (
                grad * magnitude.clamp(min=-clamp_max, max=clamp_max) / magnitude
            )  # min=-0.02,
        return grad

    # 使用DDIM
    if timestep_respacing.startswith("ddim"):
        sample_fn = diffusion.ddim_sample_loop_progressive
    else:
        sample_fn = diffusion.p_sample_loop_progressive

    image_display = Output()

    # with batches_display:
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

        if timestep_respacing.startswith("ddim"):
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
        else:
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
            )

        for j, sample in enumerate(samples):
            cur_t -= 1
            intermediateStep = False
            if steps_per_checkpoint:
                if j % steps_per_checkpoint == 0 and j > 0:
                    intermediateStep = True
            elif j in intermediate_saves:
                intermediateStep = True

            with image_display:
                if j % display_rate == 0 or cur_t == -1 or intermediateStep:
                    for k, image in enumerate(sample["pred_xstart"]):
                        # tqdm.write(f'Batch {i}, step {j}, output {k}:')
                        current_time = datetime.now().strftime("%y%m%d-%H%M%S_%f")
                        percent = math.ceil(j / total_steps * 100)
                        if num_batches > 0:
                            # if intermediates are saved to the subfolder, don't append a step or percentage to the name
                            if cur_t == -1 and intermediates_in_subfolder == True:
                                filename = f"{batch_name}_{i:04}.png"
                            else:
                                # If we're working with percentages, append it
                                if steps_per_checkpoint:
                                    filename = f"{batch_name}_{i:04}-{percent:02}%.png"
                                # Or else, iIf we're working with specific steps, append those
                                else:
                                    filename = f"{batch_name}_{i:04}-{j:03}.png"
                        image = transforms.functional.to_pil_image(
                            image.add(1).div(2).clamp(0, 1)
                        )
                        image.save("progress.png")
                        if j % display_rate == 0 or cur_t == -1:
                            display.clear_output(wait=True)
                            display.display(display.Image("progress.png"))
                        if steps_per_checkpoint is not None:
                            if j % steps_per_checkpoint == 0 and j > 0:
                                if intermediates_in_subfolder is True:
                                    image.save(f"{partial_foler}/{filename}")
                                else:
                                    image.save(f"{batch_folder}/{filename}")
                        else:
                            if j in intermediate_saves:
                                if intermediates_in_subfolder is True:
                                    image.save(f"{partial_foler}/{filename}")
                                else:
                                    image.save(f"{batch_folder}/{filename}")
                        if cur_t == -1:
                            if i == 0:
                                save_settings()
                            image.save(f"{batch_folder}/{filename}")
                            display.clear_output()

        plt.plot(np.array(loss_values), "r")
