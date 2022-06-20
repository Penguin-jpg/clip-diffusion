import torch
from torchvision import transforms as T
from torchvision.transforms import functional as TF
import gc
import lpips
import anvil
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm, trange
from ipywidgets import Output
from IPython import display
from ldm.models.diffusion.ddim import DDIMSampler
from einops import rearrange
from torchvision.utils import make_grid
from clip_diffusion.config import config
from clip_diffusion.preprocess_utils import (
    translate_zh_to_en,
    set_seed,
    get_embedding_and_weights,
)
from clip_diffusion.perlin_utils import regen_perlin, regen_perlin_no_expand
from clip_diffusion.model_utils import (
    alpha_sigma_to_t,
    load_clip_models_and_preprocessings,
    load_latent_diffusion_model,
    load_model_and_diffusion,
    load_secondary_model,
)
from clip_diffusion.cutouts import MakeCutoutsDango
from clip_diffusion.loss import spherical_dist_loss, tv_loss, range_loss
from clip_diffusion.dir_utils import *
from clip_diffusion.image_utils import *


chosen_clip_models = {
    "ViT-B/16": True,
    "ViT-B/32": True,
    "ViT-L/14": False,
    "RN50": True,
    "RN50x4": True,
    "RN50x16": False,
    "RN50x64": False,
    "RN101": False,
}
normalize = T.Normalize(
    mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711]
)
lpips_model = lpips.LPIPS(net="vgg").to(config.device)
clip_models, preprocessings = load_clip_models_and_preprocessings(chosen_clip_models)
secondary_model = load_secondary_model()
latent_diffusion_model = load_latent_diffusion_model()

# 參考並修改自：disco diffusion
@anvil.server.background_task
def generate(
    prompts=[
        "A cute golden retriever.",
    ],
    init_image=None,
    use_perlin=False,
    perlin_mode="mixed",
    batch_name="diffusion",
):
    """
    生成圖片(和anvil client互動)
    prompts: 要生成的東西
    init_image: 模型會參考該圖片生成初始噪聲(會是anvil的Media類別)
    use_perlin: 是否要使用perlin noise
    perlin_mode: 使用的perlin noise模式
    batch_name: 本次生成的名稱
    """

    prompts = translate_zh_to_en(prompts)  # 將prompts翻成英文
    model, diffusion = load_model_and_diffusion()  # 載入diffusion model和diffusion
    batch_folder = f"{out_dir_path}/{batch_name}"  # 儲存圖片的資料夾
    make_dir(batch_folder)
    remove_old_files(batch_folder)  # 移除舊的圖片

    # 設定種子
    set_seed(config.seed)

    # 取得prompt的embedding及weight
    model_stats = get_embedding_and_weights(prompts, clip_models)

    init = None  # init_image或perlin noise只能擇一
    loss_values = []

    # 如果初始圖片不為空
    if init_image is not None:
        # 透過anvil傳來的image_file的bytes開啟圖片
        init = get_image_from_bytes(init_image.get_bytes()).convert("RGB")
        init = init.resize((config.side_x, config.side_y), Image.LANCZOS)
        init = TF.to_tensor(init).to(config.device).unsqueeze(0).mul(2).sub(1)

    # 使用perlin noise
    if use_perlin:
        init = regen_perlin_no_expand(perlin_mode)

    current_timestep = None  # 目前的timestep

    # 透過clip引導guided diffusion(參考自disco diffusion)
    def cond_fn(x, t, y=None):
        with torch.enable_grad():
            x_is_NaN = False
            x = x.detach().requires_grad_()
            n = x.shape[0]

            # 使用secondary_model加速生成
            if config.use_secondary_model:
                alpha = torch.tensor(
                    diffusion.sqrt_alphas_cumprod[current_timestep],
                    device=config.device,
                    dtype=torch.float32,
                )
                sigma = torch.tensor(
                    diffusion.sqrt_one_minus_alphas_cumprod[current_timestep],
                    device=config.device,
                    dtype=torch.float32,
                )
                cosine_t = alpha_sigma_to_t(alpha, sigma)
                out = secondary_model(x, cosine_t[None].repeat([n])).pred
                fac = diffusion.sqrt_one_minus_alphas_cumprod[current_timestep]
                x_in = out * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)
            else:
                my_t = (
                    torch.ones([n], device=config.device, dtype=torch.long)
                    * current_timestep
                )
                out = diffusion.p_mean_variance(
                    model, x, my_t, clip_denoised=False, model_kwargs={"y": y}
                )
                fac = diffusion.sqrt_one_minus_alphas_cumprod[current_timestep]
                x_in = out["pred_xstart"] * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)

            for model_stat in model_stats:
                for _ in range(config.cutn_batches):
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

    # 在server端顯示
    image_display = Output()

    for current_batch in range(config.num_batches):
        display.clear_output(wait=True)
        progress_bar = tqdm(range(config.num_batches), desc="Batches")
        progress_bar.n = current_batch
        progress_bar.refresh()
        print("")
        display.display(image_display)
        gc.collect()
        torch.cuda.empty_cache()
        current_timestep = (
            diffusion.num_timesteps - config.skip_timesteps - 1
        )  # 將目前timestep的值初始化為總timestep數-1

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

        # current_timestep從總timestep數開始；step_index從0開始
        for step_index, sample in enumerate(samples):
            current_timestep -= 1  # 每次都將目前的timestep減1

            # 紀錄目前的step
            anvil.server.task_state["current_step"] = step_index + 1

            with image_display:
                # 如果需要更新、儲存圖片或生成結束時進入
                if step_index in config.intermediate_saves or current_timestep == -1:
                    for _, image in enumerate(sample["pred_xstart"]):
                        filename = f"{batch_name}_{current_batch:04}-{step_index:03}.png"  # 圖片名稱
                        image = TF.to_pil_image(
                            image.add(1).div(2).clamp(0, 1)
                        )  # 轉換為Pillow Image

                        # 需要更新和儲存圖片
                        if step_index in config.intermediate_saves:
                            image.save(f"{batch_folder}/{filename}")
                            # 將目前圖片結果的url存到current_result
                            url = upload_png(f"{batch_folder}/{filename}")
                            anvil.server.task_state["current_result"] = url
                            display.clear_output(wait=True)
                            display.display(display.Image(f"{batch_folder}/{filename}"))

                        # 生成結束
                        if current_timestep == -1:
                            image.save(f"{batch_folder}/{filename}")
                            display.display(display.Image(f"{batch_folder}/{filename}"))
                            display.clear_output()

        gc.collect()
        torch.cuda.empty_cache()
        anvil.server.task_state["current_result"] = upload_png(
            f"{batch_folder}/{filename}"
        )
        anvil.server.task_state["generation_process"] = upload_gif(
            batch_folder, batch_name
        )


# 參考並修改自：https://huggingface.co/spaces/multimodalart/latentdiffusion/blob/main/app.py
@anvil.server.background_task
def latent_diffusion_generate(
    prompts=[
        "A cute golden retriever.",
    ],
    latent_diffusion_guidance_scale=5,
    num_iterations=3,
    num_samples=3,
    latent_diffusion_steps=50,
    latent_diffusion_eta=0.0,
    sample_width=256,
    sample_height=256,
    batch_name="latent",
):
    """
    使用latent diffusion生成圖片
    prompts: 要生成的東西
    latent_diffusion_guidance_scale: latent diffusion unconditional的引導強度(介於0~15，多樣性隨著數值升高)
    num_iterations: 做幾次latent diffusion生成
    num_samples: 要生成的圖片數
    latent_diffusion_steps: latent diffusion要跑的step數
    latent_diffusion_eta: latent diffusion的eta
    sample_width:  sample圖片的寬(latent diffusion sample的圖片不能太大，後續再用sr提高解析度)
    sample_height: sample圖片的高
    batch_name: 本次生成的名稱
    """

    prompts = translate_zh_to_en(prompts)  # 將prompts翻成英文
    sampler = DDIMSampler(latent_diffusion_model)  # 建立DDIM sampler
    batch_folder = f"{out_dir_path}/{batch_name}"  # 儲存圖片的資料夾
    make_dir(batch_folder)
    remove_old_files(batch_folder)  # 移除舊的圖片

    # 設定種子
    set_seed(config.seed)

    samples = []  # 儲存所有sample
    count = 0  # 圖片編號

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with latent_diffusion_model.ema_scope():

                uncoditional_conditioning = None
                if latent_diffusion_guidance_scale > 0:
                    uncoditional_conditioning = (
                        latent_diffusion_model.get_learned_conditioning(
                            num_samples * [""]  # ""代表不考慮的prompt
                        )
                    )

                for _ in trange(num_iterations, desc="Sampling"):
                    gc.collect()
                    torch.cuda.empty_cache()

                    conditioning = latent_diffusion_model.get_learned_conditioning(
                        num_samples * [prompts[0]]
                    )
                    shape = [4, sample_height // 8, sample_width // 8]

                    # sample，只取第一個變數(samples)，不取第二個變數(intermediates)
                    samples_ddim, _ = sampler.sample(
                        S=latent_diffusion_steps,
                        conditioning=conditioning,
                        batch_size=num_samples,
                        shape=shape,
                        verbose=False,
                        unconditional_guidance_scale=latent_diffusion_guidance_scale,
                        unconditional_conditioning=uncoditional_conditioning,
                        eta=latent_diffusion_eta,
                    )

                    x_samples_ddim = latent_diffusion_model.decode_first_stage(
                        samples_ddim
                    )
                    x_samples_ddim = torch.clamp(
                        (x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0
                    )

                    for x_sample in x_samples_ddim:
                        x_sample = 255.0 * rearrange(
                            x_sample.cpu().numpy(), "c h w -> h w c"
                        )
                        filename = os.path.join(
                            batch_folder, f"{batch_name}_{count:04}.png"
                        )
                        image_vector = Image.fromarray(x_sample.astype(np.uint8))
                        image_preprocess = (
                            preprocessings[1](image_vector)
                            .unsqueeze(0)
                            .to(config.device)
                        )  # 配合ViT-B/32進行preprocess

                        with torch.no_grad():
                            image_embeddings = clip_models[1].encode_image(
                                image_preprocess
                            )  # 使用ViT-B/32進行encode

                        image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)
                        image_vector.save(filename)
                        count += 1
                    samples.append(x_samples_ddim)

    # 轉成grid形式
    grid = torch.stack(samples, 0)
    grid = rearrange(grid, "n b c h w -> (n b) c h w")
    grid = make_grid(grid, nrow=num_samples)
    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
    Image.fromarray(grid.astype(np.uint8)).save(
        os.path.join(batch_folder, f"latent_diffusion.png")
    )  # 儲存grid圖片

    gc.collect()
    torch.cuda.empty_cache()
