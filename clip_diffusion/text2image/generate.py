import torch
import os
import gc
import lpips
import anvil
import numpy as np
from PIL import Image
from tqdm.notebook import tqdm, trange
from ipywidgets import Output
from IPython import display
from ldm.models.diffusion.ddim import DDIMSampler
from einops import rearrange, repeat
from torchvision.utils import make_grid
from clip_diffusion.config import config
from clip_diffusion.utils.preprocessing import prompts_preprocessing, set_seed, get_embeddings_and_weights, create_init_noise, preprocess_mask_image
from clip_diffusion.utils.perlin import regen_perlin
from clip_diffusion.models import (
    load_clip_models_and_preprocessings,
    load_guided_diffusion_model,
    load_secondary_model,
    alpha_sigma_to_t,
    load_latent_diffusion_model,
    load_real_esrgan_upsampler,
)
from clip_diffusion.text2image.cutouts import MakeCutouts
from clip_diffusion.text2image.loss import spherical_dist_loss, tv_loss, range_loss
from clip_diffusion.utils.dir_utils import make_dir, OUTPUT_PATH
from clip_diffusion.utils.image_utils import (
    CLIP_NORMALIZE,
    unnormalize_image_zero_to_one,
    tensor_to_pillow_image,
    upload_png,
    upload_gif,
    images_to_grid_image,
    super_resolution,
)

_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
lpips_model = lpips.LPIPS(net="vgg").to(_device)
clip_models, preprocessings = load_clip_models_and_preprocessings(config.chosen_clip_models, _device)
secondary_model = None
latent_diffusion_model = None
real_esrgan_upsampler = None

# 參考並修改自：disco diffusion
@anvil.server.background_task
def guided_diffusion_generate(
    prompts=[
        "A cute golden retriever.",
    ],
    styles=[],
    init_image=None,
    use_perlin=False,
    perlin_mode="mixed",
    steps=200,
    skip_timesteps=0,
    clip_guidance_scale=8000,
    eta=0.8,
    init_scale=1000,
    num_batches=1,
    use_grid_image=False,
    num_rows=1,
    num_cols=1,
    display_rate=25,
    gif_duration=400,
):
    """
    生成圖片(和anvil client互動)
    prompts: 要生成的東西
    init_image: 模型會參考該圖片生成初始雜訊(會是anvil的Media類別)
    use_perlin: 是否要使用perlin noise
    perlin_mode: 使用的perlin noise模式
    steps: 每個batch要跑的step數
    skip_timesteps: 控制要跳過的step數(從第幾個step開始)，當使用init_image時最好調整為diffusion_steps的0~50%
    clip_guidance_scale: clip引導的強度(生成圖片要多接近prompt)
    eta: DDIM與DDPM的比例(0.0: 純DDIM; 1.0: 純DDPM)，越高每個timestep加入的雜訊越多
    init_scale: 增強init_image的效果
    num_batches: 要生成的圖片數量
    use_grid_iamge: 是否要生成grid圖片
    num_rows: grid圖片的row數量
    num_cols: grid圖片的col數量
    display_rate: 生成過程的gif多少個step要更新一次
    gif_duration: gif的播放時間
    """

    # 使用全域變數
    global secondary_model

    if secondary_model is None:
        secondary_model = load_secondary_model(_device)

    prompts = prompts_preprocessing(prompts, styles)  # prompts的前處理
    model, diffusion = load_guided_diffusion_model(steps, device=_device)  # 載入diffusion model和diffusion
    batch_folder = os.path.join(OUTPUT_PATH, "guided")  # 儲存圖片的資料夾
    make_dir(batch_folder, remove_old=True)

    # 設定種子
    set_seed()

    # 取得prompt的embedding及weight
    clip_model_stats = get_embeddings_and_weights(prompts, clip_models, _device)

    # 建立初始雜訊
    init = create_init_noise(init_image, True, use_perlin, perlin_mode, _device)

    loss_values = []
    current_timestep = None  # 目前的timestep

    def cond_fn(x, t, y=None):
        """
        透過clip引導guided diffusion(計算grad(log(p(y|x))))
        x: 上一個timestep的圖片tensor
        t: diffusion timestep tensor
        y: class
        """
        with torch.enable_grad():
            x_is_NaN = False  # x是否為NaN
            x = x.detach().requires_grad_()  # 將x從目前的計算圖中取出
            batch_size = x.shape[0]

            # 使用secondary_model加速生成
            if config.use_secondary_model:
                alpha = torch.tensor(
                    diffusion.sqrt_alphas_cumprod[current_timestep],
                    dtype=torch.float32,
                    device=_device,
                )
                sigma = torch.tensor(
                    diffusion.sqrt_one_minus_alphas_cumprod[current_timestep],
                    dtype=torch.float32,
                    device=_device,
                )
                cosine_t = alpha_sigma_to_t(alpha, sigma)
                out = secondary_model(x, cosine_t[None].repeat([batch_size])).pred
                fac = diffusion.sqrt_one_minus_alphas_cumprod[current_timestep]
                x_in = out * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)
            else:
                my_t = torch.ones([batch_size], device=_device, dtype=torch.long) * current_timestep
                out = diffusion.p_mean_variance(model, x, my_t, clip_denoised=False, model_kwargs={"y": y})
                fac = diffusion.sqrt_one_minus_alphas_cumprod[current_timestep]
                x_in = out["pred_xstart"] * fac + x * (1 - fac)
                x_in_grad = torch.zeros_like(x_in)

            for clip_model_stat in clip_model_stats:
                # 做cutout
                for _ in range(config.num_cutout_batches):
                    # 將t的值從tensor取出(t每次進入cond_fn時會減掉(1000/steps))
                    t_value = int(t.item()) + 1
                    # 做cutouts(用(1000-t_value)是因為MakeCutouts以1000當做基準線)
                    cuts = MakeCutouts(
                        cut_size=clip_models[clip_model_stat["clip_model_name"]].visual.input_resolution,  # 將輸入的圖片切成Clip model的輸入大小
                        overview=config.overview_cut_schedule[1000 - t_value],
                        inner_cut=config.inner_cut_schedule[1000 - t_value],
                        inner_cut_size_pow=config.inner_cut_size_pow,
                        cut_gray_portion=config.cut_gray_portion_schedule[1000 - t_value],
                        use_augmentations=config.use_augmentations,
                    )
                    clip_in = CLIP_NORMALIZE(cuts(unnormalize_image_zero_to_one(x_in)))
                    image_embeddings = clip_models[clip_model_stat["clip_model_name"]].encode_image(clip_in).float()
                    dists = spherical_dist_loss(
                        image_embeddings.unsqueeze(1),
                        clip_model_stat["text_embeddings"].unsqueeze(0),
                    )
                    dists = dists.view([config.overview_cut_schedule[1000 - t_value] + config.inner_cut_schedule[1000 - t_value], batch_size, -1])
                    losses = dists.mul(clip_model_stat["text_weights"]).sum(2).mean(0)
                    loss_values.append(losses.sum().item())
                    x_in_grad += torch.autograd.grad(losses.sum() * clip_guidance_scale, x_in)[0] / config.num_cutout_batches
            tv_losses = tv_loss(x_in)

            if config.use_secondary_model:
                range_losses = range_loss(out)
            else:
                range_losses = range_loss(out["pred_xstart"])

            sat_losses = torch.abs(x_in - x_in.clamp(min=-1, max=1)).mean()
            loss = tv_losses.sum() * config.tv_scale + range_losses.sum() * config.range_scale + sat_losses.sum() * config.sat_scale

            # 透過LPIPS計算初始圖片的loss
            if init is not None and init_scale:
                init_losses = lpips_model(x_in, init)
                loss = loss + init_losses.sum() * init_scale
            x_in_grad += torch.autograd.grad(loss, x_in)[0]

            if not torch.isnan(x_in_grad).any():
                grad = -torch.autograd.grad(x_in, x, x_in_grad)[0]
            else:
                x_is_NaN = True
                grad = torch.zeros_like(x)

        if config.clamp_grad and not x_is_NaN:
            magnitude = grad.square().mean().sqrt()
            return grad * magnitude.clamp(min=-config.clamp_max, max=config.clamp_max) / magnitude

        return grad

    image_display = Output()  # 在server端顯示圖片
    gif_urls = []  # 生成過程的gif url
    images = []  # 最後一個timestep的圖片

    for batch_index in range(num_batches):
        display.clear_output(wait=True)
        progress_bar = tqdm(range(num_batches), desc="Batches")
        progress_bar.n = batch_index + 1
        progress_bar.refresh()
        display.display(image_display)

        # 將目前的batch index存到current_batch
        anvil.server.task_state["current_batch"] = batch_index + 1

        gc.collect()
        torch.cuda.empty_cache()

        # 將目前timestep的值初始化為總timestep數-1
        current_timestep = diffusion.num_timesteps - skip_timesteps - 1

        if use_perlin:
            init = regen_perlin(perlin_mode, _device)

        # 使用DDIM進行sample
        samples = diffusion.ddim_sample_loop_progressive(
            model,
            (1, 3, config.height, config.width),  # shape=(batch_size, num_channels, height, width)
            clip_denoised=False,
            model_kwargs={},
            cond_fn=cond_fn,
            progress=True,
            skip_timesteps=skip_timesteps,
            init_image=init,
            randomize_class=True,
            eta=eta,
        )

        # current_timestep從總timestep數開始；step_index從0開始
        for step_index, sample in enumerate(samples):
            current_timestep -= 1  # 每次都將目前的timestep減1

            with image_display:
                # 更新、儲存圖片
                for _, image_tensor in enumerate(sample["pred_xstart"]):
                    filename = f"guided_{batch_index}_{step_index:04}.png"  # 圖片名稱
                    image_path = os.path.join(batch_folder, filename)  # 圖片路徑
                    unnormalized_image = unnormalize_image_zero_to_one(image_tensor).clamp(min=0.0, max=1.0)  # 將image_tensor範圍轉回[0, 1]，並用clamp確保範圍正確
                    image = tensor_to_pillow_image(unnormalized_image)  # 轉換為Pillow Image
                    image.save(image_path)
                    display.clear_output(wait=True)
                    display.display(display.Image(image_path))

                    # 生成結束
                    if current_timestep == -1:
                        image.save(image_path)
                        display.display(display.Image(image_path))
                        display.clear_output()
                        # 將最後一個timestep的url存到current_result
                        anvil.server.task_state["current_result"] = upload_png(image_path)
                        # 儲存生成過程的gif url
                        gif_urls.append(
                            upload_gif(
                                batch_folder,
                                batch_index,
                                display_rate,
                                gif_duration,
                                append_last_timestep=(steps - skip_timesteps - 1) % display_rate,
                            )
                        )
                        # 儲存最後一個timestep的圖片
                        images.append(Image.open(image_path))
                    elif step_index % 10 == 0:  # 每10個timestep更新上傳一次圖片
                        # 將目前圖片的url存到current_result
                        anvil.server.task_state["current_result"] = upload_png(image_path)

            # 紀錄目前的step
            anvil.server.task_state["current_step"] = step_index + 1

        gc.collect()
        torch.cuda.empty_cache()

    if use_grid_image:
        # 儲存grid圖片的url到grid_image_url
        grid_image_url = images_to_grid_image(batch_folder, images, num_rows, num_cols)
        return gif_urls, grid_image_url

    return gif_urls  # 回傳gif url


# 參考並修改自： https://github.com/CompVis/latent-diffusion/blob/main/scripts/txt2img.py
@anvil.server.background_task
def latent_diffusion_generate(
    prompts=[
        "A cute golden retriever.",
    ],
    init_image=None,
    mask_image=None,
    diffusion_steps=50,
    eta=0.0,
    latent_diffusion_guidance_scale=5,
    num_iterations=3,
    num_batches=3,
    chosen_models=["ViT-B/32", "ViT-B/16"],
    sample_width=256,
    sample_height=256,
):
    """
    使用latent diffusion生成圖片
    prompts: 要生成的東西
    init_image: 要配合inpaint使用的圖片
    mask_image: inpaint用的遮罩
    diffusion_steps: latent diffusion要跑的step數
    eta: latent diffusion的eta
    latent_diffusion_guidance_scale: latent diffusion unconditional的引導強度(介於0~15，多樣性隨著數值升高)
    num_iterations: 做幾次latent diffusion生成
    num_batches: 要生成的圖片數
    chosen_models: 要用來引導的Clip模型名稱
    sample_width:  sample圖片的寬(latent diffusion sample的圖片不能太大，後續再用sr提高解析度)
    sample_height: sample圖片的高
    """

    global latent_diffusion_model, real_esrgan_upsampler

    if latent_diffusion_model is None:
        latent_diffusion_model = load_latent_diffusion_model(_device)

    if real_esrgan_upsampler is None:
        real_esrgan_upsampler = load_real_esrgan_upsampler(_device)

    prompts = prompts_preprocessing(prompts)  # 將prompts翻成英文
    sampler = DDIMSampler(latent_diffusion_model)  # 建立DDIM sampler
    batch_folder = os.path.join(OUTPUT_PATH, "latent")  # 儲存圖片的資料夾
    make_dir(batch_folder, remove_old=True)

    # 設定種子
    set_seed()

    # sample的shape
    shape = (4, sample_height // 8, sample_width // 8)
    # encode過的init_image
    encoded_init = None
    # 遮罩tensor
    mask = None

    # 處理inpaint的參數
    if init_image is not None:
        init = create_init_noise(init_image, False, device=_device)  # 將init_image當成初始雜訊
        init = repeat(init, "c h w -> b c h w", b=num_batches)  # 將shape變成(batch_size, num_channels, height, width)
        encoder_posterior = latent_diffusion_model.encode_first_stage(init)  # 使用encoder對init encode
        encoded_init = latent_diffusion_model.get_first_stage_encoding(encoder_posterior)  # 取出encode的結果

        mask = preprocess_mask_image(mask_image, shape[2], shape[1], _device)  # 處理mask
        # 將shape變成(batch_size, num_channels, height, width)，黑白圖片的num_channels=1
        mask = mask.expand(num_batches, -1, -1).unsqueeze(1)

    urls = {}  # grid圖片的url
    exception_paths = []  # 不做sr的圖片路徑

    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with latent_diffusion_model.ema_scope():
                uncoditional_conditioning = None
                if latent_diffusion_guidance_scale > 0:
                    # ""代表不考慮的prompt
                    uncoditional_conditioning = latent_diffusion_model.get_learned_conditioning(num_batches * [""])

                for model_name in chosen_models:
                    samples = []  # 儲存所有sample
                    count = 0  # 圖片編號
                    anvil.server.task_state["current_clip_model"] = model_name
                    for current_iteration in trange(num_iterations, desc="Sampling"):
                        gc.collect()
                        torch.cuda.empty_cache()

                        conditioning = latent_diffusion_model.get_learned_conditioning(num_batches * [prompts[0]])

                        # sample，只取第一個變數(samples)，不取第二個變數(intermediates)
                        samples_ddim, _ = sampler.sample(
                            S=diffusion_steps,
                            batch_size=num_batches,
                            conditioning=conditioning,
                            x0=encoded_init,
                            mask=mask,
                            shape=shape,
                            verbose=False,
                            unconditional_guidance_scale=latent_diffusion_guidance_scale,
                            unconditional_conditioning=uncoditional_conditioning,
                            eta=eta,
                        )

                        x_samples_ddim = latent_diffusion_model.decode_first_stage(samples_ddim)
                        x_samples_ddim = unnormalize_image_zero_to_one(x_samples_ddim).clamp(min=0.0, max=1.0)
                        # x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)

                        for x_sample in x_samples_ddim:
                            x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
                            filename = os.path.join(
                                batch_folder,
                                f"latent_{model_name.replace('/', '-')}_{count}.png",
                            )  # 將"/"替換為"-"避免誤認為路徑
                            image_vector = Image.fromarray(x_sample.astype(np.uint8))
                            image_preprocess = preprocessings[model_name](image_vector).unsqueeze(0).to(_device)

                            with torch.no_grad():
                                image_embeddings = clip_models[model_name].encode_image(image_preprocess)

                            image_embeddings /= image_embeddings.norm(dim=-1, keepdim=True)  # 對image_embeddings做L2 normalization，因為不在乎長度，只看特徵
                            image_vector.save(filename)
                            count += 1

                            # 做完時才記錄current_iteration
                            anvil.server.task_state["current_iteration"] = current_iteration + 1

                        samples.append(x_samples_ddim)

                    # 轉成grid形式
                    grid = torch.stack(samples, 0)
                    grid = rearrange(grid, "n b c h w -> (n b) c h w")
                    grid = make_grid(grid, nrow=num_batches)
                    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
                    grid_filename = f"{model_name.replace('/', '-')}_grid_image.png"
                    exception_paths.append(os.path.join(batch_folder, grid_filename))
                    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(batch_folder, grid_filename))  # 儲存grid圖片

                    urls[model_name] = upload_png(os.path.join(batch_folder, grid_filename))  # 儲存url

                    gc.collect()
                    torch.cuda.empty_cache()

    # 提高解析度
    super_resolution(real_esrgan_upsampler, batch_folder, exception_paths)

    return urls  # 回傳每個Clip模型生成的grid image url
