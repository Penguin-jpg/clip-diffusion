import torch
import os
import lpips
import anvil
import numpy as np
from PIL import Image
from ipywidgets import Output
from einops import rearrange, repeat
from torchvision.utils import make_grid
from clip_diffusion.config import Config
from clip_diffusion.prompt import Prompt
from clip_diffusion.preprocessing import (
    get_text_embeddings_and_text_weights,
    create_init_noise,
    create_mask_tensor,
)
from clip_diffusion.models import (
    load_clip_models,
    load_guided_diffusion_model,
    load_latent_diffusion_model,
    load_real_esrgan_upsampler,
    load_aesthetic_predictors,
)
from clip_diffusion.utils.functional import (
    random_seed,
    clear_output,
    set_seed,
    clear_gpu_cache,
    get_sample_function,
    embed_image,
    get_sampler,
    set_display_widget,
    display_image,
    store_task_state,
    draw_index_on_grid_image,
)
from clip_diffusion.cutouts import make_cutouts
from clip_diffusion.losses import (
    square_spherical_distance_loss,
    total_variational_loss,
    LPIPS_loss,
    aesthetic_loss,
)
from clip_diffusion.utils.dir_utils import make_dir, OUTPUT_PATH
from clip_diffusion.utils.image_utils import (
    unnormalize_image_zero_to_one,
    tensor_to_pillow_image,
    upload_image,
    super_resolution,
)

LPIPS_model = lpips.LPIPS(net="vgg").to(Config.device)
clip_models = load_clip_models(Config.chosen_clip_models, Config.device)
aesthetic_predictors = load_aesthetic_predictors(Config.chosen_predictors, Config.device)
latent_diffusion_model = None
real_esrgan_upsampler = None


@anvil.server.background_task
def guided_diffusion_sample(
    prompt="A cute golden retriever.",
    use_auto_modifiers=False,
    num_modifiers=1,
    custom_model_path=None,
    dynamic_thresholding_percentile=0.995,
    seed=None,
    init_image=None,
    sample_mode="ddim",
    steps=200,
    skip_timesteps=0,
    eta=0.8,
    num_batches=1,
    display_rate=25,
    gif_duration=500,
):
    """
    生成圖片(和anvil client互動)
    prompt: 生成敘述
    use_auto_modifiers: 是否要使用自動補上修飾詞
    num_modifiers: 補上的修飾詞數量
    custom_model_path: 自訂模型的路徑
    dynamic_thresholding_percentile: dynamic thresholding中選擇的百分位數(當作threshold)
    seed: 生成種子
    init_image: 模型會參考該圖片生成初始雜訊(會是anvil的Media類別)
    sample_mode: 使用的sample模式(ddim, plms)
    steps: 每個batch要跑的step數
    skip_timesteps: 控制要跳過的step數(從第幾個step開始)，當使用init_image時最好調整為diffusion_steps的0~50%
    eta: DDIM與DDPM的比例(0.0: 純DDIM; 1.0: 純DDPM)，越高每個timestep加入的雜訊越多
    num_batches: 要生成的圖片數量
    display_rate: 生成過程的gif多少個step要更新一次
    gif_duration: gif的播放時間
    """
    global clip_models, LPIPS_model, aesthetic_predictors

    prompt = Prompt(prompt, use_auto_modifiers, num_modifiers)  # 建立Prompt物件
    if use_auto_modifiers:
        store_task_state("new_prompt", prompt.text)
    # 載入diffusion model和diffusion
    model, diffusion = load_guided_diffusion_model(custom_model_path=custom_model_path, steps=steps, device=Config.device)
    batch_folder = os.path.join(OUTPUT_PATH, "guided")  # 儲存圖片的資料夾
    make_dir(batch_folder, remove_old=True)

    # 設定種子
    if not seed:
        seed = random_seed()
    set_seed(int(seed))

    # 取得prompt的embedding及weight
    text_embeddings_and_weights = get_text_embeddings_and_text_weights(prompt, clip_models, Config.device)
    # 建立初始雜訊
    init_noise = create_init_noise(init_image, (Config.width, Config.height), Config.device)
    current_timestep = None  # 目前的timestep

    def denoised_function(x_start):
        """
        在計算p_mean_variance時套用在x_start上的function
        在此使用Imagen的dynamic thresholding方法
        """

        threshold = torch.quantile(
            rearrange(x_start, "b ... -> b (...)").abs(),
            dynamic_thresholding_percentile,
            dim=-1,
        )
        threshold = threshold.clamp(min=1.0)  # 最小值要為1
        threshold = threshold.view(*threshold.shape, *((1,) * (x_start.ndim - threshold.ndim)))  # pad到和x_start一樣的維度
        x_start = x_start.clamp(min=-threshold, max=threshold) / threshold
        return x_start

    @torch.enable_grad()
    def conditon_function(x, t, y=None):
        """
        透過clip引導guided diffusion(計算grad(log(p(y|x))))
        x: 上一個timestep的圖片tensor
        t: diffusion timestep tensor
        y: class
        """
        # 將x從目前的計算圖中取出
        x = x.detach().requires_grad_()
        batch_size = x.shape[0]
        # 目前timestep轉tensor
        current_timestep_tensor = torch.ones([batch_size], device=Config.device, dtype=torch.long) * current_timestep
        p_mean_var = diffusion.p_mean_variance(model, x, current_timestep_tensor, clip_denoised=False, model_kwargs={"y": y})
        factor = diffusion.sqrt_one_minus_alphas_cumprod[current_timestep]
        x_in = p_mean_var["pred_xstart"] * factor + x * (1 - factor)  # 將x0與目前x以一定比例相加並當成輸入
        grad_tensor = torch.zeros_like(x_in)  # 在計算最後梯度時和x_in做內積
        # 總共1000個diffusion steps，每次會減掉(1000/steps)
        total_diffusion_steps_minus_passed_steps = int(t.item()) + 1
        # 目前的diffusion timestep
        current_diffusion_step = 1000 - total_diffusion_steps_minus_passed_steps

        for clip_model_name, clip_model in clip_models.items():
            for _ in range(Config.num_cutout_batches):
                aesthetic_score = None  # aesthetic loss計算的值
                # 做cutouts
                cutout_images = make_cutouts(
                    input=x_in,
                    cut_size=clip_model.visual.input_resolution,
                    num_overview_cuts=Config.num_overview_cuts_schedule[current_diffusion_step],
                    num_inner_cuts=Config.num_inner_cuts_schedule[current_diffusion_step],
                    inner_cut_size_power=Config.inner_cut_size_power_schedule[current_diffusion_step],
                    cut_gray_portion=Config.cut_gray_portion_schedule[current_diffusion_step],
                )
                image_embeddings = embed_image(clip_model, cutout_images, clip_normalize=True)

                if clip_model_name in aesthetic_predictors.keys():
                    aesthetic_score = aesthetic_loss(aesthetic_predictors[clip_model_name], image_embeddings)

                # 計算square spherical distance loss
                distances = square_spherical_distance_loss(
                    image_embeddings.unsqueeze(1),
                    text_embeddings_and_weights[clip_model_name]["embeddings"].unsqueeze(0),
                )
                # 將shape調整為(num_cuts, batch_size, 1) (-1是把剩下的維度都補進來)
                distances = distances.view(
                    [
                        Config.num_overview_cuts_schedule[current_diffusion_step]
                        + Config.num_inner_cuts_schedule[current_diffusion_step],
                        batch_size,
                        -1,
                    ]
                )

                # 對最後一個維度取平均
                distance_loss = distances.mul(text_embeddings_and_weights[clip_model_name]["weights"]).sum(dim=2).mean(dim=0)
                if aesthetic_score is not None:
                    grad_tensor += (
                        torch.autograd.grad(
                            distance_loss.sum() * Config.clip_guidance_scale - aesthetic_score * Config.aesthetic_scale, x_in
                        )[0]
                        / Config.num_cutout_batches
                    )
                else:
                    grad_tensor += (
                        torch.autograd.grad(distance_loss.sum() * Config.clip_guidance_scale, x_in)[0] / Config.num_cutout_batches
                    )

        # 計算total variational loss
        denoise_loss = total_variational_loss(x_in)
        loss_sum = denoise_loss.sum() * Config.denoise_scale
        # 計算perceptual loss
        if init_noise is not None:
            perceptual_loss = LPIPS_loss(LPIPS_model, x_in, init_noise)
            loss_sum += perceptual_loss.sum() * Config.LPIPS_scale
        grad_tensor += torch.autograd.grad(loss_sum, x_in)[0]

        if not torch.isnan(grad_tensor).any():
            grad = -torch.autograd.grad(x_in, x, grad_tensor)[0]  # 取負是因為使用的每項loss均為值越低越好，所以改為最大化負數(最小化正數)
        else:
            return torch.zeros_like(x)

        # 使用梯度的RMS當作調整用的強度
        magnitude = grad.square().mean().sqrt()
        # 限制cond_fn中的梯度大小(避免產生一些極端生成結果)
        return grad * magnitude.clamp(min=-Config.grad_threshold, max=Config.grad_threshold) / magnitude

    image_display = Output()  # 在server端顯示圖片
    gif_urls = []  # 生成過程的gif url
    images = []  # 最後一個timestep的圖片
    for batch_index in range(num_batches):
        clear_output(wait=True)
        set_display_widget(image_display)
        store_task_state("current_batch", batch_index)  # 將目前的batch index存到current_batch
        store_task_state("current_result", None)  # 初始化
        clear_gpu_cache()

        # 將目前timestep的值初始化為總timestep數-1
        current_timestep = diffusion.num_timesteps - skip_timesteps - 1
        # 根據sample_mode選擇`sample_function
        sample_function = get_sample_function(diffusion, mode=sample_mode)

        # 根據不同function傳入參數
        if sample_mode == "ddim":  # ddim
            samples = sample_function(
                model=model,
                shape=(1, 3, Config.height, Config.width),  # shape=(batch_size, num_channels, height, width)
                clip_denoised=False,
                denoised_fn=denoised_function,
                model_kwargs={},
                cond_fn=conditon_function,
                progress=True,
                skip_timesteps=skip_timesteps,
                init_image=init_noise,
                randomize_class=True,
                eta=eta,
            )
        else:  # plms
            samples = sample_function(
                model=model,
                shape=(1, 3, Config.height, Config.width),
                clip_denoised=False,
                denoised_fn=denoised_function,
                model_kwargs={},
                cond_fn=conditon_function,
                progress=True,
                skip_timesteps=skip_timesteps,
                init_image=init_noise,
                randomize_class=True,
                order=2,
            )

        # current_timestep從總timestep數開始；step_index從0開始
        for step_index, sample in enumerate(samples):
            current_timestep -= 1  # 每次都將目前的timestep減1
            with image_display:
                # 更新、儲存圖片
                for image_tensor in sample["pred_xstart"]:
                    filename = f"guided_{batch_index}_{step_index:04}.png"  # 圖片名稱
                    image_path = os.path.join(batch_folder, filename)  # 圖片路徑
                    # 將image_tensor範圍轉回[0, 1]，並用clamp確保範圍正確
                    image_tensor = unnormalize_image_zero_to_one(image_tensor).clamp(min=0.0, max=1.0)
                    image = tensor_to_pillow_image(image_tensor)  # 轉換為Pillow Image
                    image.save(image_path)
                    display_image(image_path=image_path)
                    clear_output(wait=(current_timestep != -1))
                    # 生成結束
                    if current_timestep == -1:
                        # 將最後一個timestep的url存到current_result
                        store_task_state("current_result", upload_image(image_path, "png"))
                        # 儲存生成過程的gif url
                        gif_urls.append(
                            upload_image(
                                batch_folder=batch_folder,
                                batch_index=batch_index,
                                display_rate=display_rate,
                                gif_duration=gif_duration,
                                append_last_timestep=(steps - skip_timesteps - 1) % display_rate,
                            )
                        )
                        # 儲存最後一個timestep的圖片
                        images.append(Image.open(image_path))
                    elif step_index % 10 == 0:  # 每10個timestep更新上傳一次圖片
                        # 將目前圖片的url存到current_result
                        store_task_state("current_result", upload_image(image_path, "png"))

            store_task_state("current_step", step_index + 1)  # 紀錄目前的step
        clear_gpu_cache()
    return gif_urls  # 回傳gif url


@anvil.server.background_task
def latent_diffusion_sample(
    prompt="A cute golden retriever.",
    seed=None,
    init_image=None,
    mask_image=None,
    sample_mode="ddim",
    diffusion_steps=50,
    eta=0.0,
    latent_diffusion_guidance_scale=5,
    num_iterations=3,
    num_batches=3,
    sample_width=256,
    sample_height=256,
):
    """
    使用latent diffusion生成圖片
    prompt: 生成敘述
    seed: 生成種子
    init_image: 要配合inpaint使用的圖片
    mask_image: inpaint用的遮罩
    sample_mode: 使用的sample模式(ddim, plms)
    diffusion_steps: latent diffusion要跑的step數
    eta: latent diffusion的eta
    latent_diffusion_guidance_scale: latent diffusion unconditional的引導強度(介於0~15，多樣性隨著數值升高)
    num_iterations: 做幾次latent diffusion生成
    num_batches: 要生成的圖片數
    sample_width:  sample圖片的寬(latent diffusion sample的圖片不能太大，後續再用sr提高解析度)
    sample_height: sample圖片的高
    """
    global latent_diffusion_model, real_esrgan_upsampler

    if latent_diffusion_model is None:
        latent_diffusion_model = load_latent_diffusion_model(Config.device)

    if real_esrgan_upsampler is None:
        real_esrgan_upsampler = load_real_esrgan_upsampler(scale=4, device=Config.device)

    prompt = Prompt(prompt, False, 0)
    sampler = get_sampler(latent_diffusion_model, mode=sample_mode)  # 根據sample_mode選擇sampler
    batch_folder = os.path.join(OUTPUT_PATH, "latent")
    make_dir(batch_folder, remove_old=True)

    # 設定種子
    if not seed:
        seed = random_seed()
    set_seed(int(seed))

    # 當使用plms時，eta沒有作用
    if sample_mode == "plms":
        eta = 0.0

    # sample的shape
    shape = (4, sample_height // 8, sample_width // 8)
    # encode過的init_image
    encoded_init = None
    # 遮罩tensor
    mask = None
    # 處理inpaint的參數
    if init_image is not None and mask_image is not None:
        # 將init_image當成初始雜訊(shape=(1, num_channels, height, width))
        init = create_init_noise(init_image, (sample_width, sample_height), device=Config.device).half()
        init = repeat(init, "1 c h w -> b c h w", b=num_batches)  # 將shape變成(batch_size, num_channels, height, width)
        encoder_posterior = latent_diffusion_model.encode_first_stage(init)  # 使用encoder對init encode
        encoded_init = latent_diffusion_model.get_first_stage_encoding(encoder_posterior).detach()  # 取出encode的結果

        # mask為黑白圖片的tensor(shape=(1, num_channels, height, width)，黑白圖片的num_channels=1)
        mask = create_mask_tensor(mask_image, (shape[2], shape[1]), Config.device)
        mask = repeat(mask, "1 c h w -> b c h w", b=num_batches)  # 將shape變成(batch_size, num_channels, height, width)

    exception_paths = []  # 不做sr的圖片路徑
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            with latent_diffusion_model.ema_scope():
                uncoditional_conditioning = None
                if latent_diffusion_guidance_scale > 0:
                    # ""代表不考慮的prompt
                    uncoditional_conditioning = latent_diffusion_model.get_learned_conditioning(num_batches * [""])
                samples = []  # 儲存所有sample
                count = 0  # 圖片編號
                for current_iteration in range(num_iterations):
                    clear_gpu_cache()
                    conditioning = latent_diffusion_model.get_learned_conditioning(num_batches * [prompt.text])
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

                    for x_sample in x_samples_ddim:
                        x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
                        image_path = os.path.join(
                            batch_folder,
                            f"latent_{count}.png",
                        )  # 將"/"替換為"-"避免誤認為路徑
                        image_vector = Image.fromarray(x_sample.astype(np.uint8))
                        image_vector.save(image_path)
                        display_image(image_path)
                        count += 1

                        # 做完時才記錄current_iteration
                        store_task_state("current_iteration", current_iteration + 1)

                    samples.append(x_samples_ddim)

                    # 轉成grid形式
                    grid = torch.stack(samples, 0)
                    grid = rearrange(grid, "n b c h w -> (n b) c h w")
                    grid = make_grid(grid, nrow=num_batches)
                    grid = 255.0 * rearrange(grid, "c h w -> h w c").cpu().numpy()
                    grid_filename = f"latent_grid_image.png"
                    exception_paths.append(os.path.join(batch_folder, grid_filename))

                    # 畫上index
                    grid_image = Image.fromarray(grid.astype(np.uint8))
                    grid_image = draw_index_on_grid_image(grid_image, num_iterations, num_batches, sample_height, sample_width)
                    grid_image.save(os.path.join(batch_folder, grid_filename))  # 儲存grid圖片
                    grid_image_url = upload_image(os.path.join(batch_folder, grid_filename), "png")  # 儲存url
                    clear_gpu_cache()

    # 提高解析度
    super_resolution(real_esrgan_upsampler, batch_folder, exception_paths)
    clear_output()
    return grid_image_url  # 格狀圖片的url
