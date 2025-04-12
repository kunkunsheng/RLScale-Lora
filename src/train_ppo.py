import numpy as np
import torch
from diffusers import AudioLDMPipeline
import scipy
from tqdm import tqdm

from audioldm_eval.audio import wav_to_fbank, TacotronSTFT
from src.modules.lora_model import LoRAModule
from src.modules.PPO import compute_log_probs_from_params
from src.train_util import calculate_psnr_ssim

if __name__ == "__main__":

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16

    model_id = "XXXXXXXXXXXhuggingface/audioldm"
    lora_weight = "XXXXXXXXXXXsrc/save_models/test_last.pt"
    pipe = AudioLDMPipeline.from_pretrained(model_id, torch_dtype=weight_dtype).to(device)
    generator = torch.Generator(device).manual_seed(0)
    prompt = "A piano track,"
    negative_prompt = "low quality, average quality,no human singing"

    lora_handler = LoRAModule(root_module=pipe.unet, rank=4, multiplier=1, alpha=1.0).to(device)
    lora_handler.load_state_dict(torch.load(lora_weight), strict=False)

    optimizer = torch.optim.AdamW(lora_handler.prepare_optimizer_params("AdjustModule"), lr=1e-4)

    lora_handler.set_lora_slider(1)


    with lora_handler:
        audio_ground = pipe(prompt, num_inference_steps=200, audio_length_in_s=10.0, negative_prompt=negative_prompt,
                     guidance_scale=3.5,
                     generator=generator.manual_seed(42)).audios[0]
    lora_handler.set_lora_slider(1)
    with torch.no_grad():
        with lora_handler:
            audio_old = pipe(prompt, num_inference_steps=200, audio_length_in_s=10.0, negative_prompt=negative_prompt,
                             guidance_scale=3.5,
                             generator=generator.manual_seed(42)).audios[0]
        psnr_val_old, ssim_val_t= calculate_psnr_ssim(audio_old, audio_ground)

    
        lora_handler.set_lora_slider(0)
    with torch.no_grad():
        with lora_handler:
            audio_old = pipe(prompt, num_inference_steps=200, audio_length_in_s=10.0, negative_prompt=negative_prompt,
                             guidance_scale=3.5,
                             generator=generator.manual_seed(42)).audios[0]
        psnr_val_old, ssim_val_s = calculate_psnr_ssim(audio_old, audio_ground)
    w = 1
    beta = 0.5

    ssim_val_r = ssim_val_s + (ssim_val_t- ssim_val_s) * w^beta

    for epoch in tqdm(range(1000), desc="Training Progress"):
        print("ssim当前的指标：", ssim_val_old)
        # 设置控制器滑块
        lora_handler.set_lora_slider(0.5)
        param_trajectories = lora_handler.perturb_lora_params()
        
        # 直接从参数权重计算 log_probs
        log_probs_old, log_probs_new = compute_log_probs_from_params(param_trajectories)

        # 计算 PPO 目标
        ratios = torch.exp(log_probs_new - log_probs_old)

        with lora_handler:
            audio = pipe(prompt, num_inference_steps=200, audio_length_in_s=10.0, 
                        negative_prompt=negative_prompt, guidance_scale=3.5, 
                        generator=generator.manual_seed(42)).audios[0]

        # 计算 PSNR 和 SSIM
        psnr_val, ssim_val = calculate_psnr_ssim(audio, audio_ground)
        
        # 定义奖励函数区间
        reward_sigma = 0.1  # 控制奖励区间的宽度
        reward_center = ssim_val_old  # 以上一次的SSIM作为中心
        reward = torch.exp(-((ssim_val - reward_center) ** 2) / (2 * reward_sigma ** 2))

        # 归一化或加权分配奖励
        num_steps = len(param_trajectories)
        scaled_rewards = [reward * (i + 1) / num_steps for i in range(num_steps)]  # 线性缩放示例

        # 计算优势函数
        advantages = ssim_val - torch.mean(torch.tensor(scaled_rewards))

        print("Advantages:", advantages)

        # 更新 SSIM 的旧值
        ssim_val_old = ssim_val



        pg_losses = - advantages* ratios
        optimizer.zero_grad()  # 清空梯度
        pg_losses.mean().backward()  # 计算梯度
        optimizer.step()  # 更新参数
    lora_handler.set_lora_slider(0.5)
    with torch.no_grad():
        with lora_handler:
            audio_old = pipe(prompt, num_inference_steps=200, audio_length_in_s=10.0, negative_prompt=negative_prompt,
                             guidance_scale=3.5,
                             generator=generator.manual_seed(32)).audios[0]
        psnr_val_old, ssim_val_old = calculate_psnr_ssim(audio, audio_ground)
        print(ssim_val_old)


