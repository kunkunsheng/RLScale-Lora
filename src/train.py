import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"  # 使用 GPU 0 和 1
import torch
from tqdm import tqdm
from pathlib import Path
from src.modules.lora_model import LoRAModule
from src.modules.PPO import compute_log_probs_from_params
from src.Dataset.Dataset import AudioTextDataset
import train_util
import warnings
import random
warnings.filterwarnings("ignore")
if __name__ == "__main__":
    from diffusers import AudioLDMPipeline,DDIMScheduler

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16


    model_id = "XXXXXXXXXXXhuggingface/audioldm"
    pipe = AudioLDMPipeline.from_pretrained(model_id, torch_dtype=weight_dtype).to(device)
    unet = pipe.unet
    unet.requires_grad_(False)
    unet.eval()
    vae = pipe.vae
    vae.eval()

    save_path = Path("XXXXXXXXXXXsrc/save_models")
    data_dir = "XXXXXXXXXXXsrc/Dataset/sad-violin"
    dataset = AudioTextDataset(data_dir)
    dataloader = torch.utils.data.DataLoader(dataset=dataset,batch_size=2)
    lora_handler = LoRAModule(root_module=unet, rank=4, multiplier=1, alpha=1.0).to(device)
    lora_handler.train()
    import torch.optim as optim

    # 只优化 LoRA 层的参数
    optimizer = optim.AdamW(lora_handler.prepare_optimizer_params("LoRAModule"), lr=0.0002)
    lr_scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer,factor=1,)

    loss_fn = torch.nn.MSELoss()

    pbar = tqdm(range(1000))
    for i in pbar:
        for mel, text, unconditional, wav_path in dataloader:
            pipe.scheduler.set_timesteps(
                50, device=device
            )
            mel = mel.to(device, dtype=weight_dtype)
            init_latents = vae.encode(mel).latent_dist.sample().detach() * vae.config.scaling_factor
            positive_prompt = train_util.encode_prompts(
                pipe.tokenizer, pipe.text_encoder, text
            )
            unconditional_prompt = train_util.encode_prompts(
                pipe.tokenizer, pipe.text_encoder, unconditional
            )
            prompt = torch.cat([unconditional_prompt, positive_prompt])

            t = torch.randint(
                1, 50, (1,)
            ).item()
            seed = random.randint(0, 2 * 15)
            generator = torch.Generator(device).manual_seed(seed)
            noise = torch.randn(init_latents.shape, generator=generator, device=device).to(weight_dtype)
            time_step = pipe.scheduler.timesteps[t:t + 1]
            latents = pipe.scheduler.add_noise(init_latents, noise, time_step)

            pipe.scheduler.set_timesteps(1000)

            current_timestep = pipe.scheduler.timesteps[
                int(t * 1000 / 50)
            ]

            lora_handler.set_lora_slider(1)
            with lora_handler:
                latent_model_input = torch.cat([latents] * 2)
                latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, current_timestep)
                noise_pred = unet(latent_model_input, current_timestep, encoder_hidden_states=None, class_labels=prompt,
                                  return_dict=False)[0]
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                guided_target = noise_pred_uncond + 1 * (
                        noise_pred_text - noise_pred_uncond
                )
                loss = loss_fn(guided_target, noise)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                lr_scheduler.step()

            pbar.set_description(f"Loss: {loss.item():.6f}")
            # if "sad" in wav_path[0]:
            #
            #     lora_handler.set_lora_slider(-1)
            #     with lora_handler:
            #         latent_model_input = torch.cat([latents] * 2)
            #         latent_model_input = pipe.scheduler.scale_model_input(latent_model_input, time_step)
            #         noise_pred = \
            #         unet(latent_model_input, time_step, encoder_hidden_states=None, class_labels=prompt, return_dict=False)[
            #             0]
            #         noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            #         guided_target = noise_pred_uncond + 1 * (
            #                 noise_pred_text - noise_pred_uncond
            #         )
            #         loss = loss_fn(guided_target, noise)
            #
            #         # **优化器步骤**
            #         optimizer.zero_grad()
            #         loss.backward()
            #         optimizer.step()
            #         lr_scheduler.step()

        if (i + 1) % 500 == 0:
            print(f"Saving at iteration {i + 1}...")
            save_path.mkdir(parents=True, exist_ok=True)
            lora_handler.save_weights(
                save_path / f"test_last_sad-violin_iter_{i + 1}.pt",
                dtype=weight_dtype,
            )