import sys
sys.path.append("XXXXXXXXXXXsrc")

import numpy as np
import torch
import torchaudio
from diffusers import AudioLDMPipeline
import scipy
from src.modules.lora_model import LoRAModule
import os


def generate_scaled_audio(
    model_id: str,
    lora_weight: str,
    prompt: str,
    negative_prompt: str,
    output_dir: str,
    step: float = 0.1,
    audio_length_in_s: float = 10.0,
    num_inference_steps: int = 200,
    guidance_scale: float = 3.5,
    sample_rate: int = 16000,
    device_index: int = 5,
):
    device = torch.device(f"cuda:{device_index}" if torch.cuda.is_available() else "cpu")
    weight_dtype = torch.bfloat16

    pipe = AudioLDMPipeline.from_pretrained(model_id, torch_dtype=weight_dtype).to(device)
    generator = torch.Generator(device).manual_seed(0)

    lora_handler = LoRAModule(root_module=pipe.unet, rank=4, multiplier=1, alpha=1.0).to(device)
    lora_handler.load_state_dict(torch.load(lora_weight), strict=True)

    os.makedirs(output_dir, exist_ok=True)

    scales = np.arange(0, 1 + step, step)
    for scale in scales:
        lora_handler.set_lora_slider(scale)

        with lora_handler:
            audio = pipe(
                prompt,
                num_inference_steps=num_inference_steps,
                audio_length_in_s=audio_length_in_s,
                negative_prompt=negative_prompt,
                guidance_scale=guidance_scale,
                generator=generator.manual_seed(10),
            ).audios[0]

        audio_path = os.path.join(output_dir, f"audio_scale_{scale:.2f}.wav")
        scipy.io.wavfile.write(audio_path, rate=sample_rate, data=audio)
        print(f"Saved audio at scale {scale:.2f} to {audio_path}")

if __name__ == "__main__":
    # Example usage:
    generate_scaled_audio(
        model_id="XXXXXXXXXXXhuggingface/audioldm",
        lora_weight="XXXXXXXXXXXsample/epic-music/model.pt",
        prompt="An orchestral piece",
        negative_prompt="low quality, average quality, no singing",
        output_dir="XXXXXXXXXXXsample/epic-music/10",
        step=0.1
    )

