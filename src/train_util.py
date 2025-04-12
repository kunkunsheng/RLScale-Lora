import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer, RobertaTokenizerFast, ClapTextModelWithProjection
from typing import Optional, List, Tuple
import torch.nn.functional as F

from audioldm_eval.audio_eval import TacotronSTFT
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
def encode_prompts(
        tokenizer: RobertaTokenizerFast,
        text_encoder: ClapTextModelWithProjection,
        prompts: List[str],
        device="cuda:0",
        num_waveforms_per_prompt=1,
):
    if prompts is not None and isinstance(prompts, str):
        batch_size = 1
    elif prompts is not None and isinstance(prompts, tuple):
        batch_size = len(prompts)

    text_inputs = tokenizer(
        prompts,
        padding="max_length",
        max_length=8,
        truncation=True,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids
    attention_mask = text_inputs.attention_mask
    prompt_embeds = text_encoder(text_input_ids.to(text_encoder.device),
                                 attention_mask=attention_mask.to(device)).text_embeds
    prompt_embeds = F.normalize(prompt_embeds, dim=-1)
    prompt_embeds = prompt_embeds.repeat(1, num_waveforms_per_prompt)

    (
        bs_embed,
        seq_len,
    ) = prompt_embeds.shape
    prompt_embeds = prompt_embeds.view(bs_embed * num_waveforms_per_prompt, seq_len)

    return prompt_embeds

def get_mel_from_wav(audio,_stft):
    audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
    audio = torch.autograd.Variable(audio, requires_grad=False)

    # =========================================================================
    # Following the processing in https://github.com/v-iashin/SpecVQGAN/blob/5bc54f30eb89f82d129aa36ae3f1e90b60e73952/vocoder/mel2wav/extract_mel_spectrogram.py#L141
    melspec, energy = _stft.mel_spectrogram(audio, normalize_fun=torch.log10)
    melspec = (melspec * 20) - 20
    melspec = (melspec + 100) / 100
    melspec = torch.clip(melspec, min=0, max=1.0)
    # =========================================================================
    # Augment
    # if(self.augment):
    #     for i in range(1):
    #         random_start = int(torch.rand(1) * 950)
    #         melspec[0,:,random_start:random_start+50] = 0.0
    # =========================================================================
    melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
    energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
    return melspec, energy

def calculate_psnr_ssim(audio_gen, audio_ground):

    n_STFT = TacotronSTFT(
        512,
        160,
        512,
        64,
        16000,
        50,
        8000,
    )
    mel_tar, _ = get_mel_from_wav(audio_gen - audio_gen.mean(), n_STFT)
    mel_sou, _ = get_mel_from_wav(audio_ground - audio_ground.mean(), n_STFT)
    psnr_val = psnr(mel_sou, mel_tar)
    data_range = max(np.max(mel_tar), np.max(mel_sou)) - min(np.min(mel_tar), np.min(mel_sou))
    ssim_val = ssim(mel_tar, mel_sou, data_range=data_range)
    return psnr_val, ssim_val