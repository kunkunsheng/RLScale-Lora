import os
import numpy as np
import torch
import torchaudio
import audio_eval as Audio
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim
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


def get_mel_from_file(_stft, audio_file):
    audio, file_sr = torchaudio.load(audio_file)
    # Only use the first channel
    audio = audio[0:1,...]
    audio = audio - audio.mean()
    if file_sr != 16000:
        audio = torchaudio.functional.resample(
            audio, orig_freq=file_sr, new_freq=16000
        )

    if _stft is not None:
        melspec, energy = get_mel_from_wav(audio[0, ...],_stft)
    else:
        melspec, energy = None, None

    return melspec, energy, audio

if __name__ == "__main__":

    stft = Audio.TacotronSTFT(512, 160, 512, 64, 16000, 50, 8000)
            # Directory containing source music files
    source_dir = "XXXXXXXXXXXsample"

    # Path to target music file
    target_music = "XXXXXXXXXXXsample"

    # List all files in the source directory
    source_files = sorted([f for f in os.listdir(source_dir) if f.endswith(".wav")])

    # Iterate through each source music file and calculate FAD score
    for source_file in source_files:
        # Construct full file path for the source music file
        source_music_path = os.path.join(source_dir, source_file)

        mel_gen,_,_ =  get_mel_from_file(stft,source_music_path)
        mel_target, _, _ =  get_mel_from_file(stft,target_music)
        min_len = min(mel_gen.shape[-1], mel_target.shape[-1])
        mel_gen = mel_gen[..., :min_len]
        mel_target = mel_target[..., :min_len]
        mel_gen = mel_gen[0]
        mel_target = mel_target[0]
        psnrval = psnr(mel_gen, mel_target)
        data_range = max(np.max(mel_gen), np.max(mel_target)) - min(np.min(mel_gen), np.min(mel_target))
        ssimval = ssim(mel_gen, mel_target, data_range=data_range)
        print(psnrval)
    for source_file in source_files:
        # Construct full file path for the source music file
        source_music_path = os.path.join(source_dir, source_file)

        mel_gen,_,_ =  get_mel_from_file(stft,source_music_path)
        mel_target, _, _ =  get_mel_from_file(stft,target_music)
        min_len = min(mel_gen.shape[-1], mel_target.shape[-1])
        mel_gen = mel_gen[..., :min_len]
        mel_target = mel_target[..., :min_len]
        mel_gen = mel_gen[0]
        mel_target = mel_target[0]
        psnrval = psnr(mel_gen, mel_target)
        data_range = max(np.max(mel_gen), np.max(mel_target)) - min(np.min(mel_gen), np.min(mel_target))
        ssimval = ssim(mel_gen, mel_target, data_range=data_range)
        print(ssimval)

