import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchaudio.transforms as T
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from audioldm_eval.metrics.fad import FrechetAudioDistance
from audioldm_eval.datasets.load_mel import load_npy_data, MelPairedDataset, WaveDataset
# import audioldm_eval.audio_eval as Audio
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim









class Evaluation:
    def __init__(self, sampling_rate,backbone="mert"):
        self.sampling_rate = sampling_rate
        self.backbone = backbone

        self.frechet = FrechetAudioDistance(
            use_pca=False,
            use_activation=False,
            verbose=True,
        )

        if self.backbone == "mert":
            self.mel_model = AutoModel.from_pretrained("m-a-p/MERT-v1-95M", trust_remote_code=True)
            self.processor = Wav2Vec2FeatureExtractor.from_pretrained("m-a-p/MERT-v1-95M",trust_remote_code=True)
            self.target_sample_rate = self.processor.sampling_rate
            self.resampler = T.Resample(orig_freq=self.sampling_rate, new_freq=self.target_sample_rate).to(self.device)

        if self.sampling_rate == 16000:
            self._stft = Audio.TacotronSTFT(512, 160, 512, 64, 16000, 50, 8000)
        elif self.sampling_rate == 32000:
            self._stft = Audio.TacotronSTFT(1024, 320, 1024, 64, 32000, 50, 14000)
        else:
            raise ValueError(
                "We only support the evaluation on 16kHz and 32kHz sampling rate."
            )
        self.fbin_mean, self.fbin_std = None, None
        generate_files_path = "XXXXXXXXXXXgenerate"
        groundtruth_path = "XXXXXXXXXXXgroundtruth"
        
        
        pairedloader = DataLoader(
            MelPairedDataset(
                generate_files_path,
                groundtruth_path,
                self._stft,
                self.sampling_rate,
                self.fbin_mean,
                self.fbin_std,
                limit_num=None,
            ),
            batch_size=1,
            sampler=None,
            num_workers=16,
        )
        
        metric_psnr_ssim = self.calculate_psnr_ssim(pairedloader, same_name=True)
        print(metric_psnr_ssim)

        metric_lsd = self.calculate_lsd(pairedloader, same_name=same_name)

    def calculate_psnr_ssim(self, pairedloader, same_name=True):
        if same_name == False:
            return {"psnr": -1, "ssim": -1}
        psnr_avg = []
        ssim_avg = []
        for mel_gen, mel_target, filename, _ in tqdm(pairedloader):
            mel_gen = mel_gen.cpu().numpy()[0]
            mel_target = mel_target.cpu().numpy()[0]
            psnrval = psnr(mel_gen, mel_target)
            if np.isinf(psnrval):
                print("Infinite value encountered in psnr %s " % filename)
                continue
            psnr_avg.append(psnrval)
            data_range = max(np.max(mel_gen), np.max(mel_target)) - min(np.min(mel_gen), np.min(mel_target))
            ssim_avg.append(ssim(mel_gen, mel_target, data_range=data_range))
        return {"psnr": np.mean(psnr_avg), "ssim": np.mean(ssim_avg)}

    def lsd(self, audio1, audio2):
        result = self.lsd_metric.evaluation(audio1, audio2, None)
        return result

    def calculate_lsd(self, pairedloader, same_name=True, time_offset=160 * 7):
        if same_name == False:
            return {
                "lsd": -1,
                "ssim_stft": -1,
            }
        print("Calculating LSD using a time offset of %s ..." % time_offset)
        lsd_avg = []
        ssim_stft_avg = []
        for _, _, filename, (audio1, audio2) in tqdm(pairedloader):
            audio1 = audio1.cpu().numpy()[0, 0]
            audio2 = audio2.cpu().numpy()[0, 0]

            # If you use HIFIGAN (verified on 2023-01-12), you need seven frames' offset
            audio1 = audio1[time_offset:]

            audio1 = audio1 - np.mean(audio1)
            audio2 = audio2 - np.mean(audio2)

            audio1 = audio1 / np.max(np.abs(audio1))
            audio2 = audio2 / np.max(np.abs(audio2))

            min_len = min(audio1.shape[0], audio2.shape[0])

            audio1, audio2 = audio1[:min_len], audio2[:min_len]

            result = self.lsd(audio1, audio2)

            lsd_avg.append(result["lsd"])
            ssim_stft_avg.append(result["ssim"])

        return {"lsd": np.mean(lsd_avg), "ssim_stft": np.mean(ssim_stft_avg)}

if __name__ == "__main__":
    evaluation = Evaluation(16000)
    generate_files_path = "/data/HDD1/tjut_makunsheng/Lora-AudioLDM2/music-mm/generate"
    groundtruth_path = "/data/HDD1/tjut_makunsheng/Lora-AudioLDM2/music-mm/groundtruth"

    fad_score = evaluation.frechet.score(generate_files_path, groundtruth_path, limit_num=None, recalculate=False)
    print(fad_score)