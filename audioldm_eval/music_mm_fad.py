import os
import numpy as np
import torch
import torchaudio
from torch import nn
from scipy import linalg

class FrechetAudioDistance:
    def __init__(
        self, use_pca=False, use_activation=False, verbose=False, audio_load_worker=8
    ):
        self.__get_model(use_pca=use_pca, use_activation=use_activation)
        self.verbose = verbose
        self.audio_load_worker = audio_load_worker

    def __get_model(self, use_pca=False, use_activation=False):
        """
        Params:
        -- x   : Either
            (i) a string which is the directory of a set of audio files, or
            (ii) a np.ndarray of shape (num_samples, sample_length)
        """
        self.model = torch.hub.load("/home/tjut_makunsheng/.cache/torch/hub/harritaylor_torchvggish_master","vggish", source='local')
        if not use_pca:
            self.model.postprocess = False
        if not use_activation:
            self.model.embeddings = nn.Sequential(
                *list(self.model.embeddings.children())[:-1]
            )
        self.model.eval()

    def calculate_embd_statistics(self, embd_lst):
        if isinstance(embd_lst, list):
            embd_lst = np.array(embd_lst)
        mu = np.mean(embd_lst, axis=0)
        sigma = np.cov(embd_lst, rowvar=False)
        return mu, sigma
    def get_embeddings(self, x, sr=16000, limit_num=None):
        """
        Get embeddings using VGGish model.
        Params:
        -- x    : Either
            (i) a string which is the directory of a set of audio files, or
            (ii) a list of np.ndarray audio samples
        -- sr   : Sampling rate, if x is a list of audio samples. Default value is 16000.
        """
        embd_lst = []
        x = x[0]
        embd = self.model.forward(x.numpy(), sr)
        if self.model.device == torch.device("cuda"):
            embd = embd.cpu()
        embd = embd.detach().numpy()

        return embd


    def calculate_frechet_distance(self, mu1, sigma1, mu2, sigma2, eps=1e-6):
        """
        Adapted from: https://github.com/mseitzer/pytorch-fid/blob/master/src/pytorch_fid/fid_score.py

        Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.
        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                inception net (like returned by the function 'get_predictions')
                for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                representative data set.
        Returns:
        --   : The Frechet Distance.
        """

        mu1 = np.atleast_1d(mu1)
        mu2 = np.atleast_1d(mu2)

        sigma1 = np.atleast_2d(sigma1)
        sigma2 = np.atleast_2d(sigma2)

        assert (
            mu1.shape == mu2.shape
        ), "Training and test mean vectors have different lengths"
        assert (
            sigma1.shape == sigma2.shape
        ), "Training and test covariances have different dimensions"

        diff = mu1 - mu2

        # Product might be almost singular
        covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
        if not np.isfinite(covmean).all():
            msg = (
                "fid calculation produces singular product; "
                "adding %s to diagonal of cov estimates"
            ) % eps
            print(msg)
            offset = np.eye(sigma1.shape[0]) * eps
            covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                raise ValueError("Imaginary component {}".format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)

        return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def pad_short_audio(audio, min_samples=32000):
    if(audio.size(-1) < min_samples):
        audio = torch.nn.functional.pad(audio, (0, min_samples - audio.size(-1)), mode='constant', value=0.0)
    return audio

def read_from_file(audio_file):
    audio, file_sr = torchaudio.load(audio_file)
    # Only use the first channel
    audio = audio[0:1, ...]
    audio = audio - audio.mean()

    # if file_sr != self.sr and file_sr == 32000 and self.sr == 16000:
    #     audio = audio[..., ::2]
    # if file_sr != self.sr and file_sr == 48000 and self.sr == 16000:
    #     audio = audio[..., ::3]
    # el

    if file_sr != 16000:
        audio = torchaudio.functional.resample(
            audio, orig_freq=file_sr, new_freq=16000,  # rolloff=0.95, lowpass_filter_width=16
        )
        # audio = torch.FloatTensor(librosa.resample(audio.numpy(), file_sr, self.sr))

    audio = pad_short_audio(audio, min_samples=16000)
    return audio


if __name__ == "__main__":
    frechet = FrechetAudioDistance(
        use_pca=False,
        use_activation=False,
        verbose=True,
    )
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
        
        # Calculate embedding statistics for source and target music
        mu_background, sigma_background = frechet.calculate_embd_statistics(
            frechet.get_embeddings(read_from_file(source_music_path))
        )
        mu_eval, sigma_eval = frechet.calculate_embd_statistics(
            frechet.get_embeddings(read_from_file(target_music))
        )

        # Calculate FAD score
        fad_score = frechet.calculate_frechet_distance(
            mu_background, sigma_background, mu_eval, sigma_eval
        )

        # Print FAD score for the current source music file
        print(f"{fad_score:.5f}")