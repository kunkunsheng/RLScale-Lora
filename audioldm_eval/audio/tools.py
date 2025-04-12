import torch
import numpy as np
import torchaudio
from scipy.io.wavfile import write
import pickle
import json
from audioldm_eval.audio.audio_processing import griffin_lim


def save_pickle(obj, fname):
	print("Save pickle at " + fname)
	with open(fname, "wb") as f:
		pickle.dump(obj, f)


def load_pickle(fname):
	print("Load pickle at " + fname)
	with open(fname, "rb") as f:
		res = pickle.load(f)
	return res


def write_json(my_dict, fname):
	print("Save json file at " + fname)
	json_str = json.dumps(my_dict)
	with open(fname, "w") as json_file:
		json_file.write(json_str)


def load_json(fname):
	with open(fname, "r") as f:
		data = json.load(f)
		return data


def get_mel_from_wav(audio, _stft):
	audio = torch.clip(torch.FloatTensor(audio).unsqueeze(0), -1, 1)
	audio = torch.autograd.Variable(audio, requires_grad=False)
	melspec, log_magnitudes_stft, energy = _stft.mel_spectrogram(audio)
	melspec = torch.squeeze(melspec, 0).numpy().astype(np.float32)
	log_magnitudes_stft = (
		torch.squeeze(log_magnitudes_stft, 0).numpy().astype(np.float32)
	)
	energy = torch.squeeze(energy, 0).numpy().astype(np.float32)
	return melspec, log_magnitudes_stft, energy


def inv_mel_spec(mel, out_filename, _stft, griffin_iters=60):
	mel = torch.stack([mel])
	mel_decompress = _stft.spectral_de_normalize(mel)
	mel_decompress = mel_decompress.transpose(1, 2).data.cpu()
	spec_from_mel_scaling = 1000
	spec_from_mel = torch.mm(mel_decompress[0], _stft.mel_basis)
	spec_from_mel = spec_from_mel.transpose(0, 1).unsqueeze(0)
	spec_from_mel = spec_from_mel * spec_from_mel_scaling

	audio = griffin_lim(
		torch.autograd.Variable(spec_from_mel[:, :, :-1]), _stft._stft_fn, griffin_iters
	)

	audio = audio.squeeze()
	audio = audio.cpu().numpy()
	audio_path = out_filename
	write(audio_path, _stft.sampling_rate, audio)


def _pad_spec(fbank, target_length=1024):
	n_frames = fbank.shape[0]
	p = target_length - n_frames
	# cut and pad
	if p > 0:
		m = torch.nn.ZeroPad2d((0, 0, 0, p))
		fbank = m(fbank)
	elif p < 0:
		fbank = fbank[0:target_length, :]

	if fbank.size(-1) % 2 != 0:
		fbank = fbank[..., :-1]

	return fbank

def pad_wav(waveform, segment_length):
	waveform_length = waveform.shape[-1]
	assert waveform_length > 100, "Waveform is too short, %s" % waveform_length
	if segment_length is None or waveform_length == segment_length:
		return waveform
	elif waveform_length > segment_length:
		return waveform[:segment_length]
	elif waveform_length < segment_length:
		temp_wav = np.zeros((1, segment_length))
		temp_wav[:, :waveform_length] = waveform

	return temp_wav


def normalize_wav(waveform):
	waveform = waveform - np.mean(waveform)
	waveform = waveform / (np.max(np.abs(waveform)) + 1e-8)
	return waveform * 0.5


def read_wav_file(filename, segment_length):
	# waveform, sr = librosa.load(filename, sr=None, mono=True) # 4 times slower
	waveform, sr = torchaudio.load(filename)  # Faster!!!
	waveform = torchaudio.functional.resample(waveform, orig_freq=sr, new_freq=16000)
	waveform = waveform.numpy()[0, ...]
	waveform = normalize_wav(waveform)
	waveform = waveform[None, ...]
	waveform = pad_wav(waveform, segment_length)

	waveform = waveform / np.max(np.abs(waveform))
	waveform = 0.5 * waveform

	return waveform

def wav_to_fbank(filename, target_length=1024, fn_STFT=None):
	assert fn_STFT is not None

	# mixup
	waveform = read_wav_file(filename, target_length * 160)  # hop size is 160

	waveform = waveform[0, ...]
	waveform = torch.FloatTensor(waveform)

	fbank, log_magnitudes_stft, energy = get_mel_from_wav(waveform, fn_STFT)

	fbank = torch.FloatTensor(fbank.T)
	log_magnitudes_stft = torch.FloatTensor(log_magnitudes_stft.T)

	fbank, log_magnitudes_stft = _pad_spec(fbank, target_length), _pad_spec(
		log_magnitudes_stft, target_length
	)

	return fbank, log_magnitudes_stft, waveform
