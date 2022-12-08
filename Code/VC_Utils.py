import sys
import numpy as np
import torch
from scipy import signal
from scipy.signal import get_window
from librosa.filters import mel
import librosa
from numpy.random import RandomState
import matplotlib.pyplot as plt
from math import ceil

from SpeakerEncoder import SpeakerEncoder


class VC_Utils:
	_DEFAULT_fs = 16000
	_DEFAULT_highpass_cutoff = 30
	_DEFAULT_highpass_order = 5
	_DEFAULT_mel_nfft = 1024
	_DEFAULT_mel_fmin = 80
	_DEFAULT_mel_fmax = 7600
	_DEFAULT_mel_n_mels = 80
	_DEFAULT_stft_fft_len = 1024
	_DEFAULT_stft_hop_len = 256


	@classmethod
	def read_audio_file(cls, filepath, fs_desired : float = None):
		if fs_desired is None:
			x, fs = librosa.load(filepath)
		else:
			x, fs = librosa.load(filepath, sr=fs_desired)
		return x, fs


	@classmethod
	def pySTFT(cls, x, fft_length=_DEFAULT_stft_fft_len, hop_length=_DEFAULT_stft_hop_len):
		x = np.pad(x, int(fft_length//2), mode='reflect')

		noverlap = fft_length - hop_length
		shape = x.shape[:-1]+((x.shape[-1]-noverlap)//hop_length, fft_length)
		strides = x.strides[:-1]+(hop_length*x.strides[-1], x.strides[-1])
		result = np.lib.stride_tricks.as_strided(x, shape=shape,
												strides=strides)

		fft_window = get_window('hann', fft_length, fftbins=True)
		result = np.fft.rfft(fft_window * result, n=fft_length).T

		return np.abs(result)


	@classmethod
	def butter_highpass(cls, cutoff, fs, order=5):
		nyq = 0.5 * fs
		normal_cutoff = cutoff / nyq
		b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
		return b, a


	@classmethod
	def invert_mel_spectrogram_rough(	cls,
										spec_dB_scaled			: np.ndarray or torch.Tensor,
										fs						: float	= _DEFAULT_fs,
										hop_length				: int	= _DEFAULT_stft_hop_len,
										mel_nfft				: int	= _DEFAULT_mel_nfft,
										mel_fmin				: float	= _DEFAULT_mel_fmin,
										mel_fmax				: float	= _DEFAULT_mel_fmax,
										mel_n_mels				: int	= _DEFAULT_mel_n_mels,
										randomize_phase			: bool	= True
									):
		if type(spec_dB_scaled) is not torch.Tensor:
			spec_dB_scaled = torch.tensor(spec_dB_scaled)
		spec_dB = (spec_dB_scaled * 100) - 100
		spec = (10 ** (spec_dB / 20)) + 0j

		mel_basis = torch.tensor(VC_Utils.get_mel_basis(fs, mel_nfft, mel_fmin=mel_fmin, mel_fmax=mel_fmax, mel_n_mels=mel_n_mels)) + 0j
		fft_frames = torch.matmul(mel_basis, spec.unsqueeze(-1)).squeeze(-1)

		if randomize_phase:
			# Randomly shifting phases so we don't get impulse-like waveforms
			randPhaseShifts = torch.exp(1j * 2 * np.pi * torch.rand(fft_frames.shape))
			randPhaseShifts[:,0] = 1	# Need DC bin to be real
			if (fft_frames.shape[1] % 2 == 1):
				randPhaseShifts[:,-1] = 1	# Need fs/2 frequency bin to be real
			fft_frames = fft_frames * randPhaseShifts

		raw_frames = torch.fft.irfft(fft_frames, dim=-1, norm='forward')
		frameLen = raw_frames.shape[1]

		# Raised cosine window
		winScale = hop_length / frameLen
		win = winScale * (torch.cos(torch.linspace(-np.pi, np.pi, raw_frames.shape[-1])) + 1)

		frames = raw_frames * win
		sig = torch.zeros((frames.shape[0]-1)*hop_length + frameLen)
		for i in range(frames.shape[0]):
			offset = i * hop_length
			sig[offset:(offset+frameLen)] = sig[offset:(offset+frameLen)] + (win * frames[i,:])

		return sig.cpu().detach().numpy()


	@classmethod
	def get_mel_basis(	cls,		
						fs						: float	= _DEFAULT_fs,
						mel_nfft				: int	= _DEFAULT_mel_nfft,
						mel_fmin				: float	= _DEFAULT_mel_fmin,
						mel_fmax				: float	= _DEFAULT_mel_fmax,
						mel_n_mels				: int	= _DEFAULT_mel_n_mels,
					):
		mel_basis = mel(sr=fs, n_fft=mel_nfft, fmin=mel_fmin, fmax=mel_fmax, n_mels=mel_n_mels).T
		return mel_basis


	@classmethod
	def generate_mel_spectrogram(	cls,
									x,

									# Provide these arguments to avoid recomputing
									mel_basis = None,
									filter_coef_b = None,
									filter_coef_a = None,

									# Minimum spectrum level
									spectrum_min_level_dB	: float = -100,
									
									# Backup arguments for if the previous arguments are not given
									fs						: float	= _DEFAULT_fs,
									highpass_cutoff			: float	= _DEFAULT_highpass_cutoff,
									highpass_order			: int	= _DEFAULT_highpass_order,
									mel_nfft				: int	= _DEFAULT_mel_nfft,
									mel_fmin				: float	= _DEFAULT_mel_fmin,
									mel_fmax				: float	= _DEFAULT_mel_fmax,
									mel_n_mels				: int	= _DEFAULT_mel_n_mels,

									spectrum_stft_fft_len	: int = _DEFAULT_stft_fft_len,
									spectrum_stft_hop_len	: int = _DEFAULT_stft_hop_len,
									
									# For adding random noise
									add_noise				: bool = False,
									noise_prng				: RandomState = None
								):
		if mel_basis is None:
			mel_basis = VC_Utils.get_mel_basis(fs, mel_nfft, mel_fmin=mel_fmin, mel_fmax=mel_fmax, mel_n_mels=mel_n_mels)

		if (filter_coef_b is None) or (filter_coef_a is None):
			b, a = VC_Utils.butter_highpass(highpass_cutoff, fs, order=highpass_order)
		else:
			b = filter_coef_b
			a = filter_coef_a

		spectrum_min_level = 10 ** (spectrum_min_level_dB / 20)

		y = signal.filtfilt(b, a, x)

		if not add_noise:
			# Add a little random noise for model roubstness
			if noise_prng is None:
				prng = RandomState()
			else:
				prng = noise_prng
			wav = y * 0.96 + (prng.rand(y.shape[0])-0.5)*1e-06
		else:
			wav = y

		# Compute spectrogram
		D = VC_Utils.pySTFT(wav, fft_length=spectrum_stft_fft_len, hop_length=spectrum_stft_hop_len).T

		# Convert to mel and normalize
		D_mel = np.dot(D, mel_basis)
		D_db = 20 * np.log10(np.maximum(spectrum_min_level, D_mel)) - 16
		S = np.clip((D_db + 100) / 100, 0, 1)

		return S.astype(np.float32)


	@classmethod
	def _pad_seq(cls, x, base=32):
		len_out = int(base * ceil(float(x.shape[0])/base))
		len_pad = len_out - x.shape[0]
		assert len_pad >= 0
		return np.pad(x, ((0,len_pad),(0,0)), 'constant'), len_pad

	
	@classmethod
	def get_spectrogram_input(cls, x, pad_base=None):
		S = VC_Utils.generate_mel_spectrogram(x)
		if pad_base is None:
			y, len_pad = VC_Utils._pad_seq(S)
		else:
			y, len_pad = VC_Utils._pad_seq(S, base=pad_base)
		return y, len_pad


	@classmethod
	def get_generator_input(	cls,
								x,
								calculateSourceSpeakerEmbeddings	: bool = True,
								sourceSpeakerEmbeddingsHistoryLen	: int = 1,
								speakerEncoderModelFilepath			: str = None,
								speakerEncoderModel					: SpeakerEncoder = None,
								speaker_encoder_dim_input			: int	= SpeakerEncoder._DEFAULT_speaker_encoder_dim_input,
								speaker_encoder_dim_cell			: int	= SpeakerEncoder._DEFAULT_speaker_encoder_dim_cell,
								speaker_encoder_dim_embedding		: int	= SpeakerEncoder._DEFAULT_speaker_encoder_dim_embedding,
								pad_base = None,
								print_progress						: bool = False
							):
		y, len_pad = VC_Utils.get_spectrogram_input(x, pad_base=pad_base)

		if not calculateSourceSpeakerEmbeddings:
			return y, None, len_pad

		if (speakerEncoderModelFilepath is None) and (speakerEncoderModel is None):
			raise Exception("Must provide either 'speakerEncoderModelFilepath' or 'speakerEncoderModel' as an argument.")
		elif (speakerEncoderModel is not None) and (speakerEncoderModelFilepath is not None):
			raise Exception("Cannot provide both 'speakerEncoderModelFilepath' or 'speakerEncoderModel' as arguments.")
		else:
			speakerEncoderModel = SpeakerEncoder(	speakerEncoderModelFilepath=speakerEncoderModelFilepath,
													speaker_encoder_dim_input=speaker_encoder_dim_input,
													speaker_encoder_dim_cell=speaker_encoder_dim_cell,
													speaker_encoder_dim_embedding=speaker_encoder_dim_embedding	)

		if print_progress:
			print("Generating input...\n\t")

		speakerEmbs = []
		for i in range(y.shape[0]):
			if print_progress:
				percentProgress = (i / y.shape[0]) * 100
				print(str(round(percentProgress, 2)) + "%...", end='')
			# melsp = torch.from_numpy(y[i, :]).cuda()
			melsp = torch.from_numpy(y[max(i-sourceSpeakerEmbeddingsHistoryLen,0):(i+1), :]).cuda()
			if melsp.shape[0] == 1:
				melsp = melsp.expand(1, 1, -1)
			else:
				melsp = melsp.expand(1, -1, -1)
			tempEmb = speakerEncoderModel(melsp).detach().squeeze().cpu().numpy()
			speakerEmbs.append(tempEmb)
		print("100%")

		return y, speakerEmbs, len_pad