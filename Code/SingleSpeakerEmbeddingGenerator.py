import os
import pickle
import numpy as np
import torch
import soundfile as sf
import librosa
from librosa.filters import mel
from numpy.random import RandomState

from VC_Utils import VC_Utils
from SpeakerEncoder import SpeakerEncoder


class SingleSpeakerEmbeddingGenerator:
	def __init__(	self,
					speakerSourceDir						: str,
					speakerEncoderModelFilepath				: str	= None,
					speakerEncoderModel						: SpeakerEncoder = None,
					randomly_sample_audio_files				: bool	= True,
					min_audio_length_seconds				: float = 0,
					multiple_sample_long_audio				: bool	= True,
					long_audio_length_threshold				: float = 12,
					long_audio_random_arrival_period		: float = 3.5,
					fs										: float	= VC_Utils._DEFAULT_fs,
					highpass_cutoff							: float	= VC_Utils._DEFAULT_highpass_cutoff,
					highpass_order							: int	= VC_Utils._DEFAULT_highpass_order,
					mel_nfft								: int	= VC_Utils._DEFAULT_mel_nfft,
					mel_fmin								: float	= VC_Utils._DEFAULT_mel_fmin,
					mel_fmax								: float	= VC_Utils._DEFAULT_mel_fmax,
					mel_n_mels								: int	= VC_Utils._DEFAULT_mel_n_mels,
					spectrum_stft_fft_len					: int	= VC_Utils._DEFAULT_stft_fft_len,
					spectrum_stft_hop_len					: int	= VC_Utils._DEFAULT_stft_hop_len,
					spectrum_min_level_dB					: float = -100,
					speaker_encoder_dim_input				: int	= SpeakerEncoder._DEFAULT_speaker_encoder_dim_input,
					speaker_encoder_dim_cell				: int	= SpeakerEncoder._DEFAULT_speaker_encoder_dim_cell,
					speaker_encoder_dim_embedding			: int	= SpeakerEncoder._DEFAULT_speaker_encoder_dim_embedding,
					embedding_spectrum_length_crop			: int	= 64,
					embedding_refinement_error_threshold	: float = 1e12
				):

		if (speakerEncoderModelFilepath is None) and (speakerEncoderModel is None):
			raise Exception("Must provide either 'speakerEncoderModelFilepath' or 'speakerEncoderModel' as an argument.")
		if speakerEncoderModel is not None:
			if speakerEncoderModelFilepath is not None:
				raise Exception("Cannot provide both 'speakerEncoderModelFilepath' or 'speakerEncoderModel' as arguments.")
			speaker_encoder_dim_input = speakerEncoderModel.speaker_encoder_dim_input
			speaker_encoder_dim_cell = speakerEncoderModel.speaker_encoder_dim_cell
			speaker_encoder_dim_embedding = speakerEncoderModel.speaker_encoder_dim_embedding
			self.speakerEncoderModel = speakerEncoderModel
		else:
			self.speakerEncoderModel = SpeakerEncoder(	speakerEncoderModelFilepath=speakerEncoderModelFilepath,
														speaker_encoder_dim_input=speaker_encoder_dim_input,
														speaker_encoder_dim_cell=speaker_encoder_dim_cell,
														speaker_encoder_dim_embedding=speaker_encoder_dim_embedding	)

		self.speakerSourceDir						= speakerSourceDir
		self.speakerEncoderModelFilepath			= speakerEncoderModelFilepath
		
		self.randomly_sample_audio_files			= randomly_sample_audio_files
		self.min_audio_length_seconds				= min_audio_length_seconds
		self.multiple_sample_long_audio				= multiple_sample_long_audio
		self.long_audio_length_threshold			= long_audio_length_threshold
		self.long_audio_random_arrival_period		= long_audio_random_arrival_period
		
		self.fs										= fs
		
		self.highpass_cutoff						= highpass_cutoff
		self.highpass_order							= highpass_order
		
		self.mel_nfft								= mel_nfft
		self.mel_fmin								= mel_fmin
		self.mel_fmax								= mel_fmax
		self.mel_n_mels 							= mel_n_mels

		self.spectrum_stft_fft_len					= spectrum_stft_fft_len
		self.spectrum_stft_hop_len					= spectrum_stft_hop_len
		self.spectrum_min_level_dB					= spectrum_min_level_dB
		
		self._speaker_encoder_dim_input				= speaker_encoder_dim_input
		self._speaker_encoder_dim_cell				= speaker_encoder_dim_cell
		self._speaker_encoder_dim_embedding			= speaker_encoder_dim_embedding
		
		self.embedding_spectrum_length_crop			= embedding_spectrum_length_crop
		self.embedding_refinement_error_threshold	= embedding_refinement_error_threshold


	def generate(self):
		self._generateSpectrums()
		self._generateEmbedding()
		return (self.speakerEmbedding, self.spectrumsArray)


	def getEmbedding(self):
		return self.speakerEmbedding


	def _generateSpectrums(self):
		mel_basis = mel(self.fs, self.mel_nfft, fmin=self.mel_fmin, fmax=self.mel_fmax, n_mels=self.mel_n_mels).T
		b, a = VC_Utils.butter_highpass(self.highpass_cutoff, self.fs, order=self.highpass_order)

		self.mel_basis = mel_basis
		self.hpf_b = b
		self.hpf_a = a

		dirName, subdirList, _ = next(os.walk(self.speakerSourceDir))
		print('Generating spectrograms from directory: %s' % dirName)

		spectrumsArray = []
		inputFilenames = []
		for subdir in sorted(subdirList):
			print("| Processing sub-directory '%s'..." % subdir)
			_,_, fileList = next(os.walk(os.path.join(dirName,subdir)))
			if 'desktop.ini' in fileList:
				fileList.remove('desktop.ini')  # Not sure why I'm seeing that file

			prng = RandomState(hash(subdir) % (2 ** 32))
			for fileName in sorted(fileList):
				# Read audio file
				# x, fs = sf.read(os.path.join(dirName,subdir,fileName))
				x, fs = librosa.load(os.path.join(dirName,subdir,fileName), sr=self.fs)

				print("| | Generating mel spectrogram for " + fileName + "...", end='')

				len_seconds = len(x) * (1/fs)
				if len_seconds < self.min_audio_length_seconds:
					print("Skipped (duration too short).")
					continue

				# Generate spectrogram
				S = VC_Utils.generate_mel_spectrogram(	x, mel_basis=mel_basis, filter_coef_b=b, filter_coef_a=a,
														spectrum_stft_fft_len=self.spectrum_stft_fft_len,
														spectrum_stft_hop_len=self.spectrum_stft_hop_len,
														spectrum_min_level_dB=self.spectrum_min_level_dB,
														add_noise=True, noise_prng=prng								)
				
				# Save spectrogram and filename
				spectrumsArray.append(S)
				inputFilenames.append(fileName)

				print("Done.")

			print("| | Done.")

		self.spectrumsArray = spectrumsArray
		self.inputFilenames = inputFilenames


	def _generateEmbedding(self):
		len_crop = self.embedding_spectrum_length_crop

		# make speaker embedding
		embs = []
		processedFileNames = []
		print("Obtaining embeddings...")
		for i in range(len(self.spectrumsArray)):
			print("| Processing " + self.inputFilenames[i] + "...", end='')

			tmp = self.spectrumsArray[i]

			# choose another utterance if the current one is too short
			if tmp.shape[0] < len_crop:
				print("Skipped (too short).")
				continue

			processedFileNames.append(self.inputFilenames[i])

			if self.randomly_sample_audio_files:
				doMultisample = False
				
				if self.multiple_sample_long_audio:
					tmpDur = self.spectrum_stft_hop_len * tmp.shape[0] * (1/self.fs)
					if tmpDur >= self.long_audio_length_threshold:
						doMultisample = True
				
				if not doMultisample:
					if (tmp.shape[0] - len_crop) != 0:
						left = np.random.randint(0, tmp.shape[0] - len_crop)
					else:
						left = 0
					melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()
					emb = self.speakerEncoderModel(melsp)
					embs.append(emb.detach().squeeze().cpu().numpy())
				else:
					t_hop = self.spectrum_stft_hop_len * (1/self.fs)
					selectionProb = t_hop / self.long_audio_random_arrival_period
					for left in range(0, tmp.shape[0] - len_crop + 1):
						if (np.random.random() < selectionProb):
							# Choosing to sample here
							melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()
							emb = self.speakerEncoderModel(melsp)
							embs.append(emb.detach().squeeze().cpu().numpy())
							# print(left * self.spectrum_stft_hop_len * (1/self.fs))
			else:
				hop_size = int(np.floor(len_crop / 2))
				for left in range(0, tmp.shape[0] - len_crop + 1, hop_size):
					melsp = torch.from_numpy(tmp[np.newaxis, left:left+len_crop, :]).cuda()
					emb = self.speakerEncoderModel(melsp)
					embs.append(emb.detach().squeeze().cpu().numpy())

			print("Done.")

		initialEmbedding = np.mean(embs, axis=0)
		errs1 = [float(((torch.tensor(embs[ind]) - torch.tensor(initialEmbedding)).abs() ** 2).sum() / ((torch.tensor(embs[ind]).abs() ** 2).sum())) for ind in range(len(embs))]

		errorThresh = self.embedding_refinement_error_threshold
		selectedInds = torch.arange(len(embs))[torch.tensor(errs1) < errorThresh].tolist()
		speakerEmbedding = np.mean(np.array(embs)[selectedInds], axis=0)
		errs2 = [float(((torch.tensor(embs[ind]) - torch.tensor(speakerEmbedding)).abs() ** 2).sum() / ((torch.tensor(embs[ind]).abs() ** 2).sum())) for ind in selectedInds]

		self.processedFileNames = processedFileNames
		self.speakerEmbedding = speakerEmbedding

		return self.speakerEmbedding


	def save_speaker_embedding(self, filepath):
		with open(filepath, 'wb') as handle:
			pickle.dump(self.speakerEmbedding, handle)