import os
import sys
import pickle
import json
import torch
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
from playsound import playsound
import numpy as np
from math import ceil

from VC_Utils import VC_Utils
from SingleSpeakerEmbeddingGenerator import SingleSpeakerEmbeddingGenerator

sys.path.append('autovc/')
sys.path.append('hifi_gan/')
from autovc.model_vc import Generator
from hifi_gan.models import Generator as HiFiGAN_Generator
from hifi_gan.env import AttrDict
from synthesis import build_model
from synthesis import wavegen

#------------------------------------------------------------------------------------------------------------------------------#


def initializeGeneratorFromStateDict(generator_state_dict : dict, downsampling_factor : int or str, device : torch.device = None):
	# Can use stored values to infer some of the arguments that were passed to Generator(...)
	dim_neck = generator_state_dict['encoder.lstm.weight_hh_l1_reverse'].shape[1]
	dim_emb = generator_state_dict['encoder.convolutions.0.0.conv.weight'].shape[1] - 80
	dim_pre = generator_state_dict['decoder.convolutions.0.0.conv.bias'].shape[0]

	# It doesn't seem like freq (i.e. temporal downsampling factor between encoder and decoder) is saved in/inferrable from
	# any of the entries in a Generator's state dictionary.
	if (type(downsampling_factor) is str):
		if downsampling_factor == 'assume_paper_value':
			freq = 32
		else:
			raise Exception("Invalid string for 'downsampling_factor' argument.")
	else:
		freq = downsampling_factor
	
	G = Generator(dim_neck,dim_emb,dim_pre,freq).eval().to(device)
	G.load_state_dict(generator_state_dict)

	return G


def generateAudioFromSpectVC_rough(spect_vc, fs, gain = 5):
	for spect in spect_vc:
		name = spect[0]
		c = spect[1]
		print(name)
		waveform = VC_Utils.invert_mel_spectrogram_rough(c) * gain
		sf.write(name+'.wav', waveform, samplerate=fs)


def generateAudioFromSpectVC_wavenet(spect_vc, fs, wavenetModelFilepath, gain = 1, device = None):
	model = build_model().to(device)
	checkpoint = torch.load(wavenetModelFilepath)
	model.load_state_dict(checkpoint["state_dict"])

	for spect in spect_vc:
		name = spect[0]
		c = spect[1]
		print(name)
		waveform = wavegen(model, c=c) * gain
		sf.write(name+'.wav', waveform, samplerate=fs)


def generateAudioFromSpectVC_hifi_gan(spect_vc, fs, hifiGanModelFilepath, config_file = 'hifi_gan/config_v1.json', device = None):
	with open(config_file) as f:
		data = f.read()
	json_config = json.loads(data)
	h = AttrDict(json_config)

	model = HiFiGAN_Generator(h).to(device)
	state_dict_g = torch.load(hifiGanModelFilepath, device)
	model.load_state_dict(state_dict_g["generator"])

	for spect in spect_vc:
		name = spect[0]
		c = spect[1]
		print(name)

		chunkSize = 100
		overlapSize = 10
		waveform = None
		hopSize = None
		curInd = 0
		for i in range(0, c.shape[0], chunkSize):
			cTemp = c[max(i-overlapSize,0):(i+chunkSize),:]
			x = torch.tensor(cTemp, device=device).permute((1, 0)).unsqueeze(0)
			waveformTemp = model(x)
			waveformTemp = waveformTemp.view(waveformTemp.shape[0] * waveformTemp.shape[2]).detach()
			if (hopSize is None):
				hopSize = int(waveformTemp.shape[0] / cTemp.shape[0])
				waveform = torch.zeros(int(hopSize) * c.shape[0], device='cpu')
			numNewSamples = min(waveform.shape[0] - i*hopSize, chunkSize*hopSize)
			waveformTemp = waveformTemp[max(waveformTemp.shape[0]-numNewSamples, 0):]
			waveform[curInd:(curInd+waveformTemp.shape[0])] = waveformTemp
			curInd = curInd + chunkSize*hopSize

		sf.write(name+'.wav', waveform.detach().cpu().numpy(), samplerate=fs)


def generateVCSpectrogram(	speechInputFilepath : str,
							fs : float,
							targetSpeakerEmbedding : str or np.ndarray,
							speakerEncoderModelFilepath : str,
							autovcGeneratorModel : Generator,
							sourceSpeakerEmbedding = None,
							device : torch.device = None,
						):
	inputSig, _ = VC_Utils.read_audio_file(speechInputFilepath, fs_desired=fs)

	if type(targetSpeakerEmbedding) is str:
		targetSpeakerEmbedding = pickle.load(open(targetSpeakerEmbedding, "rb"))

	if sourceSpeakerEmbedding is None:
		calculateSourceSpeakerEmbeddings = True
	else:
		calculateSourceSpeakerEmbeddings = False

	x_org, speakerEmbs, len_pad = VC_Utils.get_generator_input(	inputSig,
																calculateSourceSpeakerEmbeddings=calculateSourceSpeakerEmbeddings,
																sourceSpeakerEmbeddingsHistoryLen=1,
																speakerEncoderModelFilepath=speakerEncoderModelFilepath,
																print_progress=False													)

	if calculateSourceSpeakerEmbeddings:
		sourceSpeakerEmbedding = speakerEmbs

	uttr_org = torch.from_numpy(x_org[np.newaxis, :, :]).to(device)
	emb_trg = torch.from_numpy(targetSpeakerEmbedding[np.newaxis, :]).to(device)
	if type(sourceSpeakerEmbedding) is np.ndarray:
		emb_org = torch.from_numpy(sourceSpeakerEmbedding[np.newaxis, :]).to(device)
	else:
		emb_org = torch.tensor(sourceSpeakerEmbedding).to(device)


	with torch.no_grad():
		_, x_identic_psnt, _ = autovcGeneratorModel(uttr_org, emb_org, emb_trg)


	if len_pad == 0:
		uttr_trg = x_identic_psnt[0, 0, :, :].cpu().numpy()
	else:
		uttr_trg = x_identic_psnt[0, 0, :-len_pad, :].cpu().numpy()

	spect_vc = []
	spect_vc.append( ('result', uttr_trg) )

	return spect_vc