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
import AutoVC_Wrapper

sys.path.append('autovc/')
sys.path.append('hifi_gan/')
from autovc.model_vc import Generator
from hifi_gan.models import Generator as HiFiGAN_Generator
from hifi_gan.env import AttrDict
from synthesis import build_model
from synthesis import wavegen

#------------------------------------------------------------------------------------------------------------------------------#


#------------------------------------------------------------------------------------------------------------------------------#
# Some basic parameters
device = 'cuda:0'
fs = 16000
#------------------------------------------------------------------------------------------------------------------------------#




#------------------------------------------------------------------------------------------------------------------------------#
# Specifies where model data is stored
speakerEncoderModelFilepath			= 'Models/SpeakerEncoder_Model.ckpt'
autovc_model_filepath				= 'Models/AutoVC_Model.ckpt'
hifi_gan_model_filepath				= 'Models/HiFiGAN_Model.ckpt'
hifi_gan_config_file_filepath		= 'hifi_gan/config_v1.json'
#------------------------------------------------------------------------------------------------------------------------------#




#------------------------------------------------------------------------------------------------------------------------------#
# Either generate a speaker embedding or load a previously generated speaker embedding
if False:
	targetEmbSourceDir = 'Speaker WAVs/speaker_p228'
	targetEmbFilepath = 'p228_speaker_emb.pkl'
	embGen = SingleSpeakerEmbeddingGenerator(	speakerSourceDir=targetEmbSourceDir,
												speakerEncoderModelFilepath=speakerEncoderModelFilepath,
												randomly_sample_audio_files=False,
												embedding_refinement_error_threshold=1e12,
												min_audio_length_seconds=0.5
											)
	embGenRet = embGen.generate()
	targetSpeakerEmbedding = embGenRet[0]
	embGen.save_speaker_embedding(targetEmbFilepath)
else:
	targetSpeakerEmbedding = 'Speaker Embeddings/speaker_p228.pkl'
#------------------------------------------------------------------------------------------------------------------------------#




#------------------------------------------------------------------------------------------------------------------------------#
# Load the AutoVC model
g_checkpoint = torch.load(autovc_model_filepath, map_location=device)
vcGenerator = AutoVC_Wrapper.initializeGeneratorFromStateDict(generator_state_dict=g_checkpoint['model'], downsampling_factor='assume_paper_value', device=device)
#------------------------------------------------------------------------------------------------------------------------------#








#------------------------------------------------------------------------------------------------------------------------------#

# This specifies the input speech, i.e. the speech that you want to transform to sound like another speaker
speechInputFilepath = 'Speaker WAVs/HarvardSentencesShortened.wav'


# This executes the AutoVC model.  The model gives a set of mel-frequency spectrograms as its output.
spect_vc = AutoVC_Wrapper.generateVCSpectrogram(	speechInputFilepath,
													fs,
													targetSpeakerEmbedding=targetSpeakerEmbedding,
													speakerEncoderModelFilepath=speakerEncoderModelFilepath,
													autovcGeneratorModel=vcGenerator,
													sourceSpeakerEmbedding=None,
													device=device
												)


# This generates speech waveforms from spectrogram frames and outputs the resulting sound file.
AutoVC_Wrapper.generateAudioFromSpectVC_hifi_gan(spect_vc, fs, hifi_gan_model_filepath, config_file = hifi_gan_config_file_filepath, device = None)


#------------------------------------------------------------------------------------------------------------------------------#




pass