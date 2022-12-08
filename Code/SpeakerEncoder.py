import sys
import numpy as np
import torch
from collections import OrderedDict

sys.path.append('autovc/')
from model_bl import D_VECTOR


class SpeakerEncoder(torch.nn.Module):
	_DEFAULT_speaker_encoder_dim_input			= 80
	_DEFAULT_speaker_encoder_dim_cell			= 768
	_DEFAULT_speaker_encoder_dim_embedding		= 256

	def __init__(	self,
					speakerEncoderModelFilepath		: str,
					speaker_encoder_dim_input		: int	= _DEFAULT_speaker_encoder_dim_input,
					speaker_encoder_dim_cell		: int	= _DEFAULT_speaker_encoder_dim_cell,
					speaker_encoder_dim_embedding	: int	= _DEFAULT_speaker_encoder_dim_embedding,
				):
		super().__init__()
		
		self.speakerEncoderModelFilepath			= speakerEncoderModelFilepath
		self.speaker_encoder_dim_input				= speaker_encoder_dim_input
		self.speaker_encoder_dim_cell				= speaker_encoder_dim_cell
		self.speaker_encoder_dim_embedding			= speaker_encoder_dim_embedding

		self._loadSpeakerEncoderModel()


	def _loadSpeakerEncoderModel(self):
		C = D_VECTOR(dim_input=self.speaker_encoder_dim_input, dim_cell=self.speaker_encoder_dim_cell, dim_emb=self.speaker_encoder_dim_embedding).eval().cuda()
		c_checkpoint = torch.load(self.speakerEncoderModelFilepath)
		new_state_dict = OrderedDict()
		for key, val in c_checkpoint['model_b'].items():
			new_key = key[7:]
			new_state_dict[new_key] = val
		C.load_state_dict(new_state_dict)
		self.model = C


	def forward(self, x):
		return self.model(x)