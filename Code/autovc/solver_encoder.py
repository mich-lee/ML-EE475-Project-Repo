from model_vc import Generator
import numpy as np
import torch
import torch.nn.functional as F
import time
import datetime


class Solver(object):

	def __init__(self, vcc_loader, config):
		"""Initialize configurations."""

		# Data loader.
		self.vcc_loader = vcc_loader

		# Model configurations.
		self.lambda_cd = config.lambda_cd
		self.dim_neck = config.dim_neck
		self.dim_emb = config.dim_emb
		self.dim_pre = config.dim_pre
		self.freq = config.freq

		# Training configurations.
		self.batch_size = config.batch_size
		self.num_iters = config.num_iters
		
		# Miscellaneous.
		self.use_cuda = torch.cuda.is_available()
		self.device = torch.device('cuda:0' if self.use_cuda else 'cpu')
		self.checkpoint_iters = config.checkpoint_iters
		self.log_step = config.log_step
		self.loss_avg_len = config.loss_avg_len

		# Build the model and tensorboard.
		self.build_model()

			
	def build_model(self):
		
		self.G = Generator(self.dim_neck, self.dim_emb, self.dim_pre, self.freq)        
		
		# self.g_optimizer = torch.optim.Adam(self.G.parameters(), 0.0001)

		self.g_optimizer = torch.optim.Adam(self.G.parameters(), 0.01)
		# self.g_lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.g_optimizer, [250, 350, 500], gamma=0.1)
		# self.g_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.g_optimizer, 1, gamma=0.1)
		self.g_lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.g_optimizer, 'min', factor=0.1, patience=100)
		
		self.G.to(self.device)
		

	def reset_grad(self):
		"""Reset the gradient buffers."""
		self.g_optimizer.zero_grad()


	def save_checkpoint(self, iteration, ckpt_pth):
		torch.save({'model': self.G.state_dict(),
					'optimizer': self.g_optimizer.state_dict(),
					'lr_scheduler': self.g_lr_scheduler,
					'iteration': iteration}, ckpt_pth)


	def askYesNoQuestion(self, questionString : str):
		while True:
			resp = input(questionString + " (y/n): ")
			if (resp == 'y'):
				return True
			elif (resp == 'n'):
				return False
			else:
				print("Invalid input.")
	  
	
	#=====================================================================================================================================#
	
	
				
	def train(self):
		# Set data loader.
		data_loader = self.vcc_loader

		dataset_size = len(data_loader.dataset.train_dataset)
		
		# Print logs in specified order
		keys = ['G/loss_id','G/loss_id_psnt','G/loss_cd']
			
		# Start training.
		print('Start training...')
		start_time = time.time()
		loss_id_hist = []
		loss_id_psnt_hist = []
		loss_cd_hist = []
		for i in range(self.num_iters):

			# =================================================================================== #
			#                             1. Preprocess input data                                #
			# =================================================================================== #

			# Fetch data.
			try:
				x_real, emb_org = next(data_iter)
			except:
				data_iter = iter(data_loader)
				x_real, emb_org = next(data_iter)
			
			
			x_real = x_real.to(self.device) 
			emb_org = emb_org.to(self.device) 
						
	   
			# =================================================================================== #
			#                               2. Train the generator                                #
			# =================================================================================== #
			
			self.G = self.G.train()
						
			# Identity mapping loss
			x_identic, x_identic_psnt, code_real = self.G(x_real, emb_org, emb_org)
			g_loss_id = F.mse_loss(x_real, x_identic)   
			g_loss_id_psnt = F.mse_loss(x_real, x_identic_psnt)   
			
			# Code semantic loss.
			code_reconst = self.G(x_identic_psnt, emb_org, None)
			g_loss_cd = F.l1_loss(code_real, code_reconst)


			# Backward and optimize.
			g_loss = g_loss_id + g_loss_id_psnt + self.lambda_cd * g_loss_cd
			self.reset_grad()
			g_loss.backward()
			self.g_optimizer.step()

			# Logging.
			loss = {}
			loss['G/loss_id'] = g_loss_id.item()
			loss['G/loss_id_psnt'] = g_loss_id_psnt.item()
			loss['G/loss_cd'] = g_loss_cd.item()

			loss_id_hist.append(loss['G/loss_id'])
			loss_id_psnt_hist.append(loss['G/loss_id_psnt'])
			loss_cd_hist.append(loss['G/loss_cd'])

			if len(loss_id_hist) > self.loss_avg_len:	# These arrays are all updated together, so the code can be simplified here.
				loss_id_hist = loss_id_hist[-self.loss_avg_len:]
				loss_id_psnt_hist = loss_id_psnt_hist[-self.loss_avg_len:]
				loss_cd_hist = loss_cd_hist[-self.loss_avg_len:]

			avgLoss = {}
			avgLoss['G/loss_id_avg'] = float(torch.tensor(loss_id_hist).mean())
			avgLoss['G/loss_id_psnt_avg'] = float(torch.tensor(loss_id_psnt_hist).mean())
			avgLoss['G/loss_cd_avg'] = float(torch.tensor(loss_cd_hist).mean())

			# Step the learning rate scheduler
				# stepSchedulerFlag = False
				# if stepSchedulerFlag:
				# 	self.g_lr_scheduler.step()
				# 	print(self.g_lr_scheduler.get_last_lr())
			self.g_lr_scheduler.step(avgLoss['G/loss_cd_avg'])


			# =================================================================================== #
			#                                 4. Miscellaneous                                    #
			# =================================================================================== #

			# Print out training information.
			if (i+1) % self.log_step == 0:
				avgLossKeys = ['G/loss_id_avg','G/loss_id_psnt_avg','G/loss_cd_avg']
				et = time.time() - start_time
				et = str(datetime.timedelta(seconds=et))[:-7]
				log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
				log += ", LR [" + str(torch.tensor(self.g_optimizer.param_groups[0]['lr']))[7:-1] + "]"
				for tag in avgLossKeys:
					log += ", {}: {:.4f}".format(tag, avgLoss[tag])
				print(log)


			# # Print out training information.
			# if (i+1) % self.log_step == 0:
			# 	et = time.time() - start_time
			# 	et = str(datetime.timedelta(seconds=et))[:-7]
			# 	log = "Elapsed [{}], Iteration [{}/{}]".format(et, i+1, self.num_iters)
			# 	for tag in keys:
			# 		log += ", {}: {:.4f}".format(tag, loss[tag])
			# 	print(log)



		# =================================================================================== #
		#                                 4. Saving stuff                                     #
		# =================================================================================== #
			if self.checkpoint_iters != -1:
				if ((i+1) % self.checkpoint_iters == 0) and (i != 0):
					saveName = "AutoVC_Model_" + str(np.random.randint(100000000)) + ".ckpt"
					doSave = self.askYesNoQuestion("Save checkpoint as " + saveName + "?")
					if doSave:
						self.save_checkpoint(i+1, saveName)

					doContinue = self.askYesNoQuestion("Continue with training?")
					if not doContinue:
						break
				

		saveName = "AutoVC_Model_" + str(np.random.randint(100000000)) + ".ckpt"
		doSave = self.askYesNoQuestion("Save final result as " + saveName + "?")
		if doSave:
			self.save_checkpoint(i+1, saveName)