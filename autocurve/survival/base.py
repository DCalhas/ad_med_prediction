import copy

import numpy as np

import pandas as pd

import pycox

import torch

import torchtuples.callbacks as cb

class _BatchRepSurv(pycox.models.cox._CoxPHBase):

	def __init__(self, *args, **kwargs):
		super(_BatchRepSurv, self).__init__(*args, **kwargs)

		self.log_mean=None
		self.log_std=None
	
	def compute_val_metrics(self, dataloader):
		raise NotImplementedError
	
	def compute_metrics(self, data, metrics): 
		return super(_BatchRepSurv, self).compute_metrics(((data.input_ids, data.values), (data.durations, data.censoring)), metrics)
	
	def lr_finder(self, dataloader, batch_size=64, lr_min=1e-8, lr_max=10.0, lr_range=(1e-7, 10.0), callbacks=None, n_steps=100, tolerance=np.inf, verbose=False, **kwargs):
		#transform 
		with self._lr_finder(lr_min, lr_max, lr_range, n_steps, tolerance, verbose) as lr_finder:
			if callbacks is None:
				callbacks = []
			callbacks.append(lr_finder)
			epochs = n_steps
			self.fit_dataloader(dataloader,epochs=epochs,callbacks=callbacks, verbose=False, **kwargs,)
		return lr_finder

	def _set_durations_dist(self, dataloader):

		with torch.no_grad():
			durations=[]

			for batch in dataloader.dataset:
				durations.append(batch.durations.unsqueeze(0))
				
			durations=torch.concatenate(durations, dim=0)

			durations_nzero=durations[durations>0.]#ignore the ones that are zero, it means they transitioned, but have not had visits at this stage yet.
			log_durations=(durations_nzero+1e-3).log()
			
			self.log_mean=float(log_durations.mean().numpy())
			self.log_std=float(log_durations.std().numpy())
	
	def fit_dataloader(self, dataloader, epochs=1, callbacks=None, verbose=True, metrics=None, val_dataloader=None, temporal_norm=100):
		self._setup_train_info(dataloader)
		self._set_durations_dist(dataloader)
		self.metrics = self._setup_metrics(metrics)
		
		self.log.verbose = verbose
		self.val_metrics.dataloader = val_dataloader
		if callbacks is None:
			callbacks = []
		
		self.callbacks = cb.TrainingCallbackHandler(
			self.optimizer, self.train_metrics, self.log, self.val_metrics, callbacks
		)
		self.callbacks.give_model(self)
		
		self.best_metric=float("inf")
		self.best_model=None

		for _ in range(epochs):

			self.batch_loss=0.
			self._batch_loop(dataloader, epochs=epochs, callbacks=callbacks, verbose=verbose, metrics=metrics, val_dataloader=val_dataloader, temporal_norm=temporal_norm)
			
		self.callbacks.on_fit_end()
		return self.log, self.best_model
	
	def compute_loss(self, batch):
		raise NotImplementedError
	
	def _batch_loop(self,dataloader, epochs=1, callbacks=None, verbose=True, metrics=None, val_dataloader=None, temporal_norm=100):
		batch_size=dataloader.batch_size if dataloader.batch_size is not None else dataloader.batch_sampler.batch_size

		self.batch_metrics={"loss": 0.}

		for i, data in enumerate(dataloader):
			#stop = self.callbacks.on_batch_start()
			#if stop:
			#	break
			
			
			self.optimizer.zero_grad()
			#self.batch_metrics = self.compute_metrics(data, self.metrics)
			self.batch_loss=self.compute_loss(data)
			self.batch_metrics["loss"]+=self.batch_loss#for logging
			#(loss+neglog_normal_gamma_prior(model._parameters_concatenated, model.raw_log_lambda, normfunc=norm_reg)).backward()
			self.batch_loss.backward()
			
			if(verbose):
				print("Loss "+"{0:.4f}".format(self.batch_loss.item())+" on instance "+str(i*batch_size)+"/"+str(dataloader.dataset.__len__()), end="\r")
			
			#stop = self.callbacks.before_step()
			#if stop:
			#	break
			self.optimizer.step()
			#stop = self.callbacks.on_batch_end()
			#if stop:
			#		break
		#else:
			#stop = self.callbacks.on_epoch_end()
		if(val_dataloader is not None):
			with torch.no_grad():
				val_loss=0
				for i, batch in enumerate(val_dataloader):
					val_loss+=self.compute_loss(batch).item()
				if(verbose): print("Val loss:", val_loss)
				metric=self.compute_val_metrics(val_dataloader)
				if(metric<self.best_metric):
					self.best_metric=metric
					self.best_model=copy.deepcopy(self.net)
					if(verbose): print("I: New best model found.")
					
		return False
		
	def predict_surv_df(self, dataloader, batch_size=8224, numpy=None, eval_=True, to_cpu=False, num_workers=0):
		surv, times=self.predict_surv(dataloader, batch_size=batch_size, numpy=numpy, eval_=eval_, to_cpu=to_cpu, num_workers=num_workers)
		
		surv_df = pd.DataFrame(surv, index=times)
		
		return surv_df, times
	
	def _predict(self, dataloader):
		self.best_model.eval()
		
		predictions=[]
		seq_len=[]
		durations=[]
		events=[]
		
		with torch.no_grad():
			for batch in dataloader:
				seq_len.append(batch.values)
				durations.append(batch.durations)
				events.append(batch.censoring)
				predictions.append(self.best_model(batch))#we need to append all parameters
		
		return predictions, seq_len, durations, events
