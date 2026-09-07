import torch

import numpy as np

import pandas as pd

from autocurve.survival.aft.losses import ExponentialLoss, WeibullLoss, GeneralizedGammaLoss

from autocurve.survival.aft import ExponentialSurvivalEvaluator, WeibullSurvivalEvaluator, GeneralizedGammaSurvivalEvaluator, DSMSurvivalEvaluator, DeepHitSurvivalEvaluator

from autocurve.survival import _BatchRepSurv

import auton_survival

from scipy._lib._bunch import _make_tuple_bunch

BatchRepresentation=_make_tuple_bunch('BatchRepresentation', ['input_ids', 'latent', 'values', 'attention_mask', 'targets', 'durations', 'censoring'],)	

class WeibullSurv(_BatchRepSurv):
	
	def __init__(self, net, loss=None, optimizer=None, device=None, gamma=0, window=2, scale=10, scale_target=0.2, temporal_norm=100):
		if(loss is None): loss=WeibullLoss(gamma=gamma, window=window, scale=scale, scale_target=scale_target)
		super(WeibullSurv, self).__init__(net, loss, optimizer, device)
		
		self.scale=scale
		self.scale_target=scale_target
		self.window=window
		self.temporal_norm=temporal_norm
	
	def compute_loss(self, batch):
		durations=batch.durations.to(device=self.device)
		censoring=batch.censoring.to(device=self.device)
		
		return self.loss(self.net(batch), durations, censoring, norm=self.temporal_norm, log_mean=self.log_mean, log_std=self.log_std)
		
	def predict_surv(self, dataloader, batch_size=8224, numpy=None, eval_=True, to_cpu=False, num_workers=0, temporal_norm=100, device="cuda:0"):
		
		all_lam, all_k, all_times, all_events = [], [], [], []
		self.net.eval()
		with torch.no_grad():
			for batch in dataloader:
				durations = batch.durations.to(device)
				event = batch.censoring.to(device)  # <-- already 1=event, 0=censored

				theta = self.net(batch)			 # <-- now model outputs lambda
				lam = ((torch.exp(self.scale*theta[:,0])))#change this to optional
				k = torch.exp(theta[:,1])+1e-6

				all_lam.append(lam.cpu())
				all_k.append(k.cpu())
				all_times.append(durations.cpu())
				all_events.append(event.cpu())

		self.lam = torch.cat(all_lam).numpy()
		self.k = torch.cat(all_k).numpy()
		self.time = torch.cat(all_times).numpy()
		self.event = torch.cat(all_events).numpy().astype(int)
		
		valid = self.time >= 0
		lam = self.lam[valid,]
		k = self.k[valid,]
		time = self.time[valid]
		event = self.event[valid]

		t_min = max(0.0, float(time.min()))
		t_max = float(time.max())
		times = np.linspace(t_min, t_max, 1000)

		norm_times=self.scale*(np.log(times)-self.log_mean)/self.log_std#normalize times, the probability is the same because we are innormalized space.

		return np.exp(- np.exp(k[None, :] * (norm_times[:, None] - np.log(lam[None, :])))), times
	
	def compute_val_metrics(self, dataloader, temporal_norm=100):
		"""Return the ibs"""
		val_evaluator=WeibullSurvivalEvaluator(self.net, dataloader, log_mean=self.log_mean, log_std=self.log_std, scale=self.scale, scale_target=self.scale_target)
		return val_evaluator(verbose=True, return_weighted=True, norm=temporal_norm)[:,1].sum().item()
	
	def compute_baseline_hazard(self, temporal_norm=100):
		
		#compute times
		times=torch.linspace(0.36,800,1000).to(device=self.device)
		
		with torch.no_grad():
			#norm_times=times/temporal_norm

			log_times=(times).log()
			norm_times=self.scale*(log_times-self.log_mean)/self.log_std
			
			k=self.net.w
			lam=self.net.head.bias

			return times.cpu().numpy(), ((k-lam)+(k.exp()-1)*(norm_times-lam)).exp().cpu().numpy()#((k/lam) * (norm_times/lam)**(k-1)).cpu().numpy()
	
class ExponentialSurv(WeibullSurv):
	
	def __init__(self, net, loss=None, optimizer=None, device=None, gamma=0, window=2, scale=10, scale_target=0.2, temporal_norm=100):
		if(loss is None): loss=ExponentialLoss(gamma=gamma, window=window, scale=scale, scale_target=scale_target)
		super(ExponentialSurv, self).__init__(net, loss=loss, device=device, gamma=gamma, optimizer=optimizer, )
		
		self.scale=scale
		self.scale_target=scale_target
		self.window=window

		self.temporal_norm=temporal_norm
		
	def predict_surv(self, dataloader, batch_size=8224, numpy=None, eval_=True, to_cpu=False, num_workers=0, temporal_norm=100, device="cuda:0"):
		
		all_lam, all_k, all_times, all_events = [], [], [], []
		self.net.eval()
		with torch.no_grad():
			for batch in dataloader:
				durations = batch.durations.to(device)
				event = batch.censoring.to(device)  # <-- already 1=event, 0=censored
				
				theta = self.net(batch)			 # <-- now model outputs lambda
				lam = ((torch.exp(self.scale*theta[:,0])))#change this to optional
				k = torch.ones_like(lam)

				all_lam.append(lam.cpu())
				all_k.append(k.cpu())
				all_times.append(durations.cpu())
				all_events.append(event.cpu())

		self.lam = torch.cat(all_lam).numpy()
		self.k = torch.cat(all_k).numpy()
		self.time = torch.cat(all_times).numpy()
		self.event = torch.cat(all_events).numpy().astype(int)
		
		valid = self.time >= 0
		lam = self.lam[valid,]
		k = self.k[valid,]
		time = self.time[valid]
		event = self.event[valid]

		t_min = max(0.0, float(time.min()))
		t_max = float(time.max())
		times = np.linspace(t_min, t_max, 1000)
		norm_times=self.scale_target*self.scale*(np.log(times+1e-3)-self.log_mean)/self.log_std#normalize times, the probability is the same because we are innormalized space.

		return np.exp(- np.exp(k[None, :]*(norm_times[:, None] - np.log(lam[None, :])))), times
		#return np.exp(- ((times[:, None] )/ lam[None, :]) ** k[None, :]), times
	
	def compute_val_metrics(self, dataloader, temporal_norm=100):
		"""Return the ibs"""
		val_evaluator=ExponentialSurvivalEvaluator(self.net, dataloader, log_mean=self.log_mean, log_std=self.log_std, scale=self.scale, scale_target=self.scale_target)
		return val_evaluator(verbose=True, return_weighted=True, norm=temporal_norm)[:,1].sum().item()
	
	def compute_baseline_hazard(self, temporal_norm=100):
		
		#compute times
		times=torch.linspace(0.36,800,1000).to(device=self.device)
		
		with torch.no_grad():

			lam=torch.exp(self.net.head.bias)
		
			return times.cpu().numpy(), (lam).repeat(times.shape[0]).cpu().numpy()

class GeneralizedGammaSurv(_BatchRepSurv):
	
	def __init__(self, net, loss=None, optimizer=None, device=None, gamma=0, window=2, scale=10, scale_target=0.2, temporal_norm=100):
		if(loss is None): loss=GeneralizedGammaLoss(gamma=gamma, window=window, scale=scale, scale_target=scale_target)
		super(GeneralizedGammaSurv, self).__init__(net, loss, optimizer, device)
		
		self.scale=scale
		self.scale_target=scale_target
		self.window=window

		self.temporal_norm=temporal_norm
	
	def compute_loss(self, batch):
		durations=batch.durations.to(device=self.device)
		censoring=batch.censoring.to(device=self.device)
		
		return self.loss(self.net(batch), durations, censoring, norm=self.temporal_norm, log_mean=self.log_mean, log_std=self.log_std)
	
	def predict_surv(self, dataloader, batch_size=8224, numpy=None, eval_=True, to_cpu=False, num_workers=0, temporal_norm=100, device="cuda:0"):
		all_a, all_d, all_p, all_times, all_events = [], [], [], [], []

		self.net.eval()
		with torch.no_grad():
			for batch in dataloader:
				durations = batch.durations.to(device)
				event = batch.censoring.to(device)  # <-- already 1=event, 0=censored

				theta = self.net(batch)			 # <-- now model outputs lambda
				a = ((torch.exp(self.scale*theta[:,0])))#change this to optional
				d = torch.exp(theta[:,1])+1e-6
				p = torch.exp(theta[:,2])+1e-6

				all_a.append(a.cpu())
				all_d.append(d.cpu())
				all_p.append(p.cpu())
				all_times.append(durations.cpu())
				all_events.append(event.cpu())

		self.a = torch.cat(all_a).numpy()
		self.d = torch.cat(all_d).numpy()
		self.p = torch.cat(all_p).numpy()
		self.time = torch.cat(all_times).numpy()
		self.event = torch.cat(all_events).numpy().astype(int)

		self.surv_dfs = []

		valid = self.time >= 0
		a = self.a[valid,]
		d = self.d[valid,]
		p = self.p[valid,]
		time = self.time[valid]
		event = self.event[valid]

		t_min = max(0.0, float(time.min()))
		t_max = float(time.max())
		times = np.linspace(t_min, t_max, 1000)

		norm_times=self.scale_target*self.scale*(np.log(times)-self.log_mean)/self.log_std#normalize times, the probability is the same because we are innormalized space.

		# Generalized Gamma survival: S(t|x) = exp(-(t/lam)^k)
		with torch.no_grad(): surv = gammaincc(torch.tensor(d[None,:]/p[None,:]), torch.tensor(((np.exp(norm_times[:, None]) )/ a[None, :]) ** p[None, :])).numpy()

		return surv, times
	
	def compute_val_metrics(self, dataloader, temporal_norm=100):
		"""Return the ibs"""
		val_evaluator=GeneralizedGammaSurvivalEvaluator(self.net, dataloader, log_mean=self.log_mean, log_std=self.log_std, scale=self.scale, scale_target=self.scale_target)
		return val_evaluator(verbose=True, return_weighted=True, norm=temporal_norm)[:,1].sum().item()
	
	def compute_baseline_hazard(self, temporal_norm=100):
		
		#compute times
		times=torch.linspace(0.36,800,1000).to(device=self.device)
		
		with torch.no_grad():
			#norm_times=times/temporal_norm

			log_times=(times).log()
			norm_times=self.scale_target*self.scale*(log_times-self.log_mean)/self.log_std
			
			d=self.net.w[0]
			p=self.net.w[1]
			a=self.net.head.bias
			
			s=d-p
			z=p*(norm_times-a)
			
			num=(p + norm_times*(d-1) + (-z)).exp()
			
			den=gammaincc(s.exp(),z.exp()) * torch.special.gammaln(s.exp()).exp() * a.exp()**d.exp()
			
			return times.cpu().numpy(), (num/den).cpu().numpy()


class DSMWrapperClinical(auton_survival.models.dsm.DeepSurvivalMachinesTorch):

	def forward(self, batch, seq_len=None, *kwargs):

		if(self.training):
			xrep = self.embedding(batch, seq_len=seq_len)[0]
		else:
			xrep = self.embedding(batch, seq_len=seq_len)

		dim=xrep.shape[0]
		
		return(self.act(self.shapeg['1'](xrep))+self.shape['1'].expand(dim, -1),
			self.act(self.scaleg['1'](xrep))+self.scale['1'].expand(dim, -1),
			self.gate['1'](xrep)/self.temp)

class DSMWrapperSUPPORT(auton_survival.models.dsm.DeepSurvivalMachinesTorch):

	def forward(self, batch, seq_len=None, *kwargs):

		xrep = self.embedding(batch, )

		dim=xrep.shape[0]
		
		return(self.act(self.shapeg['1'](xrep))+self.shape['1'].expand(dim, -1),
			self.act(self.scaleg['1'](xrep))+self.scale['1'].expand(dim, -1),
			self.gate['1'](xrep)/self.temp)

class DSMSurv(_BatchRepSurv):

	def __init__(self, net, loss=None, optimizer=None, device=None,
			 gamma=0, window=0.5, scale=1., scale_target=0.2, temporal_norm=100):
		_net = DSMWrapperClinical(1, k=5)
		_net.embedding = net
		_net.to(device="cuda:0")

		if loss is not None:
			loss = loss(_net)
		else:
			raise NotImplementedError

		super(DSMSurv, self).__init__(_net, loss, optimizer, device)

		self.scale		= scale
		self.scale_target = scale_target
		self.temporal_norm = temporal_norm

	def _predict(self, dataloader, device="cuda:0"):
		all_preds, all_seq_len, all_durations, all_events = [], [], [], []
		self.net.eval()
		with torch.no_grad():
			for batch in dataloader:
				shape, scale, logits = self.net(batch)		# each [B, K]
				# Stack into [B, 3K] so it concatenates like Weibull's [B, 2]

				preds = torch.cat([shape, scale, logits], dim=1)
				all_preds.append(preds.cpu())
				all_seq_len.append(batch.values.cpu())
				all_durations.append(batch.durations.cpu())
				all_events.append(batch.censoring.cpu())
		return all_preds, all_seq_len, all_durations, all_events

	def compute_loss(self, batch):
		durations = batch.durations.to(device=self.device)
		censoring = batch.censoring.to(device=self.device)
		
		return self.loss(*self.net(batch), durations, censoring, scale_target=self.scale_target, log_std=self.log_std, log_mean=self.log_mean, _scale=self.scale)

	def predict_surv(self, dataloader, device="cuda:0", **kwargs):
		"""
		Returns mixture Weibull survival curves, consistent with
		WeibullSurv.predict_surv.
		DSM forward: (shape [B,K], scale [B,K], logits [B,K])
			k_g   = exp(shape[:,g])
			lam_g = exp(-scale[:,g])
			S(t)  = sum_g  softmax(logits)_g * exp(-(t/lam_g)^k_g)
		"""
		all_shapes, all_scales, all_logits = [], [], []
		all_times, all_events = [], []

		self.net.eval()
		with torch.no_grad():
			for batch in dataloader:
				durations = batch.durations.to(device)
				event	 = batch.censoring.to(device)

				shape, scale, logits = self.net(batch)   # each [B, K]

				all_shapes.append(shape.cpu())
				all_scales.append(scale.cpu())
				all_logits.append(logits.cpu())
				all_times.append(durations.cpu())
				all_events.append(event.cpu())

		shapes  = torch.cat(all_shapes).numpy()						 # [N, K]
		scales  = torch.cat(all_scales).numpy()						 # [N, K]
		weights = torch.softmax(
			torch.cat(all_logits), dim=1
		).numpy()														# [N, K]
		self.time  = torch.cat(all_times).numpy()
		self.event = torch.cat(all_events).numpy().astype(int)

		# Natural-scale Weibull params per component
		k   = np.exp(shapes)	# [N, K]
		lam = np.exp(-scales)   # [N, K]

		valid  = self.time >= 0
		k	  = k[valid]
		lam	= lam[valid]
		weights = weights[valid]
		time   = self.time[valid]

		t_max = float(time.max())
		times = np.linspace(0., t_max, 1000)[1:]

		# Normalise time axis the same way as WeibullSurv
		norm_t = (self.scale_target* self.scale* (np.log(times + 1e-6) - self.log_mean)/ self.log_std)  # [T]

		# Broadcast to [T, N, K]
		t_	= norm_t[:, None, None]
		k_	= k[None, :, :]
		lam_  = np.log(lam + 1e-6)[None, :, :]

		component_surv = np.exp(-np.exp(k_ * (t_ - lam_)))   # [T, N, K]
		surv = (weights[None, :, :] * component_surv).sum(axis=2)  # [T, N]
		surv = np.clip(surv, 1e-6, 1. - 1e-6)

		return surv, times

	def compute_val_metrics(self, dataloader, temporal_norm=100):
		val_evaluator = DSMSurvivalEvaluator(
			self.net, dataloader,
			log_mean=self.log_mean,
			log_std=self.log_std,
			scale=self.scale,
			scale_target=self.scale_target,
		)
		return val_evaluator(verbose=True, return_weighted=True)[:, 1].sum().item()

	def compute_baseline_hazard(self, *args, **kwargs):
		raise NotImplementedError(
			"DSMSurv is a parametric model; baseline hazard estimation "
			"is not defined. Use predict_surv() instead."
		)

class DeepHitWrapperClinical(torch.nn.Module):
	def __init__(self, embedding, num_time_bins: int):
		super().__init__()
		self.embedding	= embedding
		self.output_head  = torch.nn.Linear(num_time_bins, num_time_bins)

	def forward(self, batch, seq_len=None, **kwargs):
		if self.training:
			xrep = self.embedding(batch, seq_len=seq_len)[0]
		else:
			xrep = self.embedding(batch, seq_len=seq_len)
		return self.output_head(xrep)

class DeepHitWrapperSUPPORT(torch.nn.Module):
	def __init__(self, embedding, num_time_bins: int):
		super().__init__()
		self.embedding	= embedding
		self.output_head  = torch.nn.Linear(num_time_bins, num_time_bins)

	def forward(self, batch, seq_len=None, **kwargs):
		
		if(type(batch).__name__!="BatchRepresentation"):

			xrep=self.embedding(batch, seq_len=seq_len)
		else:
			xrep = self.embedding(batch, seq_len=seq_len)

		if(self.embedding.training):
			xrep=xrep[0]

		return self.output_head(xrep)

class DeepHitSurv(_BatchRepSurv):

	def __init__(self, net, loss=None, optimizer=None, device=None,
				 num_time_bins: int = 100, time_grid: np.ndarray = None, gamma=2, window=0.5, scale=1, scale_target=0.2,):
		_net = DeepHitWrapperSUPPORT(net, num_time_bins)
		_net.to(device="cuda:0")

		if loss is None:
			raise NotImplementedError
		loss_fn=loss(alpha=0.2, sigma=0.1, time_grid=time_grid, num_time_bins=num_time_bins)

		super().__init__(_net, loss_fn, optimizer, device)

		self.num_time_bins = num_time_bins
		self.time_grid = time_grid		# [num_bins]
		self.scale=scale
		self.scale_target=scale_target
		self.gamma=gamma
		self.window=window
		self.log_std=None
		self.log_mean=None

	def compute_loss(self, batch):
		durations = batch.durations.to(device=self.device)
		censoring = batch.censoring.to(device=self.device)
		logits	= self.net(batch)

		return self.loss(logits, durations, censoring, )

	def predict_surv(self, dataloader, device="cuda:0", **kwargs):
		all_logits, all_times, all_events = [], [], []

		self.net.eval()
		with torch.no_grad():
			for batch in dataloader:
				logits = self.net(batch)					# [B, num_bins]
				all_logits.append(logits.cpu())
				all_times.append(batch.durations.cpu())
				all_events.append(batch.censoring.cpu())

		logits = torch.cat(all_logits)					  # [N, num_bins]
		self.time  = torch.cat(all_times).numpy()
		self.event = torch.cat(all_events).numpy().astype(int)

		pmf  = torch.softmax(logits, dim=1).numpy()		 # [N, num_bins]
		cif  = np.cumsum(pmf, axis=1)					   # [N, num_bins]  F(t)
		surv = 1.0 - cif									# [N, num_bins]  S(t)
		surv = np.clip(surv, 1e-6, 1.0 - 1e-6)

		surv = surv.T

		times = self.time_grid if self.time_grid is not None \
				else np.arange(self.num_time_bins, dtype=float)

		return surv, times

	def _predict(self, dataloader, device="cuda:0"):
		all_preds, all_seq_len, all_durations, all_events = [], [], [], []
		self.net.eval()
		with torch.no_grad():
			for batch in dataloader:
				logits = self.net(batch)
				all_preds.append(logits.cpu())
				all_seq_len.append(batch.values.cpu())
				all_durations.append(batch.durations.cpu())
				all_events.append(batch.censoring.cpu())
		return all_preds, all_seq_len, all_durations, all_events

	#def compute_metrics(self, data, metrics): 
	#	return super(_BatchRepSurv, self).compute_metrics(((data.input_ids, data.values), (data.durations, data.censoring)), metrics)

	def compute_val_metrics(self, dataloader, temporal_norm=100):
		val_evaluator = DeepHitSurvivalEvaluator(
			self.net, dataloader,
			log_mean=self.log_mean,
			log_std=self.log_std,
			scale=self.scale,
			scale_target=self.scale_target,
			time_grid=self.time_grid,
			num_time_bins=self.num_time_bins
		)
		return val_evaluator(verbose=True, return_weighted=True)[:, 1].sum().item()

	def compute_baseline_hazard(self, *args, **kwargs):
		raise NotImplementedError(
			"DeepHitSurv is a discrete PMF model; baseline hazard is not defined. "
			"Use predict_surv() instead."
		)

