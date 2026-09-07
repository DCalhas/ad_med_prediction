import numpy as np

import pandas as pd

import torch

from autocurve.survival.aft._gamma import gammainc, gammaincc#fully differentiable regularized incomplete gamma functions

from pycox.evaluation import EvalSurv

class WeibullSurvivalEvaluator():
	def __init__(self, model, loader, k=0.5, n_times=100, log_mean=0., log_std=1, scale=1, scale_target=0.2):
		self.scale=scale
		self.scale_target=scale_target
		self.log_mean=log_mean
		self.log_std=log_std
		self.model = model
		self.loader = loader
		self.k = k
		self.n_times = n_times

		self.lam = None
		self.time = None
		self.event = None
		self.surv_dfs = None

	def compute_lambdas(self, device="cuda:0", norm=1):
		all_lam, all_k, all_times, all_events = [], [], [], []

		self.model.eval()
		with torch.no_grad():
			for batch in self.loader:
				durations = batch.durations.to(device)
				event = batch.censoring.to(device)  # <-- already 1=event, 0=censored

				theta = self.model(batch)			 # <-- now model outputs lambda
				lam = theta[:,0]*self.scale#change this to optional
				k = torch.exp(theta[:,1])+1e-6

				all_lam.append(lam.cpu())
				all_k.append(k.cpu())
				all_times.append(durations.cpu())
				all_events.append(event.cpu())

		self.lam = torch.cat(all_lam).numpy()
		self.k = torch.cat(all_k).numpy()
		self.time = torch.cat(all_times).numpy()
		self.event = torch.cat(all_events).numpy().astype(int)

	def compute_metrics(self, verbose=False):
		self.surv_dfs = []
		metrics = []

		valid = self.time >= 0
		event = self.event[valid]
		time = self.time[valid]
		
		surv, times=self.compute_surv_df()

		surv_df = pd.DataFrame(surv, index=times)

		self.surv_dfs.append(surv_df)

		ev = EvalSurv(surv_df, durations=time, events=event, censor_surv='km')
		cindex = ev.concordance_td()
		ibs = ev.integrated_brier_score(times)

		metrics.append([cindex, ibs])
		
		if verbose:
			print(type(self).__name__+f" c-index={cindex:.4f}, IBS={ibs:.4f}")

		return np.nan_to_num(np.array(metrics))

	def __call__(self, verbose=False, return_weighted=False, device="cuda:0", norm=1):
		self.compute_lambdas(device=device, norm=norm)
		metrics = self.compute_metrics(verbose=verbose)

		return metrics
	
	def set_params(self, theta, norm=100):
		self.k=(torch.exp(theta[:,1]*self.scale)+1e-6).cpu().numpy()
		self.lam=(theta[:,0]*self.scale).cpu().numpy()
		
	def compute_surv_df(self,):
		valid = self.time >= 0
		lam = self.lam[valid,]
		k = self.k[valid,]
		time = self.time[valid]

		t_min = max(0.0, float(time.min()))
		t_max = float(time.max())
		times = np.linspace(0., t_max, 1000)[1:]

		norm_times=self.scale_target*self.scale*(np.log(times+ 1e-6)-self.log_mean)/self.log_std#normalize times, the probability is the same because we are innormalized space.
		#norm_times=np.log(times+1e-6)

		# Weibull survival: S(t|x) = exp(-(t/lam)^k)
		surv = np.exp(- np.exp(k[None, :]*(norm_times[:, None] -lam[None, :])))
		
		return surv, times

class ExponentialSurvivalEvaluator(WeibullSurvivalEvaluator):

	def compute_lambdas(self, device="cuda:0", norm=1):
		all_lam, all_k, all_times, all_events = [], [], [], []

		self.model.eval()
		with torch.no_grad():
			for batch in self.loader:
				durations = batch.durations.to(device)
				event = batch.censoring.to(device)  # <-- already 1=event, 0=censored

				theta = self.model(batch)			 # <-- now model outputs lambda
				lam = theta[:,0]*self.scale#change this to optional
				k = torch.ones_like(lam)

				all_lam.append(lam.cpu())
				all_k.append(k.cpu())
				all_times.append(durations.cpu())
				all_events.append(event.cpu())

		self.lam = torch.cat(all_lam).numpy()
		self.k = torch.cat(all_k).numpy()
		self.time = torch.cat(all_times).numpy()
		self.event = torch.cat(all_events).numpy().astype(int)
		
	def set_params(self, theta, norm=100):
		self.k=torch.ones_like(theta[:,0]).cpu().numpy()
		self.lam=(theta[:,0]*self.scale).cpu().numpy()

class GeneralizedGammaSurvivalEvaluator():
	def __init__(self, model, loader, k=0.5, n_times=100, log_mean=0., log_std=1., scale=1., scale_target=0.2):
		self.scale=scale
		self.scale_target=scale_target
		self.log_mean=log_mean
		self.log_std=log_std
		self.model = model
		self.loader = loader
		self.k = k
		self.n_times = n_times

		self.lam = None
		self.time = None
		self.event = None
		self.surv_dfs = None

	def compute_lambdas(self, device="cuda:0", norm=1):
		all_a, all_d, all_p, all_times, all_events = [], [], [], [], []

		self.model.eval()
		with torch.no_grad():
			for batch in self.loader:
				durations = batch.durations.to(device)
				event = batch.censoring.to(device)  # <-- already 1=event, 0=censored

				theta = self.model(batch)			 # <-- now model outputs lambda
				a = theta[:,0]*self.scale#change this to optional
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

	def compute_metrics(self, verbose=False):
		self.surv_dfs = []
		metrics = []
		
		valid = self.time >= 0
		
		time = self.time[valid]
		event = self.event[valid]

		surv, times=self.compute_surv_df()
		surv_df = pd.DataFrame(surv, index=times)

		self.surv_dfs.append(surv_df)

		ev = EvalSurv(surv_df, durations=time, events=event, censor_surv='km')
		cindex = ev.concordance_td()
		ibs = ev.integrated_brier_score(times)

		metrics.append([cindex, ibs])
		
		if verbose:
			print(type(self).__name__+f" c-index={cindex:.4f}, IBS={ibs:.4f}")

		return np.nan_to_num(np.array(metrics))

	def __call__(self, verbose=False, return_weighted=False, device="cuda:0", norm=1):
		self.compute_lambdas(device=device, norm=norm)
		metrics = self.compute_metrics(verbose=verbose)

		return metrics
	
	def set_params(self, theta, norm=100):
		
		self.d=(torch.exp(theta[:,1])+1e-6).cpu().numpy()
		self.p=(torch.exp(theta[:,2])+1e-6).cpu().numpy()
		self.a=(theta[:,0]*self.scale).cpu().numpy()
		
	def compute_surv_df(self,):
		valid = self.time >= 0
		a = self.a[valid,]
		d = self.d[valid,]
		p = self.p[valid,]
		time = self.time[valid]
		event = self.event[valid]

		t_min = max(0.0, float(time.min()))
		t_max = float(time.max())
		times = np.linspace(0, t_max, 1000)[1:]

		norm_times=self.scale_target*self.scale*(np.log(times+ 1e-6)-self.log_mean)/self.log_std#normalize times, the probability is the same because we are innormalized space.
		#norm_times=np.log(times+1e-6)

		with torch.no_grad(): surv = gammaincc(torch.tensor(d/p), torch.tensor(p[None, :]*(norm_times[:, None] -  a[None, :]) ).exp()).numpy()
		
		return surv, times


class DSMSurvivalEvaluator(WeibullSurvivalEvaluator):

	def __init__(self, model, loader, n_times=100,
				 log_mean=0., log_std=1., scale=1., scale_target=0.2):
		super().__init__(
			model=model,
			loader=loader,
			n_times=n_times,
			log_mean=log_mean,
			log_std=log_std,
			scale=scale,
			scale_target=scale_target,
		)
		self.shapes  = None   # raw log-shape  output of DSM
		self.scales  = None   # raw log-scale  output of DSM
		self.weights = None   # softmax mixture weights

	def compute_lambdas(self, device="cuda:0", norm=1):
		all_shapes, all_scales, all_logits = [], [], []
		all_times, all_events = [], []

		self.model.eval()
		with torch.no_grad():
			for batch in self.loader:
				durations = batch.durations.to(device)
				event	 = batch.censoring.to(device)

				shape, scale, logits = self.model(batch)   # each [B, K]

				all_shapes.append(shape.cpu())
				all_scales.append(scale.cpu())
				all_logits.append(logits.cpu())
				all_times.append(durations.cpu())
				all_events.append(event.cpu())

		self.shapes  = torch.cat(all_shapes).numpy()						# [N, K]
		self.scales  = torch.cat(all_scales).numpy()						# [N, K]
		self.weights = torch.softmax(
			torch.cat(all_logits), dim=1
		).numpy()															# [N, K]
		self.time	= torch.cat(all_times).numpy()
		self.event   = torch.cat(all_events).numpy().astype(int)

		self.k   = np.exp(self.shapes)	# [N, K]  actual Weibull shape
		self.lam = np.exp(-self.scales)   # [N, K]  actual Weibull scale

	def set_params(self, theta, norm=100):
		# theta: [N, 3K] — first K cols shape, next K scale, last K logits
		K = self.model.k if self.model is not None else theta.shape[1] // 3
		shape  = theta[:, :K]
		scale  = theta[:, K:2*K]
		logits = theta[:, 2*K:]

		self.shapes  = shape.numpy() if isinstance(shape, torch.Tensor) else shape
		self.scales  = scale.numpy() if isinstance(scale, torch.Tensor) else scale
		self.weights = torch.softmax(
			torch.tensor(logits) if not isinstance(logits, torch.Tensor) else logits,
			dim=1
		).numpy()
		self.k   = np.exp(self.shapes)	# [N, K]
		self.lam = np.exp(-self.scales)   # [N, K]

	def compute_surv_df(self):
		valid   = self.time >= 0
		shapes  = self.shapes[valid]
		scales  = self.scales[valid]
		weights = self.weights[valid]
		time	= self.time[valid]

		t_max = float(time.max())
		times = np.linspace(0., t_max, 1000)[1:]

		norm_times = (self.scale_target * self.scale* (np.log(times + 1e-6) - self.log_mean)/ max(self.log_std, 1e-6))

		t_ = norm_times[:, None, None]
		k_ = np.exp(shapes)[None, :, :]
		b_ = scales[None, :, :]

		# norm_times is already in log-space — use log-linear formula, no np.log(t_)
		component_surv = np.exp(-np.exp(k_ * (t_ - b_)))
		component_surv = np.nan_to_num(component_surv, nan=0.0, posinf=0.0, neginf=0.0)

		surv = (weights[None, :, :] * component_surv).sum(axis=2)
		surv = np.clip(surv, 1e-6, 1. - 1e-6)

		return surv, times

class DeepHitSurvivalEvaluator(WeibullSurvivalEvaluator):
	def __init__(self, model, loader, time_grid: np.ndarray, n_times=100, num_time_bins=50, log_mean=0., log_std=1., scale=1., scale_target=1.,):
		super().__init__(
			model=model,
			loader=loader,
			n_times=n_times,
			log_mean=log_mean, log_std=log_std, scale=scale, scale_target=scale_target,
		)
		self.time_grid = time_grid   # [num_bins] real-time values
		self.num_time_bins=num_time_bins
		self.logits	= None		# raw network output [N, num_bins]
		self.pmf	   = None		# softmax(logits)	[N, num_bins]

	def compute_lambdas(self, device="cuda:0", norm=1):
		all_logits, all_times, all_events = [], [], []

		self.model.eval()
		with torch.no_grad():
			for batch in self.loader:
				durations = batch.durations.to(device)
				event	 = batch.censoring.to(device)
				logits	= self.model(batch)		   # [B, num_bins]

				all_logits.append(logits.cpu())
				all_times.append(durations.cpu())
				all_events.append(event.cpu())

		self.logits = torch.cat(all_logits)			 # [N, num_bins]
		self.pmf	= torch.softmax(self.logits, dim=1).numpy()
		self.time   = torch.cat(all_times).numpy()
		self.event  = torch.cat(all_events).numpy().astype(int)

	def set_params(self, theta, norm=None):
		if isinstance(theta, torch.Tensor):
			self.logits = theta
		else:
			self.logits = torch.tensor(theta)
		self.pmf = torch.softmax(self.logits, dim=1).numpy()

	def compute_surv_df(self):
		valid = self.time >= 0
		pmf   = self.pmf[valid]						 # [N, num_bins]
		
		cif  = np.cumsum(pmf, axis=1)				   # [N, num_bins]
		surv = 1.0 - cif								# [N, num_bins]
		surv = np.clip(surv, 1e-6, 1.0 - 1e-6)

		# surv is [N, num_bins], transpose to [num_bins, N] to match parent contract
		return surv.T, self.time_grid