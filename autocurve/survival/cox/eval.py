import torch

import numpy as np

import pandas as pd

from pycox.evaluation import EvalSurv

def cox_breslow_baseline_hazard(risk, time, event):
	order=np.argsort(time)
	time=time[order]
	event=event[order]
	risk=risk[order]

	unique_times=np.unique(time[event == 1])
	exp_risk=np.exp(risk)

	cum_hazard=[]
	H=0.0

	for t in unique_times:
		d = np.sum(event[time == t])
		at_risk = exp_risk[time >= t].sum()
		H += d / at_risk
		cum_hazard.append(H)

	return unique_times, np.array(cum_hazard)

def cox_make_surv_df(risk, base_times, base_cumhaz):
	surv=np.exp(-np.outer(base_cumhaz, np.exp(risk)))
	return surv, base_times

class CoxSurvivalEvaluator():

	def __init__(self, model, loader):

		self.model=model
		self.loader=loader

		self.risk=None
		self.time=None
		self.event=None

		self.surv_dfs=None

		self.log_mean=None
		self.log_std=None

	def compute_risks(self, device="cuda:0"):

		all_risks=[]
		all_times=[]
		all_events=[]
		
		self.model.eval()
		with torch.no_grad():
			for batch in self.loader:

				durations=batch.durations.to(device)
				censoring=batch.censoring.to(device)
		
				risks=self.model(batch)[:,-1]
		
				all_risks.append(risks.cpu())
				all_times.append(durations.cpu())
				all_events.append(censoring.cpu())
		
		self.risk = torch.cat(all_risks).numpy()
		self.time = torch.cat(all_times).numpy()
		self.event = torch.cat(all_events).numpy().astype(int)

	def compute_metrics(self, verbose=False):
		self.surv_dfs=[]
		metrics=[]
		
		valid=self.time>=0
		risk, time, event=self.risk[valid], self.time[valid], self.event[valid]
		base_times, base_cumhaz=cox_breslow_baseline_hazard(risk, time, event)
		surv, times=cox_make_surv_df(risk, base_times, base_cumhaz)

		surv_df=pd.DataFrame(surv, index=times,)
		self.surv_dfs.append(surv_df)
		
		ev=EvalSurv(surv_df, durations=time, events=event, censor_surv='km')
		cindex=ev.concordance_td()
		ibs=ev.integrated_brier_score(np.linspace(base_times.min()-1e-6, base_times.max()+1e-6, 100))
		metrics+=[[cindex, ibs]]
		
		if(verbose):
			print(type(self).__name__+f" c-index={cindex:.4f}, IBS={ibs:.4f}")
		
		return np.nan_to_num(metrics, nan=np.array([0,1]))

	def __call__(self, verbose=False, return_weighted=False):

		self.compute_risks()

		metrics=self.compute_metrics(verbose=verbose,)
		
		return metrics

	def compute_surv_df(self,):
		valid=self.time>=0
		risk, time, event=self.risk[valid], self.time[valid], self.event[valid]
		base_times, base_cumhaz=cox_breslow_baseline_hazard(risk, time, event)
		surv, times=cox_make_surv_df(risk, base_times, base_cumhaz)

		return surv, times

	def set_params(self, theta, norm=100):
		
		self.risk=(theta[:,-1]).cpu().numpy()