import torch

import numpy as np

import pandas as pd	

import pycox

from autocurve.survival import _BatchRepSurv

from autocurve.survival.cox import CoxSurvivalEvaluator

class CoxSurv(_BatchRepSurv):
	
	def __init__(self, net, loss=None, optimizer=None, gamma=2, device=None, temporal_norm=10, window=0.5, scale=1, scale_target=0.2,):
		class CoxPHLoss(pycox.models.loss.CoxPHLoss):
			def forward(self, x, durations, events, log_mean=0, log_std=1., eps=1e-6):
				if(type(x) is tuple):
					x=x[0]
				if(len(x.shape)>1):
					return super(CoxPHLoss, self,).forward(x[:,-1], durations, events)
				elif(len(x.shape)==1):
					return super(CoxPHLoss, self,).forward(x, durations, events)
				else:
					raise NotImplementedError
		
		if(loss is None): loss=CoxPHLoss()
		super(CoxSurv, self).__init__(net, loss, optimizer, device)
	
	def compute_loss(self, batch):
		durations=batch.durations.to(device=self.device)
		censoring=batch.censoring.to(device=self.device)
		
		return self.loss(self.net(batch), durations, censoring, )
	
	def predict_surv(self, dataloader, batch_size=8224, numpy=None, eval_=True, to_cpu=False, num_workers=0):
		all_risks=[]
		all_times=[]
		all_events=[]
		self.net.eval()
		with torch.no_grad():
			for batch in dataloader:
				durations=batch.durations.to(device)
				censoring=batch.censoring.to(device)
				risks=self.net(batch)[:,-1]
				all_risks.append(risks.cpu())
				all_times.append(durations.cpu())
				all_events.append(censoring.cpu())
		self.risk = torch.cat(all_risks).numpy()
		self.time = torch.cat(all_times).numpy()
		self.event = torch.cat(all_events).numpy().astype(int)
		self.surv_dfs=[]
		valid=self.time>=0
		risk, time, event=self.risk[valid], self.time[valid], self.event[valid]
		base_times, base_cumhaz=cox_breslow_baseline_hazard(risk, time, event)
		return cox_make_surv_df(risk, base_times, base_cumhaz), base_times
	
	def compute_val_metrics(self, dataloader, temporal_norm=100):
		"""Return the ibs"""
		val_evaluator=CoxSurvivalEvaluator(self.net, dataloader)
		return val_evaluator(verbose=True, return_weighted=True,)[:,1].sum().item()
	
	def compute_baseline_hazard(self, loader, temporal_norm=100):
		with torch.no_grad():
			input_ids, seq_len, durations, events=[], [], [], []
			
			for batch in loader.dataset:
				input_ids.append(batch.input_ids.to(device="cuda:0").unsqueeze(0))
				seq_len.append(batch.values.to(device="cuda:0").unsqueeze(0))
				durations.append(batch.durations.unsqueeze(0))
				events.append(batch.censoring.unsqueeze(0))
				
			input_ids=torch.concatenate(input_ids, dim=0)
			seq_len=torch.concatenate(seq_len, dim=0)
			durations=torch.concatenate(durations, dim=0).numpy()
			events=torch.concatenate(events, dim=0).numpy()
				
			df = self.target_to_df((durations, events))
			
			return (df
					.assign(expg=np.exp(self.net(input_ids, seq_len,).cpu().numpy()[:,-1]))
					.groupby(self.duration_col)
					.agg({'expg': 'sum', self.event_col: 'sum'})
					.sort_index(ascending=False)
					.assign(expg=lambda x: x['expg'].cumsum())
					.pipe(lambda x: x[self.event_col]/x['expg'])
					.fillna(0.)
					.iloc[::-1]
					.rename('baseline_hazards'))