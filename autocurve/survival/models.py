import torchtuples.callbacks as cb
	
import torch

import numpy as np

import pandas as pd

from autocurve.survival.aft.losses import ExponentialLoss, WeibullLoss, GeneralizedGammaLoss

from autocurve.survival.aft import ExponentialSurvivalEvaluator, WeibullSurvivalEvaluator, GeneralizedGammaSurvivalEvaluator
from autocurve.survival.cox import CoxSurv

from autocurve.survival import _BatchRepSurv

class CoxWeibullSurv(CoxSurv):
	
	def __init__(self, net, loss=None, optimizer=None, device=None, gamma=0, temporal_norm=100):
		super(CoxWeibullSurv, self).__init__(net, CoxWeibullLoss(gamma=gamma), optimizer, device)
		
		self.temporal_norm=temporal_norm
	
	def compute_loss(self, batch):
		durations=batch.durations.to(device=self.device)
		censoring=batch.censoring.to(device=self.device)
		
		return self.loss(self.net(batch), durations, censoring, norm=self.temporal_norm)
	
	def compute_val_metrics(self, dataloader, temporal_norm=100):
		"""Return the ibs"""
		val_evaluator=WeibullSurvivalEvaluator(self.net, dataloader)
		val_evaluator(verbose=True, return_weighted=True, norm=temporal_norm)[:,1].sum().item()
		val_evaluator=CoxSurvivalEvaluator(self.net, dataloader)
		return val_evaluator(verbose=True, return_weighted=True,)[:,1].sum().item()
	
	def compute_baseline_hazard(self, temporal_norm=100):
		
		#compute times
		times=torch.linspace(0.36,800,1000).to(device=self.device)
		
		with torch.no_grad():
			norm_times=times/temporal_norm
			
			k=torch.exp(self.net.w)
			lam=torch.exp(self.net.head.bias[0])#get first element
		
			return times.cpu().numpy(), ((k/lam) * (norm_times/lam)**(k-1)).cpu().numpy()
	
class CoxExponentialSurv(CoxSurv):
	
	def __init__(self, net, loss=None, optimizer=None, device=None, gamma=0, temporal_norm=100):
		super(CoxExponentialSurv, self).__init__(net, CoxExponentialLoss(gamma=gamma), optimizer, device)
		
		self.temporal_norm=temporal_norm
	
	def compute_loss(self, batch):
		durations=batch.durations.to(device=self.device)
		censoring=batch.censoring.to(device=self.device)
		
		return self.loss(self.net(batch), durations, censoring, norm=self.temporal_norm)
	
	def compute_val_metrics(self, dataloader, temporal_norm=100):
		"""Return the ibs"""
		val_evaluator=ExponentialSurvivalEvaluator(self.net, dataloader)
		val_evaluator(verbose=True, return_weighted=True, norm=temporal_norm)[:,1].sum().item()
		val_evaluator=CoxSurvivalEvaluator(self.net, dataloader)
		return val_evaluator(verbose=True, return_weighted=True,)[:,1].sum().item()
	
	def compute_baseline_hazard(self, temporal_norm=100):
		
		#compute times
		times=torch.linspace(0.36,800,1000).to(device=self.device)
		
		with torch.no_grad():

			lam=torch.exp(self.net.head.bias)
		
			return times.cpu().numpy(), (lam*temporal_norm).repeat(times.shape[0]).cpu().numpy()
		
class CoxGeneralizedGammaSurv(CoxSurv):
	
	def __init__(self, net, loss=None, optimizer=None, device=None, gamma=0, temporal_norm=100):
		super(CoxGeneralizedGammaSurv, self).__init__(net, CoxGeneralizedGammaLoss(gamma=gamma), optimizer, device)
		
		self.temporal_norm=temporal_norm
	
	def compute_loss(self, batch):
		durations=batch.durations.to(device=self.device)
		censoring=batch.censoring.to(device=self.device)
		
		return self.loss(self.net(batch), durations, censoring, norm=self.temporal_norm)
	
	def compute_val_metrics(self, dataloader, temporal_norm=100):
		"""Return the ibs"""
		val_evaluator=GeneralizedGammaSurvivalEvaluator(self.net, dataloader)
		val_evaluator(verbose=True, return_weighted=True, norm=temporal_norm)[:,1].sum().item()
		val_evaluator=CoxSurvivalEvaluator(self.net, dataloader)
		return val_evaluator(verbose=True, return_weighted=True,)[:,1].sum().item()
	
	def compute_baseline_hazard(self, temporal_norm=100):
		
		#compute times
		times=torch.linspace(0.36,800,1000).to(device=self.device)
		
		with torch.no_grad():
			norm_times=times/temporal_norm
			
			d=torch.exp(self.net.w[0])+1e-6
			p=torch.exp(self.net.w[1])+1e-6
			a=torch.exp(self.net.head.bias)+1e-6
			
			s=d/p
			z=(norm_times/a)**p
			
			num=p * norm_times**(d-1) * (-z).exp()
			
			den=gammaincc(s,z) * torch.special.gammaln(s).exp() * a**d
			
			return times.cpu().numpy(), (num/den).cpu().numpy()