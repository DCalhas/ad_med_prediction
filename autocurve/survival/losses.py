import torch

import pycox

from autocurve.survival.aft.losses import ExponentialLoss, WeibullLoss, GeneralizedGammaLoss

from autocurve.survival.cox.losses import CoxPHLoss, CoxFocalLoss, MultiStateCoxLoss

"""
All the loss functions forward pass receive theta, duration, event, eps, and norm
"""


__all__=["ExponentialLoss",
		"WeibullLoss",
		"GeneralizedGammaLoss",
		"CoxPHLoss",
		"CoxFocalLoss"
		"MultiStateCoxLoss",
		"CoxExponentialLoss",
		"CoxWeibullLoss",
		"CoxGeneralizedGammaLoss",
		]

class CoxExponentialLoss(torch.nn.Module):
	def __init__(self, gamma=0):
		super(CoxExponentialLoss, self).__init__()
		
		self.exponential_loss=ExponentialLoss(gamma=gamma)
		
		self.cox_loss=pycox.models.loss.CoxPHLoss()
		
	def forward(self, theta, duration, event, eps=1e-6, norm=1):
		
		exponential_loss=self.exponential_loss(theta, duration, event, eps=eps, norm=norm)
		
		cox_loss=self.cox_loss(theta[:,-1], duration, event)
		
		return exponential_loss.mean()+cox_loss.mean()
	
class CoxWeibullLoss(torch.nn.Module):
	def __init__(self, gamma=0):
		super(CoxWeibullLoss, self).__init__()
		
		self.weibull_loss=WeibullLoss(gamma=gamma)
		
		self.cox_loss=pycox.models.loss.CoxPHLoss()
		
	def forward(self, theta, duration, event, eps=1e-6, norm=1):
		
		weibull_loss=self.weibull_loss(theta, duration, event, eps=eps, norm=norm)
		
		cox_loss=self.cox_loss(theta[:,-1], duration, event)
		
		return weibull_loss.mean()+cox_loss.mean()
	
class CoxGeneralizedGammaLoss(torch.nn.Module):
	def __init__(self, gamma=0):
		super(CoxGeneralizedGammaLoss, self).__init__()
		
		self.gg_loss=GeneralizedGammaLoss(gamma=gamma)
		
		self.cox_loss=pycox.models.loss.CoxPHLoss()
		
	def forward(self, theta, duration, event, eps=1e-6, norm=1):
		
		gg_loss=self.gg_loss(theta, duration, event, eps=eps, norm=norm)
		
		cox_loss=self.cox_loss(theta[:,-1], duration, event)
		
		return gg_loss.mean()+cox_loss.mean()