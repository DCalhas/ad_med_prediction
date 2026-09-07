import torch

import pycox

import numpy as np

__all__=["CoxPHLoss", "CoxFocalLoss", "MultiStateCoxLoss"]

CoxPHLoss=pycox.models.loss.CoxPHLoss

class CoxFocalLoss(torch.nn.Module):
	def __init__(self, ):
		super(CoxFocalLoss, self).__init__()
		
	def forward(self, risks, duration, censoring, eps=1e-6, gamma=2., tau=1e0):
		with torch.cuda.amp.autocast(enabled=False):

			valid=~( (~censoring.bool()) & (duration == -1) )

			if(valid.sum() == 0):
				return torch.tensor(0., device=risks.device)
				
			risks=risks[valid]
			duration=duration[valid]
			censoring=censoring[valid]
			
			order=torch.argsort(duration, descending=True)
			risks=risks[order]
			censoring=censoring[order].float()
			
			risks/=tau

			risks=risks-risks.amax(dim=0)
	
			loglik=risks-torch.logcumsumexp(risks.float(), dim=0)

			pt = torch.exp(loglik.detach())
			focal_weight = (1 - pt) ** gamma
	
			return -torch.sum(focal_weight * loglik * censoring) / (censoring.sum() + 1e-8)

class MultiStateCoxLoss(CoxFocalLoss):
	def __init__(self, gamma=0.):
		super(MultiStateCoxLoss, self).__init__()

		self.gamma=gamma
	
	def forward(self, risks, duration, censoring, eps=1e-6, ):
		losses = []
		weights = []

		T = risks.shape[1]

		for t in range(T):
			censor_t = censoring[:, t]
			num_events = censor_t.sum()

			if num_events > 2:
				loss_t = super().forward(risks[:, t],duration[:, t],censor_t, eps=eps, gamma=self.gamma)

				weight_t = (1.0 / (num_events + eps)) ** self.gamma
				losses.append(loss_t)
				weights.append(weight_t)

		if len(losses) == 0:
			return torch.tensor(0., device=risks.device)

		losses = torch.stack(losses)
		weights = torch.stack(weights).to(losses.device)

		return (weights * losses).sum() / weights.sum()