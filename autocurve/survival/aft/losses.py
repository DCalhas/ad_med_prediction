import numpy as np

import torch

import pycox

from autocurve.survival.aft import gammainc, gammaincc

from pycox.models.data import pair_rank_mat

__all__=["ExponentialLoss", "WeibullLoss", "GeneralizedGammaLoss"]


class ExponentialLoss(torch.nn.Module):
	def __init__(self,gamma=0, window=2, scale=10, scale_target=0.2):
		super(ExponentialLoss, self).__init__()
		self.gamma=gamma

		self.scale_target=scale_target
		self.window=window
		self.scale=scale
	
	def forward(self, theta, durations, events, eps=1e-6, norm=1, log_mean=0, log_std=1):
		losses = []
		weights = []

		seq_len=0
		if(type(theta) is tuple):
			theta, seq_len=theta
		
		exp_func=torch.exp#torch.nn.functional.softplus
		mask=(durations>0.)# ignoring -1 durations
		#durations = (torch.log(0.36+durations)-np.log(norm))+eps
		#durations=self.scale_target*self.scale*(torch.log(durations+eps)-log_mean)/log_std#log normalization
		#durations=torch.log(durations+eps)

		#durations_plus=durations+self.window#log normalization
		#durations_minus=durations-self.window#log normalization
		
		#correction to avoid nan * 0 or infty*0
		#durations[torch.where(mask==0)]=0

		safe_durations = durations.clamp_min(eps)   # avoids log of <=0
		durations = self.scale_target * self.scale * (torch.log(safe_durations) - log_mean) / max(log_std, eps)

		durations = torch.where(mask, durations, torch.zeros_like(durations))
		durations_plus = torch.where(mask, durations + self.window, torch.zeros_like(durations))
		durations_minus = torch.where(mask, durations - self.window, torch.zeros_like(durations))
		
		mask=mask.float()

		lam=(theta[:,0])*self.scale
		k=torch.ones_like(lam)
		delta=events.float()
		
		truth=durations-lam
		truth=truth
		
		#loss=(truth)**k - delta*( torch.log(k/lam) + (k-1) * torch.log(truth) )
		z=torch.pow(truth.exp(), k)
		event_nll=z - torch.log(k) - k * truth
		censored_nll = z
		
		censored_nll_minus = torch.pow((durations_minus-lam).exp(), k)
		censored_nll_plus = torch.pow((durations_plus-lam).exp(), k)

		log_other=-torch.log(1-(1-(-censored_nll_minus).exp()+(-censored_nll_plus).exp())+eps)#to minimize what is not right.
		
		p_surv=(-censored_nll).exp()+eps
		p_event=(-censored_nll_minus).exp()-(-censored_nll_plus).exp()
		p_other=1-(-censored_nll_minus).exp()+(-censored_nll_plus).exp()+eps
		
		loss= (1-delta)*censored_nll*(1-p_surv)**self.gamma  + delta*event_nll*(p_other)**self.gamma# + 0*delta*log_other*(p_other)**self.gamma
		loss*=mask
		
		return loss.sum() / mask.sum()
	
class WeibullLoss(torch.nn.Module):
	def __init__(self,gamma=0, window=2, scale=10, scale_target=0.2):
		super(WeibullLoss, self).__init__()
		self.gamma=gamma

		self.scale_target=scale_target
		self.window=window
		self.scale=scale
	
	def forward(self, theta, durations, events, eps=1e-6, norm=1, log_mean=0, log_std=1):
		losses = []
		weights = []

		seq_len=0
		if(type(theta) is tuple):
			theta, seq_len=theta
		
		exp_func=torch.exp#torch.nn.functional.softplus
		mask=(durations>0.)# ignoring -1 durations
		#durations = (torch.log(0.36+durations)-np.log(norm))+eps
		#durations=self.scale_target*self.scale*(torch.log(durations+eps)-log_mean)/log_std#log normalization
		#durations=torch.log(durations+eps)

		#durations_plus=durations+self.window#log normalization
		#durations_minus=durations-self.window#log normalization

		safe_durations = durations.clamp_min(eps)   # avoids log of <=0
		durations = self.scale_target * self.scale * (torch.log(safe_durations) - log_mean) / max(log_std, eps)

		durations = torch.where(mask, durations, torch.zeros_like(durations))
		durations_plus = torch.where(mask, durations + self.window, torch.zeros_like(durations))
		durations_minus = torch.where(mask, durations - self.window, torch.zeros_like(durations))
		
		mask=mask.float()

		lam=(theta[:,0])*self.scale
		k=torch.exp(theta[:,1])+eps
		delta=events.float()

		#print(durations_minus, durations_plus, lam, k)
		censored_nll_minus = torch.pow(((durations_minus-lam)).exp()+eps, k)
		censored_nll_plus = torch.pow(((durations_plus-lam)).exp()+eps, k)

		truth=durations-lam
		
		#loss=(truth)**k - delta*( torch.log(k/lam) + (k-1) * torch.log(truth) )
		z=torch.pow(truth.exp()+eps, k)
		event_nll = z - torch.log(k) - k * truth
		#event_nll = z - delta*( k.log()-lam + (k-1) * truth )
		censored_nll = z

		log_other=-torch.log(1-(1-(-censored_nll_minus).exp()+(-censored_nll_plus).exp())+eps)#to minimize what is not right.
		
		p_surv=(-censored_nll).exp()+eps
		p_event=(-censored_nll_minus).exp()-(-censored_nll_plus).exp()
		p_surv=p_surv
		p_event=p_event
		p_other=1-(-censored_nll_minus).exp()+(-censored_nll_plus).exp()+eps

		loss= (1-delta)*censored_nll*(1-p_surv)**self.gamma + delta*event_nll*(p_other)**self.gamma
		loss*=mask

		#print(loss)
		#print()
		#loss*=((1-p)**self.gamma)*mask
		
		return loss.sum() / mask.sum()


class GeneralizedGammaLoss(torch.nn.Module):
	
	def __init__(self, gamma=0, window=2, scale=10, scale_target=0.2):
		super(GeneralizedGammaLoss, self).__init__()
		
		self.window=window
		self.gamma=gamma#focal parameter

		self.scale_target=scale_target
		self.scale=scale

	def forward(self, theta, durations, events, eps=1e-6, norm=100, log_mean=0, log_std=1):
		
		mask=(durations>0.)
		
		delta=events.float()

		seq_len=0
		if(type(theta) is tuple):
			theta, seq_len=theta
		
		#durations=torch.log(0.36+durations/norm)
		#durations=self.scale_target*self.scale*(torch.log(durations+eps)-log_mean)/log_std#log normalization
		#durations=torch.log(durations+eps)

		#durations_plus=durations+self.window#log normalization
		#durations_minus=durations-self.window#log normalization

		safe_durations = durations.clamp_min(eps)   # avoids log of <=0
		durations = self.scale_target * self.scale * (torch.log(safe_durations) - log_mean) / max(log_std, eps)

		durations = torch.where(mask, durations, torch.zeros_like(durations))
		durations_plus = torch.where(mask, durations + self.window, torch.zeros_like(durations))
		durations_minus = torch.where(mask, durations - self.window, torch.zeros_like(durations))
		
		mask=mask.float()
		
		#they are all still in logspace
		log_a, log_d, log_p = theta[:,0], theta[:,1], theta[:,2]
		
		log_a=log_a*self.scale
		log_d=log_d+eps
		log_p=log_p+eps

		k = (log_d - log_p).exp()

		# z = (t / a)^p, computed in log-space for stability
		z = torch.exp(log_p.exp() * (durations - log_a) + eps)
		
		log_pdf = (
			log_p
			- log_d.exp() * log_a
			- torch.special.gammaln(k)
			+ (log_d.exp() - 1.0) * durations
			- z
		)
		
		S = gammaincc(k, z)
		S_eps_plus=gammaincc(k, torch.exp(log_p.exp() * (durations_plus - log_a))+eps)
		S_eps_minus=gammaincc(k, torch.exp(log_p.exp() * (durations_minus - log_a))+eps)
		
		log_survival = torch.log(S+eps)
		log_other=torch.log(1-((1-S_eps_minus)+S_eps_plus)+eps)#to minimize what is not right.

		p_surv=S+eps
		p_pdf=(S_eps_minus-S_eps_plus)
		p_other=((1-S_eps_minus)+S_eps_plus)+eps

		loss = -(delta*log_pdf*(p_other)**self.gamma + (1-delta)*log_survival*(1-p_surv)**self.gamma)
		
		loss*=mask

		return (loss.sum() / mask.sum()).to(theta.dtype)#+neglog_normal_gamma_prior(model._parameters_concatenated, model.raw_log_lambda, normfunc=norm_reg)


class DSMLoss(torch.nn.Module):
	def __init__(self, model):
		super(DSMLoss, self).__init__()
		
		self.alpha=model.discount
		self.k=model.k

	def forward(self, shape, scale, logits, t, e, elbo=True, risk='1', eps=1e-6, norm=100, scale_target=0.2, log_std=1, log_mean=0., _scale=10):
		alpha=self.alpha
		k=self.k

		mask=(t>0.)
		t=t.clamp_min(eps)
		
		t = (scale_target * _scale* (torch.log(t) - log_mean)			/ max(log_std, eps))

		k_ = shape
		b_ = scale
		
		lossf = []
		losss = []
		
		for g in range(k):
			
			k = k_[:, g]
			b = b_[:, g]
			
			s = (b+t)*torch.exp(k)
			f = k + b + ((torch.exp(k)-1)*(b+t)) - torch.exp(s)
			f = f + s
			
			lossf.append(f)
			losss.append(s)
			
		losss = torch.stack(losss, dim=1)
		lossf = torch.stack(lossf, dim=1)
		
		if elbo:
			
			lossg = torch.nn.Softmax(dim=1)(logits)
			losss = lossg*losss
			lossf = lossg*lossf
			losss = losss.sum(dim=1)
			lossf = lossf.sum(dim=1)
			
		else:
			
			lossg = torch.nn.LogSoftmax(dim=1)(logits)
			losss = lossg + losss
			lossf = lossg + lossf
			losss = torch.logsumexp(losss, dim=1)
			lossf = torch.logsumexp(lossf, dim=1)
			
		uncens = torch.logical_and(e, mask)
		cens = torch.logical_and(~e, mask)
		ll = lossf[uncens].sum() + alpha*losss[cens].sum()
		
		return -ll/float(len(uncens)+len(cens))

class DSMUnconditionalLoss(DSMLoss):

	def __init__(self, model, ):
		super(DSMUnconditionalLoss, self).__init__(model)

		self.model=model


	def forward(self, shape, scale, logits, t, e, elbo=True, risk='1', eps=1e-6, norm=100, log_mean=0, log_std=1, _scale=10, scale_target=0.2):

		shape, scale = self.model.get_shape_scale(risk)

		mask=(t>0.)
		t=t.clamp_min(eps)

		#t = (scale_target * _scale* (torch.log(t) - log_mean)/ max(log_std, eps)).clamp(min=eps)
		
		k_ = shape.expand(t.shape[0], -1)
		b_ = scale.expand(t.shape[0], -1)
		
		ll = 0.
		for g in range(self.k):

			k = k_[:, g]
			b = b_[:, g]
			
			s = - (torch.pow(torch.exp(b)*t, torch.exp(k)))
			f = k + b + ((torch.exp(k)-1)*(b+torch.log(t)))
			f = f + s
			
			uncens = torch.logical_and(e, mask)
			cens = torch.logical_and(~e, mask)
			ll += f[uncens].sum() + s[cens].sum()
	
		return -ll.mean()

class DeepHitLoss(pycox.models.loss.DeepHitSingleLoss):

	def __init__(self, alpha=0.2, sigma=0.1, time_grid=None, num_time_bins=50, **kwargs):
		self.time_grid=time_grid
		self.num_time_bins=num_time_bins

		super(DeepHitLoss, self).__init__(alpha=alpha, sigma=sigma, **kwargs)
	
	def forward(self, logits, durations, censoring, **kwargs):
		time_grid_tensor = torch.tensor(self.time_grid, dtype=durations.dtype, device=durations.device)
		idx_durations = torch.bucketize(durations, time_grid_tensor, right=True)
		idx_durations = idx_durations.clamp(0, self.num_time_bins - 1).long()

		rank_mat = torch.tensor(pair_rank_mat(idx_durations.detach().cpu().numpy(), censoring.detach().cpu().numpy()), dtype=idx_durations.dtype, device=idx_durations.device)
		
		return super(DeepHitLoss, self).forward(logits, idx_durations, censoring, rank_mat)