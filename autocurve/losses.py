import torch

from autocurve.survival.losses import CoxPHLoss, CoxFocalLoss, MultiStateCoxLoss, ExponentialLoss, WeibullLoss, GeneralizedGammaLoss, CoxExponentialLoss, CoxWeibullLoss, CoxGeneralizedGammaLoss

def l1norm(x):
	return x.abs().sum()

def l2norm(x):
	return (x ** 2).sum()

def neglog_normal_gamma_prior(beta: torch.Tensor, raw_log_lambda: torch.Tensor, normfunc=l2norm, a: float = 2.0,b: float = 1.0,eps: float = 1e-8):
	if(normfunc=="l1"):
		normfunc=l1norm
	elif(normfunc=="l2"):
		normfunc=l2norm

	lam = torch.nn.functional.softplus(raw_log_lambda) + eps

	p = beta.numel()
	beta2 = normfunc(beta)

	nlp_beta = 0.5 * lam * beta2 - 0.5 * p * torch.log(lam)

	nlp_lam = b * lam - (a - 1.0) * torch.log(lam)

	return nlp_beta + nlp_lam