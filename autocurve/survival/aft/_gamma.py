import math

import warnings

import torch

from torch.autograd import Function

from torch.autograd.function import once_differentiable

__all__=["gammainc", "gammaincc"]

def _work_dtype(*tensors: torch.Tensor) -> torch.dtype:
	dtype = tensors[0].dtype
	for t in tensors[1:]:
		dtype = torch.promote_types(dtype, t.dtype)
	if dtype in (torch.float16, torch.bfloat16, torch.float32):
		return torch.float64
	return dtype


def _safe_pos(x: torch.Tensor) -> torch.Tensor:
	tiny = torch.finfo(x.dtype).tiny
	return torch.where(x > 0, x, torch.full_like(x, tiny))


def _grad_x_lower(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
	# d/dx P(a, x) = x^(a-1) e^{-x} / Gamma(a)
	x_safe = _safe_pos(x)
	return torch.exp((a - 1) * torch.log(x_safe) - x_safe - torch.special.gammaln(a))


def _grad_x_upper(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
	# d/dx Q(a, x) = -d/dx P(a, x)
	return -_grad_x_lower(a, x)


def _igammac_grad_a(result: torch.Tensor,a: torch.Tensor,x: torch.Tensor,*,n_asympt_terms: int = 9,max_series_terms: int = 100000,series_precision: float = 1e-10,) -> torch.Tensor:
	"""
	d/da Q(a, x), where Q is the regularized upper incomplete gamma.
	Port of the logic in the PR snippet.
	"""
	log_x = torch.log(_safe_pos(x))
	digamma_a = torch.special.digamma(a)
	gammaln_a = torch.special.gammaln(a)

	large = (x >= 8) & (x >= a)

	# Stabilize near 0/1 by recomputing result if needed.
	eps = torch.finfo(result.dtype).eps
	invalid = (result < eps) | ((1 - result) < eps)
	safe_result = torch.where(invalid, torch.special.gammaincc(a, x), result)

	out = torch.zeros_like(a)

	# Large-x asymptotic expansion.
	if large.any().item():
		sum_asympt = torch.zeros_like(a)
		uk = torch.ones_like(a)
		du = torch.ones_like(a)
		u_term = a.clone()

		for k in range(1, n_asympt_terms + 1):
			u_term = u_term - 1
			if k > 1:
				du = uk + du * u_term
			uk = uk * u_term
			sum_asympt = sum_asympt + du / x.pow(k)

		asympt = safe_result * (log_x - digamma_a)
		asympt = asympt + sum_asympt * torch.exp((a - 1) * log_x - x - gammaln_a)
		out = torch.where(large, asympt, out)

	# Small-x series expansion.
	small = ~large
	if small.any().item():
		sum_series = torch.zeros_like(a)
		subseries_term = torch.zeros_like(a)
		sign = 1.0
		log_prec = math.log(series_precision)
		converged = False

		for k in range(max_series_terms + 1):
			if k > 0:
				subseries_term = subseries_term + log_x - math.log(k)

			series_term = subseries_term - 2 * torch.log(a + k)
			sum_series = sum_series + sign * torch.exp(series_term)
			sign = -sign

			done = torch.where(small, series_term <= log_prec, torch.ones_like(small))
			if done.all().item():
				converged = True
				break

		if not converged:
			warnings.warn(
				"igammac grad-a series did not converge to the requested precision; "
				"returning the current partial sum.",
				RuntimeWarning,
			)

		series = (1 - safe_result) * (digamma_a - log_x)
		series = series + sum_series * torch.exp(a * log_x - gammaln_a)
		out = torch.where(small, series, out)

	return out


def _igamma_grad_a(result: torch.Tensor,a: torch.Tensor,x: torch.Tensor,*,max_terms: int = 100000,precision: float = 1e-10,) -> torch.Tensor:
	"""
	d/da P(a, x), where P is the regularized lower incomplete gamma.
	Port of the logic in the PR snippet.
	"""
	radicand = 60.0 * x - x.pow(2.0) - 756.0
	use_gammac_grad = (
		((a < 0.8) & (x > 15.0))
		| ((a < 12.0) & (x > 30.0))
		| (a < torch.sqrt(radicand))
	)

	eps = torch.finfo(result.dtype).eps
	invalid = (result < eps) | ((1 - result) < eps)
	safe_result = torch.where(invalid, torch.special.gammainc(a, x), result)

	out = torch.zeros_like(a)

	# Reuse upper-gradient when that branch is preferred.
	if use_gammac_grad.any().item():
		out = torch.where(use_gammac_grad, -_igammac_grad_a(1 - safe_result, a, x), out)

	# Otherwise use Gautschi-style series.
	direct = ~use_gammac_grad
	if direct.any().item():
		log_x = torch.log(_safe_pos(x))
		term_gammaln = torch.special.gammaln(a)
		term_dgammaln = torch.special.digamma(a)
		sum_a = torch.zeros_like(a)
		sum_b = torch.zeros_like(a)
		conv_a = False
		conv_b = False

		for k in range(max_terms + 1):
			term_gammaln = term_gammaln + torch.log(a + k)
			term_exp = torch.exp((a + k) * log_x - term_gammaln)

			if not conv_a:
				term_a = term_exp * log_x
				sum_a = sum_a + term_a
				done_a = torch.where(
					direct, torch.abs(term_a) <= precision, torch.ones_like(direct)
				)
				conv_a = done_a.all().item()

			if not conv_b:
				term_dgammaln = term_dgammaln + 1.0 / (a + k)
				term_b = term_exp * term_dgammaln
				sum_b = sum_b + term_b
				done_b = torch.where(
					direct, torch.abs(term_b) <= precision, torch.ones_like(direct)
				)
				conv_b = done_b.all().item()

			if conv_a and conv_b:
				break
		else:
			warnings.warn(
				"igamma grad-a series did not converge to the requested precision; "
				"returning the current partial sums.",
				RuntimeWarning,
			)

		series = torch.exp(-x) * (sum_a - sum_b)
		out = torch.where(direct, series, out)

	return out


class _GammaincFn(Function):
	@staticmethod
	def forward(ctx, a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
		a_b, x_b = torch.broadcast_tensors(a, x)
		y = torch.special.gammainc(a_b, x_b)
		ctx.save_for_backward(a_b, x_b, y)
		ctx.a_shape = a.shape
		ctx.x_shape = x.shape
		return y

	@staticmethod
	@once_differentiable
	def backward(ctx, grad_output: torch.Tensor):
		a, x, y = ctx.saved_tensors
		work_dtype = _work_dtype(a, x, grad_output)

		a_w = a.to(work_dtype)
		x_w = x.to(work_dtype)
		y_w = y.to(work_dtype)
		g_w = grad_output.to(work_dtype)

		grad_a = None
		grad_x = None

		if ctx.needs_input_grad[0]:
			grad_a = g_w * _igamma_grad_a(y_w, a_w, x_w)
			grad_a = grad_a.sum_to_size(ctx.a_shape).to(a.dtype)

		if ctx.needs_input_grad[1]:
			grad_x = g_w * _grad_x_lower(a_w, x_w)
			grad_x = grad_x.sum_to_size(ctx.x_shape).to(x.dtype)

		return grad_a, grad_x


class _GammainccFn(Function):
	@staticmethod
	def forward(ctx, a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
		a_b, x_b = torch.broadcast_tensors(a, x)
		y = torch.special.gammaincc(a_b, x_b)
		ctx.save_for_backward(a_b, x_b, y)
		ctx.a_shape = a.shape
		ctx.x_shape = x.shape
		return y

	@staticmethod
	@once_differentiable
	def backward(ctx, grad_output: torch.Tensor):
		a, x, y = ctx.saved_tensors
		work_dtype = _work_dtype(a, x, grad_output)

		a_w = a.to(work_dtype)
		x_w = x.to(work_dtype)
		y_w = y.to(work_dtype)
		g_w = grad_output.to(work_dtype)

		grad_a = None
		grad_x = None

		if ctx.needs_input_grad[0]:
			grad_a = g_w * _igammac_grad_a(y_w, a_w, x_w)
			grad_a = grad_a.sum_to_size(ctx.a_shape).to(a.dtype)

		if ctx.needs_input_grad[1]:
			grad_x = g_w * _grad_x_upper(a_w, x_w)
			grad_x = grad_x.sum_to_size(ctx.x_shape).to(x.dtype)

		return grad_a, grad_x

def gammainc(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
	return _GammaincFn.apply(a, x)


def gammaincc(a: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
	return _GammainccFn.apply(a, x)