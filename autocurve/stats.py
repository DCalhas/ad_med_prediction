import numpy as np

from scipy.stats import linregress as _linregress

import scipy.stats._stats_py as _stats

from scipy._lib._bunch import _make_tuple_bunch

import scipy.special as special

class _SimpleStudentT:
	# A very simple, array-API compatible t distribution for use in
	# hypothesis tests. May be replaced by new infrastructure t
	# distribution in due time.
	def __init__(self, df):
		self.df = df

	def cdf(self, t):
		return special.stdtr(self.df, t)

	def sf(self, t):
		return special.stdtr(self.df, -t)

LinregressResult=_make_tuple_bunch('LinregressResult', ['slope', 'intercept', 'rvalue', 'pvalue', 'x_train', 'y_train', 'x_test', 'y_test', 'wgcna'],)

def linregress(X_train, y_train, X_test, y_test, wgcna):
	res=_linregress(X_train, y_train)
	
	x=X_test
	y=y_test
	
	TINY = 1.0e-20
	
	n = len(x)
	xmean = np.mean(x, None)
	ymean = np.mean(y, None)
	
	x=x-xmean
	y=y-xmean

	y_pred=res.slope * x + res.intercept
	
	ss_res = np.sum((y - y_pred)**2)
	ss_tot = np.sum((y - np.mean(y))**2)
	r2 = 1 - ss_res/ss_tot
	ssxm, ssxym, _, ssym = np.cov(x, y, bias=1).flat
	r = ssxym / np.sqrt(ssxm * ssym)
	
	slope = ssxym / ssxm
	intercept = ymean - slope*xmean
	df = n - 2
	t = r * np.sqrt(df / ((1.0 - r + TINY)*(1.0 + r + TINY)))
	dist=_SimpleStudentT(df)
	prob=_stats._get_pvalue(t, dist, "two-sided")
	
	return LinregressResult(res.slope, res.intercept, r, prob, X_train, y_train, x, y, wgcna)
