__all__=["_Dataset", "_GetTransitions", "preprocess_step_curve"]

import numpy as np

from bisect import bisect_right

def lnds_indices(y):
	y = np.asarray(y)
	tails, tails_idx = [], []
	prev = np.full(len(y), -1, dtype=int)
	for i, val in enumerate(y):
		j = bisect_right(tails, val)
		if j == len(tails):
			tails.append(val)
			tails_idx.append(i)
		else:
			tails[j] = val
			tails_idx[j] = i
		prev[i] = tails_idx[j - 1] if j > 0 else -1
	k = tails_idx[-1] if tails_idx else -1
	keep = []
	while k != -1:
		keep.append(k)
		k = prev[k]
	keep.reverse()
	return np.array(keep, dtype=int)


def confirmed_cummax(y, min_confirm=1):
	"""Monotone-increasing clamp that keeps every visit.

	A rise to level v at visit i is accepted only if v is corroborated by at least
	`min_confirm` observations at or after i, so an isolated spike (1,1,4,1,1) does
	not lock the patient at the higher stage forever. min_confirm=1 is plain
	np.maximum.accumulate.

	Near the end of follow-up there may be fewer than `min_confirm` visits left, so the
	requirement is capped at the number of remaining observations -- otherwise a rise on
	the final visit could never be confirmed and every patient would lose their last
	transition. A late rise is therefore taken at face value; there is nothing left to
	corroborate it against.

	Unlike an LNDS filter this never deletes a visit, so the stage a patient reaches
	is stamped at the visit where it was first observed rather than being re-timed or,
	when the drop and the rise tie, discarded outright.
	"""
	y = np.asarray(y, dtype=float)
	if y.size == 0:
		return y.copy()

	out = np.empty_like(y)
	cur = y[0]
	out[0] = cur
	for i in range(1, y.size):
		v = y[i]
		required = min(min_confirm, y.size - i)
		if v > cur and int((y[i:] >= v).sum()) >= required:
			cur = v
		out[i] = cur
	return out


def preprocess_step_curve(
	y,
	t=None,
	values=(0.0, 0.5, 1.0, 2.0, 3.0),
	snap=False,
	eps_frac=1e-6,
	return_indices=False,
	method="cummax",
	min_confirm=1,
):
	"""Enforce a monotonically non-decreasing step curve.

	method="cummax": clamp each visit to the running (confirmed) maximum. Keeps every
		visit, so transition times stay at the visit where the stage was first seen.
	method="lnds": the old longest-non-decreasing-subsequence filter, which *deletes*
		visits. On a tie it can delete a genuine stage rather than the dip, which both
		re-times transitions late and can drop a stage from the path entirely.
	"""
	y = np.asarray(y, dtype=float)

	vals = np.asarray(values, dtype=float)
	vals = np.unique(vals)
	vals.sort()

	if t is None:
		t = np.arange(len(y), dtype=float)
	else:
		t = np.asarray(t, dtype=float)

	if snap:
		idx = np.abs(y[:, None] - vals[None, :]).argmin(axis=1)
		y = vals[idx]

	if method == "cummax":
		keep = np.arange(len(y), dtype=int)
		y = confirmed_cummax(y, min_confirm=min_confirm)
	elif method == "lnds":
		keep = lnds_indices(y)
	else:
		raise ValueError(f"unknown method {method!r}; expected 'cummax' or 'lnds'")

	t_hat = t[keep]
	y_hat = y[keep]

	if return_indices:
		return np.array(y_hat), np.array(t_hat), keep
	return np.array(y_hat), np.array(t_hat)

class classproperty(property):
	def __get__(self, obj, cls):
		return super().__get__(cls, cls)

class _Dataset:

	@classproperty
	def get_dataset(cls):
		raise NotImplementedError

	@classproperty
	def state(cls):
		raise NotImplementedError

	@classproperty
	def curve(cls):
		raise NotImplementedError

	@classproperty
	def transition(cls):
		raise NotImplementedError

	@classproperty
	def get_transitions(cls):
		raise NotImplementedError

class _GetTransitions():

	def __init__(self, transition_class, state_class, curves_fn):
		self.transition_class=transition_class
		self.state_class=state_class
		self.curves_fn=curves_fn
	
	def __call__(self, dataset_dir="/ceph/hpc/data/s25r10-02-users/calhasd/adni", state="ON"):
		pat_curves = self.curves_fn(dataset_dir=dataset_dir, state=state)

		pat_transitions = {pat: [] for pat in pat_curves}

		for pat in pat_curves:
			progression = np.asarray(pat_curves[pat].progression, dtype=float)
			progression[np.where(progression==-1.)]=0.
			tt  = np.asarray(pat_curves[pat].t, dtype=float)

			#age_weeks=pat_curves[pat].age_weeks#to assert the age at which the observation at that stage started! it is supposed to be the input of the model so there should be no ground truth as input, age at transition is ground truth
			gender=pat_curves[pat].gender

			n = min(len(progression), len(tt))
			progression = progression[:n]
			tt  = tt[:n]
			#age_weeks  = age_weeks[:n]

			if n == 0:
				continue

			#start_age = int(age_weeks[0])

			if np.all(np.diff(progression) == 0):
				final_time = int(tt[-1])
				s = self.state_class(progression[0])
				pat_transitions[pat].append(self.transition_class(pat, final_time, gender, s, s, final_time))
				continue

			changes=np.argwhere(np.diff(progression) > 0.)[:, 0]
			for c in range(len(changes)-1):
				idx=changes[c]
				next_idx=changes[c+1]
				time = int(tt[next_idx])
				#start_age=age_weeks[idx]#non censor we get the age at transition
				#next_age=age_weeks[idx+1]

				orig = self.state_class(progression[idx])
				dest = self.state_class(progression[next_idx])

				if len(pat_transitions[pat]) > 0:
					prev = pat_transitions[pat][-1]
					duration = int(tt[next_idx] - prev.t)

					if orig.x == prev.orig and dest.x == prev.dest:
						pat_transitions[pat][-1] = self.transition_class(pat,prev.duration + duration,gender,orig,dest,time)
						continue
				else:
					duration = int(tt[next_idx])

				pat_transitions[pat].append(self.transition_class(pat, duration, gender, orig, dest, time))

			if len(pat_transitions[pat]) > 0:
				last_time = int(pat_transitions[pat][-1].t)
				final_time = int(tt[-1])
				if final_time > last_time:
					last_state = self.state_class(self.state_class.values[pat_transitions[pat][-1].dest])
					pat_transitions[pat].append(self.transition_class(pat, final_time - last_time, gender, last_state, last_state, final_time))
			else:
				final_time = int(tt[-1])
				#final_age=age_weeks[0]
				s = self.state_class(progression[0])
				pat_transitions[pat].append(self.transition_class(pat, final_time, gender, s, s, final_time))

		return pat_transitions