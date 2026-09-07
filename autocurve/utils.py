import random

import torch

import numpy as np

import pandas as pd

from pycox.evaluation import EvalSurv

__all__=["collect_survival_targets", "build_transition_pools", "MultiTransitionCoxBatchSampler", ]

def collect_survival_targets(loader):
	durations = []
	events = []

	for batch in loader:
		durations.append(batch.durations.numpy())
		events.append(batch.censoring.numpy())

	return np.concatenate(durations), np.concatenate(events)

def build_transition_pools(loader):
	event_pools = None
	censor_pools = None

	for j, batch in enumerate(loader):

		durations = batch.durations.cpu()[0]
		events = batch.censoring.cpu()[0]

		if event_pools is None:
			T = len(durations)
			event_pools = [[] for _ in range(T)]
			censor_pools = [[] for _ in range(T)]

		for t in range(T):
			if durations[t] == -1:
				continue

			if bool(events[t]):
				event_pools[t].append(j)
			else:
				censor_pools[t].append(j)

	return event_pools, censor_pools

def build_event_censor_indices(loader):
    event_indices = []
    censor_indices = []

    for i, sample in enumerate(loader):

        duration = sample.durations
        event = sample.censoring

        if torch.is_tensor(duration):
            duration = duration.item()
        if torch.is_tensor(event):
            event = event.item()

        if duration == -1:
            continue
        
        if bool(event):
            event_indices.append(i)
        else:
            censor_indices.append(i)
            
    return event_indices, censor_indices

class MultiTransitionCoxBatchSampler(torch.utils.data.BatchSampler):
	def __init__(self,event_pools,censor_pools,batch_size: int,min_events: int = 2,min_transitions: int = 2,steps_per_epoch: int = 2000,seed: int = 0,drop_last: bool = True,use_global_censored: bool = True,prefer_unused_padding: bool = True,recycle_events: bool = True,max_resample_tries: int = 50,):
		self.event_pools = {t: list(p) for t, p in enumerate(event_pools)}
		self.censor_pools = {t: list(p) for t, p in enumerate(censor_pools)}

		self.batch_size = int(batch_size)
		self.min_events = int(min_events)
		self.min_transitions = int(min_transitions)
		self.steps_per_epoch = int(steps_per_epoch)
		self.seed = int(seed)
		self.drop_last = bool(drop_last)

		self.use_global_censored = bool(use_global_censored)
		self.prefer_unused_padding = bool(prefer_unused_padding)
		self.recycle_events = bool(recycle_events)
		self.max_resample_tries = int(max_resample_tries)

		self.valid_ts = [t for t in self.event_pools if len(self.event_pools[t]) >= self.min_events]
		if len(self.valid_ts) < self.min_transitions:
			raise ValueError(
				f"Not enough transitions with events: "
				f"have {len(self.valid_ts)} valid transitions, need {self.min_transitions}."
			)

		self.global_censored = []
		for t in self.censor_pools:
			self.global_censored.extend(self.censor_pools[t])

		if self.batch_size < self.min_events * self.min_transitions:
			raise ValueError(
				f"batch_size={self.batch_size} is smaller than the event core "
				f"min_events*min_transitions={self.min_events*self.min_transitions}."
			)

		for t in range(len(self.event_pools)):
			print(t, "events", len(self.event_pools[t]), "censored", len(self.censor_pools[t]))

	def __len__(self):
		return self.steps_per_epoch

	def __iter__(self):
		rng = random.Random(self.seed)

		for t in self.valid_ts:
			rng.shuffle(self.event_pools[t])

		event_ptr = {t: 0 for t in self.valid_ts}

		if self.use_global_censored:
			pad_pool = self.global_censored
		else:
			pad_pool = None

		if self.prefer_unused_padding and self.use_global_censored and len(self.global_censored) > 0:
			unused_pad = list(self.global_censored)
			rng.shuffle(unused_pad)
			unused_pad_ptr = 0
		else:
			unused_pad = None
			unused_pad_ptr = 0

		def recycle_all_events():
			for tt in self.valid_ts:
				rng.shuffle(self.event_pools[tt])
				event_ptr[tt] = 0

		for _step in range(self.steps_per_epoch):
			active_ts = [t for t in self.valid_ts if event_ptr[t] + self.min_events <= len(self.event_pools[t])]

			if len(active_ts) < self.min_transitions:
				return

			built = False
			for _try in range(self.max_resample_tries):
				ts = rng.sample(active_ts, self.min_transitions)

				batch = []
				for t in ts:
					p = event_ptr[t]
					batch.extend(self.event_pools[t][p:p + self.min_events])

				remaining = self.batch_size - len(batch)
				if remaining > 0:
					if not self.use_global_censored:
						local_pad = []
						for t in ts:
							local_pad.extend(self.censor_pools[t])
						current_pad_pool = local_pad
					else:
						current_pad_pool = pad_pool

					if self.prefer_unused_padding and unused_pad is not None and remaining > 0:
						while remaining > 0 and unused_pad_ptr < len(unused_pad):
							idx = unused_pad[unused_pad_ptr]
							unused_pad_ptr += 1
							if idx in batch:
								continue
							batch.append(idx)
							remaining -= 1

					if remaining > 0 and current_pad_pool and len(current_pad_pool) > 0:
						for _ in range(remaining):
							for _inner in range(10):
								idx = rng.choice(current_pad_pool)
								if idx not in batch:
									break
							batch.append(idx)
						remaining = 0

				if len(batch) < self.batch_size and self.drop_last:
					continue

				for t in ts:
					event_ptr[t] += self.min_events

				rng.shuffle(batch)
				yield batch
				built = True
				break

			if not built:
				return

class SingleTransitionCoxBatchSampler(torch.utils.data.Sampler):
    def __init__(self, event_indices, censor_indices, batch_size, min_events=2,
                 steps_per_epoch=100, seed=0, recycle_events=True):
        self.event_indices = list(event_indices)
        self.censor_indices = list(censor_indices)
        self.batch_size = batch_size
        self.min_events = min_events
        self.steps_per_epoch = steps_per_epoch
        self.seed = seed
        self.recycle_events = recycle_events

        if len(self.event_indices) < self.min_events:
            raise ValueError("Not enough event indices")
        if self.batch_size < self.min_events:
            raise ValueError("batch_size must be >= min_events")

    def __len__(self):
        return self.steps_per_epoch

    def __iter__(self):
        rng = random.Random(self.seed)

        event_pool = self.event_indices[:]
        censor_pool = self.censor_indices[:]

        rng.shuffle(event_pool)
        rng.shuffle(censor_pool)

        e_ptr = 0
        c_ptr = 0

        for _ in range(self.steps_per_epoch):
            if e_ptr + self.min_events > len(event_pool):
                rng.shuffle(event_pool)
                e_ptr = 0

            batch = event_pool[e_ptr:e_ptr + self.min_events]
            e_ptr += self.min_events

            while len(batch) < self.batch_size:
                if not censor_pool:
                    break
                if c_ptr >= len(censor_pool):
                    rng.shuffle(censor_pool)
                    c_ptr = 0
                batch.append(censor_pool[c_ptr])
                c_ptr += 1
                

            rng.shuffle(batch)
            yield batch