import numpy as np

import torch

from torch.utils.data import DataLoader

from scipy._lib._bunch import _make_tuple_bunch

__all__=["_ModelModule",]

BatchRepresentation=_make_tuple_bunch('BatchRepresentation', ['input_ids', 'latent', 'values', 'attention_mask', 'targets', 'durations', 'censoring'],)

class classproperty(property):
	def __get__(self, obj, cls):
		return super().__get__(cls, cls)
		
class _ModelModule:

	@classproperty
	def cv_class(cls):
		raise NotImplementedError

	@classproperty
	def model(cls):
		raise NotImplementedError

	@classproperty
	def dataset_class(cls):
		raise NotImplementedError

class _CV_Model():

	@property
	def vocab_list(self,):
		raise NotImplementedError

	def create_h5(self, model=None, force=False, file_name="data"):
		raise NotImplementedError

	def get_split(self, i, model=None):
		assert i < self.n_splits

		train_idx=[]
		val_idx=[]
		test_idx=[]

		for state in self.event_splits:
			#events			
			split_idx=self.event_splits[state][i]
			_train_idx=self.event_idx[state][split_idx[0]]
			_test_idx=self.event_idx[state][split_idx[1]]
			split=int(0.8*len(_train_idx))
			_val_idx=_train_idx[split:]
			_train_idx=_train_idx[:split]
			train_idx+=[np.argwhere(np.isin(self.individuals,self.unique_individuals[_train_idx])).reshape(-1)]
			val_idx+=[np.argwhere(np.isin(self.individuals,self.unique_individuals[_val_idx])).reshape(-1)]
			test_idx+=[np.argwhere(np.isin(self.individuals,self.unique_individuals[_test_idx])).reshape(-1)]

		for state in self.censored_splits:
			split_idx=self.censored_splits[state][i]
			_train_idx=self.censored_idx[state][split_idx[0]]
			_test_idx=self.censored_idx[state][split_idx[1]]
			split=int(0.8*len(_train_idx))
			_val_idx=_train_idx[split:]
			_train_idx=_train_idx[:split]
			train_idx+=[np.argwhere(np.isin(self.individuals,self.unique_individuals[_train_idx])).reshape(-1)]
			val_idx+=[np.argwhere(np.isin(self.individuals,self.unique_individuals[_val_idx])).reshape(-1)]
			test_idx+=[np.argwhere(np.isin(self.individuals,self.unique_individuals[_test_idx])).reshape(-1)]
		
		return np.concatenate(train_idx, axis=0), np.concatenate(val_idx, axis=0), np.concatenate(test_idx, axis=0)


class _BaseFoundationalModel(torch.nn.Module):

	def __init__(self, **kwargs):

		super(_BaseFoundationalModel, self).__init__(**kwargs)

		#gamma prior
		self.raw_log_lambda=torch.nn.Parameter(torch.tensor(0.0))

	@property
	def _parameters_concatenated(self):
		return torch.concat([p.view(-1) for n, p in self._trainable_params], axis=0)

	def _initialize(self):
		for m in self.modules():
			if isinstance(m, torch.nn.Linear):
				if hasattr(m, "parametrizations") and hasattr(m.parametrizations, "weight"):
					w = m.parametrizations.weight.original
					torch.nn.init.uniform_(w)
				else:
					w = m.weight
					torch.nn.init.xavier_uniform_(w)
				if m.bias is not None:
					torch.nn.init.zeros_(m.bias)
			elif isinstance(m, torch.nn.LayerNorm):
				torch.nn.init.ones_(m.weight)
				torch.nn.init.zeros_(m.bias)


	@staticmethod
	def build(n_classes, pretrained_dir):
		raise NotImplementedError

	@property
	def _trainable_params(self,):
		raise NotImplementedError

	@property
	def hidden_size(self,):
		raise NotImplementedError

class BatchDataLoader(DataLoader):
	def __iter__(self):
		base_iter=super().__iter__()

		for batch in base_iter:
			new_batch=self.transform_batch(batch)
			yield new_batch

	def transform_batch(self, batch):
		return BatchRepresentation(*batch)