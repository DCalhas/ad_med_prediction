from scipy._lib._bunch import _make_tuple_bunch

#need to update this to the correct one for clinical. this comes from the rna that was done first in the project
BatchRepresentation=_make_tuple_bunch('BatchRepresentation', ['input_ids', 'latent', 'values', 'attention_mask', 'targets', 'durations', 'censoring'],)

from ucimlrepo import fetch_ucirepo

from sklearn.model_selection import train_test_split, KFold
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scipy import sparse

import pandas as pd

import numpy as np

import random

import torch

def get_scaler(X):
	if not isinstance(X, pd.DataFrame):
		X=pd.DataFrame(X)

	num_pipe = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),("scale", StandardScaler()),])

	cat_pipe = Pipeline([("impute", SimpleImputer(strategy="most_frequent")),("onehot", OneHotEncoder(handle_unknown="ignore",sparse_output=False,min_frequency=10,)),])

	scaler = ColumnTransformer([("num", num_pipe, selector(dtype_include=["number"])),("cat", cat_pipe, selector(dtype_exclude=["number"])),])

	return scaler.fit(X)


class DatasetSUPPORT():
	
	def __init__(self, indices=None, scaler=None):
		
		self.dataset=fetch_ucirepo(id=880)
		
		self.X = self.dataset.data.features.copy()
		
		full = self.dataset.data.original
		self.times = full["d.time"].to_numpy().astype(np.float32)
		self.events = full["death"].to_numpy().astype(np.bool_)
		
		shortcut_cols = [c for c in ["surv2m", "surv6m", "prg2m", "prg6m"] if c in self.X.columns]
		self.X = self.X.drop(columns=shortcut_cols)
		
		support_fill = {"alb": 3.5,"pafi": 333.3,"bili": 1.01,"crea": 1.01,"bun": 6.51,"wblc": 9.0,"urine": 2502.0,}
		
		for col, val in support_fill.items():
			if col in self.X.columns:
				self.X[col] = self.X[col].fillna(val)
				
		if(scaler is not None):
			self.X=scaler.transform(self.X)
				
		if sparse.issparse(self.X):
			self.X = self.X.toarray()
		elif hasattr(self.X, "to_numpy"):
			self.X = self.X.to_numpy()
		else:
			self.X = np.asarray(self.X)
		
		if(indices is not None):
			self.X, self.times, self.events=self.X[indices], self.times[indices], self.events[indices]
		
		self.X=np.asarray(self.X)
	
	def __len__(self,):
		return self.X.shape[0]
	
	def __getitem__(self,idx):
		
		return BatchRepresentation(torch.tensor(self.X[idx]).float(), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(0), torch.tensor(self.times[idx]).long(), torch.tensor(self.events[idx]).bool())
	
	
class CV_SUPPORT():
	
	def __init__(self, folds=5,):
		
		self.dataset=DatasetSUPPORT(indices=None)
		
		self.individuals=np.array(list(np.arange(0, self.dataset.X.shape[0], 1)))
		
		self.unique_individuals=np.unique(self.individuals)
		
		random.shuffle(self.unique_individuals)

		self.event_inds=[]
		self.censored_inds=[]
		
		for ind in self.unique_individuals:
			if(ind in self.individuals):
				if(self.dataset.events[ind]):
					self.event_inds+=[ind]
				else:
					self.censored_inds+=[ind]

		self.event_idx=[]
		self.censored_idx=[]
		
		self.event_splits=[]
		self.censored_splits=[]
		
		self.event_idx=self.unique_individuals[np.argwhere(np.isin(self.unique_individuals, np.array(self.event_inds), ))]
		self.censored_idx=self.unique_individuals[np.argwhere(np.isin(self.unique_individuals, np.array(self.censored_inds), ))]
	
		self.event_idx=np.random.permutation(self.event_idx)
		self.censored_idx=np.random.permutation(self.censored_idx)
		
		self.n_splits=folds

		self.kfold=KFold(n_splits=self.n_splits)

		if(len(self.event_idx)>=self.n_splits): self.event_splits=list(self.kfold.split(self.event_idx))
		if(len(self.censored_idx)>=self.n_splits):  self.censored_splits=list(self.kfold.split(self.censored_idx))
				
		self.train_dataset=None
		self.test_dataset=None
		
	def get_split_scaler(self, i):
		train_dataset,_,_=self.get_split(i)
		
		X = self.dataset.X
		
		return get_scaler(X[train_dataset])
		
	
	def get_split(self, i, model=None):
		assert i < self.n_splits

		train_idx=[]
		val_idx=[]
		test_idx=[]

		#events			
		split_idx=self.event_splits[i]
		_train_idx=self.event_idx[split_idx[0]]
		_test_idx=self.event_idx[split_idx[1]]
		split=int(0.8*len(_train_idx))
		_val_idx=_train_idx[split:]
		_train_idx=_train_idx[:split]
		train_idx+=[np.argwhere(np.isin(self.individuals,self.unique_individuals[_train_idx])).reshape(-1)]
		val_idx+=[np.argwhere(np.isin(self.individuals,self.unique_individuals[_val_idx])).reshape(-1)]
		test_idx+=[np.argwhere(np.isin(self.individuals,self.unique_individuals[_test_idx])).reshape(-1)]

		
		split_idx=self.censored_splits[i]
		_train_idx=self.censored_idx[split_idx[0]]
		_test_idx=self.censored_idx[split_idx[1]]
		split=int(0.8*len(_train_idx))
		_val_idx=_train_idx[split:]
		_train_idx=_train_idx[:split]
		train_idx+=[np.argwhere(np.isin(self.individuals,self.unique_individuals[_train_idx])).reshape(-1)]
		val_idx+=[np.argwhere(np.isin(self.individuals,self.unique_individuals[_val_idx])).reshape(-1)]
		test_idx+=[np.argwhere(np.isin(self.individuals,self.unique_individuals[_test_idx])).reshape(-1)]
	
		return np.concatenate(train_idx, axis=0), np.concatenate(val_idx, axis=0), np.concatenate(test_idx, axis=0)


class MLP(torch.nn.Module):
	
	def __init__(self, in_dim=1686, hidden_dim=256, classes=1, dropout_p=0.2):
		
		super(MLP, self, ).__init__()
		
		#3 layer mlp
		self.l1=torch.nn.Linear(in_dim, hidden_dim)
		self.a1=torch.nn.ReLU()
		self.bn1=torch.nn.BatchNorm1d(hidden_dim)
		self.d1=torch.nn.Dropout(dropout_p)
		
		self.l2=torch.nn.Linear(hidden_dim, hidden_dim)
		self.a2=torch.nn.ReLU()
		self.bn2=torch.nn.BatchNorm1d(hidden_dim)
		self.d2=torch.nn.Dropout(dropout_p)
		
		self.l3=torch.nn.Linear(hidden_dim, classes)
		
		#init weights
		torch.nn.init.xavier_uniform_(self.l1.weight)
		torch.nn.init.xavier_uniform_(self.l2.weight)
		torch.nn.init.xavier_uniform_(self.l3.weight)
		torch.nn.init.zeros_(self.l1.bias)
		torch.nn.init.zeros_(self.l2.bias)
		torch.nn.init.zeros_(self.l3.bias)
		torch.nn.init.ones_(self.bn1.weight)
		torch.nn.init.ones_(self.bn2.weight)
		torch.nn.init.zeros_(self.bn1.bias)
		torch.nn.init.zeros_(self.bn2.bias)

		from autocurve.clinical.models.base import BatchRepresentation as _BatchRepresentation
		self.BatchRepresentation=_BatchRepresentation
	
	def forward(self, x, seq_len=None):
		if(type(x) is BatchRepresentation or type(x) is self.BatchRepresentation): x=x.input_ids.to(self.l1.weight.device)
		
		x=self.d1(self.bn2(self.a1(self.l1(x))))
		
		x=self.d2(self.bn2(self.a2(self.l2(x))))
		
		return self.l3(x)
	
class BaselineHazardModel(MLP):
	
	def __init__(self, params, **kwargs):
		super(BaselineHazardModel, self).__init__(**kwargs)
		
		self.w=torch.nn.Parameter(torch.ones((params, )))#only use this parameter for shapes paramters of weibull and generalized gamma, the loc paramters should be output of ClinicalModel
		
		torch.nn.init.zeros_(self.w)
		
	def forward(self, x, seq_len=None):
		output=super(BaselineHazardModel, self).forward(x, seq_len=seq_len)#batch times param

		if(type(output) is tuple):
			output, seq_len=output
		
			return torch.concatenate((output, self.w.unsqueeze(0).repeat(output.shape[0], 1)), dim=1), seq_len

		return torch.concatenate((output, self.w.unsqueeze(0).repeat(output.shape[0], 1)), dim=1)
	
	def baseline_hazard(self,):
		raise NotImplementedError

class ModelBuilds():
	
	"""_summary_
	This class has staticmethods to build the models that are use as nets for the PyCox wrapper classes
	
	ClinicalModel subtype 
	"""
	
	@staticmethod
	def exponential(input_dim=1686, hidden_dim=128, classes=1, ):
		
		return BaselineHazardModel(0, in_dim=input_dim, hidden_dim=hidden_dim, classes=classes)
	
	@staticmethod
	def weibull(input_dim=1686, hidden_dim=128, classes=1, ):
		
		return BaselineHazardModel(1, in_dim=input_dim, hidden_dim=hidden_dim, classes=classes)
	
	@staticmethod
	def generalized_gamma(input_dim=1686, hidden_dim=128, classes=1, ):
		
		return BaselineHazardModel(2, in_dim=input_dim, hidden_dim=hidden_dim, classes=classes)
	
	@staticmethod
	def cox(input_dim=7, hidden_dim=64, classes=1, ):
		
		return BaselineHazardModel(0, in_dim=input_dim, hidden_dim=hidden_dim, classes=classes)