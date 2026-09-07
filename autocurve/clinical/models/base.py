import copy

import random

import numpy as np

import torch

from scipy._lib._bunch import _make_tuple_bunch

#need to update this to the correct one for clinical. this comes from the rna that was done first in the project
BatchRepresentation=_make_tuple_bunch('BatchRepresentation', ['input_ids', 'latent', 'values', 'attention_mask', 'targets', 'durations', 'censoring'],)	

from sklearn.model_selection import KFold

import autocurve.clinical.datasets as datasets

from autocurve.clinical.models import RNN, LSTM

class BatchDataLoader(torch.utils.data.DataLoader):
	def __iter__(self):
		base_iter=super().__iter__()

		for batch in base_iter:
			new_batch=self.transform_batch(batch)
			yield new_batch

	def transform_batch(self, batch):
		return BatchRepresentation(*batch)

class DatasetClinical(torch.utils.data.Dataset):

	def __init__(self, indices, i_transition=2, transitions=None, individuals=None, dataset_dir=None, file_name="data"):

		super(DatasetClinical, self).__init__()

		self.indices=indices
		self.dataset_dir=dataset_dir
		self.file_name=file_name

		dataset_name=dataset_dir.split("/")[-1].upper()
		dataset_class=getattr(datasets, dataset_name.upper())

		if(transitions is None):
			transitions=dataset_class.get_transitions(dataset_dir=dataset_dir)
			
			self.filled_transitions={}
			
			for patno in transitions:
				
				possible_transitions=[dataset_class.transition(patno, -1, -1, -1, dataset_class.state(i), dataset_class.state(i), -1, 101*np.ones((1,1), dtype=np.int32)) for i in dataset_class.state.values[:-1]]
				
				for t in transitions[patno]:
					if(t.orig>=len(dataset_class.state.values)-1):
						continue
					
					if(t.orig==i_transition):
						possible_transitions[t.orig]=t

						self.filled_transitions[patno]=t
						
			transitions=self.filled_transitions
		
		self.filled_transitions=copy.deepcopy(transitions)
		
		self.individuals=np.array(list(self.filled_transitions.keys()))
		
	def __len__(self,): return len(self.indices)

	def __getitem__(self, idx):
		
		idx=self.indices[idx]

		patno=self.individuals[idx]

		return BatchRepresentation(torch.tensor(self.filled_transitions[patno].measures).float(), torch.tensor(0), torch.tensor(self.filled_transitions[patno].seq_len).long(), torch.tensor(0), torch.tensor(0), torch.tensor(self.filled_transitions[patno].duration).long(), torch.tensor(self.filled_transitions[patno].event).bool())


class CV_Clinical():
	
	def __init__(self, folds=5, i_transition=2, dataset_dir=None):
		
		dataset_name=dataset_dir.split("/")[-1].upper()
		dataset_class=getattr(datasets, dataset_name.upper())
		self.transitions=dataset_class.get_transitions(dataset_dir=dataset_dir,)
		
		self.filled_transitions={}
		for patno in self.transitions:
			
			possible_transitions=[dataset_class.transition(patno, -1, -1, -1, dataset_class.state(i), dataset_class.state(i), -1, 101*np.ones((1,1), dtype=np.int32)) for i in dataset_class.state.values[:-1]]
			
			#if(len(self.transitions[patno])>0 and self.transitions[patno][0].orig!=self.transitions[patno][0].dest):
			for t in self.transitions[patno]:
				if(t.orig>=len(dataset_class.state.values)-1):
					continue
				
				if(t.orig==i_transition):
					possible_transitions[t.orig]=t

					self.filled_transitions[patno]=t

		self.transitions=self.filled_transitions
		
		self.individuals=np.array(list(self.transitions.keys()))
		
		self.unique_individuals=np.unique(self.individuals)
		
		random.shuffle(self.unique_individuals)

		self.event_inds=dict(zip(range(len(dataset_class.state.states)-1), [[] for i in range(len(dataset_class.state.states)-1)]))
		self.censored_inds=dict(zip(range(len(dataset_class.state.states)-1), [[] for i in range(len(dataset_class.state.states)-1)]))
		
		for ind in self.unique_individuals:
			#for transition in self.transitions[ind]:
			if(self.transitions[ind].orig in self.event_inds.keys() and ind in self.individuals):
				if(self.transitions[ind].event):
					self.event_inds[self.transitions[ind].orig]+=[ind]
				else:
					self.censored_inds[self.transitions[ind].orig]+=[ind]

		self.event_idx=dict(zip(range(len(dataset_class.state.states)-1), [[] for i in range(len(dataset_class.state.states)-1)]))
		self.censored_idx=dict(zip(range(len(dataset_class.state.states)-1), [[] for i in range(len(dataset_class.state.states)-1)]))

		self.event_splits={}
		self.censored_splits={}

		for state in self.event_inds:
			self.event_idx[state]=self.unique_individuals[np.argwhere(np.isin(self.unique_individuals, np.array(self.event_inds[state]), ))]
			self.censored_idx[state]=self.unique_individuals[np.argwhere(np.isin(self.unique_individuals, np.array(self.censored_inds[state]), ))]
		
			self.event_idx[state]=np.random.permutation(self.event_idx[state])
			self.censored_idx[state]=np.random.permutation(self.censored_idx[state])

		#remove unique inds from lower levels
		for state in list(range(1, len(dataset_class.state.states)-1))[::-1]:
			self.event_idx[state-1]=self.event_idx[state-1][~np.isin(self.event_idx[state-1], self.event_idx[state])]
			self.censored_idx[state-1]=self.censored_idx[state-1][~np.isin(self.censored_idx[state-1], self.censored_idx[state])]
			
		#assert event and censored have different individuals
		for state_i in range(len(dataset_class.state.states)-1):
			for state_j in range(len(dataset_class.state.states)-1):
				self.censored_idx[state_i]=self.censored_idx[state_i][~np.isin(self.censored_idx[state_i], self.event_idx[state_j])]

		#convert to indices
		for state in list(range(len(dataset_class.state.states)-1)):
			self.event_idx[state]=np.argwhere(np.isin(self.unique_individuals, self.event_idx[state]))
			self.censored_idx[state]=np.argwhere(np.isin(self.unique_individuals, self.censored_idx[state]))
		
		for state in self.event_inds:
			self.n_splits=folds

			self.kfold=KFold(n_splits=self.n_splits)

			if(len(self.event_idx[state])>=self.n_splits): self.event_splits[state]=list(self.kfold.split(self.event_idx[state]))
			if(len(self.censored_idx[state])>=self.n_splits):  self.censored_splits[state]=list(self.kfold.split(self.censored_idx[state]))
					
		self.train_dataset=None
		self.test_dataset=None
		
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

class ClinicalEncoder(torch.nn.Module):
	
	def __init__(self, n_features=65, input_dim=7, hidden_dim=128, encoder=LSTM):
		super(ClinicalEncoder, self).__init__()
		
		self.emb=torch.nn.Linear(input_dim, hidden_dim)
		
		self.rnn=encoder(input_dim=hidden_dim, hidden_dim=hidden_dim)
		
		self.head=torch.nn.Linear(n_features*hidden_dim, hidden_dim)
		
		torch.nn.init.xavier_uniform_(self.head.weight)
		torch.nn.init.xavier_uniform_(self.emb.weight)
		
		torch.nn.init.zeros_(self.head.bias)
		torch.nn.init.zeros_(self.emb.bias)
		
		
	def forward(self, x):
		
		x=self.emb(x)
		
		x=self.rnn(x)
		
		return self.head(x.view(*x.shape[:2], -1))


class ClinicalModel(torch.nn.Module):
	def __init__(self, n_features=65, input_dim=7, hidden_dim=128, n_transitions=5, n_estimations=2, encoder=LSTM):
		super(ClinicalModel, self).__init__()
		
		self.n_estimations=n_estimations
		
		self.encoder=ClinicalEncoder(n_features=n_features, input_dim=input_dim, hidden_dim=hidden_dim, encoder=encoder)
		
		self.head=torch.nn.Linear(hidden_dim, self.n_estimations)
		
		torch.nn.init.xavier_uniform_(self.head.weight)
		torch.nn.init.zeros_(self.head.bias)
		
		
	def forward(self, x, seq_len=None, device="cuda:0"):
		
		if(type(x).__name__=="BatchRepresentation"):
			seq_len=x.values.to(device=device)
			x=x.input_ids.to(device=device)
		else:
			x=x.to(device=device)
			seq_len=seq_len.to(device=device)
			
		#output=self.temporal(output)
		seq_len_flag=(seq_len>1).float().unsqueeze(1)
		
		output=self.encoder(x)
		
		output=self.head(output)

		seq_len = (seq_len - 1).clamp(0, output.size(1)-1).long()#running in gpu makes this unstable... need to clamp and cast to long

		if(self.training):
			return torch.gather(output, dim=1, index=seq_len.unsqueeze(1).unsqueeze(2).repeat(1,1,self.n_estimations)).squeeze(1), seq_len_flag
		return torch.gather(output, dim=1, index=seq_len.unsqueeze(1).unsqueeze(2).repeat(1,1,self.n_estimations)).squeeze(1)
	
	def predict(self, x, seq_len):
		return self.forward(x, seq_len)

class ClinicalModule():

	cv_class=CV_Clinical
	model=ClinicalModel
	dataset_class=DatasetClinical