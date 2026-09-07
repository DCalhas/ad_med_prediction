import json

import os

import h5py

import numpy as np

import torch

from sklearn.model_selection import KFold

from autocurve.rna.models import _ModelModule, _BaseFoundationalModel, _CV_Model

import autocurve.rna.datasets as datasets

from teddy.models.model_directory import get_architecture, model_dict

from captum.attr import IntegratedGradients

__all__=["TEDDYModule"]

class CV_TEDDY(_CV_Model):


	def __init__(self, n_splits=5, dataset_dir="/ceph/hpc/data/s25r10-02-users/calhasd/adni", pretrained_dir=None, state="ON"):
		"""
		
		
		"""
		self.topk_genes=2000
		self.seq_len=2048

		self.pretrained_dir=pretrained_dir
		
		self.dataset_dir=dataset_dir

		with open(self.pretrained_dir+"/tokenizer/vocab.json", "r") as f: self.vocab=json.load(f)

		dataset_name=dataset_dir.split("/")[-1].upper()
		dataset_class=getattr(datasets, dataset_name)

		self.gene_ind_matrix, self.gene_ids, self.gene_names, self.individuals, self.targets=dataset_class.get_dataset(dataset_dir=dataset_dir)

		self.transitions=dataset_class.get_transitions(dataset_dir=dataset_dir, state=state)
		self.individuals_transitions=np.array(list(self.transitions.keys())).astype('U25')

		self.gene_ind_matrix=self.gene_ind_matrix[np.isin(self.individuals, self.individuals_transitions)]
		self.targets=self.targets[np.isin(self.individuals, self.individuals_transitions)]
		self.individuals=self.individuals[np.isin(self.individuals, self.individuals_transitions)]
		
		disease_indices=np.where(self.targets>0)
		self.gene_ind_matrix=self.gene_ind_matrix[disease_indices]
		self.individuals=self.individuals[disease_indices]
		self.targets=self.targets[disease_indices]

		self.unique_individuals=np.unique(self.individuals)

		self.event_inds=dict(zip(range(len(dataset_class.state.states)-1), [[] for i in range(len(dataset_class.state.states)-1)]))
		self.censored_inds=dict(zip(range(len(dataset_class.state.states)-1), [[] for i in range(len(dataset_class.state.states)-1)]))
		
		for ind in self.transitions:
			for transition in self.transitions[ind]:
				if(transition.orig in self.event_inds.keys() and ind in self.individuals):
					if(transition.event):
						self.event_inds[transition.orig]+=[ind]
					else:
						self.censored_inds[transition.orig]+=[ind]

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
			self.n_splits=n_splits

			self.kfold=KFold(n_splits=self.n_splits)

			if(len(self.event_idx[state])>=self.n_splits): self.event_splits[state]=list(self.kfold.split(self.event_idx[state]))
			if(len(self.censored_idx[state])>=self.n_splits):  self.censored_splits[state]=list(self.kfold.split(self.censored_idx[state]))
					
		self.train_dataset=None
		self.test_dataset=None

	@property
	def vocab_list(self,):
		return list(self.vocab.keys())


	def create_h5(self, model=None, force=False, file_name="data"):
		file_name=file_name+type(self).__name__
		
		if(os.path.isfile(self.dataset_dir+"/"+file_name+".h5")):
			print("I: Data pointers h5 file already exists. Toggle force to create a new one.", end="\n")
			if(not force): return
		
		N=self.individuals.shape[0]
			
		if self.vocab is None:
			with open(self.pretrained_dir+"/tokenizer/vocab.json", "r") as f: self.vocab=json.load(f)

		input_ids=np.zeros((N, self.seq_len, ), dtype=np.int32)
		attention_mask=np.zeros((N, self.seq_len, ), dtype=np.bool_)
		
		for i in range(N):
			sorted_idx=np.argsort(-self.gene_ind_matrix[i])
	
			k=min(self.topk_genes, len(sorted_idx))
			
			top_indices=sorted_idx[:k]
			
			top_gene_names=np.array(self.gene_ids)[top_indices]
	
			pad_id=self.vocab.get("<unknown>", 43803,)
			token_ids=[ self.vocab.get(g, pad_id) for g in top_gene_names ]
			meta_ids=[ self.vocab.get(t, pad_id) for t in ["<disease>", ] ]
			seq=meta_ids + token_ids
	
			seq = seq[:self.seq_len]
			_input_ids = np.zeros(self.seq_len, dtype=np.int32)
			_attention_mask = np.zeros(self.seq_len, dtype=np.bool_)
			_input_ids[:len(seq)] = np.array(seq, dtype=np.int32)
			_attention_mask[:len(seq)] = True

			input_ids[i]=_input_ids
			attention_mask[i]=_attention_mask

		data_pointer=h5py.File(self.dataset_dir+"/"+file_name+".h5", "w")
		data_pointer.create_dataset("data", (N, self.seq_len,), chunks=(1, self.seq_len,), dtype=np.int32, compression='gzip')
		data_pointer.create_dataset("attention_mask", (N, self.seq_len), chunks=(1, self.seq_len,), dtype=np.bool_, compression='gzip')
		data_pointer.create_dataset("targets", (N, 1), chunks=(1, 1,), dtype=np.int32, compression='gzip')
		
		for i in range(N):
			
			data_pointer["data"][i]=input_ids[i].astype(np.int32)
			data_pointer["attention_mask"][i]=attention_mask[i]
			data_pointer["targets"][i]=self.targets[i].astype(np.int32)

		data_pointer.close()
		print("H5 file created at "+self.dataset_dir+"/"+file_name+".h5")

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


class TEDDY(_BaseFoundationalModel):

	@staticmethod
	def build(n_classes, pretrained_dir):
		return TEDDY(n_classes=n_classes, model_type='70M', model_path=pretrained_dir+"/models/teddy_g/", rank=2, )

	def __init__(self, model_type, model_path, n_classes=1, rank=2, train_layers=[11]):
		super(TEDDY, self).__init__()

		model_name_or_path=model_path+model_type
		arch=get_architecture(model_name_or_path)
		config_cls=model_dict[arch]["config_cls"]
		model_cls=model_dict[arch]["model_cls"]
		
		config=config_cls.from_pretrained(model_name_or_path)
		self.model=model_cls.from_pretrained(model_name_or_path, config=config)

		self.model.encoder.layers[11]=torch.nn.TransformerEncoderLayer(d_model=self.model.encoder.layers[11].linear1.in_features, 
																						nhead=self.model.encoder.layers[11].self_attn.num_heads,
																						dim_feedforward=self.model.encoder.layers[11].linear1.out_features,
																						batch_first=True,
																						norm_first=self.model.encoder.layers[11].norm_first,)
		
		self.model.decoder_head=torch.nn.Linear(512, n_classes)

		self._initialize()

	@property
	def hidden_size(self,):
		return 512

	@property
	def _trainable_params(self):
		return [(n,p) for n,p in self.model.decoder_head.named_parameters()]+[(n,p) for n,p in self.model.encoder.layers[11].named_parameters()]
	
	@property
	def _device(self,):
		return self.model.decoder_head.weight.device

	def first_attention_scores(self, batch):

		input_ids, attention_mask=batch.input_ids.to(device=self._device), batch.attention_mask.to(device=self._device)
		
		embeddings=self.model.embeddings(input_ids)

		position_ids=torch.arange(0, input_ids.shape[1], device=self.model.position_embeddings.weight.device)
		position_embeddings=self.model.position_embeddings(position_ids)
		embeddings+=position_embeddings
		
		attention_mask=~attention_mask.bool()

		attention_mask=torch.nn.functional._canonical_mask(mask=attention_mask,mask_name="src_key_padding_mask",other_type=torch.nn.functional._none_or_dtype(None),other_name="src_mask", target_type=None)

		embeddings=self.model.encoder.layers[0].norm1(embeddings)
		
		return self.model.encoder.layers[0].self_attn(embeddings,embeddings,embeddings,attn_mask=None,key_padding_mask=attention_mask,need_weights=True,is_causal=False,)[1]

	def _explain_helper(self, embeddings, attention_mask):
		output=self.model.encoder(embeddings, src_key_padding_mask=attention_mask, is_causal=False)[:,0]
		
		return self.model.decoder_head(output)

	def explain_embeddings(self, batch, transition=-1, vocab=None, verbose=False):
		input_ids, attention_mask=batch.input_ids.to(device=self._device), batch.attention_mask.to(device=self._device)

		genes=np.zeros(input_ids.shape)
		if(vocab is not None):
			genes=np.array(vocab)[input_ids.cpu()[attention_mask.cpu()].numpy()]

		self.eval()

		embeddings=self.model.embeddings(input_ids)
		
		position_ids=torch.arange(0, input_ids.shape[1], device=self._device)
		position_embeddings=self.model.position_embeddings(position_ids)
		embeddings+=position_embeddings
		
		attention_mask=~attention_mask.bool()

		ig=IntegratedGradients(self._explain_helper)

		if(verbose):
			print("Explaining ...", end="\r")
		
		return ig.attribute(inputs=embeddings, baselines=torch.zeros_like(embeddings), additional_forward_args=(attention_mask,), target=transition, n_steps=2,), genes

	#@torch.compile
	def encode(self, batch):

		input_ids, attention_mask=batch.input_ids.to(device=self._device), batch.attention_mask.to(device=self._device)

		embeddings=self.model.embeddings(input_ids)
		
		position_ids=torch.arange(0, input_ids.shape[1], device=self._device)
		position_embeddings=self.model.position_embeddings(position_ids)
		embeddings+=position_embeddings
		
		attention_mask=~attention_mask.bool()

		embeddings=self.model.encoder(embeddings, src_key_padding_mask=attention_mask, is_causal=False)[:,0]

		return embeddings

	#@torch.compile
	def forward(self, batch):

		output=self.encode(batch)

		return self.model.decoder_head(output)

class DatasetTEDDY(torch.utils.data.Dataset):

	def __init__(self, indices, transitions=None, individuals=None, dataset_dir=None, file_name="data"):

		super(DatasetTEDDY, self).__init__()

		self.indices=indices
		self.dataset_dir=dataset_dir
		self.file_name=file_name

		dataset_name=dataset_dir.split("/")[-1].upper()
		dataset_class=getattr(datasets, dataset_name)

		self.transitions=transitions
		if(self.transitions is None):
			self.transitions=dataset_class.get_transitions(dataset_dir=dataset_dir)

		self.filled_transitions={}
		for patno in self.transitions:
			
			possible_transitions=[dataset_class.transition(patno, -1, -1, -1, dataset_class.state(i), dataset_class.state(i), -1) for i in dataset_class.state.values[:-1]]
			
			for t in self.transitions[patno]:
				if(t.orig>=len(dataset_class.state.values)-1):
					continue

				possible_transitions[t.orig]=t

			self.filled_transitions[patno]=possible_transitions

		self.individuals=individuals
		
	def __len__(self,): return len(self.indices)

	def __getitem__(self, idx):
		
		idx=self.indices[idx]
		
		if not hasattr(self, 'data_pointer'):
			self.data_pointer=h5py.File(self.dataset_dir+"/"+self.file_name+".h5", "r")

		patno=self.individuals[idx]

		return (torch.tensor(self.data_pointer["data"][idx]).long(), torch.tensor(0.), torch.tensor(0.), torch.tensor(self.data_pointer["attention_mask"][idx]), 
									torch.tensor(self.data_pointer["targets"][idx]).long()[0], 
									torch.tensor([t.duration for t in self.filled_transitions[patno]]).long(), 
									torch.tensor([t.event for t in self.filled_transitions[patno]]).bool())

class TEDDYModule(_ModelModule):

	cv_class=CV_TEDDY
	model=TEDDY
	dataset_class=DatasetTEDDY