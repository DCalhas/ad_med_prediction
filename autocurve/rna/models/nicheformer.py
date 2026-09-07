import os

import h5py

import numpy as np

import pandas as pd

import torch

from transformers import AutoTokenizer, AutoModelForMaskedLM

from sklearn.model_selection import KFold

import autocurve.rna.datasets as datasets

from autocurve.rna.models import _ModelModule, _BaseFoundationalModel, _CV_Model

from captum.attr import IntegratedGradients

__all__=["NicheformerModule"]

class CV_Nicheformer(_CV_Model):


	def __init__(self, n_splits=5, dataset_dir="/ceph/hpc/data/s25r10-02-users/calhasd/adni", pretrained_dir=None, state="ON"):
		"""
		A dataset_func is a fuction that receives gene/individuals matrix, genes, and individuals ids and return a Dataset instance
		
		"""

		self.topk_genes=1450
		self.seq_len=1500

		self.pretrained_dir=pretrained_dir
		
		self.dataset_dir=dataset_dir

		self.vocab=AutoTokenizer.from_pretrained(pretrained_dir, trust_remote_code=True)
		
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
		return list(self.vocab._vocabulary.keys())

	def create_h5(self, model=None, force=False, file_name="data"):
		
		if(os.path.isfile(self.dataset_dir+"/"+file_name+".h5")):
			print("I: Data pointers h5 file already exists. Toggle force to create a new one.", end="\n")
			if(not force): return

		N=self.individuals.shape[0]
			
		if self.vocab is None:
			self.vocab=AutoTokenizer.from_pretrained(self.pretrained_dir, trust_remote_code=True)


		input_ids=np.zeros((N, self.seq_len, ), dtype=np.int32)
		attention_mask=np.zeros((N, self.seq_len, ), dtype=np.bool_)
		
		for i in range(N):
			sorted_idx=np.argsort(-self.gene_ind_matrix[i])
	
			k=min(self.topk_genes, len(sorted_idx))
			
			top_indices=sorted_idx[:k]
			
			top_gene_names=np.array(self.gene_ids)[top_indices]
	
			pad_id=self.vocab.get_vocab().get("[PAD]", 0)
			token_ids=[ self.vocab.get_vocab().get(g, pad_id) for g in top_gene_names ]
			meta_ids=[ self.vocab.get_vocab().get(t, pad_id) for t in ["[CLS]"] ]
			seq=meta_ids + token_ids
	
			seq = seq[:self.seq_len]
			_input_ids = np.zeros(self.seq_len, dtype=np.int32)
			_attention_mask = np.zeros(self.seq_len, dtype=np.bool_)
			_input_ids[:len(seq)] = np.array(seq, dtype=np.int32)
			_attention_mask[:len(seq)] = True

			input_ids[i]=_input_ids
			attention_mask[i]=_attention_mask

		data_pointer=h5py.File(self.dataset_dir+"/"+file_name+".h5", "w")
		data_pointer.create_dataset("data", (N, 1500,), chunks=(1, 1500,), dtype=np.int32, compression='gzip')
		data_pointer.create_dataset("attention_mask", (N, 1500), chunks=(1, 1500,), dtype=np.bool_, compression='gzip')
		data_pointer.create_dataset("targets", (N, 1), chunks=(1, 1,), dtype=np.int32, compression='gzip')
		
		for i in range(N):
			
			data_pointer["data"][i]=input_ids[i].astype(np.float16)
			data_pointer["attention_mask"][i]=attention_mask[i]
			data_pointer["targets"][i]=self.targets[i].astype(np.int32)

		data_pointer.close()
		print("H5 file created at "+self.dataset_dir+"/"+file_name+".h5")


class Nicheformer(_BaseFoundationalModel):

	@staticmethod
	def build(n_classes, pretrained_dir):
		return Nicheformer('cell', pretrained_dir, n_classes=n_classes, rank=2, )

	def __init__(self, model_type, model_path, n_classes=1, rank=2, train_layers=[11]):
		super(Nicheformer, self).__init__()

		self.model=AutoModelForMaskedLM.from_pretrained(model_path, trust_remote_code=True)
		
		self.model.nicheformer.encoder.layers[11]=torch.nn.TransformerEncoderLayer(d_model=self.model.nicheformer.encoder.layers[11].linear1.in_features, 
																						nhead=self.model.nicheformer.encoder.layers[11].self_attn.num_heads,
																						dim_feedforward=self.model.nicheformer.encoder.layers[11].linear1.out_features,
																						batch_first=True,
																						norm_first=self.model.nicheformer.encoder.layers[11].norm_first,)
		
		self.model.nicheformer.classifier_head=torch.nn.Linear(512, n_classes)

		self._initialize()
		
	@property
	def hidden_size(self,):
		return 512

	@property
	def _trainable_params(self):
		return [(n,p) for n,p in self.model.nicheformer.classifier_head.named_parameters()]+[(n,p) for n,p in self.model.nicheformer.encoder.layers[11].named_parameters()]

	@property
	def _device(self,):
		return self.model.nicheformer.classifier_head.weight.device

	def first_attention_scores(self, batch):

		input_ids, attention_mask=batch.input_ids.to(device=self._device), batch.attention_mask.to(device=self._device)

		token_embedding=self.model.nicheformer.embeddings(input_ids)

		pos_embedding=self.model.nicheformer.positional_embedding(self.model.nicheformer.pos.to(token_embedding.device))
		embeddings=token_embedding + pos_embedding
		#embeddings=self.model.nicheformer.dropout(token_embedding + pos_embedding)
		
		attention_mask=~attention_mask.bool()

		attention_mask=torch.nn.functional._canonical_mask(mask=attention_mask,mask_name="src_key_padding_mask",other_type=torch.nn.functional._none_or_dtype(None),other_name="src_mask", target_type=None)

		embeddings=self.model.nicheformer.encoder.layers[0].norm1(embeddings)
		
		return self.model.nicheformer.encoder.layers[0].self_attn(embeddings,embeddings,embeddings,attn_mask=None,key_padding_mask=attention_mask,need_weights=True,is_causal=False,)[1]
	
	def _explain_helper(self, embeddings, attention_mask):
		output=self.model.nicheformer.encoder(embeddings, src_key_padding_mask=attention_mask, is_causal=False)[:,0]

		return self.model.nicheformer.classifier_head(output)

	def explain_embeddings(self, batch, transition=-1, vocab=None, verbose=False):
		input_ids, attention_mask=batch.input_ids.to(device=self._device), batch.attention_mask.to(device=self._device)

		genes=np.zeros(input_ids.shape)
		if(vocab is not None):
			genes=np.array(vocab)[input_ids.cpu().numpy()[attention_mask.cpu().numpy()]]

		self.eval()

		token_embedding=self.model.nicheformer.embeddings(input_ids)

		pos_embedding=self.model.nicheformer.positional_embedding(self.model.nicheformer.pos.to(token_embedding.device))
		embeddings=token_embedding + pos_embedding
		#embeddings=self.model.nicheformer.dropout(token_embedding + pos_embedding)
		
		attention_mask=~attention_mask.bool()

		attention_mask=torch.nn.functional._canonical_mask(mask=attention_mask,mask_name="src_key_padding_mask",other_type=torch.nn.functional._none_or_dtype(None),other_name="src_mask", target_type=None)

		ig=IntegratedGradients(self._explain_helper)

		if(verbose):
			print("Explaining ...", end="\r")
		
		return ig.attribute(inputs=embeddings, baselines=torch.zeros_like(embeddings), additional_forward_args=(attention_mask,), target=transition, n_steps=1,), genes

	@torch.compile
	def encode(self, batch):

		input_ids, attention_mask=batch.input_ids.to(device=self._device), batch.attention_mask.to(device=self._device)

		token_embedding=self.model.nicheformer.embeddings(input_ids)

		pos_embedding=self.model.nicheformer.positional_embedding(self.model.nicheformer.pos.to(token_embedding.device))
		embeddings=token_embedding + pos_embedding
		#embeddings=self.model.nicheformer.dropout(token_embedding + pos_embedding)
		
		attention_mask=~attention_mask.bool()

		output=self.model.nicheformer.encoder(embeddings, src_key_padding_mask=attention_mask, is_causal=False)
		
		return output[:,0]

	@torch.compile
	def forward(self, batch):

		output=self.encode(batch)

		return self.model.nicheformer.classifier_head(output)

class DatasetNicheformer(torch.utils.data.Dataset):

	def __init__(self, indices, dataset_dir=None, file_name="data", individuals=None, transitions=None):

		super(DatasetNicheformer, self).__init__()

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

class NicheformerModule(_ModelModule):

	cv_class=CV_Nicheformer
	model=Nicheformer
	dataset_class=DatasetNicheformer