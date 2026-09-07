import os

import h5py

import numpy as np

import pandas as pd

import pickle

import torch

from transformers.models.bert.configuration_bert import BertConfig

from transformers.models.bert import BertOnlyMLMHead

from geneformer.emb_extractor import pu

from geneformer import TOKEN_DICTIONARY_FILE

from geneformer.emb_extractor import pu

from peft import LoraConfig, get_peft_model

from sklearn.model_selection import KFold

import autocurve.rna.datasets as datasets

from datasets import load_from_disk

from autocurve.rna.models import _ModelModule, _BaseFoundationalModel, _CV_Model

from captum.attr import IntegratedGradients

__all__=["GeneformerModule",]

class CV_Geneformer(_CV_Model):

	def __init__(self, n_splits=5, dataset_dir="/ceph/hpc/data/s25r10-02-users/calhasd/adni", pretrained_dir=None, state="ON"):
		"""
		A dataset_func is a fuction that receives gene/individuals matrix, genes, and individuals ids and return a Dataset instance
		
		"""

		self.topk_genes=2000
		self.seq_len=1024

		self.n_splits=n_splits

		self.pretrained_dir=pretrained_dir
		
		self.dataset_dir=dataset_dir

		with open(TOKEN_DICTIONARY_FILE, "rb") as f: self.vocab=pickle.load(f)

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

		if(os.path.isfile(self.dataset_dir+"/"+file_name+".h5")):
			print("I: Data pointers h5 file already exists. Toggle force to create a new one.", end="\n")
			if(not force): return

		N=self.individuals.shape[0]
			
		if self.vocab is None:
			with open(TOKEN_DICTIONARY_FILE, "rb") as f: self.vocab=pickle.load(f)


		input_ids=np.zeros((N, self.seq_len, ), dtype=np.int32)
		attention_mask=np.zeros((N, self.seq_len, ), dtype=np.bool_)
		
		for i in range(N):
			sorted_idx=np.argsort(-self.gene_ind_matrix[i])
	
			k=min(self.topk_genes, len(sorted_idx))
			
			top_indices=sorted_idx[:k]
			
			top_gene_names=np.array(self.gene_ids)[top_indices]
	
			pad_id=self.vocab.get("[PAD]", 0)
			token_ids=[ self.vocab.get(g, pad_id) for g in top_gene_names ]
			meta_ids=[ self.vocab.get(t, pad_id) for t in ["[CLS]"] ]
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
			
			data_pointer["data"][i]=input_ids[i].astype(np.float16)
			data_pointer["attention_mask"][i]=attention_mask[i]
			data_pointer["targets"][i]=self.targets[i].astype(np.int32)

		data_pointer.close()
		print("H5 file created at "+self.dataset_dir+"/"+file_name+".h5")


class Geneformer(_BaseFoundationalModel):


	@staticmethod
	def build(n_classes, pretrained_dir):
		return Geneformer("Pretrained", pretrained_dir+"/fine_tuned_models/Geneformer-V2-104M", rank=2, n_classes=n_classes)

	def __init__(self, model_type, model_path, n_classes=1, rank=2, train_layers=[11]):
		super(Geneformer, self).__init__()
		
		self.BERT=pu.load_model(model_type, None, model_path, mode="train")

		config=BertConfig(hidden_size=self.BERT.cls.predictions.transform.dense.out_features, vocab_size=n_classes, hidden_act="relu", is_decoder=True)
		
		self.BERT.cls=BertOnlyMLMHead(config, )

		self._initialize()
	
	@property
	def hidden_size(self,):
		return self.BERT.cls.predictions.transform.dense.out_features
	
	@property
	def _trainable_params(self):
		return [(n,p) for n,p in self.BERT.cls.named_parameters()]+[(n,p) for n,p in self.BERT.bert.encoder.layer[11].named_parameters()]

	@property
	def _device(self,):
		return self.BERT.cls.predictions.decoder.bias.device

	@property
	def config(self,): return self.BERT.config

	def first_attention_scores(self, batch):

		input_ids, attention_mask=batch.input_ids.to(device=self._device), batch.attention_mask.to(device=self._device)

		embedding_output=self.BERT.bert.embeddings(input_ids=input_ids, position_ids=None, token_type_ids=None, inputs_embeds=None, past_key_values_length=0,)

		attention_weights=self.BERT.bert.encoder.layer[0].attention(embedding_output, attention_mask, past_key_values=None, cache_position=None, output_attentions=True)[1]
		
		return attention_weights.mean(dim=1)

	def _explain_helper(self, embeddings, attention_mask=None):

		output=self.BERT.bert.encoder(embeddings, attention_mask)[0][:,0]

		return self.BERT.cls(output)

	def explain_embeddings(self, batch, transition=-1, vocab=None, verbose=False):
		input_ids, attention_mask=batch.input_ids.to(device=self._device), batch.attention_mask.to(device=self._device)

		genes=np.zeros(input_ids.shape)
		if(vocab is not None):
			genes=np.array(vocab)[input_ids.cpu().numpy()]

		self.eval()

		embedding_output=self.BERT.bert.embeddings(input_ids=input_ids, position_ids=None, token_type_ids=None, inputs_embeds=None, past_key_values_length=0,)

		ig=IntegratedGradients(self._explain_helper)

		attention_mask=attention_mask.float()

		attention_mask=self.BERT.bert.get_extended_attention_mask(attention_mask, input_ids.shape)

		if(verbose):
			print("Explaining ...", end="\r")
		
		return ig.attribute(inputs=embedding_output, baselines=torch.zeros_like(embedding_output), additional_forward_args=(attention_mask,), target=transition, n_steps=5,), genes

	def encode(self, batch):

		input_ids, attention_mask=batch.input_ids.to(device=self._device), batch.attention_mask.to(device=self._device)
		
		output=self.BERT.bert(input_ids,attention_mask=attention_mask,)
		
		return output[0][:,0]

	def forward(self, batch,):
		
		output=self.encode(batch)
		
		return self.BERT.cls(output)


class DatasetGeneformer(torch.utils.data.Dataset):

	def __init__(self, indices, dataset_dir=None, file_name="data", individuals=None, transitions=None):

		super(DatasetGeneformer, self).__init__()

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


class GeneformerModule(_ModelModule):

	cv_class=CV_Geneformer
	model=Geneformer
	dataset_class=DatasetGeneformer