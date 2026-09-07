import os

import h5py

import numpy as np

import pandas as pd

import pickle

import json

import torch

import scgpt

import inspect

from scgpt.utils import set_seed, eval_scib_metrics, load_pretrained 

from scgpt.tokenizer.gene_tokenizer import GeneVocab, tokenize_and_pad_batch

from sklearn.model_selection import KFold

import autocurve.rna.datasets as datasets

from autocurve.rna.models import _ModelModule, _BaseFoundationalModel, _CV_Model

from captum.attr import IntegratedGradients

__all__=["scGPTModule"]


class CV_scGPT(_CV_Model):


	def __init__(self, n_splits=5, dataset_dir="/ceph/hpc/data/s25r10-02-users/calhasd/adni", pretrained_dir=None, state="ON"):
		"""
		A dataset_func is a fuction that receives gene/individuals matrix, genes, and individuals ids and return a Dataset instance
		
		"""

		self.topk_genes=2000
		self.seq_len=1201

		self.pretrained_dir=pretrained_dir
		self.dataset_dir=dataset_dir

		self.vocab=GeneVocab.from_file(self.pretrained_dir+"/pretrained/vocab.json")

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
		return list(self.vocab.get_stoi().keys())

	def create_h5(self, model=None, force=False, file_name="data"):

		if(os.path.isfile(self.dataset_dir+"/"+file_name+".h5")):
			print("I: Data pointers h5 file already exists. Toggle force to create a new one.", end="\n")
			if(not force): return

		N=self.individuals.shape[0]
			
		if self.vocab is None:
			self.vocab=GeneVocab.from_file(self.pretrained_dir+"/pretrained/vocab.json")

		vocab=self.vocab

		mask_value=-1
		pad_value=-2
		pad_token="<pad>"
		max_seq_len=self.seq_len
		mask_ratio=[0.25, 0.5, 0.75]

		special_tokens=["<pad>", "<cls>"]
		for s in special_tokens:
			if s not in vocab:
				vocab.append_token(s)
		vocab.set_default_index(vocab["<pad>"])

		input_ids=np.zeros((N, self.seq_len, ), dtype=np.int32)
		values=np.zeros((N, self.seq_len, ), dtype=np.float32)
		attention_mask=np.zeros((N, self.seq_len, ), dtype=np.bool_)

		gene_ids_in_vocab=np.array([i if g in vocab else -1 for i, g in enumerate(self.gene_names)])
		keep_mask=gene_ids_in_vocab >= 0
		array=self.gene_ind_matrix[:, keep_mask]
		
		genes_filtered=[g for g, keep in zip(self.gene_names, keep_mask) if keep]
		_gene_names=np.array([vocab[g] for g in genes_filtered], dtype=int)

		data_pointer=h5py.File(self.dataset_dir+"/"+file_name+".h5", "w")
		data_pointer.create_dataset("data", (N, self.seq_len,), chunks=(1, self.seq_len,), dtype=np.int32, compression='gzip')
		data_pointer.create_dataset("values", (N, self.seq_len), chunks=(1, self.seq_len,), dtype=np.float32, compression='gzip')
		data_pointer.create_dataset("targets", (N, 1), chunks=(1, 1,), dtype=np.int32, compression='gzip')

		tokenized_data=tokenize_and_pad_batch(array,_gene_names,max_len=max_seq_len,vocab=vocab,pad_token=pad_token,pad_value=pad_value,append_cls=True,include_zero_gene=True)
		
		for i in range(N):
			data_pointer["data"][i]=tokenized_data["genes"][i].numpy().astype(np.int32)
			data_pointer["values"][i]=tokenized_data["values"][i].numpy().astype(np.float32)
			data_pointer["targets"][i]=self.targets[i].astype(np.int32)

		data_pointer.close()
		print("H5 file created at "+self.dataset_dir+"/"+file_name+".h5")


class scGPT(_BaseFoundationalModel):

	@staticmethod
	def build(n_classes, pretrained_dir):
		return scGPT(None, pretrained_dir+"/pretrained/", n_classes=n_classes, rank=2, )

	def __init__(self, model_type, model_path, n_classes=1, rank=2, train_layers=[11]):
		super(scGPT, self).__init__()
		
		vocab=GeneVocab.from_file(model_path+"vocab.json")
		special_tokens=["<pad>", "<cls>"]
		for s in special_tokens:
			if s not in vocab:
				vocab.append_token(s)
		vocab.set_default_index(vocab["<pad>"])

		with open(model_path+"args.json", "r") as f:
			args=json.load(f)

		#load model
		model_params=inspect.signature(scgpt.model.TransformerModel.__init__).parameters
		model_arg_names=set(model_params.keys()) - {"self"}
		model_args={k: v for k, v in args.items() if k in model_arg_names}
		model_args['ntoken']=len(vocab)
		model_args['d_model']=args['embsize']
		model_args['nhead']=4
		model_args['vocab']=vocab
		model_args['use_fast_transformer']=True
		model=scgpt.model.TransformerModel(**model_args)
		self.model=load_pretrained(model, torch.load(model_path+"/best_model.pt"), verbose=False)

		#low rank adaptation for scGPT
		#train_layers=[11]
		#lora_config=LoraConfig(r=2,lora_alpha=16, target_modules=
		#				["transformer_encoder.layers."+str(l)+".self_attn.out_proj" for l in train_layers]+ \
		#				["transformer_encoder.layers."+str(l)+".linear1" for l in train_layers]+ \
		#				["transformer_encoder.layers."+str(l)+".linear2" for l in train_layers] ,lora_dropout=0.05,bias="none",task_type="SEQ_CLS")
		#self.model=get_peft_model(self.model, lora_config)

		#create new cls layer
		self.model.cls_decoder=scgpt.model.model.ClsDecoder(d_model=512, n_cls=n_classes,)

		self._initialize()

	@property
	def hidden_size(self,):
		return 512

	@property
	def _trainable_params(self):
		return [(n,p) for n,p in self.model.cls_decoder.named_parameters()]+[(n,p) for n,p in self.model.transformer_encoder.layers[11].named_parameters()]

	@property
	def _device(self,):
		return self.model.cls_decoder.out_layer.weight.device

	def _explain_helper(self, embeddings,):
		output=self.model.transformer_encoder(embeddings, None)[:,0,:]
		
		return self.model.cls_decoder(output)

	def explain_embeddings(self, batch, transition=-1, vocab=None, verbose=False):
		src, values=batch.input_ids.to(device=self._device), batch.values.to(device=self._device)

		genes=np.zeros(src.shape)
		if(vocab is not None):
			genes=np.array(vocab)[src.cpu().numpy()]

		self.eval()

		src = self.model.encoder(src)
		self.model.cur_gene_token_embs = src

		values = self.model.value_encoder(values)

		if self.model.input_emb_style == "scaling":
			values = values.unsqueeze(2)
			total_embs = src * values
		else:
			total_embs = src + values

		if getattr(self.model, "bn", None) is not None:
			total_embs = self.model.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

		ig=IntegratedGradients(self._explain_helper)

		if(verbose):
			print("Explaining ...", end="\r")
		
		return ig.attribute(inputs=total_embs, baselines=torch.zeros_like(total_embs), target=transition, n_steps=20,), genes

	def first_attention_scores(self, batch):

		src, values=batch.input_ids.to(device=self._device), batch.values.to(device=self._device)

		src = self.model.encoder(src)
		self.model.cur_gene_token_embs = src

		values = self.model.value_encoder(values)

		if self.model.input_emb_style == "scaling":
			values = values.unsqueeze(2)
			total_embs = src * values
		else:
			total_embs = src + values

		if getattr(self.model, "bn", None) is not None:
			total_embs = self.model.bn(total_embs.permute(0, 2, 1)).permute(0, 2, 1)

		total_embs=self.model.transformer_encoder.layers[0].norm1(total_embs)
		
		return self.model.transformer_encoder.layers[0].self_attn(total_embs, total_embs, total_embs,attn_mask=None,key_padding_mask=None,need_weights=True,is_causal=False,)[1]
	
	def encode(self, batch):

		src, values=batch.input_ids.to(device=self._device), batch.values.to(device=self._device)

		return self.model._encode(src, values, None)[:,0,:]

	def forward(self, batch):

		output=self.encode(batch)

		return self.model.cls_decoder(output)



class DatasetscGPT(torch.utils.data.Dataset):

	def __init__(self, indices, dataset_dir=None, file_name="data", individuals=None, transitions=None):

		super(DatasetscGPT, self).__init__()

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

		return (torch.tensor(self.data_pointer["data"][idx]).long(), torch.tensor(0.), torch.tensor(self.data_pointer["values"][idx]), torch.tensor(0.), 
						torch.tensor(self.data_pointer["targets"][idx]).long()[0], 
						torch.tensor([t.duration for t in self.filled_transitions[patno]]).long(), 
						torch.tensor([t.event for t in self.filled_transitions[patno]]).bool())

class scGPTModule(_ModelModule):

	cv_class=CV_scGPT
	model=scGPT
	dataset_class=DatasetscGPT