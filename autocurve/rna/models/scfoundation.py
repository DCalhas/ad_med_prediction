import os

import h5py

import numpy as np

import pandas as pd

import torch

from sklearn.model_selection import KFold

import autocurve.rna.datasets as datasets

import scfoundation

from scfoundation.load import getEncoerDecoderData, gatherData

from autocurve.rna.models import _ModelModule, _BaseFoundationalModel, _CV_Model

from captum.attr import IntegratedGradients

__all__=["scFoundationModule"]

class CV_scFoundation(_CV_Model):

	def __init__(self, n_splits=5, dataset_dir="/ceph/hpc/data/s25r10-02-users/calhasd/adni", pretrained_dir=None, state="ON"):
		"""
		A dataset_func is a fuction that receives gene/individuals matrix, genes, and individuals ids and return a Dataset instance
		
		"""

		self.n_splits=n_splits

		self.pretrained_dir=pretrained_dir
		
		self.dataset_dir=dataset_dir

		self.pretrainedconfig=scfoundation.load.load_model_frommmf(pretrained_dir+"/models/models.ckpt", key='cell')[1]

		self.vocab=pd.read_csv(pretrained_dir+"/OS_scRNA_gene_index.19264.tsv", sep='\t')

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
		return list(self.vocab['gene_name'])

	def create_h5(self, model=None, force=False, file_name="data"):
		if(os.path.isfile(self.dataset_dir+"/"+file_name+".h5")):
			print("I: Data pointers h5 file already exists. Toggle force to create a new one.", end="\n")
			if(not force): return
			
		if self.vocab is None:
			self.vocab=pd.read_csv(self.pretrained_dir + "/OS_scRNA_gene_index.19264.tsv",sep="\t")
			
		vocab_genes=self.vocab["gene_name"].astype(str).values
		
		N=self.gene_ind_matrix.shape[0]
		M=vocab_genes.shape[0]
		assert M == 19264, "Unexpected vocab size"
	
		name_to_idx={g: i for i, g in enumerate(self.gene_names)}
	
		map_idx=np.array([name_to_idx.get(g, -1) for g in vocab_genes])
		
		values=np.zeros((N, M + 2), dtype=np.float32)
		gene_ids=np.zeros((N, M + 2), dtype=np.int32)
		base_gene_ids=np.arange(M + 2)
	
		for i in range(N):
			v = np.zeros(M, dtype=np.float32)
			present = map_idx >= 0
			if present.sum() > 0:
				v[present] = self.gene_ind_matrix[i, map_idx[present]]
	
			totalcount = np.log10(v.sum() + 1e-8)
	
			values[i, :M]=v
			values[i, M]=totalcount
			values[i, M+1]=totalcount
			gene_ids[i]=base_gene_ids
			
		encoder_data_padding=None

		data_pointer=h5py.File(self.dataset_dir+"/"+file_name+".h5", "w")
		data_pointer.create_dataset("data", (N, 19266,), chunks=(1, 19266,), dtype=np.float16, compression='gzip')
		n_tokens=16791 if "adni" in self.dataset_dir else 18753#problem with ppmi!
		data_pointer.create_dataset("latent", (N, n_tokens, 768), chunks=(1, n_tokens, 768,), dtype=np.float16, compression='gzip')
		data_pointer.create_dataset("pad", (N, n_tokens), chunks=(1, n_tokens,), dtype=np.bool_, compression='gzip')
		data_pointer.create_dataset("targets", (N, 1), chunks=(1, 1,), dtype=np.int32, compression='gzip')
		
		loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.tensor(values,), torch.tensor(self.targets)), batch_size=1, shuffle=False, num_workers=2, prefetch_factor=4)
		
		model.eval()
		model.to(device="cuda:0")
		for i, batch in enumerate(loader):
			with torch.autocast(device_type='cuda', dtype=torch.float16):
				input_ids=batch[0].to(device="cuda:0")
				output=model.nograd_encode(input_ids,)
			
				print("Mapping embeddings", i, end="\r")
				
				data_pointer["data"][i]=batch[0].numpy().astype(np.float16)
				data_pointer["latent"][i]=output[0].cpu().numpy().astype(np.float16)
				data_pointer["pad"][i]=output[1].cpu().numpy()
				data_pointer["targets"][i]=batch[1].numpy().astype(np.int32)
				
			print("Mapping embeddings", i, end="\r")

		data_pointer.close()
		print("H5 file created at "+self.dataset_dir+"/"+file_name+".h5")


class scFoundation(_BaseFoundationalModel):

	@staticmethod
	def build(n_classes, pretrained_dir):
		return scFoundation(model_type='cell', model_path=pretrained_dir+"/models/models.ckpt", n_classes=n_classes, rank=2, )

	def __init__(self, model_type, model_path, n_classes=1, rank=2, train_layers=[11]):
		super(scFoundation, self).__init__()

		model=scfoundation.load.load_model_frommmf(model_path, key=model_type)
		self.model=model[0]
		self.pretrainconfig=model[1]

		self.model.encoder.transformer_encoder[11]=torch.nn.TransformerEncoderLayer(d_model=self.model.encoder.transformer_encoder[11].linear1.in_features, 
																						nhead=self.model.encoder.transformer_encoder[11].self_attn.num_heads,
																						dim_feedforward=self.model.encoder.transformer_encoder[11].linear1.out_features,
																						batch_first=True,
																						norm_first=self.model.encoder.transformer_encoder[11].norm_first,)

		
		self.model.to_final=torch.nn.Sequential(torch.nn.Linear(768*4, n_classes))

		self._initialize()

	@property
	def hidden_size(self,):
		return 768*4

	@property
	def _trainable_params(self):
		return [(n,p) for n,p in self.model.to_final.named_parameters()]+[(n,p) for n,p in self.model.encoder.transformer_encoder[11].named_parameters()]

	@property
	def _device(self,):
		return self.model.to_final[0].weight.device

	@torch.compile
	def first_attention_scores(self, batch):
		x=batch.input_ids.to(device=self._device)

		with torch.no_grad():
			with torch.cuda.amp.autocast(dtype=torch.float16):
				encoder_data, encoder_position_gene_ids, encoder_data_padding, \
				encoder_labels, decoder_data, decoder_data_padding, \
				new_data_raw, data_mask_labels, decoder_position_gene_ids = getEncoerDecoderData(x, x, self.pretrainconfig)
				
				x=self.model.token_emb(encoder_data.unsqueeze(2).float(), output_weight=0) + self.model.pos_emb(encoder_position_gene_ids)
				
				encoder_data_padding=torch.nn.functional._canonical_mask(mask=encoder_data_padding,mask_name="src_key_padding_mask",other_type=torch.nn.functional._none_or_dtype(None),other_name="src_mask", target_type=None)
		
				x=self.model.encoder.transformer_encoder[0].norm1(x)
				
				return self.model.encoder.transformer_encoder[0].self_attn(x,x,x,attn_mask=None,key_padding_mask=encoder_data_padding,need_weights=True,is_causal=False,)[1]

	def _explain_helper(self, embeddings, attention_mask=None):

		for layer in self.model.encoder.transformer_encoder:
			embeddings=layer(embeddings, src_key_padding_mask=attention_mask)

		geneemb=self.model.encoder.norm(embeddings,)
			
		geneemb1 = geneemb[:, -1, :]		  # second last position
		geneemb2 = geneemb[:, -2, :]		  # third last
		geneemb3, _ = torch.max(geneemb[:, :-2, :], dim=1)   # max pool
		geneemb4 = torch.mean(geneemb[:, :-2, :], dim=1)	 # mean pool

		output=torch.concat([geneemb1, geneemb2, geneemb3, geneemb4], 1)

		return self.model.to_final(output)

	def explain_embeddings(self, batch, transition=-1, vocab=None, verbose=False):
		input_ids=batch.input_ids.to(device=self._device)

		genes=np.zeros(input_ids.shape)
		if(vocab is not None):
			genes=np.array(vocab+["helper1", "helper2"])

		self.eval()

		with torch.cuda.amp.autocast(dtype=torch.float16):
			

			encoder_data, encoder_position_gene_ids, encoder_data_padding, \
			encoder_labels, decoder_data, decoder_data_padding, \
			new_data_raw, data_mask_labels, decoder_position_gene_ids = getEncoerDecoderData(input_ids, input_ids, self.pretrainconfig)
			
			x=self.model.token_emb(encoder_data.unsqueeze(2).float(), output_weight=0) + self.model.pos_emb(encoder_position_gene_ids)

			ig=IntegratedGradients(self._explain_helper)

			if(verbose):
				print("Explaining ...", end="\r")
			
			return ig.attribute(inputs=x, baselines=torch.zeros_like(x), additional_forward_args=(encoder_data_padding,), target=transition, n_steps=6,), genes#6 steps occupies 30G in GPU

	@torch.compile
	def nograd_encode(self, x):

		with torch.no_grad():
			with torch.cuda.amp.autocast(dtype=torch.float16):
				encoder_data, encoder_position_gene_ids, encoder_data_padding, \
				encoder_labels, decoder_data, decoder_data_padding, \
				new_data_raw, data_mask_labels, decoder_position_gene_ids = getEncoerDecoderData(x, x, self.pretrainconfig)
				
				x=self.model.token_emb(encoder_data.unsqueeze(2).float(), output_weight=0) + self.model.pos_emb(encoder_position_gene_ids)
				
				for layer in self.model.encoder.transformer_encoder[:11]:
					x=layer(x, src_key_padding_mask=encoder_data_padding)
	
				return x, encoder_data_padding
			
	
	@torch.compile
	def encode(self, batch):

		if(batch.latent is None):
			x=batch.input_ids.to(device=self._device)
			x, encoder_data_padding=self.nograd_encode(x)
		else:
			x, encoder_data_padding=batch.latent.to(device=self._device), batch.attention_mask.to(device=self._device)

		with torch.cuda.amp.autocast(dtype=torch.float16):
			geneemb=self.model.encoder.transformer_encoder[11](x, src_key_padding_mask=encoder_data_padding)
			geneemb=self.model.encoder.norm(geneemb,)
			
			geneemb1 = geneemb[:, -1, :]		  # second last position
			geneemb2 = geneemb[:, -2, :]		  # third last
			geneemb3, _ = torch.max(geneemb[:, :-2, :], dim=1)   # max pool
			geneemb4 = torch.mean(geneemb[:, :-2, :], dim=1)	 # mean pool
			
			return torch.concat([geneemb1, geneemb2, geneemb3, geneemb4], 1)

	@torch.compile
	def forward(self, batch,):

		with torch.cuda.amp.autocast(dtype=torch.float16):
	
			geneembmerge=self.encode(batch)
	
			return self.model.to_final(geneembmerge)


class DatasetscFoundation(torch.utils.data.Dataset):

	def __init__(self, indices, dataset_dir=None, file_name="data", individuals=None, transitions=None):

		super(DatasetscFoundation, self).__init__()

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

		return (torch.tensor(self.data_pointer["data"][idx]), torch.tensor(self.data_pointer["latent"][idx]), torch.tensor(0.), 
								torch.tensor(self.data_pointer["pad"][idx]), 
								torch.tensor(self.data_pointer["targets"][idx]).long()[0], 
								torch.tensor([t.duration for t in self.filled_transitions[patno]]).long(), 
								torch.tensor([t.event for t in self.filled_transitions[patno]]).bool())
		
class scFoundationModule(_ModelModule):

	cv_class=CV_scFoundation
	model=scFoundation
	dataset_class=DatasetscFoundation