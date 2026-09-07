import torch

import numpy as np

import pandas as pd

from scipy._lib._bunch import _make_tuple_bunch

#need to update this to the correct one for clinical. this comes from the rna that was done first in the project
BatchRepresentation=_make_tuple_bunch('BatchRepresentation', ['input_ids', 'latent', 'values', 'attention_mask', 'targets', 'durations', 'censoring'],)	

from sklearn.model_selection import train_test_split, KFold

from sklearn.preprocessing import StandardScaler

from autocurve.rna.models.geneformer import GeneformerModule

import random

from geneformer import TOKEN_DICTIONARY_FILE

import pickle

def get_metabrics(dataset_dir="/shared/workspace/lui/AUTH_PERSONEL_ONLY_diferre/brca_metabric/"):
	
	clin = pd.read_csv(dataset_dir+"data_clinical_patient.txt", sep="\t", comment="#", dtype=str)
	clin["OS_MONTHS"] = pd.to_numeric(clin["OS_MONTHS"], errors="coerce")
	clin["event"] = clin["OS_STATUS"].fillna("").str.contains(r"DECEASED|^1$", case=False, regex=True).astype(int)
	clin = clin.dropna(subset=["PATIENT_ID", "OS_MONTHS"]).copy()
	gene = "ESR1"
	expr = pd.read_csv(dataset_dir+"data_mrna_illumina_microarray.txt", sep="\t", comment="#", dtype=str)
	samp = pd.read_csv(dataset_dir+"data_clinical_sample.txt", sep="\t", comment="#", dtype=str)
	gene_col = "Hugo_Symbol" if "Hugo_Symbol" in expr.columns else expr.columns[0]
	gene_row = expr.loc[expr[gene_col].str.upper() == gene.upper()]
	if gene_row.empty:
		raise ValueError(f"{gene} not found in expression file")
	gene_row = gene_row.iloc[0]
	non_sample_cols = [c for c in ["Hugo_Symbol", "Entrez_Gene_Id"] if c in gene_row.index]
	sample_expr = gene_row.drop(labels=non_sample_cols).rename("expr").to_frame()
	sample_expr.index.name = "SAMPLE_ID"
	sample_expr = sample_expr.reset_index()
	sample_expr["expr"] = pd.to_numeric(sample_expr["expr"], errors="coerce")
	sample_map = samp[["SAMPLE_ID", "PATIENT_ID"]].dropna().drop_duplicates()
	df = (sample_expr.merge(sample_map, on="SAMPLE_ID", how="inner").merge(clin[["PATIENT_ID", "OS_MONTHS", "event"]], on="PATIENT_ID", how="inner").dropna(subset=["expr", "OS_MONTHS"]).copy())
	cut = df["expr"].median()
	df["group"] = (df["expr"] >= cut).map({True: f"{gene}_high", False: f"{gene}_low"})

	expr = pd.read_csv(dataset_dir+"data_mrna_illumina_microarray.txt", sep="\t", comment="#")
	meta_cols = [c for c in ["Hugo_Symbol", "Entrez_Gene_Id"] if c in expr.columns]
	sample_cols = [c for c in expr.columns if c not in meta_cols]
	X = expr[sample_cols].apply(pd.to_numeric, errors="coerce")
	gene_var = X.var(axis=1)
	keep = gene_var > 0.5
	expr_filtered = expr.reset_index(drop=True)#expr.loc[keep].reset_index(drop=True)
	
	individuals=expr_filtered.columns[2:].to_numpy()

	gene_ind_matrix=expr_filtered[individuals].to_numpy()

	genes_names=expr_filtered['Hugo_Symbol'].to_numpy()
	
	genes_ids=np.load(dataset_dir+"genes_ids.npy", allow_pickle=True)
	
	valid_genes=np.where(genes_ids!=None)
	
	#filter genes
	genes_names=genes_names[valid_genes]
	genes_ids=genes_ids[valid_genes]
	gene_ind_matrix=gene_ind_matrix[valid_genes]
	#drop with nans
	valid_genes=~np.any(np.isnan(gene_ind_matrix), axis=1)
	genes_names=genes_names[valid_genes]
	genes_ids=genes_ids[valid_genes]
	gene_ind_matrix=gene_ind_matrix[valid_genes].T#individuals X genes matrix
	
	return gene_ind_matrix, genes_ids, genes_names, individuals, None



class DatasetMETABRIC():
	
	
	def __init__(self, indices, dataset_dir="/shared/workspace/lui/AUTH_PERSONEL_ONLY_diferre/brca_metabric/", scaler=None, **kwargs):
		
		if("gene_ind_matrix" in kwargs):
			self.gene_ind_matrix, self.genes_ids, self.genes_names, self.individuals, self.targets=kwargs['gene_ind_matrix'], kwargs['gene_ids'], kwargs['gene_names'], kwargs['individuals'], kwargs['targets']
		else:
			self.gene_ind_matrix, self.genes_ids, self.genes_names, self.individuals, self.targets=get_metabrics(dataset_dir=dataset_dir)
		
		if(indices is not None):
			self.gene_ind_matrix, self.individuals=self.gene_ind_matrix[indices], self.individuals[indices]
				
		clin = pd.read_csv(dataset_dir+"data_clinical_patient.txt", sep="\t", comment="#", dtype=str)
		clin["OS_MONTHS"] = pd.to_numeric(clin["OS_MONTHS"], errors="coerce")
		clin["event"] = clin["OS_STATUS"].fillna("").str.contains(r"DECEASED|^1$", case=False, regex=True).astype(int)
		clin = clin.dropna(subset=["PATIENT_ID", "OS_MONTHS"]).copy()
		clin=clin[np.isin(clin['PATIENT_ID'].to_numpy(), self.individuals)]
		
		self.events=clin['event'].to_numpy().astype(np.bool_)
		self.times=clin['OS_MONTHS'].to_numpy().astype(np.float32)
		
		self.scaler=scaler
		
		with open(TOKEN_DICTIONARY_FILE, "rb") as f: self.vocab=pickle.load(f)
		topk_genes=2000
		seq_len=1024
		N=self.gene_ind_matrix.shape[0]

		self.input_ids=np.zeros((N, seq_len, ), dtype=np.int32)
		self.attention_mask=np.zeros((N, seq_len, ), dtype=np.bool_)

		N=self.individuals.shape[0]

		for i in range(N):
			sorted_idx=np.argsort(-self.gene_ind_matrix[i])
			k=min(topk_genes, len(sorted_idx))
			top_indices=sorted_idx[:k]
			top_gene_names=np.array(self.genes_ids)[top_indices]
			pad_id=self.vocab.get("[PAD]", 0)
			token_ids=[ self.vocab.get(g, pad_id) for g in top_gene_names ]
			meta_ids=[ self.vocab.get(t, pad_id) for t in ["[CLS]"] ]
			seq=meta_ids + token_ids
			seq = seq[:seq_len]
			_input_ids = np.zeros(seq_len, dtype=np.int32)
			_attention_mask = np.zeros(seq_len, dtype=np.bool_)
			_input_ids[:len(seq)] = np.array(seq, dtype=np.int32)
			_attention_mask[:len(seq)] = True
			
			self.input_ids[i]=_input_ids
			self.attention_mask[i]=_attention_mask
		
	def __len__(self,):
		
		return self.gene_ind_matrix.shape[0]
	
	def __getitem__(self, idx):
		
		return BatchRepresentation(torch.tensor(self.input_ids[idx]).long(), torch.tensor(0), torch.tensor(self.scaler.transform(self.gene_ind_matrix[idx].reshape(1,-1))).view(-1).float(), torch.tensor(self.attention_mask[idx]), torch.tensor(0), torch.tensor(self.times[idx]).long(), torch.tensor(self.events[idx]).bool())
	
def get_scaler(X):
	scaler = StandardScaler()
	
	scaler.fit(X)
	
	return scaler

class CV_METABRIC():
	
	def __init__(self, folds=5, **dataset_kwargs):
		
		self.dataset=DatasetMETABRIC(None, **dataset_kwargs)
		
		self.individuals=np.array(list(np.arange(0, self.dataset.gene_ind_matrix.shape[0], 1)))
		
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
		
		X = self.dataset.gene_ind_matrix
		
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

class BaselineHazardModel(GeneformerModule.model):
	
	def __init__(self, params, **kwargs):
		super(BaselineHazardModel, self).__init__("Pretrained", "/shared/home/david.calhas/ppmi_diferre/venv/lib/python3.11/site-packages/geneformer/fine_tuned_models/Geneformer-V2-104M", rank=2, n_classes=kwargs['classes'])
		
		self.w=torch.nn.Parameter(torch.ones((params, )))#only use this parameter for shapes paramters of weibull and generalized gamma, the loc paramters should be output of ClinicalModel
		
		torch.nn.init.zeros_(self.w)
		
	def forward(self, batch, *args, **kwargs):
		if(not type(batch).__base__ is tuple): return torch.concatenate((batch.view(batch.shape[0], -1).sum(dim=1, keepdim=True)*0, self.w.unsqueeze(0).repeat(batch.shape[0], 1)), dim=1)
		
		output=super(BaselineHazardModel, self).forward(batch)#batch times param
		
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
	def exponential(input_dim=7, hidden_dim=64, classes=1, ):
		
		return BaselineHazardModel(0, in_dim=input_dim, hidden_dim=hidden_dim, classes=classes)
	
	@staticmethod
	def weibull(input_dim=7, hidden_dim=64, classes=1, ):
		
		return BaselineHazardModel(1, in_dim=input_dim, hidden_dim=hidden_dim, classes=classes)
	
	@staticmethod
	def generalized_gamma(input_dim=7, hidden_dim=64, classes=1, ):
		
		return BaselineHazardModel(2, in_dim=input_dim, hidden_dim=hidden_dim, classes=classes)
	
	@staticmethod
	def cox(input_dim=7, hidden_dim=64, classes=1, ):
		
		return BaselineHazardModel(0, in_dim=input_dim, hidden_dim=hidden_dim, classes=classes)