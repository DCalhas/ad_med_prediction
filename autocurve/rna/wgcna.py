import pandas as pd

import numpy as np

from sklearn.preprocessing import scale

import json

import PyWGCNA as pywgcna

import matplotlib.colors as mcolors

from sklearn.preprocessing import StandardScaler

import sys

import os

np.set_printoptions(threshold=sys.maxsize)

class WGCNA():
	"""
	This is a personal implementation of the PyWGCNA package code, because I needed to distinguish the train set from the test set.

	Usage:
		>>> wgcna=WGCNA(gene_ind_matrix, genes_ids, pretrained_dir+"/tokenizer/vocab.json")
		>>> wgcna.fit(gene_ind_matrix, genes_ids=genes_ids)
		>>> wgcna.transform(gene_ind_matrix, genes_ids)
	"""

	@staticmethod
	def filter(X, genes_ids, genes_names, vocab, k=2000):
		#vocab=np.array(vocab).astype(genes_ids.dtype)
		#valid=np.isin(genes_ids, vocab)
		#if(valid.astype(np.float32).sum()==0):
		valid=np.isin(genes_names, genes_names)

		X=X[:,valid]

		topgenes=np.argsort(-np.var(X[:,:,0], axis=0))[:k]
		bottomgenes=np.argsort(-np.var(X[:,:,0], axis=0))[k:]
		X=X[:,topgenes]
		valid2 = valid.copy()
		valid_idx = np.flatnonzero(valid)
		valid2[ valid_idx[bottomgenes] ] = False
		return X, valid2

	def __init__(self, X_train, genes_ids, genes_names, vocab, topk=2000, beta=6, minClusterSize=20, minModuleSize=10, RsquaredCut=0.8):
		_, valid_genes=WGCNA.filter(X_train, genes_ids, genes_names, vocab, k=topk)

		self.scaler=StandardScaler()
		self.valid_genes=valid_genes
		self.modules=[]
		self.eigengenes=[]
		self.beta=beta
		self.minClusterSize=minClusterSize
		self.minModuleSize=minModuleSize
		self.RsquaredCut=RsquaredCut
		
	def fit(self, X, genes_ids, genes_names):
		print("I: Running WGCNA", end="\r")
		old_stdout = sys.stdout
		sys.stdout = open(os.devnull, "w")

		len_valid_genes=self.valid_genes.astype(np.float32).sum()
		
		if(X.shape[1]!=len_valid_genes):
			X=X[:, self.valid_genes]
			genes_ids=genes_ids[self.valid_genes]

		X=X.reshape((X.shape[0], -1))#flatten in case of multiple slopes

		self.scaler.fit(X)
		X=self.scaler.transform(X)
		
		df_wgcna=pd.DataFrame(X, dtype=float)
		#df_wgcna.columns = df_wgcna.columns.astype(str)
		#df_wgcna.index = df_wgcna.index.astype(str)
		df_wgcna = df_wgcna.apply(pd.to_numeric)
		wgcna=pywgcna.WGCNA(geneExp=df_wgcna, minModuleSize=self.minModuleSize, RsquaredCut=self.RsquaredCut, )
		
		wgcna.adjacency=pywgcna.WGCNA.adjacency(wgcna.datExpr.to_df(), power=self.beta, adjacencyType=wgcna.networkType,)

		wgcna.adjacency=pd.DataFrame(wgcna.adjacency, columns=wgcna.datExpr.to_df().columns, index=wgcna.datExpr.to_df().columns)

		wgcna.TOM=pywgcna.WGCNA.TOMsimilarity(wgcna.adjacency.to_numpy(), TOMType=wgcna.TOMType,)
		wgcna.TOM.columns=wgcna.datExpr.to_df().columns
		wgcna.TOM.index=wgcna.datExpr.to_df().columns
		dissTOM=1 - wgcna.TOM
		dissTOM=dissTOM.round(decimals=8)

		diss = np.asarray(dissTOM.values, dtype=float)
		a=pywgcna.squareform(dissTOM.values, checks=False)
		wgcna.geneTree=pywgcna.linkage(a, method="average")
		
		dissDF = pd.DataFrame(diss,index=np.arange(diss.shape[0]),columns=np.arange(diss.shape[1]),dtype=float)
		
		dynamicMods = pywgcna.WGCNA.cutreeHybrid(dendro=wgcna.geneTree,distM=dissDF,deepSplit=1,minClusterSize=self.minClusterSize,pamStage=False,pamRespectsDendro=False,useMedoids=False,respectSmallClusters=False)

		colors = pywgcna.WGCNA.labels2colors(dynamicMods, colorSeq=list(mcolors.CSS4_COLORS.keys()))
		
		colors = np.array(colors)
		
		moduleSizes = {c: (colors == c).sum() for c in np.unique(colors)}
		
		keep = [c for c in moduleSizes if moduleSizes[c] >= 10]
		colors_filtered = np.where(np.isin(colors, keep), colors, 'grey')
		
		expr=wgcna.datExpr.to_df()
		colors=colors_filtered
		impute=True
		nPC=1
		align="along average"
		excludeGrey=False
		grey="grey"
		subHubs=True
		softPower=6
		scaleVar=True
		trapErrors=False
		check = True
		pc = None
		returnValidOnly = trapErrors
		grey=0
		
		maxVarExplained=10
		nVarExplained = min(nPC, maxVarExplained)
		modlevels = pd.Categorical(colors).categories
		
		PrinComps = np.empty((expr.shape[0], len(modlevels)))
		PrinComps[:] = np.nan
		PrinComps = pd.DataFrame(PrinComps)
		PrinCompsVector = np.empty((expr.shape[1], len(modlevels)))
		PrinCompsVector[:] = np.nan
		PrinCompsVector = pd.DataFrame(PrinCompsVector)
		averExpr = np.empty((expr.shape[0], len(modlevels)))
		averExpr[:] = np.nan
		averExpr = pd.DataFrame(averExpr)
		varExpl = np.empty((nVarExplained, len(modlevels)))
		varExpl[:] = np.nan
		varExpl = pd.DataFrame(varExpl)
		validMEs = np.repeat(True, len(modlevels))
		validAEs = np.repeat(False, len(modlevels))
		isPC = np.repeat(True, len(modlevels))
		isHub = np.repeat(False, len(modlevels))
		validColors = colors
		PrinComps.columns = ["ME" + str(modlevel) for modlevel in modlevels]
		PrinCompsVector.columns = ["MV" + str(modlevel) for modlevel in modlevels]
		averExpr.columns = ["AE" + str(modlevel) for modlevel in modlevels]
			
		if expr.index is not None:
			PrinComps.index = expr.index
			averExpr.index = expr.index
		for i in range(len(modlevels)):
			modulename = modlevels[i]
			restrict1 = (colors == modulename)
			datModule = expr.loc[:, restrict1].T
			n = datModule.shape[0]
			p = datModule.shape[1]
			if datModule.shape[0] > 1 and impute:
				seedSaved = True
				if datModule.isnull().values.any():
					# define imputer
					imputer = KNNImputer(n_neighbors=np.min(10, datModule.shape[0] - 1))
					# fit on the dataset
					imputer.fit(datModule)
					# transform the dataset
					datModule = imputer.transform(
						datModule)  # datModule = impute.knn(datModule, k = min(10, nrow(datModule) - 1))
			if scaleVar:
				datModule = pd.DataFrame(scale(datModule.T).T, index=datModule.index, columns=datModule.columns)

			Xc=np.array(datModule).T
			C=np.cov(Xc, rowvar=False)
			S, V=np.linalg.eig(C)
			V=V.real
			S=S.real
			highest_eig=np.argsort(S)[-1]
			
			self.eigengenes.append(restrict1)
			_u=np.zeros(restrict1.shape)
			_u[restrict1]=V[:,highest_eig]
			PrinCompsVector.iloc[:, i] = _u

		self.W=np.array(PrinCompsVector)

		sys.stdout.close()
		sys.stdout = old_stdout

		print("I: Successfully ran WGCNA.", end="\n")


	def transform(self, X, genes_ids):
		
		_X=X[:, self.valid_genes]
		_X=_X.reshape((_X.shape[0],-1))
		scaled_X=self.scaler.transform(_X)
		
		W=self.W

		return scaled_X@W