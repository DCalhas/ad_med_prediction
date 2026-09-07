from autocurve.rna.lookup import GENE_LOOKUP

from autocurve.rna.wgcna import WGCNA

from autocurve.rna.utils import get_topk_genes

import torch as _torch
if _torch.cuda.is_available():

	from autocurve.rna import models
else:
	import warnings as _warnings
	_warnings.filterwarnings("default")
	_warnings.warn("Your machine does not have a GPU or it was not recognized. For this reason the autocurve.rna.models module is not available.")
	_warnings.filterwarnings("ignore")

from autocurve.rna import datasets