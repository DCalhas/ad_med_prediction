from autocurve.rna.models.base import _ModelModule, _BaseFoundationalModel, BatchRepresentation, BatchDataLoader, _CV_Model

from autocurve.rna.models.scgpt import scGPTModule
from autocurve.rna.models.geneformer import GeneformerModule
from autocurve.rna.models.nicheformer import NicheformerModule
from autocurve.rna.models.teddy import TEDDYModule
from autocurve.rna.models.scfoundation import scFoundationModule


#extract the models' classes
TEDDY=TEDDYModule.model
scFoundation=scFoundationModule.model
Nicheformer=NicheformerModule.model
Geneformer=GeneformerModule.model
scGPT=scGPTModule.model