from autocurve.clinical.models.rnn import RNN, LSTM

from autocurve.clinical.models.base import ClinicalModule, BatchDataLoader

import torch

#a ClinicalModel receives an encoder class. this can be anything.

class BaselineHazardModel(ClinicalModule.model):
	
	def __init__(self, params, **kwargs):
		super(BaselineHazardModel, self).__init__(**kwargs)
		
		self.w=torch.nn.Parameter(torch.ones((params, )))#only use this parameter for shapes paramters of weibull and generalized gamma, the loc paramters should be output of ClinicalModel
		
		torch.nn.init.zeros_(self.w)
		
	def forward(self, x, seq_len=None):
		output=super(BaselineHazardModel, self).forward(x, seq_len=seq_len)#batch times param

		if(type(output) is tuple):
			output, seq_len=output
		
			return torch.concatenate((output, self.w.unsqueeze(0).repeat(output.shape[0], 1)), dim=1), seq_len

		return torch.concatenate((output, self.w.unsqueeze(0).repeat(output.shape[0], 1)), dim=1)
	
	def baseline_hazard(self,):
		raise NotImplementedError		
		
class BaselineHazardCoxModel(BaselineHazardModel):
	
	def forward(self, x, seq_len=None):
		output=super(BaselineHazardCoxModel, self).forward(x, seq_len=seq_len)#batch times param
		
		return torch.concatenate((output[:,:-1], self.w.unsqueeze(0).repeat(output.shape[0], 1), output[:,-1:]), dim=1)
	
class ModelBuilds():
	
	"""_summary_
	This class has staticmethods to build the models that are use as nets for the PyCox wrapper classes
	
	ClinicalModel subtype 
	"""
	
	@staticmethod
	def exponential(n_features=65, input_dim=7, hidden_dim=64, n_transitions=5, n_estimations=1):
		
		return ClinicalModule.model(n_features, input_dim, hidden_dim, n_transitions, n_estimations=n_estimations)
	
	@staticmethod
	def weibull(n_features=65, input_dim=7, hidden_dim=64, n_transitions=5, n_estimations=1):
		
		return BaselineHazardModel(1, n_features=n_features, input_dim=input_dim, hidden_dim=hidden_dim, n_transitions=n_transitions, n_estimations=n_estimations)
	
	@staticmethod
	def generalized_gamma(n_features=65, input_dim=7, hidden_dim=64, n_transitions=5, n_estimations=1):
		
		return BaselineHazardModel(2, n_features=n_features, input_dim=input_dim, hidden_dim=hidden_dim, n_transitions=n_transitions, n_estimations=n_estimations)
	
	@staticmethod
	def cox(n_features=65, input_dim=7, hidden_dim=64, n_transitions=5, n_estimations=1):
		
		return ClinicalModule.model(n_features, input_dim, hidden_dim, n_transitions, n_estimations=n_estimations)
	
	@staticmethod
	def cox_exponential(n_features=65, input_dim=7, hidden_dim=64, n_transitions=5, n_estimations=2):
		
		return ClinicalModule.model(n_features, input_dim, hidden_dim, n_transitions, n_estimations=n_estimations)
	
	@staticmethod
	def cox_weibull(n_features=65, input_dim=7, hidden_dim=64, n_transitions=5, n_estimations=2):
		
		return BaselineHazardModel(1, n_features=n_features, input_dim=input_dim, hidden_dim=hidden_dim, n_transitions=n_transitions, n_estimations=n_estimations)
	
	@staticmethod
	def cox_generalized_gamma(n_features=65, input_dim=7, hidden_dim=64, n_transitions=5, n_estimations=2):
		
		return BaselineHazardModel(2, n_features=n_features, input_dim=input_dim, hidden_dim=hidden_dim, n_transitions=n_transitions, n_estimations=n_estimations)