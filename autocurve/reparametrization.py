import torch

class SingularValuesParametrization(torch.nn.Module):
	def __init__(self, U, V):
		super().__init__()
		
		self.register_buffer("U", U)
		self.register_buffer("V", V)
		
	def forward(self, S):
		return (self.U*S)@self.V
	
def register_parametrization(module, name, tensor):	

	with torch.no_grad():
		U, S, V=torch.linalg.svd(tensor, full_matrices=False)

	del module._parameters[name]
	module.register_parameter(name, torch.nn.Parameter(S.clone()))

	torch.nn.utils.parametrize.register_parametrization(module, name, SingularValuesParametrization(U, V), unsafe=True)