import torch

def get_topk_genes(model, instance, vocab):
	"""
	Inputs:
		model: torch.nn.Module
		instance: 
		vocab: list of gene names
		k: top k genes to return 
	Outputs:
		indices np.ndarray
		expression
	"""
	with torch.no_grad():
		
		attention_scores=model.first_attention_scores(instance)
		scores=attention_scores.sum(1)
		top=torch.argsort(scores, descending=True, dim=-1)
		
		topexpression=torch.gather(scores, dim=-1, index=top).cpu().numpy()
		
	return top.cpu().numpy(), topexpression