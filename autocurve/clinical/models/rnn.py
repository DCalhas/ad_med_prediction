import torch

class RNN(torch.nn.Module):
	
	def __init__(self, input_dim=65, hidden_dim=128,):
		super(RNN, self).__init__()
		
		self.hidden_size = hidden_dim

		# weight matrices
		self.W_ih = torch.nn.Linear(input_dim, hidden_dim)
		self.W_hh = torch.nn.Linear(hidden_dim, hidden_dim)
		self.activation = torch.tanh

	def forward(self, x):
		#batch, _, seq_len, n_features, _ = x.shape#B, T, L, F, C
		batch, seq_len, n_features, _ = x.shape#B, T, L, F, C
		
		#h=torch.zeros(batch, 5, n_features, self.hidden_size, device=x.device)
		h=torch.zeros(batch, n_features, self.hidden_size, device=x.device)
		
		outputs = []
		
		for t in range(seq_len):
			#xt = x[:, :, t, :, :]
			xt = x[:, t, :, :]
			h = self.activation(self.W_ih(xt) + self.W_hh(h))
			
			outputs.append(h.unsqueeze(1))
			
		return torch.cat(outputs, dim=1)
	
class LSTM(torch.nn.Module):
	
	def __init__(self, input_dim=65, hidden_dim=128):
		super(LSTM, self).__init__()
		
		self.hidden_size = hidden_dim
		
		self.W_ii = torch.nn.Linear(input_dim, hidden_dim)
		self.W_if = torch.nn.Linear(input_dim, hidden_dim)
		self.W_ig = torch.nn.Linear(input_dim, hidden_dim)
		self.W_io = torch.nn.Linear(input_dim, hidden_dim)
		
		self.W_hi = torch.nn.Linear(hidden_dim, hidden_dim)
		self.W_hf = torch.nn.Linear(hidden_dim, hidden_dim)
		self.W_hg = torch.nn.Linear(hidden_dim, hidden_dim)
		self.W_ho = torch.nn.Linear(hidden_dim, hidden_dim)
		
		self.ln_i = torch.nn.LayerNorm(hidden_dim, eps=1e-3)
		self.ln_f = torch.nn.LayerNorm(hidden_dim, eps=1e-3)
		self.ln_g = torch.nn.LayerNorm(hidden_dim, eps=1e-3)
		self.ln_o = torch.nn.LayerNorm(hidden_dim, eps=1e-3)
		
		torch.nn.init.xavier_uniform_(self.W_ii.weight)
		torch.nn.init.xavier_uniform_(self.W_if.weight)
		torch.nn.init.xavier_uniform_(self.W_ig.weight)
		torch.nn.init.xavier_uniform_(self.W_io.weight)
		torch.nn.init.xavier_uniform_(self.W_hi.weight)
		torch.nn.init.xavier_uniform_(self.W_hf.weight)
		torch.nn.init.xavier_uniform_(self.W_hg.weight)
		torch.nn.init.xavier_uniform_(self.W_ho.weight)
		
		torch.nn.init.zeros_(self.W_ii.bias)
		torch.nn.init.zeros_(self.W_if.bias)
		torch.nn.init.zeros_(self.W_ig.bias)
		torch.nn.init.zeros_(self.W_io.bias)
		torch.nn.init.zeros_(self.W_hi.bias)
		torch.nn.init.ones_(self.W_hf.bias)
		torch.nn.init.zeros_(self.W_hg.bias)
		torch.nn.init.zeros_(self.W_ho.bias)
		
		torch.nn.init.ones_(self.ln_i.weight)
		torch.nn.init.ones_(self.ln_f.weight)
		torch.nn.init.ones_(self.ln_g.weight)
		torch.nn.init.ones_(self.ln_o.weight)
		
		torch.nn.init.zeros_(self.ln_i.bias)
		torch.nn.init.zeros_(self.ln_f.bias)
		torch.nn.init.zeros_(self.ln_g.bias)
		torch.nn.init.zeros_(self.ln_o.bias)
		
		self.sigmoid = torch.sigmoid
		self.tanh = torch.tanh

	def forward(self, x):
		#batch, T, seq_len, n_features, _ = x.shape  # B, T, L, F, C
		batch, seq_len, n_features, _ = x.shape  # B, T, L, F, C
		
		#h = torch.zeros(batch, T, n_features, self.hidden_size, device=x.device)
		
		#c = torch.zeros(batch, T, n_features, self.hidden_size, device=x.device)
		h = torch.zeros(batch, n_features, self.hidden_size, device=x.device)
		
		c = torch.zeros(batch, n_features, self.hidden_size, device=x.device)
		
		outputs = []
		
		for t in range(seq_len):
			xt = x[:, t, :, :]  # (B, T, F, input_dim)
			
			i = self.sigmoid(self.W_ii(xt) + self.W_hi(h))
			f = self.sigmoid(self.W_if(xt) + self.W_hf(h))
			g = self.tanh(self.W_ig(xt) + self.W_hg(h))
			o = self.sigmoid(self.W_io(xt) + self.W_ho(h))
			
			c = f * c + i * g
			
			h = o * self.tanh(c)
			
			outputs.append(h.unsqueeze(1))
		
		return torch.cat(outputs, dim=1)