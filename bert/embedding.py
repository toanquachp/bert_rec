import torch.nn as nn
import torch
import math

class PositionalEmbedding(nn.Module):
	
	def __init__(self, max_len, d_model, device='cuda'):
		super().__init__()

		self.device = device
		self.pe = nn.Embedding(max_len, d_model)
  
	def forward(self, x):
		batch_size, sequence_length = x.size(0), x.size(1)
		pos_encoder = (
      		torch.arange(0, sequence_length, device=self.device)
        	.unsqueeze(0)
         	.repeat(batch_size, 1)
        )
		# return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)
		return self.pe(pos_encoder)

class ItemEmbedding(nn.Embedding):
	
	def __init__(self, items_size, embed_size=512):
		super().__init__(items_size, embed_size, padding_idx=0)
		
class BertEmbedding(nn.Module):
	
	def __init__(self, items_size, embed_size, max_len, dropout=0.1, device='cuda'):
		
		super().__init__()
		
		self.items = ItemEmbedding(items_size=items_size, embed_size=embed_size)
		self.position = PositionalEmbedding(max_len=max_len, d_model=embed_size, device=device)
		
		self.dropout = nn.Dropout(p=dropout)
		self.embed_size = embed_size
		
	def forward(self, sequence):
		
		x = self.items(sequence) + self.position(sequence)
		return self.dropout(x)