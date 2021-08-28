import torch.nn as nn
import torch
import torch.nn.functional as F

import math

class Attention(nn.Module):
	def forward(self, query, key, value, mask=None, dropout=None):
		scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
		
		if mask is not None:
			scores = scores.masked_fill(mask == 0, -1e9)
			
		p_attn = F.softmax(scores, dim=1)
		
		if dropout is not None:
			p_attn = dropout(p_attn)
			
		return torch.matmul(p_attn, value), p_attn
	
class MultiHeadedAttention(nn.Module):
	def __init__(self, h, d_model, dropout=0.1):
		super().__init__()
		
		assert d_model % h == 0
	
		self.d_k = d_model // h
		self.h = h

		self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(3)])
		self.output_linear = nn.Linear(d_model, d_model)
		self.attention = Attention()
		
		self.dropout = nn.Dropout(p=dropout)
		
	def forward(self, query, key, value, mask=None):
		batch_size = query.size(0)
		
		# do linear projection in batch 
		query, key, value = [l(x).view(batch_size, -1, self.h, self.d_k).transpose(1, 2) 
							 for l, x in zip(self.linear_layers, (query, key, value))]
		
		# apply attention
		x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)
	
		# concat with view and apply final linear layer
		x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)
		
		return self.output_linear(x)