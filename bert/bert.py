import numpy as np
import random

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from attention import MultiHeadedAttention
from embedding import BertEmbedding
from utils import SublayerConnection, PositionwiseFeedForward

def fix_random_seed_as(random_seed):
	random.seed(random_seed)
	torch.manual_seed(random_seed)
	torch.cuda.manual_seed_all(random_seed)
	np.random.seed(random_seed)
	cudnn.deterministic = True
	cudnn.benchmark = False

class TransformerBlock(nn.Module):
	
	def __init__(self, hidden, attn_heads, feed_forward_hidden, dropout):
		
		super().__init__()
		
		self.attention = MultiHeadedAttention(h=attn_heads, d_model=hidden, dropout=dropout)
		self.feed_forward = PositionwiseFeedForward(d_model=hidden, d_ff=feed_forward_hidden, dropout=dropout)
		self.input_sublayer = SublayerConnection(size=hidden, dropout=dropout)
		self.output_sublayer = SublayerConnection(size=hidden, dropout=dropout)
		self.dropout = nn.Dropout(p=dropout)
  
	def forward(self, x, mask):
		x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask=mask))
		x = self.output_sublayer(x, self.feed_forward)
		return self.dropout(x)

class BERT(nn.Module):
	def __init__(self, args):
		super().__init__()
		
		print(args)
		
		model_init_seed = args['model_init_seed']
		fix_random_seed_as(model_init_seed)

		max_len = args['max_len']
		num_items = args['num_items']
		n_layers = args['num_blocks']
		heads = args['num_heads']
		items_size = num_items + 2
		hidden = args['hidden_units']
		self.hidden = hidden
		dropout = args['dropout']
		num_output_class = args['num_output_class']
		device = args['device']

		self.embeddings = BertEmbedding(items_size=items_size, 
										embed_size=self.hidden, 
										max_len=max_len,
										dropout=dropout,
										device=device)
	
		self.transformer_blocks = nn.ModuleList([TransformerBlock(hidden, heads, hidden * 4, dropout) for _ in range(n_layers)])
		self.linear_out = nn.Linear(hidden, items_size)
  
	def forward(self, x):
		mask = (x > 0).unsqueeze(1).repeat(1, x.size(1), 1).unsqueeze(1)
  
		# embedding indexes
		x = self.embeddings(x)
		
		for transformer in self.transformer_blocks:
			x = transformer.forward(x, mask)
   
		x = self.linear_out(x)

		return x

	def init_weights(self):
		pass