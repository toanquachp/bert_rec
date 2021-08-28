import numpy as np
import pandas as pd
import random

import torch
import torch.nn as nn


class Recommender(nn.Module):
	def __init__(self, vocab_size, channels=128, dropout=0.4):
		super().__init__()

		self.dropout = dropout
		self.vocab_size = vocab_size

		self.item_embeddings = nn.Embedding(self.vocab_size, embedding_dim=channels)
		self.input_pos_embedding = nn.Embedding(512, embedding_dim=channels)

		encoder_layer = nn.TransformerEncoderLayer(d_model=channels, nhead=4, dropout=self.dropout)

		self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
		self.linear_out = nn.Linear(channels, self.vocab_size)

	def encode_src(self, src_items):
		src_items = self.item_embeddings(src_items)

		batch_size, in_sequence_len = src_items.size(0), src_items.size(1)

		pos_encoder = (torch.arange(0, in_sequence_len, device=src_items.device).unsqueeze(0).repeat(batch_size, 1))
		pos_encoder = self.input_pos_embedding(pos_encoder)

		src_items += pos_encoder
		src = src_items.permute(1, 0, 2)

		src = self.encoder(src)

		return src.permute(1, 0, 2)

	def forward(self, src_items):
		src = self.encode_src(src_items)
		out = self.linear_out(src)

		return out
