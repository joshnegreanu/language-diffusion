import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

# dynamically select device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


"""
PositionalEncoding
	Applies positional encoding to sequential word embeddings
	via sinusoidal encoding.
"""
class PositionalEncoding(nn.Module):
	def __init__(self, embed_dim):
		super().__init__()
		"""
		PositionalEncoding.__init__
			Initializes encoding with proper embedding dimension.

		Args:
			embed_dim: int embedding dimensionality
		"""

		self.embed_dim = embed_dim


	def forward(self, x):
		"""
		PositionalEncoding.forward
			Applies sinusoidal positional encoding to input.
		
		Args:
			x: torch.Tensor of size (B, N, D)

		Returns:
			torch.Tensor of size (B, N, D)
		"""

		batch_size = x.shape[0]
		seq_len = x.shape[1]

		# calcualte sinusoidal encodings
		pe = torch.zeros(1, seq_len, self.embed_dim).to(x.device)
		pos = torch.arange(0, seq_len, dtype=torch.float32)
		enc = torch.exp((-math.log(10000.0)) * (torch.arange(0, self.embed_dim, step=2, dtype=torch.float32) / self.embed_dim))

		# calculate positional encoding
		prod = torch.outer(pos, enc)
		pe[0, :, 0::2] = torch.sin(prod)
		pe[0, :, 1::2] = torch.cos(prod)
		pe = pe.expand(batch_size, -1, -1)

		# apply as residual
		x = x + pe
		return x


class MultiheadAttention(nn.Module):
	def __init__(self, emb_dim, num_heads):
		super().__init__()

		assert emb_dim % num_heads == 0
		self.head_dim = emb_dim / num_heads
		
		# set up key, query, and value linear transformations
		self.q_linear = nn.Linear(emb_dim, emb_dim)
		self.k_linear = nn.Linear(emb_dim, emb_dim)
		self.v_linear = nn.Linear(emb_dim, emb_dim)

		self.concat_linear = nn.Linear(emb_dim, emb_dim)


	def scaled_dot_product_attention(self, q, k, v, causal_mask):
		# dot product self attention
		dots = torch.matmul(q, k.transpose(1, 2)) / (self.head_dim ** 0.5)

		# apply causal mask
		if causal_mask is not None:
			dots = dots.masked_fill(causal_mask == 0, float('-inf'))
		
		attn = F.softmax(dots, dim=-1)
		return torch.matmul(attn, v)


	def forward(self, x, causal_mask):
		bs = x.shape[0]

		# run through query, key, and value transformations
		q = self.q_linear(x).view(bs, -1, self.num_heads, self.head_dim).transpose(1, 2)
		k = self.k_linear(x).view(bs, -1, self.num_heads, self.head_dim).transpose(1, 2)
		v = self.v_linear(x).view(bs, -1, self.num_heads, self.head_dim).transpose(1, 2)

		# calculate attentions, concatenate multiple heads
		attn = self.scaled_dot_product_attention(q, k, v, causal_mask).transpose(1, 2)
		attn = x.view(bs, -1, self.emb_dim)
		return self.concat_linear(attn)


class TransformerLayer(nn.Module):
	def __init__(self, emb_dim, num_heads, causal_mask = None):
		self.attn_layer = MultiheadAttention(emb_dim, num_heads)

		self.feed_forward = nn.Sequential(
			nn.Linear(emb_dim, 1024),
			nn.ReLU(),
			nn.Linear(1024, emb_dim)
		)

		self.emb_dim = emb_dim
		self.num_heads = num_heads
	
	def forward(self, x, causal_mask):
		# run through residual attention layer
		x += self.attn_layer(x, causal_mask)
		x = torch.layer_norm(x, normalized_shape=[self.emb_dim])

		# run through feed forward network
		x += self.feed_forward(x)
		return torch.layer_norm(x, normalized_shape=[self.emb_dim])


class Transformer(nn.Module):
	def __init__(self, emb_dim, num_heads, num_layers):
		super().__init__()

		# build tranformer layers
		transformer_layers = []
		for _ in range(num_layers):
			transformer_layers.append(TransformerLayer(emb_dim, num_heads, causal_mask))
		self.transformer_layers = nn.Sequential(*transformer_layers)
		
	
	def forward(self, x, causal_mask = None):
		return self.transformer_layers(x, causal_mask)