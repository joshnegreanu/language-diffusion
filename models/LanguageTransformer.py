import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import math
import torch.nn as nn

from utils import Transformer, PositionalEncoding

# dynamically select device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


"""
LanguageTransformer
	A decoder-only multiheaded transformer with
	MLP vocabulary classifier.
"""
class LanguageTransformer(nn.Module):

	"""
	LanguageTransformer.__init__
		Constructs necessary internal modules for language
		model. Creates (if necessary) word embeddings,
		positional encoding, transformer layers, transformer,
		and linear classifier.

	Args:
		vocab_size: int size of vocab
		embed_dim: int embedding dimensioanlity
		num_layers: int number of transformer layers
		num_heads: int number of attention heads per layer
		word_emb: None or torch.Tensor of size (V, D)
	"""
	def __init__(
		self,
		vocab_size,
		embed_dim,
		num_layers,
		num_heads,
		word_emb=None
	):
		super().__init__()

		if word_emb is not None:
			# freeze embeddings if provided
			self.token_embedding = nn.Embedding.from_pretrained(word_emb)
			# self.token_embedding.weight.data[:].requires_grad_(False)
		else:
			self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

		# positional encodings
		self.pos_enc = PositionalEncoding(embed_dim=embed_dim)

		# custom transformer
		self.transformer = Transformer(embed_dim, num_heads, num_layers)

		# vocab classifier
		self.classifier = nn.Linear(in_features=embed_dim, out_features=vocab_size)
	
	"""
	LanguageTransformer.causal_mask
		Constructs a causal mask for training on a
		single input sequence.

	Args:
		dim: int length of training sequence
	
	Returns:
		torch.Tensor causal mask
	"""
	def generate_causal_mask(self, seq_len):
		# mask out appropriate triangle half
		inf_tensor = torch.ones(seq_len, seq_len, dtype=torch.float32) * float('-inf')
		mask = torch.tril(inf_tensor, diagonal=-1).T
		return mask
	

	"""
	LanguageTransformer.forward
		Applies custom (provided) or learned embeddings across
		vocabulary. Positionally encodes embeddings. Generates
		causal mask and feeds embeddings through transformer.
		Classifies over vocabulary.

	Args:
		seq: torch.Tensor of size (B, N)

	Returns:
		torch.Tensor of size (B, N, V)
	"""
	def forward(self, seq):
		seq_len = seq.shape[1]

		# embed sequence w poisitional encodings
		seq_embed = self.token_embedding(seq)
		seq_embed = self.pos_enc(seq_embed)

		# generate causal mask
		mask = self.generate_causal_mask(seq_len)
		seq_out = self.transformer(src=seq_embed, mask=mask)

		# next token classification
		out = self.classifier(seq_out)
		return out