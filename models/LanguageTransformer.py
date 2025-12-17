import torch
import math
import torch.nn as nn

# set appropriate device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')


"""
PositionalEncoding
	Applies positional encoding to sequential word embeddings
	via sinusoidal encoding. Used by LanguageTransformer for
	internal positional encoding.
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


'''
LanguageTransformer
	A decoder-only multiheaded transformer with two layer
	MLP classifier.
'''
class LanguageTransformer(nn.Module):

	def __init__(
		self,
		vocab_size,
		embed_dim,
		num_layers,
		num_heads,
		word_emb=None
	):
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
		super().__init__()

		if word_emb is not None:
			# freeze embeddings if provided
			self.token_embedding = nn.Embedding.from_pretrained(word_emb)
			self.token_embedding.weight.data[:].requires_grad_(False)
		else:
			self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_dim)

		# positional encodings
		self.pos_enc = PositionalEncoding(embed_dim=embed_dim)

		# custom transformer
		transformer_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
		self.transformer = nn.TransformerEncoder(encoder_layer=transformer_layer, num_layers=num_layers)

		# vocab classifier (two layer MLP)
		self.classifier = nn.Sequential(
			nn.Linear(in_features=embed_dim, out_features=2048),
			nn.ReLU(),
			nn.Linear(in_features=2048, out_features=vocab_size)
		)
		
	def forward(self, seq):
		'''
		LanguageTransformer.forward
			Applies custom (provided) or learned embeddings across
			vocabulary. Positionally encodes embeddings. Generates
			causal mask and feeds embeddings through transformer.
			Classifies over vocabulary.

		Args:
			seq: torch.Tensor of size (B, N)

		Returns:
			torch.Tensor of size (B, N, V)
		'''

		seq_len = seq.shape[1]

		# embed sequence w poisitional encodings
		seq_embed = self.token_embedding(seq)
		seq_embed = self.pos_enc(seq_embed)

		# generate causal mask
		mask = torch.zeros(seq_len, seq_len, dtype=torch.float32)
		mask = mask.masked_fill(torch.triu(torch.ones(seq_len, seq_len), diagonal=1) == 1, float('-inf')).to(device)
		seq_out = self.transformer(src=seq_embed, mask=mask)

		# next token classification
		out = self.classifier(seq_out)
		return out