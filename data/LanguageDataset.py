import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import pandas as pd

from torch.utils.data import DataLoader

from datasets import load_dataset
from transformers import AutoTokenizer
from staticvectors import StaticVectors

# dynamically select device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


"""
LanguageDataset
    Custom dataset class for storing poetry dataset
    and custom dataloader.
"""
class PoetryDataset():
    def __init__(self, bs):
        """
        PoetryDataset.__init__
            Creates bert tokenizer, tokenizes huggingface poetry dataset.
            Creates word2vec embeddings for tokenizer vocabulary. Creates
            custom dataloader.
        
        Args:
            bs: int batch size
        """
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.vocab_len = len(self.tokenizer.get_vocab())

        # word2vec embeddings (300 dim)
        self.word2vec = StaticVectors("neuml/word2vec")
        self.word2vec_embeddings = torch.tensor(self.word2vec.embeddings(self.tokenizer.get_vocab())).type(torch.float32).to(device)

        # load huggingface dataset
        poetry_dataset = load_dataset("merve/poetry")

        # tokenize (create tuple pair)
        dataset_tokens = self.tokenizer(list(poetry_dataset['train']['content']), padding='max_length', max_length=512, truncation=True, return_tensors="pt", add_special_tokens=True)
        inputs = dataset_tokens.input_ids[:, :-1]
        labels = dataset_tokens.input_ids[:, 1:]
        self.tokenized_dataset = torch.utils.data.TensorDataset(inputs, labels)

        # create custom dataloader
        self.dataloader = DataLoader(self.tokenized_dataset, batch_size=bs, num_workers=4, shuffle=True, drop_last=True)