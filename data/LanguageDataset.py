import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import pandas as pd
import random

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
class LanguageDataset():
    def __init__(self, max_examples, max_len, bs):
        """
        LanguageDataset.__init__
            Creates bert tokenizer, tokenizes huggingface language dataset.
            Creates word2vec embeddings for tokenizer vocabulary. Creates
            custom dataloader.
        
        Args:
            max_examples: int max number of training examples to sample
            max_len: max length of each example
            bs: int batch size
        """
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
        self.vocab_len = len(self.tokenizer.get_vocab())

        # word2vec embeddings (300 dim)
        self.word2vec = StaticVectors("neuml/word2vec")
        self.word2vec_embeddings = torch.tensor(self.word2vec.embeddings(self.tokenizer.get_vocab())).type(torch.float32).to(device)

        # load huggingface dataset
        dataset = load_dataset("roneneldan/TinyStories")
        print("[dataset] loaded")

        # tokenize (create tuple pair)
        self.tokenized_dataset = self.tokenizer([x['text'] for x in random.sample(list(dataset['train']), max_examples)], padding='max_length', max_length=max_len, truncation=True, return_tensors="pt", add_special_tokens=True).input_ids
        print("[dataset] tokenized")

        # create custom dataloader
        self.dataloader = DataLoader(self.tokenized_dataset, batch_size=bs, num_workers=4, shuffle=True, drop_last=True)