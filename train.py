import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import math
import pandas as pd
import random
import wandb

from torch import nn, optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from datasets import load_dataset
from transformers import AutoTokenizer
from staticvectors import StaticVectors
from datetime import datetime
from tqdm import tqdm

# custom modules
from models.LanguageTransformer import LanguageTransformer
from data.LanguageDataset import PoetryDataset

# set appropriate device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('mps')


"""
dry_run
    Runs language model through random to ensure proper
    dimensionality. Asserts correct shape.

    Args:
        model: torch.nn.Module language model
        bs: int batch size
        vocab_len: int size of vocab
        seq_len: int length of token sequence
"""
def dry_run(model, bs, vocab_len, seq_len):
    seq = torch.randint(0, vocab_len, (bs, seq_len)).to(device)
    out = model(seq)
    assert out.shape == (bs, seq_len, vocab_len)
    print("[dry_run] passed")


"""
Training, model, and generation configurations.
Can be changed prior to training/autoregressive generation.
"""
train_config = {
    'bs': 32,
    'lr': 0.0001,
    'weight_decay': 0.00001,
    'max_epochs': 10
}

model_config = {
    'emb_dim': 300,
    'num_layers': 12,
    'num_heads': 4
}

generation_config = {
    'max_length': 50,
    'temperature': 0.9,
    'top_p': 0.9
}


"""
Main function (to deal with multiprocessing)
"""
if __name__ == '__main__':
    # create poetry dataset
    poetry_dataset = PoetryDataset(train_config['bs'])

    # create language model
    model = LanguageTransformer(
        vocab_size=poetry_dataset.vocab_len,
        embed_dim=model_config['emb_dim'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        word_emb=poetry_dataset.word2vec_embeddings
    ).to(device)
    dry_run(model, train_config['bs'], poetry_dataset.vocab_len, 100)

    # set up wandb and checkpoint path
    now = datetime.now()
    project_name = "diffusion-language-model"
    run_name = "dlm-" + now.strftime("%Y_%m_%d_%H_%m")
    wandb.login()
    wandb.init(project=project_name, name=run_name, config=train_config)
    os.makedirs(f"./checkpoints/{project_name}/{run_name}", exist_ok=True)

    # optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=train_config['lr'], weight_decay=train_config['weight_decay'])
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    # construct linear warmup and cosine annealing cooldown
    warmup_epochs = int(train_config['max_epochs'] / 10)
    cooldown_epochs = train_config['max_epochs'] - warmup_epochs
    epoch_len = len(poetry_dataset.tokenized_dataset) // train_config['bs']

    linear = LinearLR(optimizer, start_factor=0.25, end_factor=1.0, total_iters=warmup_epochs*epoch_len)
    cosine = CosineAnnealingLR(optimizer, T_max=cooldown_epochs*epoch_len, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[linear, cosine], milestones=[warmup_epochs*epoch_len])

    # main training loop
    pbar = tqdm(total=(train_config['max_epochs'])*epoch_len, desc="Training Iterations", unit="batch")
    iteration = 0
    for epoch in range(train_config['max_epochs']):
        model.train()

        # minibatch gradient descent
        for batch_idx, batch in enumerate(poetry_dataset.dataloader):
            wandb.log({'learning-rate': scheduler.get_last_lr()[0]}, step=iteration)

            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            out = model(inputs).permute(0, 2, 1)

            # compute loss
            loss = criterion(out, labels)
            wandb.log({"loss": loss.item()}, step=iteration)

            # optimization
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            pbar.update(1)
            iteration += 1
            scheduler.step()

        # save model each epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_config': train_config,
            'model_config': model_config,
            'generation_config': generation_config},
            f"./checkpoints/{project_name}/{run_name}/{epoch}"
        )

    wandb.finish()
    pbar.close()