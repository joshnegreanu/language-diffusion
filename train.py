import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import torch
import math
import pandas as pd
import random
import wandb

import sys
import signal
from functools import partial

from torch import nn, optim
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
from torch.utils.data import DataLoader, Dataset, ConcatDataset

from datasets import load_dataset
from transformers import AutoTokenizer
from staticvectors import StaticVectors
from datetime import datetime
from tqdm import tqdm

from models.LanguageTransformer import LanguageTransformer
from data.LanguageDataset import LanguageDataset

# dynamically select device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


"""
Training, model, and generation configurations.
Can be changed prior to training/autoregressive generation.
"""
train_config = {
    'bs': 32,
    'lr': 0.0001,
    'weight_decay': 0.000001,
    'max_epochs': 10
}

model_config = {
    'emb_dim': 300,
    'num_layers': 24,
    'num_heads': 10
}

generation_config = {
    'max_length': 50,
    'temperature': 0.9,
    'top_p': 0.9
}


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
interrupt_handler
    Save model checkpoint in case of terminal interrupt.
    Special checkpoint file tag not to override current
    epoch checkpoint.
"""
def interrupt_handler(
    epoch,
    model,
    scheduler,
    train_config,
    model_config,
    generation_config,
    project_name,
    run_name,
    sig,
    frame
):
    # save model on interrupt
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_config': train_config,
        'model_config': model_config,
        'generation_config': generation_config},
        f"./checkpoints/{project_name}/{run_name}/{epoch}int"
    )
    sys.exit(0)


"""
main
    Builds a model, checks through a dry run, runs
    through training cycle.
"""
def main():
    # create dataset
    dataset = LanguageDataset(max_examples=100000, max_len=512, bs=train_config['bs'])

    # create language model
    model = LanguageTransformer(
        vocab_size=dataset.vocab_len,
        embed_dim=model_config['emb_dim'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        word_emb=dataset.word2vec_embeddings
    ).to(device)
    dry_run(model, train_config['bs'], dataset.vocab_len, 100)

    # enter training cycle
    train(model, dataset.dataloader)


"""
train
    Sets up wandb logging, creates AdamW optimizer,
    custom linear warmup and cosine annealing scheduler,
    trains for designated number of epochs, saving model
    checkpoints each epoch.

    Args:
        model: torch.nn.Module language model
        data_loader: torch.DataLoader training data
"""
def train(model, dataloader):
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
    epoch_len = len(dataloader)

    linear = LinearLR(optimizer, start_factor=0.25, end_factor=1.0, total_iters=warmup_epochs*epoch_len)
    cosine = CosineAnnealingLR(optimizer, T_max=cooldown_epochs*epoch_len, eta_min=1e-6)
    scheduler = SequentialLR(optimizer, schedulers=[linear, cosine], milestones=[warmup_epochs*epoch_len])

    model.train()

    # main training loop
    pbar = tqdm(total=(train_config['max_epochs'])*epoch_len, desc="Training Iterations", unit="batch")
    iteration = 0
    for epoch in range(train_config['max_epochs']):
        # ignal catching to save model on interrupt
        signal.signal(signal.SIGINT, partial(interrupt_handler,
            epoch, model,
            scheduler, train_config,
            model_config, generation_config,
            project_name, run_name))
        signal.signal(signal.SIGTERM, partial(interrupt_handler,
            epoch, model,
            scheduler, train_config,
            model_config, generation_config,
            project_name, run_name))

        # minibatch gradient descent
        for batch_idx, batch in enumerate(dataloader):
            wandb.log({'learning-rate': scheduler.get_last_lr()[0]}, step=iteration)

            batch = batch.to(device)
            out = model(batch)[:,:-1,:]
            labels = batch[:,1:]

            # compute loss
            loss = criterion(out.permute(0, 2, 1), labels)
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


if __name__ == '__main__':
    main()