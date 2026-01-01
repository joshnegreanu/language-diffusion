# language-diffusion

This repo is a playground for me to explore modern language generation methods, namely autoregressive and the emerging diffuse language models.

## Project Structure

Below is a repo folder structure diagram:
```
language-diffusion/
├── data/
│   └── AutoregressiveLanguage.py
│   └── DiffuseLanguage.py
├── models/
│   └── LanguageTransformer.py
│   └── utils.py
...
└── README.md
```

## Models

A base decoder-only BERT-style language transformer used for both autoregressive and diffuse language generation is implemented under [`models/LanguageTransformer.py`](./models/LanguageTransformer.py), using transformer-specific modules in [`models/utils.py`](./models/utils.py).

The model can be initialized with the following arguments:
```python
model_config = {
    'embed_dim': 256,
    'num_layers': 8,
    'num_heads': 8
}

model = LanguageTransformer(
    vocab_size=model_config['vocab_len'],
    embed_dim=model_config['embed_dim'],
    num_layers=model_config['num_layers'],
    num_heads=model_config['num_heads'],
    is_causal=True
)
```

The `is_causal` arguments will automatically generate a standard autoregressive causal mask and apply during training if `True`, and omit any masking if `False`. Ensure `emb_dim` is divisible by `num_heads`.

## Autoregressive Language Generation

Autoregressive language generation assumes a conditionally probabilistic interpretation of language generation. Given a sequence of language tokens, the model predicts a probability mass function $p(x_n \mid x_1, x_2, \dots, x_{n-1})$ as the vector output of the $x_{n-1}$ input token. Training such a model results in a GPT-style transformer. Generation can be customized with the following sampling variables:

```python
generation_config = {
    'max_len': 100,
    'temperature': 0.75,
    'top_p': 0.95
}
```

`max_len` is the maximum length of a generated sequence if the model does not terminate prematurely on a `</s>` token. `temperature` modifies the underlying distribution, scaling the weights prior to the softmax. `top_p` denotes the size of the nucleus prior to sampling.

## Discrete Diffuse Language Generation

Diffuse language generation assumes a reverse diffusion interpretation of language generation...
