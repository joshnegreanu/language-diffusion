import os
import torch
import math
import torch.nn as nn

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# dynamically select device
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

# to be written later...