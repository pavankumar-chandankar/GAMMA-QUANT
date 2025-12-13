# ------------------------------------------------------------------------
# Gamma-Quant related functions and classes.
# ------------------------------------------------------------------------
# Adaption by: Marius Bock
# E-Mail: marius.bock(at)uni-siegen.de
# ------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

    
class UniformQuantizerSTE(nn.Module):
    """
    Uniform quantization with straight-through estimator.

    Args:
        n_bits: int
            Number of bits for quantization.
    Forward:
        x: torch.Tensor
            Input tensor to be quantized.
    """
    def __init__(self, n_bits):
        super().__init__()
        self.n_bits = n_bits
        self.n_levels = 2 ** n_bits

    def forward(self, x):
        # Scale x from [-1, 1] to [0, n_levels - 1]
        x_clipped = torch.clamp(x, -1, 1)
        x_scaled = (x_clipped + 1) * (self.n_levels - 1) / 2
        x_quant = torch.round(x_scaled)
        x_dequant = x_quant * 2 / (self.n_levels - 1) - 1

        # Straight-through estimator: use quantized value in forward, but pass gradients as if identity
        return x + (x_dequant - x).detach()
        

class gammaFunction(nn.Module):
    """
    Gamma function as proposed in paper. 

    Args:
        init: str
            Initialization type of gamma function ('id' for identity, 's_shaped' for s-shaped curve).
        offset: float
            Offset value for the gamma function.
    Forward:
        x_query: torch.Tensor
            Input tensor to be transformed by the gamma function.
    """
    def __init__(self, init = "id", offset=0):
        super().__init__()
        if init == "id":
            self.gamma = nn.Parameter(torch.ones(1))
        elif init == "s_shaped":
            self.gamma = nn.Parameter(0.4*torch.ones(1))
        self.offset = nn.Parameter(offset*torch.ones(1))  

    def forward(self, x_query):
        x_query = torch.clamp(x_query, -1, 1)
        return torch.sign(x_query-self.offset)*(torch.abs(x_query-self.offset) + 1e-3)**self.gamma
