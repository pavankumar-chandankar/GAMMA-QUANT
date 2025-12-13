from typing import Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContinuousConv2d(nn.Module):
    """
    2D convolution where the temporal kernel weights are generated
    as a continuous function of (normalized) time positions.

    Expected input shape: [B, C_in, T, W]

    Currently supports only kernels of shape (k_t, 1), i.e. convolution
    along the temporal dimension.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        rank: int = 8,
        mlp_hidden_dim: int = 32,
        bias: bool = True,
        padding: str = "valid",
    ) -> None:
        super().__init__()

        # Enforce kernel_size = (k_t, 1) or int
        if isinstance(kernel_size, tuple):
            assert len(kernel_size) == 2, "kernel_size must be int or (k_t, 1)"
            assert (
                kernel_size[1] == 1
            ), "ContinuousConv2d currently only supports kernel_size=(k_t, 1)"
            self.kernel_size_time = int(kernel_size[0])
        else:
            self.kernel_size_time = int(kernel_size)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.rank = rank
        self.mlp_hidden_dim = mlp_hidden_dim
        assert padding in ("valid", "same"), 'padding must be "valid" or "same"'
        self.padding_mode = padding

        # MLP that maps (time position) -> rank-dimensional embedding
        self.mlp = nn.Sequential(
            nn.Linear(1, mlp_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(mlp_hidden_dim, rank),
        )

        # Learnable per-(out,in) channel low-rank weights
        # Shape: [out_channels, in_channels, rank]
        self.channel_weights = nn.Parameter(
            torch.randn(out_channels, in_channels, rank) * 0.1
        )

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channels))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, C_in, T, W]
        """
        if x.dim() != 4:
            raise ValueError(
                f"ContinuousConv2d expected 4D input [B, C_in, T, W], got shape {tuple(x.shape)}"
            )

        B, C_in, T, W = x.shape

        # Positions along the kernel support (normalized -1 to 1)
        positions = torch.linspace(
            -1.0,
            1.0,
            self.kernel_size_time,
            device=x.device,
            dtype=x.dtype,
        )
        pos = positions.unsqueeze(-1)  # [k_t, 1]

        # MLP: [k_t, 1] -> [k_t, rank]
        g = self.mlp(pos)  # [k_t, rank]

        # Combine low-rank factors:
        # channel_weights: [O, I, R]
        # g:               [k_t, R]
        # -> K_t:          [O, I, k_t]
        K_t = torch.einsum("oir,kr->oik", self.channel_weights, g)

        # Expand to 2D conv kernel: [O, I, k_t, 1]
        K = K_t.unsqueeze(-1)

        # Padding only in time dimension
        if self.padding_mode == "same":
            pad_t = self.kernel_size_time // 2
        else:
            pad_t = 0

        y = F.conv2d(x, K, bias=self.bias, padding=(pad_t, 0))
        return y
