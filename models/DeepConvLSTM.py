# ------------------------------------------------------------------------
# DeepConvLSTM model based on architecture suggested by Ordonez and Roggen
# https://www.mdpi.com/1424-8220/16/1/115
# ------------------------------------------------------------------------
# Adaption by: Marius Bock
# Extended to support:
#   - Gamma-Quant quantization
#   - ContinuousConv2d (time-continuous convolution) via conv_type="continuous"
# ------------------------------------------------------------------------

from torch import nn
import torch

from models.gamma_quant import gammaFunction
from models.gamma_quant import UniformQuantizerSTE
from models.continuous_conv import ContinuousConv2d


class DeepConvLSTM(nn.Module):
    """
    DeepConvLSTM model as described in
        "Deep Convolutional and LSTM Recurrent Neural Networks for
         Multimodal Wearable Activity Recognition"
    and adapted for Gamma-Quant and continuous convolutions.

    Parameters
    ----------
    channels : int
        Number of sensor channels in the input.
    classes : int
        Number of output classes.
    window_size : int
        Length of the input window (time dimension).
    conv_kernels : int, default=64
        Number of convolution kernels in each conv layer.
    conv_kernel_size : int, default=5
        Temporal kernel size (height) of the convolutions.
    lstm_units : int, default=128
        Number of hidden units in the LSTM.
    lstm_layers : int, default=2
        Number of stacked LSTM layers.
    dropout : float, default=0.5
        Dropout rate before the final classifier.
    gamma_quant : {"quantize", "quantize_and_gamma", None}, optional
        If not None, enables quantization (and optionally gamma transform).
    quant_bits : int, default=4
        Number of bits for quantization.
    apply_gamma : {"global", "local"}, default="global"
        If "global", a single gamma function is used for all channels.
        If "local", one gamma function per channel is used.
    gamma_type : {"id", "s_shaped"}, default="id"
        Initialization of the gamma function(s).
    conv_type : {"standard", "continuous"}, default="standard"
        - "standard": use nn.Conv2d with fixed discrete kernels.
        - "continuous": use ContinuousConv2d with time-conditioned kernels.
    conv_rank : int, default=8
        Low-rank dimension for ContinuousConv2d (if used).
    conv_mlp_hidden_dim : int, default=32
        Hidden dimension of the MLP inside ContinuousConv2d (if used).
    """

    def __init__(
        self,
        channels,
        classes,
        window_size,
        conv_kernels=64,
        conv_kernel_size=5,
        lstm_units=128,
        lstm_layers=2,
        dropout=0.5,
        gamma_quant=None,
        quant_bits=4,
        apply_gamma="global",
        gamma_type="id",
        conv_type="standard",
        conv_rank=8,
        conv_mlp_hidden_dim=32,
    ):
        super(DeepConvLSTM, self).__init__()

        self.gamma_quant = gamma_quant
        self.channels = channels
        self.classes = classes
        self.window_size = window_size
        self.conv_kernels = conv_kernels
        self.conv_kernel_size = conv_kernel_size
        self.lstm_units = lstm_units
        self.lstm_layers = lstm_layers
        self.dropout_p = dropout
        self.conv_type = conv_type

        # ------------------------------------------------------------------
        # Convolutional feature extractor
        # ------------------------------------------------------------------
        if conv_type == "standard":
            # Original discrete convolutions (no padding -> shorter sequence)
            self.conv1 = nn.Conv2d(1, conv_kernels, (conv_kernel_size, 1))
            self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
            self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
            self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
            # Effective sequence length after 4 valid convs
            self.final_seq_len = window_size - (conv_kernel_size - 1) * 4
        elif conv_type == "continuous":
            # Time-continuous convs with "same" padding -> preserve sequence length
            self.conv1 = ContinuousConv2d(
                1,
                conv_kernels,
                (conv_kernel_size, 1),
                rank=conv_rank,
                mlp_hidden_dim=conv_mlp_hidden_dim,
                padding="same",
            )
            self.conv2 = ContinuousConv2d(
                conv_kernels,
                conv_kernels,
                (conv_kernel_size, 1),
                rank=conv_rank,
                mlp_hidden_dim=conv_mlp_hidden_dim,
                padding="same",
            )
            self.conv3 = ContinuousConv2d(
                conv_kernels,
                conv_kernels,
                (conv_kernel_size, 1),
                rank=conv_rank,
                mlp_hidden_dim=conv_mlp_hidden_dim,
                padding="same",
            )
            self.conv4 = ContinuousConv2d(
                conv_kernels,
                conv_kernels,
                (conv_kernel_size, 1),
                rank=conv_rank,
                mlp_hidden_dim=conv_mlp_hidden_dim,
                padding="same",
            )
            self.final_seq_len = window_size
        else:
            raise ValueError(f"Unknown conv_type: {conv_type}")

        # LSTM over flattened conv features
        self.lstm = nn.LSTM(
            channels * conv_kernels,
            lstm_units,
            num_layers=lstm_layers,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(lstm_units, classes)
        self.activation = nn.ReLU()

        # ------------------------------------------------------------------
        # Gamma-Quant components (optional)
        # ------------------------------------------------------------------
        self.apply_gamma = apply_gamma
        if self.gamma_quant is not None:
            if apply_gamma == "global":
                # Single gamma function for all channels
                self.gamma_function = gammaFunction(init=gamma_type, offset=0.0)
            elif apply_gamma == "local":
                # One gamma function per channel
                self.gamma_functions = nn.ModuleList(
                    [
                        gammaFunction(init=gamma_type, offset=0.0)
                        for _ in range(channels)
                    ]
                )
            else:
                raise ValueError(f"Unknown apply_gamma mode: {apply_gamma}")

            self.quantizer = UniformQuantizerSTE(n_bits=quant_bits)

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape [batch, seq_len, channels].

        Returns
        -------
        torch.Tensor
            Logits of shape [batch, classes].
        """
        # [B, T, C] -> [B, 1, T, C] for 2D convs
        x = x.unsqueeze(1)  # batch, 1, sequence, axes

        # Optional Gamma-Quant processing
        if self.gamma_quant == "quantize":
            x = self.quantizer(x)
        elif self.gamma_quant == "quantize_and_gamma":
            if hasattr(self, "gamma_functions"):
                # Per-axis gamma
                x_split = torch.split(x, 1, dim=-1)  # list of (B, 1, T, 1)
                x = torch.cat(
                    [
                        self.gamma_functions[i](x_split[i])
                        for i in range(len(x_split))
                    ],
                    dim=-1,
                )
                x = self.quantizer(x)
            else:
                # Global gamma
                x = self.quantizer(self.gamma_function(x))

        # Convolutional feature extractor
        x = self.activation(self.conv1(x))  # [B, K, T, C]
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        # Prepare for LSTM: [B, K, T, C] -> [T, B, C*K]
        x = x.permute(2, 0, 3, 1)  # [T, B, C, K]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # [T, B, C*K]

        # LSTM over time
        x, _ = self.lstm(x)
        x = x[-1, :, :]  # last time step: [B, H]
        x = x.view(-1, self.lstm_units)
        x = self.dropout(x)
        return self.classifier(x)
