from typing import List, Optional, Tuple, Dict

import torch
from torch import nn
import torch.nn.functional as F

from models.DeepConvLSTM import DeepConvLSTM


class MultiFreqDeepConvLSTM(nn.Module):
    """
    Wrapper model that holds three DeepConvLSTM experts, one per sampling rate
    (25 Hz, 12 Hz, 6 Hz), and an oracle-like gating mechanism that can
    dynamically choose which expert to use based on the past predictions.

    Notes
    -----
    - Each expert is a standard DeepConvLSTM instance.
    - The gating/oracle is *not* trainable; it is a deterministic rule that
      uses a sliding window of the last K predictions and a per-class mapping
      from activity -> preferred frequency.
    - For simplicity, the `step(...)` interface assumes batch_size = 1 during
      evaluation. For larger batch sizes, call it in a loop.
    """

    def __init__(
        self,
        model_kwargs_25: Dict,
        model_kwargs_12: Dict,
        model_kwargs_6: Dict,
        freq_for_class: List[int],
        history_len: int = 5,
        majority_threshold: float = 0.6,
        confidence_threshold: float = 0.7,
    ):
        """
        Parameters
        ----------
        model_kwargs_25, model_kwargs_12, model_kwargs_6 : dict
            Keyword arguments passed to the DeepConvLSTM constructor for each
            frequency-specific expert. All three must produce the same number
            of output classes.
        freq_for_class : list of int
            List of length C (number of classes) where each entry is one of
            {6, 12, 25}, specifying the preferred sampling rate for that class,
            as obtained from the per-class F1 analysis.
        history_len : int
            Number of past predictions to consider when computing the majority
            class and average confidence.
        majority_threshold : float
            Minimum fraction of past predictions that must agree on the same
            class to consider the history "stable".
        confidence_threshold : float
            Minimum mean softmax-max confidence over the history to trust the
            majority class. Otherwise, the gate falls back to 25 Hz.
        """
        super().__init__()

        # Instantiate the three experts
        self.model_25 = DeepConvLSTM(**model_kwargs_25)
        self.model_12 = DeepConvLSTM(**model_kwargs_12)
        self.model_6 = DeepConvLSTM(**model_kwargs_6)

        # Sanity check: all experts must have the same number of classes
        self.n_classes = model_kwargs_25.get("classes", None)
        if self.n_classes is None:
            raise ValueError("model_kwargs_25 must contain a 'classes' key.")
        if model_kwargs_12.get("classes", None) != self.n_classes:
            raise ValueError("All experts must have the same 'classes' parameter.")
        if model_kwargs_6.get("classes", None) != self.n_classes:
            raise ValueError("All experts must have the same 'classes' parameter.")

        if len(freq_for_class) != self.n_classes:
            raise ValueError(
                f"freq_for_class must have length {self.n_classes}, "
                f"but got {len(freq_for_class)}."
            )

        # Store per-class preferred frequency as a buffer (no gradients)
        self.register_buffer(
            "freq_for_class",
            torch.as_tensor(freq_for_class, dtype=torch.long),
            persistent=False,
        )

        # Gating / oracle hyperparameters
        self.history_len = int(history_len)
        self.majority_threshold = float(majority_threshold)
        self.confidence_threshold = float(confidence_threshold)

        # Internal history buffers; we keep them as Python lists and not as
        # tensors, because they are small and not part of the computational graph.
        self._history_classes: List[int] = []
        self._history_conf: List[float] = []

    # ------------------------------------------------------------------
    # Helper functions for the oracle / gating
    # ------------------------------------------------------------------
    def reset_history(self) -> None:
        """
        Clear the stored prediction history. Call this at the beginning
        of each new sequence or test run.
        """
        self._history_classes = []
        self._history_conf = []

    def _update_history(self, cls: int, conf: float) -> None:
        """
        Append a new prediction and confidence value to the history,
        truncating it to the configured history_len.
        """
        self._history_classes.append(int(cls))
        self._history_conf.append(float(conf))

        if len(self._history_classes) > self.history_len:
            self._history_classes.pop(0)
            self._history_conf.pop(0)

    def _decide_frequency(self) -> int:
        """
        Decide which frequency to use for the *next* step based on the
        history of past predictions.

        Returns
        -------
        int
            One of {25, 12, 6}. If the history is too short or not stable /
            confident enough, this falls back to 25 Hz.
        """
        # If not enough history, or no history at all, use 25 Hz as the safe default
        if len(self._history_classes) < 1:
            return 25

        # Compute majority class and its fraction
        classes = torch.as_tensor(self._history_classes)
        unique, counts = torch.unique(classes, return_counts=True)
        majority_idx = torch.argmax(counts)
        majority_class = int(unique[majority_idx])
        majority_fraction = float(counts[majority_idx].item()) / float(len(classes))

        mean_conf = float(sum(self._history_conf) / max(len(self._history_conf), 1))

        stable_enough = majority_fraction >= self.majority_threshold
        confident_enough = mean_conf >= self.confidence_threshold

        if stable_enough and confident_enough:
            # Use per-class oracle mapping
            # freq_for_class is in {6, 12, 25}
            freq = int(self.freq_for_class[majority_class].item())
            if freq in (6, 12, 25):
                return freq

        # Fallback
        return 25

    # ------------------------------------------------------------------
    # Forward functions
    # ------------------------------------------------------------------
    def forward_all(
        self,
        x_25: torch.Tensor,
        x_12: Optional[torch.Tensor] = None,
        x_6: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Run all three experts and return their logits.

        Parameters
        ----------
        x_25, x_12, x_6 : torch.Tensor
            Input windows for each expert, with shape (B, T, C).
            For convenience, x_12 and x_6 default to x_25 if omitted.

        Returns
        -------
        logits_25, logits_12, logits_6 : torch.Tensor
            Logits of shape (B, n_classes) for each expert.
        """
        if x_12 is None:
            x_12 = x_25
        if x_6 is None:
            x_6 = x_25

        logits_25 = self.model_25(x_25)
        logits_12 = self.model_12(x_12)
        logits_6 = self.model_6(x_6)

        return logits_25, logits_12, logits_6

    def forward(
        self,
        x_25: torch.Tensor,
        x_12: Optional[torch.Tensor] = None,
        x_6: Optional[torch.Tensor] = None,
        freq_choice: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Standard forward interface.

        Parameters
        ----------
        x_25, x_12, x_6 : torch.Tensor
            Input windows for each frequency, shape (B, T, C).
        freq_choice : torch.Tensor, optional
            Tensor of shape (B,) with entries in {6, 12, 25}. If provided,
            only the corresponding expert's logits are returned for each
            batch element. If None, all experts are run and the logits for
            25 Hz are returned (to keep the default behaviour simple).

        Returns
        -------
        torch.Tensor
            Logits of shape (B, n_classes).
        """
        logits_25, logits_12, logits_6 = self.forward_all(x_25, x_12, x_6)

        if freq_choice is None:
            # Default: behave like a plain 25 Hz model
            return logits_25

        # freq_choice is a per-sample indicator in {6, 12, 25}
        if freq_choice.dim() != 1 or freq_choice.shape[0] != logits_25.shape[0]:
            raise ValueError(
                "freq_choice must be 1D of shape (B,), with values in {6,12,25}."
            )

        out_logits = []
        for i, f in enumerate(freq_choice.tolist()):
            if f == 25:
                out_logits.append(logits_25[i : i + 1])
            elif f == 12:
                out_logits.append(logits_12[i : i + 1])
            elif f == 6:
                out_logits.append(logits_6[i : i + 1])
            else:
                raise ValueError(f"Unsupported frequency value in freq_choice: {f}")
        return torch.cat(out_logits, dim=0)

    # ------------------------------------------------------------------
    # Step-wise interface for cascading during evaluation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def step(
        self,
        x_25: torch.Tensor,
        x_12: Optional[torch.Tensor] = None,
        x_6: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, int, float, int]:
        """
        Single time-step evaluation with cascading oracle.

        This method:
        1) Runs all three experts for the current window.
        2) Decides which frequency to use based on the stored history.
        3) Returns the logits from the selected expert, the predicted class,
           the max-softmax confidence, and the chosen frequency.

        Parameters
        ----------
        x_25, x_12, x_6 : torch.Tensor
            Input windows for each expert (B, T, C). For cascading behaviour,
            this is intended for B = 1. If B > 1, the same oracle decision
            will be used for all elements in the batch.

        Returns
        -------
        logits_selected : torch.Tensor
            Logits from the selected expert, shape (B, n_classes).
        y_hat : int
            Predicted class index (for B=1).
        conf : float
            Maximum softmax probability (for B=1).
        freq_used : int
            Frequency (6, 12, or 25) selected by the oracle.
        """
        logits_25, logits_12, logits_6 = self.forward_all(x_25, x_12, x_6)

        freq_used = self._decide_frequency()

        if freq_used == 25:
            logits = logits_25
        elif freq_used == 12:
            logits = logits_12
        elif freq_used == 6:
            logits = logits_6
        else:
            raise ValueError(f"Unsupported frequency selected by oracle: {freq_used}")

        # Softmax and prediction (assume B=1 for the oracle statistics)
        probs = F.softmax(logits, dim=-1)
        conf, y_hat = torch.max(probs, dim=-1)

        # For history, we only track the first element of the batch
        self._update_history(int(y_hat[0].item()), float(conf[0].item()))

        return logits, int(y_hat[0].item()), float(conf[0].item()), freq_used


def build_multifreq_from_single_cfg(
    cfg: Dict,
    freq_for_class: List[int],
    conv_type: str = "continuous",
    conv_rank: int = 8,
) -> MultiFreqDeepConvLSTM:
    """
    Convenience helper to construct a MultiFreqDeepConvLSTM from a single
    config dictionary `cfg` that already contains the common architecture
    hyperparameters.

    This assumes that:
    - All three experts share the same architecture settings (kernels, units, etc.).
    - Only the data pipeline / sampling rate differs between 25 / 12 / 6 Hz.

    Parameters
    ----------
    cfg : dict
        Full experiment configuration.
    freq_for_class : list of int
        Per-class preferred frequency, as in the table you computed.
    conv_type : str
        Passed through to DeepConvLSTM (e.g. "continuous" or "standard").
    conv_rank : int
        Rank parameter for ContinuousConv2d, if used.

    Returns
    -------
    MultiFreqDeepConvLSTM
        Instantiated multi-frequency model.
    """
    dataset_cfg = cfg["dataset"]
    net_cfg = cfg["network"]

    common_kwargs = dict(
        channels=dataset_cfg["input_dim"],
        classes=dataset_cfg["n_classes"],
        window_size=dataset_cfg["window_size"],
        conv_kernels=net_cfg.get("conv_kernels", 64),
        conv_kernel_size=net_cfg.get("conv_kernel_size", 5),
        lstm_units=net_cfg.get("lstm_units", 128),
        gamma_quant=net_cfg.get("gamma_quant", None),
        quant_bits=net_cfg.get("quant_bits", 4),
        apply_gamma=net_cfg.get("apply_gamma", "global"),
        gamma_type=net_cfg.get("gamma_type", "id"),
        conv_type=conv_type,
        conv_rank=conv_rank,
    )

    model_kwargs_25 = common_kwargs.copy()
    model_kwargs_12 = common_kwargs.copy()
    model_kwargs_6 = common_kwargs.copy()

    return MultiFreqDeepConvLSTM(
        model_kwargs_25=model_kwargs_25,
        model_kwargs_12=model_kwargs_12,
        model_kwargs_6=model_kwargs_6,
        freq_for_class=freq_for_class,
    )
