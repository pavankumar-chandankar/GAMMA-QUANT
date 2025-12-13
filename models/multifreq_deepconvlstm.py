from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .DeepConvLSTM import DeepConvLSTM
except ImportError:  # pragma: no cover
    from DeepConvLSTM import DeepConvLSTM  # type: ignore


# Per-class best frequency (from your final mean-F1 table)
FREQ_FOR_CLASS: List[str] = [
    "6hz",   # 0: null
    "25hz",  # 1: bench-dips
    "25hz",  # 2: burpees
    "6hz",   # 3: jogging
    "25hz",  # 4: jogging (butt-kicks)
    "25hz",  # 5: jogging (rotating arms)
    "25hz",  # 6: jogging (sidesteps)
    "25hz",  # 7: jogging (skipping)
    "12hz",  # 8: lunges
    "12hz",  # 9: lunges (complex)
    "25hz",  # 10: push-ups
    "25hz",  # 11: push-ups (complex)
    "25hz",  # 12: sit-ups
    "12hz",  # 13: sit-ups (complex)
    "6hz",   # 14: stretching (hamstrings)
    "6hz",   # 15: stretching (lumbar rotation)
    "12hz",  # 16: stretching (lunging)
    "12hz",  # 17: stretching (shoulders)
    "12hz",  # 18: stretching (triceps)
]


class MultiFreqDeepConvLSTM(nn.Module):
    """
    Wrapper that holds three DeepConvLSTM experts (6 / 12 / 25 Hz) and
    implements the oracle-like cascade:

      * Maintain a history of predictions and confidences.
      * If history is stable and confident, pick the expert suggested by
        FREQ_FOR_CLASS[majority_class].
      * Otherwise, fall back to the 25 Hz expert.
    """

    def __init__(
        self,
        model_6hz: nn.Module,
        model_12hz: nn.Module,
        model_25hz: nn.Module,
        freq_for_class: Optional[Sequence[str]] = None,
        history_len: int = 5,
        stability_threshold: float = 0.8,
        min_confidence: float = 0.6,
    ) -> None:
        super().__init__()

        self.model_6hz = model_6hz
        self.model_12hz = model_12hz
        self.model_25hz = model_25hz

        self.freq_for_class: List[str] = list(freq_for_class) if freq_for_class is not None else list(FREQ_FOR_CLASS)

        self.history_len = history_len
        self.stability_threshold = stability_threshold
        self.min_confidence = min_confidence

        self.reset_history()

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------
    def reset_history(self) -> None:
        """Reset stored predictions and confidences (start of a new sequence)."""
        self.history_classes: List[int] = []
        self.history_confidences: List[float] = []

    def _update_history(self, classes, confidences) -> None:
        for cls, conf in zip(classes, confidences):
            self.history_classes.append(int(cls))
            self.history_confidences.append(float(conf))

            if len(self.history_classes) > self.history_len:
                self.history_classes.pop(0)
            if len(self.history_confidences) > self.history_len:
                self.history_confidences.pop(0)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _softmax_confidence(logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute softmax probabilities and max confidence per sample.

        logits : (B, num_classes)
        """
        probs = F.softmax(logits, dim=1)
        conf, _ = probs.max(dim=1)
        return probs, conf

    def _choose_frequency_from_history(self) -> str:
        """
        Decide which expert to use based on the recent history.

        If not enough history, or unstable / low confidence, we pick 25 Hz.
        """
        if len(self.history_classes) == 0:
            return "25hz"

        classes = torch.tensor(self.history_classes, dtype=torch.long)
        majority_class = int(torch.mode(classes).values.item())
        stability = (classes == majority_class).float().mean().item()

        if self.history_confidences:
            avg_conf = float(sum(self.history_confidences) / len(self.history_confidences))
        else:
            avg_conf = 0.0

        if stability >= self.stability_threshold and avg_conf >= self.min_confidence:
            if 0 <= majority_class < len(self.freq_for_class):
                return self.freq_for_class[majority_class]
            else:
                return "25hz"
        else:
            return "25hz"

    # ------------------------------------------------------------------
    # Main step for evaluation
    # ------------------------------------------------------------------
    def step(
        self,
        x_25hz: torch.Tensor,
        x_12hz: torch.Tensor,
        x_6hz: torch.Tensor,
    ):
        """
        Process a batch of windows using the cascade logic.

        x_25hz : (B, T_25, C)
        x_12hz : (B, T_12, C)
        x_6hz  : (B, T_6, C)

        Returns:
            logits       : (B, num_classes)
            chosen_freqs : list[str] of length B
        """
        assert x_25hz.size(0) == x_12hz.size(0) == x_6hz.size(0), "Batch sizes must match."

        device = next(self.parameters()).device
        x_25hz = x_25hz.to(device)
        x_12hz = x_12hz.to(device)
        x_6hz = x_6hz.to(device)

        batch_size = x_25hz.size(0)
        all_logits = []
        chosen_freqs: List[str] = []

        # Process sequentially so temporal history is meaningful
        for i in range(batch_size):
            freq = self._choose_frequency_from_history()

            if freq == "6hz":
                logits_i = self.model_6hz(x_6hz[i : i + 1])
            elif freq == "12hz":
                logits_i = self.model_12hz(x_12hz[i : i + 1])
            else:
                logits_i = self.model_25hz(x_25hz[i : i + 1])

            _, conf_i = self._softmax_confidence(logits_i)
            pred_i = torch.argmax(logits_i, dim=1)

            self._update_history(pred_i.cpu().tolist(), conf_i.cpu().tolist())

            all_logits.append(logits_i)
            chosen_freqs.append(freq)

        return torch.cat(all_logits, dim=0), chosen_freqs
