# ------------------------------------------------------------------------
# Frequency cascade controller for multi-rate HAR inference
# ------------------------------------------------------------------------
#   - You have three DeepConvLSTM models:
#       model_low  : trained on 12.5 Hz data
#       model_mid  : trained on 25   Hz data
#       model_high : trained on 50   Hz data
#   - All three models see the same label space (all activities).
#   - Each activity has a *preferred* frequency level:
#       class -> "low" | "mid" | "high"
#   - During inference, you:
#       1) Start at LOW (12.5 Hz) model to save power.
#       2) If confidence is LOW -> escalate to MID, then HIGH.
#       3) If confidence is HIGH and the predicted class prefers a LOWER
#          level than the current one, you can *fall back* (after some
#          stable predictions) to that lower frequency.
#
# This file implements ONLY the controller logic.
# You must:
#   - Train and load model_low / model_mid / model_high separately.
#   - Provide class_to_pref_level = {class_id: "low"|"mid"|"high"}.
#   - Call .step(window) with a window sampled at the CURRENT frequency.
# ------------------------------------------------------------------------

from typing import Optional, Tuple, Dict

import torch
import torch.nn.functional as F


FreqLevel = str  # one of {"low", "mid", "high"}


def _level_rank(level: FreqLevel) -> int:
    """Map level string to an integer rank for comparisons."""
    if level == "low":
        return 0
    if level == "mid":
        return 1
    if level == "high":
        return 2
    raise ValueError(f"Unknown freq level '{level}' (expected 'low'|'mid'|'high').")


class FrequencyCascadeController:
    """
    Controller that manages three HAR models at different sampling frequencies
    and performs conditional computation based on confidence + class->freq mapping.

    Behaviour:
        - Start at low frequency ("low" = 12.5 Hz).
        - If confidence is below an upward threshold, escalate:
              low -> mid  (12.5 -> 25 Hz)
              mid -> high (25   -> 50 Hz)
        - If confidence is high and the predicted class prefers a lower level,
          then (after some consecutive stable predictions) fall back (downsample)
          to that preferred level.

    Typical usage (pseudo-code):

        model_low  = load_model_12_5hz(...)
        model_mid  = load_model_25hz(...)
        model_high = load_model_50hz(...)

        CLASS_TO_PREF = {
            # e.g. 0: lying, 1: sitting, 2: standing, 3: walking, 4: running, ...
            0: "low",
            1: "low",
            2: "low",
            3: "mid",
            4: "high",
            # ...
        }

        controller = FrequencyCascadeController(
            model_low=model_low,
            model_mid=model_mid,
            model_high=model_high,
            class_to_pref_level=CLASS_TO_PREF,
            threshold_up_low=0.80,   # low -> mid
            threshold_up_mid=0.80,   # mid -> high
            threshold_down=0.90,     # need higher confidence to downsample
            down_stable_windows=3,   # require 3 consecutive stable predictions
            fs_low=12.5,
            fs_mid=25.0,
            fs_high=50.0,
        )

        controller.reset(start_level="low")

        while True:
            current_fs = controller.get_current_fs()
            window = read_sensor_window_at(current_fs)  # shape [T, C]

            pred, conf, level = controller.step(window)

            # Use pred, conf, and new level
            use_prediction(pred, conf, level)

            # If level changed, reconfigure sensor rate
            new_fs = controller.get_current_fs()
            if new_fs != current_fs:
                reconfigure_sensor_sampling_rate(new_fs)
    """

    def __init__(
        self,
        model_low: torch.nn.Module,
        model_mid: torch.nn.Module,
        model_high: torch.nn.Module,
        class_to_pref_level: Dict[int, FreqLevel],
        threshold_up_low: float = 0.8,
        threshold_up_mid: float = 0.8,
        threshold_down: float = 0.9,
        down_stable_windows: int = 3,
        device: Optional[torch.device] = None,
        fs_low: float = 12.5,
        fs_mid: float = 25.0,
        fs_high: float = 50.0,
    ):
        """
        Parameters
        ----------
        model_low : nn.Module
            Model trained for the lowest sampling frequency (e.g., 12.5 Hz).
        model_mid : nn.Module
            Model trained for the middle sampling frequency (e.g., 25 Hz).
        model_high : nn.Module
            Model trained for the highest sampling frequency (e.g., 50 Hz).
        class_to_pref_level : dict[int, {"low","mid","high"}]
            Mapping from activity class-id to preferred frequency level.
            Any class not present in this dict defaults to "high".
        threshold_up_low : float
            Confidence threshold for escalating from low -> mid.
        threshold_up_mid : float
            Confidence threshold for escalating from mid -> high.
        threshold_down : float
            Confidence threshold required (or higher) to allow falling
            back to a lower preferred level.
        down_stable_windows : int
            Number of consecutive windows with the same predicted class
            and enough confidence before actually falling back.
        device : torch.device or None
            Device to run models on. If None, uses cuda if available, else cpu.
        fs_low, fs_mid, fs_high : float
            Sampling frequencies (Hz) for the three levels.
        """
        super().__init__()

        self.models: Dict[FreqLevel, torch.nn.Module] = {
            "low": model_low,
            "mid": model_mid,
            "high": model_high,
        }

        # Device setup
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        for m in self.models.values():
            m.to(self.device)
            m.eval()

        # Class -> preferred freq-level
        self.class_to_pref_level = dict(class_to_pref_level)

        # Upward thresholds (for escalation)
        self.threshold_up_low = float(threshold_up_low)
        self.threshold_up_mid = float(threshold_up_mid)

        # Downward threshold (for fallback)
        self.threshold_down = float(threshold_down)
        self.down_stable_windows = int(down_stable_windows)

        # Mapping from level to sampling frequency
        self.fs_map: Dict[FreqLevel, float] = {
            "low": float(fs_low),
            "mid": float(fs_mid),
            "high": float(fs_high),
        }

        # State variables
        self.current_level: FreqLevel = "low"
        self.seen_windows: int = 0

        # For downward hysteresis: keep track of last stable prediction
        self._last_pred_class: Optional[int] = None
        self._stable_count: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, start_level: FreqLevel = "low") -> None:
        """
        Reset the controller state at the beginning of a new sequence.

        Parameters
        ----------
        start_level : {"low", "mid", "high"}
            Initial frequency level to use. Usually "low" (e.g., 12.5 Hz).
        """
        if start_level not in self.models:
            raise ValueError(f"Unknown start_level '{start_level}' (expected 'low'|'mid'|'high').")
        self.current_level = start_level
        self.seen_windows = 0
        self._last_pred_class = None
        self._stable_count = 0

    def get_current_level(self) -> FreqLevel:
        """Return the current frequency level: 'low', 'mid', or 'high'."""
        return self.current_level

    def get_current_fs(self) -> float:
        """Return the sampling frequency (Hz) corresponding to the current level."""
        return self.fs_map[self.current_level]

    @torch.no_grad()
    def step(self, window) -> Tuple[int, float, FreqLevel]:
        """
        Process one window at the current frequency level.

        Parameters
        ----------
        window : numpy array or torch.Tensor
            Shape [T, C] or [1, T, C], already sampled at the CURRENT frequency
            (12.5, 25, or 50 Hz depending on self.current_level).

        Returns
        -------
        pred_class : int
            Predicted class index.
        confidence : float
            Maximum softmax probability.
        new_level : {"low", "mid", "high"}
            Frequency level to be used for the NEXT window. This can be higher
            (escalation) or lower (fallback) than the current level, depending
            on confidence and the preferred level of the predicted class.
        """
        x = self._prepare_input(window)
        level = self.current_level
        model = self.models[level]

        # Forward pass
        logits = model(x)  # [1, num_classes] or [num_classes]
        if logits.ndim == 1:
            probs = F.softmax(logits, dim=0)
        else:
            probs = F.softmax(logits, dim=-1)[0]

        conf, pred = torch.max(probs, dim=-1)
        confidence = float(conf.item())
        pred_class = int(pred.item())

        # Preferred level for this predicted class
        pref_level = self.class_to_pref_level.get(pred_class, "high")

        # --- Upward transitions (escalation) ---
        new_level = level  # default: stay

        if level == "low":
            if confidence < self.threshold_up_low:
                new_level = "mid"
        elif level == "mid":
            if confidence < self.threshold_up_mid:
                new_level = "high"

        # --- Downward transitions (fallback) ---
        # Only if:
        #   - we did NOT just escalate,
        #   - confidence is high enough (>= threshold_down),
        #   - preferred level is strictly LOWER than current one.
        if new_level == level and confidence >= self.threshold_down:
            current_rank = _level_rank(level)
            pref_rank = _level_rank(pref_level)

            if pref_rank < current_rank:
                # Potential fallback candidate
                if self._last_pred_class == pred_class:
                    self._stable_count += 1
                else:
                    self._stable_count = 1
                    self._last_pred_class = pred_class

                if self._stable_count >= self.down_stable_windows:
                    # Fall all the way down to the preferred level
                    new_level = pref_level
                    self._stable_count = 0  # reset after transition
            else:
                # Preferred level is >= current, no fallback
                self._stable_count = 0
                self._last_pred_class = pred_class
        else:
            # We escalated or confidence too low for fallback; reset stability
            self._stable_count = 0
            self._last_pred_class = pred_class

        # Update controller state
        self.current_level = new_level
        self.seen_windows += 1

        return pred_class, confidence, self.current_level

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prepare_input(self, window) -> torch.Tensor:
        """
        Convert input window to a tensor on the correct device.

        Expects:
            - window: numpy array or torch.Tensor of shape [T, C] or [1, T, C]

        Returns:
            - x: torch.Tensor of shape [1, T, C] on self.device
        """
        if isinstance(window, torch.Tensor):
            x = window
        else:
            x = torch.tensor(window, dtype=torch.float32)

        if x.ndim == 2:
            # [T, C] -> [1, T, C]
            x = x.unsqueeze(0)
        elif x.ndim == 3:
            # [B, T, C]
            if x.shape[0] != 1:
                raise ValueError(
                    f"Expected batch size 1 for cascade, got batch dimension {x.shape[0]}."
                )
        else:
            raise ValueError(
                f"Expected window with 2 or 3 dimensions, got shape {tuple(x.shape)}."
            )

        return x.to(self.device)
