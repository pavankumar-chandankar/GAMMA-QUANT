"""
train2.py

Thin wrappers around train1.py to FORCE DeepConvLSTM to use ContinuousConv2d
(conv_type="continuous") for multi-frequency expert training + cascade testing.

Why wrapper?
- Keeps train1.py stable (your working baseline)
- Lets you retrain 6/12/25 experts with continuous conv without touching train1 logic

Note:
ContinuousConv2d is a layer (not a full model). DeepConvLSTM must support the
conv_type switch internally.
"""

import copy

from models.train1 import run_inertial_expert_702010, test_multifreq_cascade


def _force_continuous_conv(cfg: dict) -> dict:
    """
    Return a deep-copied cfg where the model conv layers are forced to continuous.
    """
    cfg2 = copy.deepcopy(cfg)
    if "model" not in cfg2 or cfg2["model"] is None:
        cfg2["model"] = {}

    # Force continuous conv
    cfg2["model"]["conv_type"] = "continuous"

    # Provide sane defaults if missing
    cfg2["model"].setdefault("conv_rank", 8)
    cfg2["model"].setdefault("conv_mlp_hidden_dim", 32)

    return cfg2


def run_inertial_expert_702010_continuous(
    cfg,
    sbj_list,
    log_dir,
    freq_tag,
    seed,
    test_split_dir,
    device,
    gamma_quant=None,
    quant_bits=4,
    gamma_type="id",
    apply_gamma="global",
    ckpt_freq=-1,
    resume="",
):
    """
    Train one expert (6/12/25 Hz) with 70/20/10 split, using continuous conv blocks.
    """
    cfg2 = _force_continuous_conv(cfg)
    return run_inertial_expert_702010(
        cfg2,
        sbj_list,
        log_dir,
        freq_tag,
        seed,
        test_split_dir,
        device,
        gamma_quant=gamma_quant,
        quant_bits=quant_bits,
        gamma_type=gamma_type,
        apply_gamma=apply_gamma,
        ckpt_freq=ckpt_freq,
        resume=resume,
    )


def test_multifreq_cascade_continuous(
    cfg,
    ckpt_6hz,
    ckpt_12hz,
    ckpt_25hz,
    test_split_dir,
    device,
    gamma_quant=None,
    quant_bits=4,
    gamma_type="id",
    apply_gamma="global",
):
    """
    Run multi-frequency cascade test using experts trained with continuous conv.
    """
    cfg2 = _force_continuous_conv(cfg)
    return test_multifreq_cascade(
        cfg2,
        ckpt_6hz=ckpt_6hz,
        ckpt_12hz=ckpt_12hz,
        ckpt_25hz=ckpt_25hz,
        test_split_dir=test_split_dir,
        device=device,
        gamma_quant=gamma_quant,
        quant_bits=quant_bits,
        gamma_type=gamma_type,
        apply_gamma=apply_gamma,
    )
