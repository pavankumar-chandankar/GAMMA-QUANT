# ------------------------------------------------------------------------
# Activity-specific target sampling rates for power-aware HAR inference
# ------------------------------------------------------------------------
# This module defines three groups of activities that can be sensed at
# different effective sampling rates (50 Hz, 25 Hz and 12.5 Hz).
#
# IMPORTANT:
#   - You MUST adapt FS_50_CLASSES, FS_25_CLASSES and FS_12_5_CLASSES
#     to the integer label IDs used in your dataset (after applying
#     cfg['label_dict']).
#   - The values below are placeholders assuming 18 classes (0..17).
# ------------------------------------------------------------------------

from typing import Dict

# Example grouping for 18 activity classes (0..17).
# EDIT these to match your actual activity semantics.
FS_50_CLASSES = [0, 1, 2, 3, 4, 5]          # high-frequency activities
FS_25_CLASSES = [6, 7, 8, 9, 10, 11]        # medium-frequency activities
FS_12_5_CLASSES = [12, 13, 14, 15, 16, 17]  # low-frequency activities

DEFAULT_FS = 50.0  # Hz


def build_activity_to_fs_map() -> Dict[int, float]:
    """
    Build a dictionary mapping activity label -> target sampling rate (Hz).
    """
    mapping: Dict[int, float] = {}
    for a in FS_50_CLASSES:
        mapping[int(a)] = 50.0
    for a in FS_25_CLASSES:
        mapping[int(a)] = 25.0
    for a in FS_12_5_CLASSES:
        mapping[int(a)] = 12.5
    return mapping


ACTIVITY_TO_FS = build_activity_to_fs_map()


def get_target_sampling_rate(activity_id: int) -> float:
    """
    Given a predicted activity id, return the sensor sampling rate that
    should be used for *future* windows to save power.

    Typical usage on-device (pseudocode):

        current_fs = 50.0  # start at 50 Hz
        while True:
            window = read_window_from_sensor(current_fs)
            logits = net(window)                 # DeepConvLSTM
            pred = argmax(softmax(logits))
            current_fs = get_target_sampling_rate(pred)
            reconfigure_sensor_sampling(current_fs)

    If an activity id is not present in the mapping, DEFAULT_FS (50 Hz)
    is returned.

    Parameters
    ----------
    activity_id : int
        Predicted activity label.

    Returns
    -------
    float
        Target sampling rate in Hz.
    """
    return ACTIVITY_TO_FS.get(int(activity_id), DEFAULT_FS)
