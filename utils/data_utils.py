# ------------------------------------------------------------------------
# Data operation utilities
# ------------------------------------------------------------------------
# Adaption by: Marius Bock
# E-Mail: marius.bock(at)uni-siegen.de
# ------------------------------------------------------------------------

import os
import numpy as np
import pandas as pd


def sliding_window_samples(data, win_len, overlap_ratio=None):
    """
    Return a sliding window measured in seconds over a data array.
    
    Args:
        data: numpy array
            Input data
        win_len: int
            Window length in samples
        overlap_ratio: int, optional
            Overlap ratio in percent (default is None)
            
    Returns:
        windows: numpy array
            Sliding windows
        indices: numpy array
            Indices of the sliding windows
    """
    windows = []
    indices = []
    curr = 0
    overlapping_elements = 0

    if overlap_ratio is not None:
        if not ((overlap_ratio / 100) * win_len).is_integer():
            float_prec = True
        else:
            float_prec = False
        overlapping_elements = int((overlap_ratio / 100) * win_len)
        if overlapping_elements >= win_len:
            print('Number of overlapping elements exceeds window size.')
            return
    changing_bool = True
    while curr < len(data) - win_len:
        windows.append(data[curr:curr + win_len])
        indices.append([curr, curr + win_len])
        if (float_prec == True) and (changing_bool == True):
            curr = curr + win_len - overlapping_elements - 1
            changing_bool = False
        else:
            curr = curr + win_len - overlapping_elements
            changing_bool = True

    return np.array(windows), np.array(indices)



def apply_sliding_window(data, window_size, window_overlap, no_context_windows=0):
    """
    Apply a sliding window to the data.
    
    Args:
        data: numpy array
            Input data
        window_size: int
            Window size
        window_overlap: int
            Window overlap
        no_context_windows: int, optional
            Number of context windows to apply (default is 0)
            
    Returns:
        output_sbj: numpy array
            Subject IDs
        output_x: numpy array
            Sliding windows
        output_y: numpy array
            Labels
    """
    output_x = None
    output_y = None
    output_sbj = []
    if no_context_windows > 0:
        look_up_windows = None
    for i, subject in enumerate(np.unique(data[:, 0])):
        subject_data = data[data[:, 0] == subject]
        subject_x, subject_y = subject_data[:, :-1], subject_data[:, -1]
        tmp_x, _ = sliding_window_samples(subject_x, window_size, window_overlap)
        tmp_y, _ = sliding_window_samples(subject_y, window_size, window_overlap)
        if no_context_windows > 0:
            tmp_lookup = tmp_x
            tmp_x = tmp_x[no_context_windows - 1:]
            tmp_y = tmp_y[no_context_windows - 1:]
        
        if output_x is None:
            if no_context_windows > 0:
                look_up_windows = tmp_lookup
            output_x = tmp_x
            output_y = tmp_y
            output_sbj = np.full(len(tmp_y), subject)
        else:
            if no_context_windows > 0:
                look_up_windows = np.concatenate((look_up_windows, tmp_lookup), axis=0)
            output_x = np.concatenate((output_x, tmp_x), axis=0)
            output_y = np.concatenate((output_y, tmp_y), axis=0)
            output_sbj = np.concatenate((output_sbj, np.full(len(tmp_y), subject)), axis=0)

    output_y = [[i[-1]] for i in output_y]
    if no_context_windows > 0:
        return output_sbj, output_x, look_up_windows, np.array(output_y).flatten()
    else:
        return output_sbj, output_x, np.array(output_y).flatten()


def apply_context_sliding_window(data, no_context_windows, window_size, window_overlap, dataset_path='dataset'):
    """
    Apply a sliding window to the data with context.

    :param data: numpy array
        Input data
    :param no_context_windos: int
        number of context windows to apply
    :param window_size: int
        Window size
    :param window_overlap: int
        Window overlap
    :param dataset_path: str
        Path to the dataset
    :return: tuple of windows and indices
    """
    context_sbj, context_x, context_y = None, None, None
    for i, subject in enumerate(np.unique(data[:, 0])):
        subject_data = data[data[:, 0] == subject]
        subject_x, subject_y = subject_data[:, :-1], subject_data[:, -1]
        tmp_x, _ = sliding_window_samples(subject_x, window_size, window_overlap)
        tmp_y, _ = sliding_window_samples(subject_y, window_size, window_overlap)
        tmp_y = [[i[-1]] for i in tmp_y]
        sbj_x = None
        sbj_y = None
        if os.path.exists(os.path.join(dataset_path, f'{subject}.npy')):
            sbj_x = np.load(os.path.join(dataset_path, f'{subject}_cw{no_context_windows}.npy'))
            sbj_y = np.load(os.path.join(dataset_path, f'{subject}_cw{no_context_windows}_y.npy'))
            print('Loading context windows from file')
        else:
            for j in range(len(tmp_x) - no_context_windows + 1):
                print('j: ', j)
                if j < no_context_windows:
                    continue
                else:
                    curr_context_windows = tmp_x[j]
                    curr_y = tmp_y[j]
                # create context windows
                if sbj_x is None:
                    sbj_x = curr_context_windows[None, ...]
                    sbj_y = curr_y
                else:
                    sbj_x = np.concatenate((sbj_x, curr_context_windows[None, ...]), axis=0)
                    sbj_y = np.append(sbj_y, curr_y)
            np.save(os.path.join(dataset_path, f'{subject}_cw{no_context_windows}.npy'), sbj_x)
            np.save(os.path.join(dataset_path, f'{subject}_cw{no_context_windows}_y.npy'), sbj_y)
        if context_y is None:
            context_x = sbj_x
            context_y = sbj_y
            context_sbj = np.full(len(sbj_y), subject)
        else:
            context_x = np.concatenate((context_x, sbj_x), axis=0)
            context_y = np.concatenate((context_y, sbj_y), axis=0)
            context_sbj = np.concatenate((context_sbj, np.full(len(sbj_y), subject)), axis=0)
    return context_sbj, context_x, np.array(context_y).flatten()


def unwindow_inertial_data(orig, ids, preds, win_size, win_overlap):
    """
    Method to unwindow the predictions of the model.
    
    Args:
        orig: numpy array
            Original data
        ids: numpy array
            Subject IDs
        preds: numpy array
            Predictions
        win_size: int
            Window size
        win_overlap: int
            Window overlap
            
    Returns:
        unseg_preds: numpy array
            Unsegmented predictions
        orig_labels: numpy array
            Original labels
    """
    unseg_preds = []

    if not ((win_overlap / 100) * win_size).is_integer():
        float_prec = True
    else:
        float_prec = False

    for sbj in np.unique(orig[:, 0]):
        sbj_data = orig[orig[:, 0] == sbj]
        sbj_preds = preds[ids==sbj]
        sbj_unseg_preds = []
        changing_bool = True
        for i, pred in enumerate(sbj_preds):
            if (float_prec == True) and (changing_bool == True):
                sbj_unseg_preds = np.concatenate((sbj_unseg_preds, [pred] * (int(win_size * (1 - win_overlap * 0.01)) + 1)))
                if i + 1 == len(preds):
                    sbj_unseg_preds = np.concatenate((sbj_unseg_preds, [pred] * (int(win_size * (win_overlap * 0.01)) + 1)))
                changing_bool = False
            else:
                sbj_unseg_preds = np.concatenate((sbj_unseg_preds, [pred] * (int(win_size * (1 - win_overlap * 0.01)))))
                if i + 1 == len(preds):
                    sbj_unseg_preds = np.concatenate((sbj_unseg_preds, [pred] * int(win_size * (win_overlap * 0.01))))
                changing_bool = True
        sbj_unseg_preds = np.concatenate((sbj_unseg_preds, np.full(len(sbj_data) - len(sbj_unseg_preds), sbj_preds[-1])))
        unseg_preds = np.concatenate((unseg_preds, sbj_unseg_preds))
    assert len(unseg_preds) == len(orig)    
    return unseg_preds, orig[:, -1]

def convert_samples_to_segments(ids, labels, sampling_rate):
    """
    Method to convert samples to segments.
    
    Args:
        ids: numpy array
            Subject IDs
        labels: numpy array
            Labels
        sampling_rate: int
            Sampling rate
    
    Returns:
        dict: Dictionary with video IDs, labels, start time, end time, and score
    """
    
    f_video_ids, f_labels, f_t_start, f_t_end, f_score = [], np.array([]), np.array([]), np.array([]), np.array([])

    for id in np.unique(ids):
        sbj_labels = labels[(ids == id)]
        curr_start_i = 0
        curr_end_i = 0
        curr_label = sbj_labels[0]
        for i, l in enumerate(sbj_labels):
            if curr_label != l:
                act_start = curr_start_i / sampling_rate
                act_end = curr_end_i / sampling_rate
                act_label = curr_label - 1
                if curr_label != 0:
                    # create annotation
                    f_video_ids.append('sbj_' + str(int(id)))
                    f_labels = np.append(f_labels, act_label)
                    f_t_start = np.append(f_t_start, act_start)
                    f_t_end = np.append(f_t_end, act_end)
                    f_score = np.append(f_score, 1)
                curr_label = l
                curr_start_i = i + 1
                curr_end_i = i + 1    
            else:
                curr_end_i += 1        
    return {
        'video-id': f_video_ids,
        'label': f_labels,
        't-start': f_t_start,
        't-end': f_t_end,
        'score': f_score
    }
