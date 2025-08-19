"""
GSSC Helper Module for Inference Tests

This module provides helper functions for sleep stage classification using the GSSC model.
It includes utilities for EEG data preprocessing, inference execution, and result comparison.

The script performs sleep stage classification on EEG data, applies bandpass filtering, 
and uses a pre-trained GSSC model to predict sleep stages for each epoch. It processes 
both EEG and EOG channels, handles signal permutations, and outputs predicted sleep 
stage classes with their corresponding probabilities for each epoch.

Functions include:
- realtime_inference: Main inference function for processing EEG data
- compare_sleep_stages: Compare predicted vs ground truth sleep stages
- preprocess_eeg_epoch_for_gssc: Preprocess EEG data for GSSC model input
- Various utility functions for data preparation and result analysis
"""
from gssc.infer import ArrayInfer
import numpy as np
import torch
import torch.nn as nn
from gssc.utils import permute_sigs, prepare_inst, epo_arr_zscore, loudest_vote
import mne
import scipy
from scipy.signal import butter, lfilter
import warnings

def compare_sleep_stages(inferred_stages, expected_stages, verbose=True):
    sleepSMG_staging_key = {
        0: 'Wake',
        1: 'N1',
        2: 'N2',
        3: 'N3',
        5: 'REM',
        6: 'Movement/Undefined',
        7: 'Unscored'
    }
    gssc_staging_key = {
        0: 'Wake',
        1: 'N1',
        2: 'N2',
        3: 'N3',
        4: 'REM'
    }
    # TODO: removed unscored data or maye also movement/undefined epochs from comparison
    # Ensure both arrays are the same length
    if len(inferred_stages) != len(expected_stages):
        raise ValueError("Inferred and expected stage arrays must be the same length")
    
    # remove the epochs where the expected stage is 7 or 6
    # First get the epoch index of all expected stages that are 7 or 6
    # then remove those epochs from the inferred stages 
    remove_epochs = np.where((expected_stages == 7) | (expected_stages == 6))[0]
    inferred_stages_cut = np.delete(inferred_stages, remove_epochs)
    expected_stages_cut = np.delete(expected_stages, remove_epochs)

    # Initialize counters
    matches = 0
    total = len(inferred_stages_cut)
    mismatches = []

    # Compare stages epoch by epoch
    print("")
    max_length = max(len(f"Epoch {len(inferred_stages)}: ") + 20, 30)  # Adjust 20 if needed

    for i, (inferred, expected) in enumerate(zip(inferred_stages_cut, expected_stages_cut)):
        if gssc_staging_key[inferred] == sleepSMG_staging_key[expected]:
            matches += 1
            print(f"Epoch {i+1}: Stage {gssc_staging_key[inferred]}")
        else:
            mismatches.append((i, inferred, expected))
            first_part = f"Epoch {i+1}: Stage {gssc_staging_key[inferred]}"
            print(f"{first_part:<{max_length}} NOT MATCHED --Expected Stage {sleepSMG_staging_key[expected]}")
    # Calculate accuracy 
    accuracy = matches / total
    
    # kappa = cohen_kappa_score(expected_stages, inferred_stages)

    # Print summary
    print(f"\nTotal Epochs: {total}")
    print(f"Matched Epochs: {matches}")
    print(f"Mismatched Epochs: {total - matches}")
    print(f"Accuracy: {(accuracy*100):.4f}%")

    if verbose:
        false_pos_rem_count = [mismatch for mismatch in mismatches if (mismatch[1] == 4.0)]
        false_neg_rem_count = [mismatch for mismatch in mismatches if (mismatch[2] == 5.0)]

        false_pos_rem_prc = len(false_pos_rem_count) / total
        false_neg_rem_prc = len(false_neg_rem_count) / total
        overall_rem_accuracy = 1-((len(false_pos_rem_count) + len(false_neg_rem_count)) / total)

        false_pos_n1_count = [mismatch for mismatch in mismatches if (mismatch[1] == 1.0)]
        false_neg_n1_count = [mismatch for mismatch in mismatches if (mismatch[2] == 1.0)]

        false_pos_n1_prc = len(false_pos_n1_count) / total
        false_neg_n1_prc = len(false_neg_n1_count) / total
        overall_n1_accuracy = 1-((len(false_pos_n1_count) + len(false_neg_n1_count)) / total)

        false_pos_n2_count = [mismatch for mismatch in mismatches if (mismatch[1] == 2.0)]
        false_neg_n2_count = [mismatch for mismatch in mismatches if (mismatch[2] == 2.0)]

        false_pos_n2_prc = len(false_pos_n2_count) / total
        false_neg_n2_prc = len(false_neg_n2_count) / total
        overall_n2_accuracy = 1-((len(false_pos_n2_count) + len(false_neg_n2_count)) / total)

        false_pos_n3_count = [mismatch for mismatch in mismatches if (mismatch[1] == 3.0)]
        false_neg_n3_count = [mismatch for mismatch in mismatches if (mismatch[2] == 3.0)]

        false_pos_n3_prc = len(false_pos_n3_count) / total
        false_neg_n3_prc = len(false_neg_n3_count) / total
        overall_n3_accuracy = 1-((len(false_pos_n3_count) + len(false_neg_n3_count)) / total)

        false_pos_wake_count = [mismatch for mismatch in mismatches if (mismatch[1] == 0.0)]
        false_neg_wake_count = [mismatch for mismatch in mismatches if (mismatch[2] == 0.0)]

        false_pos_wake_prc = len(false_pos_wake_count) / total
        false_neg_wake_prc = len(false_neg_wake_count) / total
        overall_wake_accuracy = 1-((len(false_pos_wake_count) + len(false_neg_wake_count)) / total)
        print("")
        print(f"False Positive REM Epochs: {len(false_pos_rem_count)}")
        print(f"False Positive REM percentage: {(false_pos_rem_prc * 100):.2f}%")
        print(f"False Negative REM Epochs: {len(false_neg_rem_count)}")
        print(f"False Negative REM:  {(false_neg_rem_prc * 100):.2f}%")
        print(f"Overall REM Accuracy: {(overall_rem_accuracy * 100):.2f}%")
        print("")
        print(f"False Positive N1 Epochs: {len(false_pos_n1_count)}")
        print(f"False Positive N1 percentage: {(false_pos_n1_prc * 100):.2f}%")
        print(f"False Negative N1 Epochs: {len(false_neg_n1_count)}")
        print(f"False Negative N1 percentage: {(false_neg_n1_prc * 100):.2f}%")
        print(f"Overall N1 Accuracy: {(overall_n1_accuracy * 100):.2f}%")
        print("")
        print(f"False Positive N2 Epochs: {len(false_pos_n2_count)}")
        print(f"False Positive N2 percentage: {(false_pos_n2_prc * 100):.2f}%")
        print(f"False Negative N2 Epochs: {len(false_neg_n2_count)}")
        print(f"False Negative N2 percentage: {(false_neg_n2_prc * 100):.2f}%")
        print(f"Overall N2 Accuracy: {(overall_n2_accuracy * 100):.2f}%")
        print("")
        print(f"False Positive N3 Epochs: {len(false_pos_n3_count)}")
        print(f"False Positive N3 percentage: {(false_pos_n3_prc * 100):.2f}%")
        print(f"False Negative N3 Epochs: {len(false_neg_n3_count)}")
        print(f"False Negative N3 percentage: {(false_neg_n3_prc * 100):.2f}%")
        print(f"Overall N3 Accuracy: {(overall_n3_accuracy * 100):.2f}%")
        print("")
        print(f"False Positive Wake Epochs: {len(false_pos_wake_count)}")
        print(f"False Positive Wake percentage: {(false_pos_wake_prc * 100):.2f}%")
        print(f"False Negative Wake Epochs: {len(false_neg_wake_count)}")
        print(f"False Negative Wake percentage: {(false_neg_wake_prc * 100):.2f}%")
        print(f"Overall Wake Accuracy: {(overall_wake_accuracy * 100):.2f}%")
        print("")
    print(f"Removed Unscored and Movement/Undefined Epochs: {len(remove_epochs)}")

def make_eeg_eog_combinations(eeg_indices, eog_indices):
    """
    Create all combinations of EEG and EOG indices.

    Args:
    eeg_indices (list): List of EEG channel indices.
    eog_indices (list): List of EOG channel indices.
    
    Returns:
    list: All combinations of EEG and EOG indices.
    """
    combinations = []
    # rmemeber to include none as an option
    for eeg_index in eeg_indices+[None]:
        for eog_index in eog_indices+[None]:
            # if both are none, skip
            if eeg_index is None and eog_index is None:
                continue
            combinations.append((eeg_index, eog_index))
    return combinations


def bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Apply a bandpass filter to the data.

    Parameters:
    - data (numpy.ndarray): Input data array of shape (n_channels, n_samples).
    - lowcut (float): Low cutoff frequency of the filter.
    - highcut (float): High cutoff frequency of the filter.
    - fs (float): Sampling frequency of the data.
    - order (int): Order of the filter.

    Returns:
    - filtered_data (numpy.ndarray): The filtered data array.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist

    # Design a Butterworth bandpass filter
    b, a = butter(order, [low, high], btype='band')

    # Apply the filter to each channel
    filtered_data = lfilter(b, a, data, axis=-1)
    return filtered_data

    
def apply_filter_raw(raw):
    # return eeg_tensor, channel_names, epochs.times
            # Store a copy of the data before filtering
    data_before = raw.get_data().copy()
    
    filter_band = [None, None]
    if round(raw.info["highpass"], 2) < 0.3:
        filter_band[0] = 0.3
    if round(raw.info["lowpass"], 2) > 30.:
        filter_band[1] = 30.
    raw.filter(*filter_band)
    
    # Get data after filtering
    data_after = raw.get_data()
    
    # Check if data actually changed
    if np.array_equal(data_before, data_after):
        warnings.warn("WARNING: Filtering did not change the data. This might indicate an issue with the filtering process.")
    else:
        print(f"Filtering changed the data. Max difference: {np.max(np.abs(data_before - data_after))}")
    if round(raw.info["highpass"], 2) != 0.3:
        warnings.warn("WARNING: GSSC was trained on data with a highpass "
                        "filter of 0.3Hz. These data have a highpass filter "
                        f"of {raw.info['highpass']}Hz")
    if round(raw.info["lowpass"], 2) != 30.:
        warnings.warn("WARNING: GSSC was trained on data with a lowpass "
                        "filter of 30Hz. These data have a lowpass filter "
                        f"of {raw.info['lowpass']}Hz")
    return raw

def validate_epoch_length(data, sfreq):
    """Validate that the data has exactly 30 seconds of samples."""
    expected_samples = int(30 * sfreq)
    if data.shape[1] != expected_samples:
        raise ValueError(f"Expected {expected_samples} samples for 30s epoch, got {data.shape[1]}")
    return data

def resample_to_target_frequency(raw, target_sfreq):
    """
    Resample raw data to target sampling frequency.
    First upsamples to 1000 Hz if current sampling rate is lower,
    then downsamples to target frequency.
    
    Args:
        raw (mne.io.Raw): Raw EEG data
        target_sfreq (float): Target sampling frequency (typically 85.3 Hz for GSSC)
    
    Returns:
        numpy.ndarray: Resampled data in microvolts
    """
    current_sfreq = raw.info['sfreq']
    # print the number of samples in the raw data
    print(f"Number of samples in raw data: {raw.n_times}")
    
    # First upsample to 1000 Hz if needed
 
    raw_upsampled = raw.copy().resample(1000)
    data = raw_upsampled.get_data(picks=raw.ch_names) * 1e+6
    print(f"Number of samples in upsampled data: {raw_upsampled.n_times}")
  

    validate_epoch_length(data, raw.info['sfreq'])
    
    # Then downsample to target frequency if needed
    if current_sfreq != target_sfreq:
        raw_resampled = raw.copy().resample(target_sfreq)
        data = raw_resampled.get_data(picks=raw.ch_names) * 1e+6
    
    return data

def preprocess_eeg_epoch_for_gssc(raw, apply_bandpass_filter=True):
    """
    Preprocess a 30-second epoch of EEG data for sleep stage classification with GSSC.
    
    This function performs the following steps:
    1. Applies bandpass filtering (optional)
    2. Extracts and converts data to microvolts
    3. Validates epoch length
    4. Resamples to target frequency (85.3 Hz)
    5. Applies z-scoring normalization using gssc.utils.epo_arr_zscore
       - Centers data around zero (removes DC offset)
       - Normalizes amplitude across channels
       - Makes data suitable for GSSC model input
    6. Converts to PyTorch tensor
    
    Args:
        raw (mne.io.Raw): Raw EEG data containing a single 30-second epoch
        apply_bandpass_filter (bool): Whether to apply bandpass filtering (default: True)
    
    Returns:
        torch.Tensor: Preprocessed EEG data ready for GSSC inference
    """
    if apply_bandpass_filter:
        apply_filter_raw(raw)
    
    # Extract and convert data
    data = raw.get_data(picks=raw.ch_names) * 1e+6
    
    # Validate epoch length
    
    # Resample to target frequency
    target_sfreq = 2560 / 30.  # ~85.3 Hz
    data = resample_to_target_frequency(raw, target_sfreq)
    
    # Apply z-scoring using GSSC's implementation
    data = epo_arr_zscore(data)  # Uses gssc.utils.epo_arr_zscore for consistency with model training
    return torch.tensor(data).float()

def convert_raw_full_data_to_gssc_tensor(raw, epoch_duration=30, filter=True):
    channel_names = raw.ch_names
    

    if filter:
        apply_filter_raw(raw)

    epo3, _ = prepare_inst(raw, 2560, 'back')
    data1 = epo3.get_data(picks=channel_names) * 1e+6

    data3 = epo_arr_zscore(data1)

    tensor_data = torch.tensor(data3).float()


    return tensor_data

# we updated prepare input to prepare the input for the updated ArrayInfer class
def prepare_input(epoch_data, eeg_index, eog_index):
    input_dict = {}
    
    # Add EEG data to the dictionary with 'eeg' key
    if eeg_index is not None:
        eeg_data = epoch_data[eeg_index].reshape(1, 1, -1)  # Reshape to [1, 1, samples]
        input_dict['eeg'] = eeg_data
    
    # Add EOG data to the dictionary with 'eog' key
    if eog_index is not None:
        eog_data = epoch_data[eog_index].reshape(1, 1, -1)  # Reshape to [1, 1, samples]
        input_dict['eog'] = eog_data
    
    return input_dict


    # Convert logits to a PyTorch tensor if they're not already
def get_predicted_classes(logits):
    """
    Convert logits to predicted classes.
    
    Args:
    logits (list or numpy.ndarray): The raw logits output from the model.
    
    Returns:
    numpy.ndarray: The predicted classes.
    """
    # Convert logits to tensor if it's not already
    logits_tensor = torch.tensor(logits)

    # Apply softmax to convert logits to probabilities
    probabilities = torch.softmax(logits_tensor, dim=-1)

    # Get the predicted classes
    predicted_classes = probabilities.argmax(dim=-1).numpy()

    return predicted_classes

def get_predicted_classes_and_probabilities(logits):
    """
    Convert logits to predicted classes and class probabilities.
    
    Args:
    logits (list or numpy.ndarray): The raw logits output from the model.
    
    Returns:
    tuple: (predicted_classes, class_probabilities)
        predicted_classes (numpy.ndarray): The predicted classes.
        class_probabilities (numpy.ndarray): Probabilities for each class.
    """
    # Convert logits to tensor if it's not already
    logits_tensor = torch.tensor(logits)

    # Apply softmax to convert logits to probabilities
    probabilities = torch.softmax(logits_tensor, dim=-1)

    # Get the predicted classes
    predicted_classes = probabilities.argmax(dim=-1).numpy()

    # Convert probabilities to numpy array
    class_probabilities = probabilities.numpy()

    return predicted_classes, class_probabilities
def get_results_for_each_combo(eeg_tensor_epoched, eeg_eog_combinations, hiddens, infer):
    input_dicts_for_each_combo = []
    for eeg_index, eog_index in eeg_eog_combinations:
        input_dicts_for_each_combo.append(prepare_input(eeg_tensor_epoched, eeg_index, eog_index))


    # eeg_tensor_epoched.shape
    # torch.Size([100, 8, 2560])
    # eeg_tensor_epoched[i].shape
    # torch.Size([8, 2560])

    # input_dict_combo_1['eeg'].shape
    # # torch.Size([1, 1, 2560])
    results_for_each_combo = []
    for i2, input_dict in enumerate(input_dicts_for_each_combo):
        results_for_each_combo.append(infer.infer(input_dict, hiddens[i2]))

    # for each result, set the hidden to the third dimension of the result
    new_hiddens= []
    for index_of_result, result in enumerate(results_for_each_combo):
        new_hiddens.append(result[2])
    index_of_logits = 0
    all_combo_logits = np.stack([result[index_of_logits].numpy() for result in results_for_each_combo])  # Only take the first element (logits) from each result tuple



    predicted_classes, class_probs = get_predicted_classes_and_probabilities(all_combo_logits)
    loudest_vote_result = loudest_vote(all_combo_logits)
    return loudest_vote_result, predicted_classes, class_probs, new_hiddens

def make_hiddens(eeg_eog_combinations):
    return torch.zeros(len(eeg_eog_combinations), 10, 1, 256)

def make_infer():
    return ArrayInfer(
        net=None,  # Use default network
        con_net=None,  # Use default context network
        use_cuda=False,  # Set to True if you want to use CUDA
        gpu_idx=None,  # Specify GPU index if needed
    )
def realtime_inference(raw, eeg_channels, eog_channels, sig_len = 2560):

    # If you want to use the epoched version:
    eeg_tensor_epoched = convert_raw_full_data_to_gssc_tensor(raw)
   
    # new
    infer = make_infer()
    # find the indices of the eeg and eog channels
    eeg_indices = [raw.ch_names.index(ch) for ch in eeg_channels]
    eog_indices = [raw.ch_names.index(ch) for ch in eog_channels]

    eeg_eog_combinations = make_eeg_eog_combinations(eeg_indices, eog_indices)

    # set up variables to store the results
    hiddens = make_hiddens(eeg_eog_combinations)
    loudest_votes = []
    predicted_classes_list = []
    class_probs_list = []

    #  for each epoch, get the logits, predicted classes, and class probabilities
    for i in range(len(eeg_tensor_epoched)): 
        loudest_vote, predicted_classes, class_probs, new_hiddens = get_results_for_each_combo(eeg_tensor_epoched[i], eeg_eog_combinations, hiddens, infer)
        # update hiddens
        hiddens = torch.stack(new_hiddens)
        loudest_votes.append(loudest_vote)
        predicted_classes_list.append(predicted_classes)
        class_probs_list.append(class_probs)

    return loudest_votes, predicted_classes_list, class_probs_list


# if __name__ == "__main__": 
    # this no logerworks, we need to passin raw not fif
    # realtime_inference("/Users/dashiellbarkhuss/Documents/openbci_and_python_playgound/kd-lucid-dream-lab/data/sleep_data/dash_data_104_session_6/alphah104ses06scoring_raw.fif", ['F3','C3', 'O1'], ['L-HEOG'])