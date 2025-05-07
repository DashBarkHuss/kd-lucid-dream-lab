"""
This script uses the gssc_array_inference_with_fif.py as a starting point. then we replace the old ArrayInfer class with the new one.

The gssc_array_inference_with_fif.py script performs sleep stage classification on EEG data stored in .fif format.
It loads EEG data, applies bandpass filtering, and uses a pre-trained GSSC model
to predict sleep stages for each epoch. The script processes both EEG and EOG channels,
handles signal permutations, and outputs predicted sleep stage classes with their
corresponding probabilities for each epoch.

The script logs the predicted classes and class probabilities for each epoch.

Example output for epoch 299:
Predicted classes for epoch 299: [[2]
 [1]
 [2]
 [2]
 [2]
 [0]
 [1]]
  Predicted class: 2
  Class probabilities:
    Class 0: 20.06%
    Class 1: 24.22%
    Class 2: 54.38%
    Class 3: 0.92%
    Class 4: 0.42%
  Predicted class: 1
  Class probabilities:
    Class 0: 2.32%
    Class 1: 61.03%
    Class 2: 36.27%
    Class 3: 0.02%
    Class 4: 0.35%
  Predicted class: 2
  Class probabilities:
    Class 0: 0.80%
    Class 1: 13.76%
    Class 2: 85.23%
    Class 3: 0.08%
    Class 4: 0.13%
  Predicted class: 2
  Class probabilities:
    Class 0: 2.44%
    Class 1: 23.62%
    Class 2: 72.93%
    Class 3: 0.45%
    Class 4: 0.56%
  Predicted class: 2
  Class probabilities:
    Class 0: 0.65%
    Class 1: 31.00%
    Class 2: 67.79%
    Class 3: 0.54%
    Class 4: 0.01%
  Predicted class: 0
  Class probabilities:
    Class 0: 44.16%
    Class 1: 36.65%
    Class 2: 18.15%
    Class 3: 0.44%
    Class 4: 0.60%
  Predicted class: 1
  Class probabilities:
    Class 0: 27.10%
    Class 1: 50.51%
    Class 2: 21.87%
    Class 3: 0.06%
    Class 4: 0.47%

It doesn't log a summary of the predicted classes and class probabilities for all epochs. It doesn't log an accuracy comparison with the ground truth.
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

def convert_raw_full_data_to_gssc_tensor(raw, epoch_duration=30, filter=True):
    channel_names = raw.ch_names
    
    events = mne.make_fixed_length_events(raw, duration=epoch_duration)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=epoch_duration, baseline=None, preload=True)
    
    if filter:
        apply_filter_raw(raw)

    epo3, _ = prepare_inst(raw, 2560, 'back')
    data1 = epo3.get_data(picks=channel_names) * 1e+6

    data3 = epo_arr_zscore(data1)

    tensor_data = torch.tensor(data3).float()

    # Step 3: Extract EEG data as a numpy array
    eeg_data = epochs.get_data()
    
    # Step 4: Preprocess the data (example: normalize)
    # Adjust this step based on your specific preprocessing needs
    eeg_data = (eeg_data - np.mean(eeg_data)) / np.std(eeg_data)

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
    return all_combo_logits, predicted_classes, class_probs, new_hiddens   
    
def realtime_inference(raw, eeg_channels, eog_channels, sig_len = 2560):
    signals = {"eeg":{"chans":eeg_channels, "drop":True, "flip":False}, "eog":{"chans":eog_channels, "drop":True, "flip":False}}

    # If you want to use the epoched version:
    eeg_tensor_epoched = convert_raw_full_data_to_gssc_tensor(raw)
   
    # new
    infer = ArrayInfer(
        net=None,  # Use default network
        con_net=None,  # Use default context network
        use_cuda=False,  # Set to True if you want to use CUDA
        gpu_idx=None,  # Specify GPU index if needed
    )

    # find the indices of the eeg and eog channels
    eeg_indices = [raw.ch_names.index(ch) for ch in eeg_channels]
    eog_indices = [raw.ch_names.index(ch) for ch in eog_channels]

    eeg_eog_combinations = make_eeg_eog_combinations(eeg_indices, eog_indices)

    # set up variables to store the results
    hiddens = torch.zeros(len(eeg_eog_combinations), 10, 1, 256)
    loudest_votes = []
    predicted_classes_list = []
    class_probs_list = []

    #  for each epoch, get the logits, predicted classes, and class probabilities
    for i in range(len(eeg_tensor_epoched)): 
        all_combo_logits, predicted_classes, class_probs, new_hiddens = get_results_for_each_combo(eeg_tensor_epoched[i], eeg_eog_combinations, hiddens, infer)
        hiddens = torch.stack(new_hiddens)
        loudest_votes.append(loudest_vote(all_combo_logits))
        predicted_classes_list.append(predicted_classes)
        class_probs_list.append(class_probs)

    return loudest_votes, predicted_classes_list, class_probs_list

if __name__ == "__main__": 
    realtime_inference("/Users/dashiellbarkhuss/Documents/openbci_and_python_playgound/kd-lucid-dream-lab/data/sleep_data/dash_data_104_session_6/alphah104ses06scoring_raw.fif", ['F3','C3', 'O1'], ['L-HEOG'])