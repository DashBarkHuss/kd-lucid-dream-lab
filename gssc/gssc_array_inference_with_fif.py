"""
This script performs sleep stage classification on EEG data stored in .fif format.
It loads EEG data, applies bandpass filtering, and uses a pre-trained GSSC model
to predict sleep stages for each epoch. The script processes both EEG and EOG channels,
handles signal permutations, and outputs predicted sleep stage classes with their
corresponding probabilities for each epoch.
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

def realtime_inference(fif_file_path, eeg_channels, eog_channels, sig_len = 2560):

    # Why is sig_len 2560? This i don't understand.

    # Initialize hidden state
    # Define your RNN configuration
    # RNN pass on abstract representations
    # hidden = torch.zeros(10, 1, 256)# this is what mne_infer uses
    num_layers = 10
    num_directions = 1
    batch_size = 1
    hidden_size = 256 # this is what mne_infer uses, I'm not sure why
    n_signals = 7  # Number of signal indices or permutations
    # TODO: make n_signals dynamic
    warnings.warn("WARNING: n_signals is currently hardcoded to 5. This should be made dynamic based on the permutation matrix size.", 
                     UserWarning)


    # Initialize the hidden state for each signal index
    hiddens = torch.zeros((n_signals, num_layers * num_directions, batch_size, hidden_size))

   

    signals = {"eeg":{"chans":eeg_channels, "drop":True, "flip":False}, "eog":{"chans":eog_channels, "drop":True, "flip":False}}
    # Create sig_combs dynamically from signals
    sig_combs = {}
    for signal_type, signal_info in signals.items():
        channels = signal_info['chans']
        # Create combinations: first empty tuple, then single-channel tuples
        sig_combs[signal_type] = [()] + [(chan,) for chan in channels]

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


    def fif_to_tensor(fif_file_path):
        # Step 1: Load the .fif file
        raw = mne.io.read_raw_fif(fif_file_path, preload=True)
        channel_names = raw.ch_names
        # ['F3', 'C3', 'O1', 'B-VEOG', 'L-HEOG', 'R-HEOG', 'EMG1', 'Airflow']
        
        # Step 2: Extract EEG data as a numpy array
        eeg_data = raw.get_data()
        
        # Step 3: Preprocess the data (example: normalize)
        # Adjust this step based on your specific preprocessing needs
        eeg_data = (eeg_data - np.mean(eeg_data)) / np.std(eeg_data)
        
        # Step 4: Convert to tensor
        eeg_tensor = torch.from_numpy(eeg_data).float()
        
        # If your model expects a specific shape, you might need to reshape
        # For example, if it expects [batch_size, channels, time_steps]:
        eeg_tensor = eeg_tensor.unsqueeze(0)  # Add batch dimension
        
        return eeg_tensor, channel_names


    def fif_to_tensor_weird_thing_epochs(fif_file_path, signals, epoch_duration=30, filter=True):
        # Step 1: Load the .fif file
        raw = mne.io.read_raw_fif(fif_file_path, preload=True)
        channel_names = raw.ch_names
        
        # Step 2: Create epochs
        epoch_samples = int(epoch_duration * raw.info['sfreq'])
        events = mne.make_fixed_length_events(raw, duration=epoch_duration)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=epoch_duration, baseline=None, preload=True)
        if filter:
            filter_band = [None, None]
            if round(raw.info["highpass"], 2) < 0.3:
                filter_band[0] = 0.3
            if round(raw.info["lowpass"], 2) > 30.:
                filter_band[1] = 30.
            raw.filter(*filter_band)
        if round(raw.info["highpass"], 2) != 0.3:
            warnings.warn("WARNING: GSSC was trained on data with a highpass "
                         "filter of 0.3Hz. These data have a highpass filter "
                         f"of {raw.info['highpass']}Hz")
        if round(raw.info["lowpass"], 2) != 30.:
            warnings.warn("WARNING: GSSC was trained on data with a lowpass "
                         "filter of 30Hz. These data have a lowpass filter "
                         f"of {raw.info['lowpass']}Hz")
        # # get the MNE inst into correct form, convert to z-scored array,pass in signals instead of doing this static code

        epo3, start_time3 = prepare_inst(raw, 2560, 'back')
        sig_combs3, perm_matrix3, all_chans3, _ = permute_sigs(epo3, signals)
        data3 = epo3.get_data(picks=channel_names) * 1e+6
        data3 = epo_arr_zscore(data3)



        tensor_data = torch.tensor(data3).float()

        # Step 3: Extract EEG data as a numpy array
        eeg_data = epochs.get_data()
        
        # Step 4: Preprocess the data (example: normalize)
        # Adjust this step based on your specific preprocessing needs
        eeg_data = (eeg_data - np.mean(eeg_data)) / np.std(eeg_data)
        
        # Step 5: Convert to tensor
        eeg_tensor = torch.from_numpy(eeg_data).float()
        
        return tensor_data, channel_names, epochs.times, perm_matrix3
        # return eeg_tensor, channel_names, epochs.times
   
    def fif_to_tensor_epochs(fif_file_path, epoch_duration=30):
        # Step 1: Load the .fif file
        raw = mne.io.read_raw_fif(fif_file_path, preload=True)
        channel_names = raw.ch_names
        
        # Step 2: Create epochs
        epoch_samples = int(epoch_duration * raw.info['sfreq'])
        events = mne.make_fixed_length_events(raw, duration=epoch_duration)
        epochs = mne.Epochs(raw, events, tmin=0, tmax=epoch_duration, baseline=None, preload=True)
        

        # tensor_data = torch.tensor(data3)

        # Step 3: Extract EEG data as a numpy array
        eeg_data = epochs.get_data()
        
        # Step 4: Preprocess the data (example: normalize)
        # Adjust this step based on your specific preprocessing needs
        eeg_data = (eeg_data - np.mean(eeg_data)) / np.std(eeg_data)
        
        # Step 5: Convert to tensor
        eeg_tensor = torch.from_numpy(eeg_data).float()
        
        # return tensor_data, channel_names, epochs.times
        return eeg_tensor, channel_names, epochs.times

    # If you want to use the epoched version:
    eeg_tensor_epoched, channel_names_epoched, times, perm_matrix1 = fif_to_tensor_weird_thing_epochs(fif_file_path, signals)
    
    eeg_tensor_epoched1, channel_names_epoched1, times1 = fif_to_tensor_epochs(fif_file_path)
    filtered_numpy = bandpass_filter(eeg_tensor_epoched1, 0.3, 35.0, 1000)
    filtered_eeg_tensor_epoched =torch.from_numpy(filtered_numpy).float() 

    perm_matrix = perm_matrix1
    # perm_matrix = np.array([np.array([1, 0])]) #testing 
    # perm_matrix = np.array([np.array([1, 0])]) # I'm not sure why we need to use this perm matrix. this doesn't make sense to me. Us ing this perm matrix gives the same result as using
    # this perm matrix in eeglab.mne_infer
    # array([[0, 1],
    #    [1, 0],
    #    [1, 1],
    #    [2, 0],
    #    [2, 1],
    #    [3, 0],
    #    [3, 1]])
    all_chans = channel_names_epoched 

    # Initialize ArrayInfer with pre-trained models 
    infer = ArrayInfer(net=None, con_net=None, sig_combs=sig_combs, perm_matrix=perm_matrix, all_chans=all_chans, sig_len=sig_len)


    # Perform inference with dummy data
    # hi = infer.infer(eeg_data_tensor, hiddens)
    # Adjust the input for the infer method
    def prepare_input(epoch_data):
        # Add a batch dimension and ensure it's the right shape for infer
        return epoch_data.unsqueeze(0)  # Shape becomes (1, 8, 30001)

    # Initialize hiddens
    hiddens = torch.zeros((n_signals, num_layers * num_directions, batch_size, hidden_size))

    # List to store all logits
    all_logits = []
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

    # Perform inference on each epoch

   
    for i in range(len(filtered_eeg_tensor_epoched)):  
        logits, res_logits, hiddens = infer.infer(prepare_input(eeg_tensor_epoched[i]), hiddens)
        all_logits.append(logits)
        # get the predicted classes
        predicted_classes = get_predicted_classes(logits)
        print(f"Predicted classes for epoch {i+1}: {predicted_classes}")
        predicted_classes, class_probs = get_predicted_classes_and_probabilities(logits)
        for i in range(len(predicted_classes)):
            print(f"  Predicted class: {predicted_classes[i][0]}")
            print("  Class probabilities:")
            for class_idx, prob in enumerate(class_probs[i][0]):
                print(f"    Class {class_idx}: {prob*100:.2f}%")
        print()




    # Convert all logits to predicted classes
    all_predicted_classes = [loudest_vote(logits) for logits in all_logits]


    # Example usage:
    # logits = ...  # Your logits from the model




    print("Done")
    return all_predicted_classes

if __name__ == "__main__":
    realtime_inference("data/sleep_data/dash_data_104_session_6/alphah104ses06scoring_raw.fif", ['F3','C3', 'O1'], ['L-HEOG'])