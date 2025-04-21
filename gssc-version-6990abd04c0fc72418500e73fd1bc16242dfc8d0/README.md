There were breaking changes in the gssc api so use commit 6990abd04c0fc72418500e73fd1bc16242dfc8d0 from September 28 2023

# Test GSSC's scoring on your scored data

To test GSSC's scoring on your data, you'll need an mne raw python object to pass into mne_infer. You create a mne object via a .fif file.

Or you can use array_infer if you have your data as a numpy array.

## `EEGInfer.mne_infer` vs `ArrayInfer.infer`

`mne_infer` is used for scoring data after it's recorded, like after your sleeping subject wakes up. And `ArrayInfer.infer` is used for scoring data in real time, like during sleep. I'm not totally sure why `mne_infer` was built for scoring data after it's recorded. These are the main differences between the two functions:

`ArrayInfer.infer`:

- takes in a numpy array
- very lightweight
- uses hiddens
- does not filter
- returns logits instead of outcomes

`EEGInfer.mne_infer`:

- takes in an mne raw object
- filters the data
- returns outcomes
- does not use hiddens
- does some formatting with the data I don't completely understand and we need to recreate that effect on the data we pass into ArrayInfer.

I believe the main different between the two inference functions is that ArrayInfer only uses past data to make a prediction, while MneInfer uses past and future data to make a prediction. This would make sense as to why `mne_infer` is used for scoring data after it's recorded, because you have both past and future data to make a prediction. However, as a novice in ML, I couldn't really make that out in the code itself.

# Create a fif file of your data from a set file

MNE python objects can take fif file directly to generate a raw object. But you can also use other file types (csv) and format them to pass into `mne.io.RawArray`.

Use make_shortened_fif_out_of_set.py to create a fif file from a set file. Set the paths to your set file and matlab file. Set `start` and `end` to the desired start and end times in seconds.

After you create your fif file, use gssc_inference_mne_array_compare.py to score your data. Set the paths to your fif file and matlab file, set the `start` and `end` times to match the times you used to create the fif file in make_shortened_fif_out_of_set.py.

You can also use make_fif_out_of_set.py to create a fif file of the entire set file, instead of a shortened version.

If you want to change the eeg and eog channels used for scoring, you can do so in gssc_inference_mne_array_compare.py.

# Why is sig_len 2560?

I don't understand why the signal length is 2560. I'd think it would be the number of samples in an epoch, which in some of our data is 30 \* 1000hz = 30000 samples and in other data is 30 time 200hz = 6000 samples.

2560 is the default seen in the code. It's also the hidden size. When I tried to change it to 30000 I couldn't get the code to work.

# Trouble Shooting: Mismatch in Sampling Rate

In sleepSMG, is seems like it doesn't matter if the .set file's sampling rate is set correctly. So there may be an issue where sleepSMG works fine but our code doesn't. This can be cause by the wrong sampling rate being set in the .set file.

This can cause raw.\_last_time to be incorrect and the horizantal axis to show the wrong number of seconds when plotting the data.

You can check the sampling rate of your set file by running the following code:

```python
# Path to your .set file
set_file_path = 'data/sleep_data/dash_data_104_session_6/alphah104ses06scoring_corrected.set'
raw = mne.io.read_raw_eeglab(set_file_path, preload=True)

sampling_rate = raw.info['sfreq'] # 250

# compare both sampling rates
print(f"Matlab sampling rate: {matlab_sampling_rate} Hz")
print(f"MNE sampling rate: {sampling_rate} Hz")

# throw error is sampling rates don't match
if matlab_sampling_rate != sampling_rate:
    raise ValueError("Sampling rates do not match. Use pop_loadset to load the data in EEGLAB and set the sampling rate to the matlab sampling rate with EEG.srate = and then save the .set file again using pop_saveset.")

```

Use pop_loadset to load the data in EEGLAB and set the sampling rate to the matlab sampling rate with EEG.srate = and then save the .set file again using pop_saveset.
