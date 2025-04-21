3/20
epochs displayed looks as if it's skipping around. could just be a display issue. it looks like it's taking 5 second epochs that aren't contiguous.

adding to buffer is adding 200 points at a time but its not in order. at the 200 index it changes to another part of the buffer

not sure why initial_data[2] is 1.0 (good) but frist new_data is not 1.0 (bad) is it because new data is after initial data?

looks lke new_data is going in order actually but maybe we are having issues with adding all data to the buffers. OK really what are the buffers? should it just be one buffer? but we pass in the epochs to infer?

we added code to check if the order of the data makes sense with the timestamps. but we are getting data coming in non consecutively. so it's almost as if new_data repeats itself. we need to makesure new data isn't redundant or too late. but the timestamps arne't catching a reduntancy when the other code is (wait notw maybe it is). ok so the timestamps are not the actual timestamps from the cvs. braindflow uses new timestamps when playing back the csv.

why are the epochs not exactly 6000?

ok lets check that new data, ok we need to still make sure that we use exactly 6000 and its in order but we cna't use the time stamps... not exactly because brianflow is making them up in hte test

seems like things are consecutive now. lets try to make epochs 6000 exactly.

3break- we're getting double of the first 200 points. numbers[200] is 1

now we're getting out of order when we hit 1000-1200. we need to pause before we process this part of the buffer.

Processing Buffer 0
Buffer range: 0 to 6000
Epoch start time: 1743057107.839177
Global recording start time: 1743057107.8390799
It keeps processing 0 - 6000 over and over.

the refectored buffer is working but we still need to test it with a scored csv (ganglion_realtime_round_robin). I am having trouble converting a bdf to a csv but Maybe we can to it with mne in the middle (like in brainflow_csv_to_mne.py). we don't need to convert a bdf to csv now, karen got a csv she can score.

in cyton_realtime_round_robin.py we never get past 3000 something collected data. the new data is 0 at some point and it breaks the script 2923

in cyton_reatime_csv_test_match.py we fixed an issue where initial data was added to the buffer twice. this probably needs to be fixed in the other scripts: cyton_reatime_round_robin.py, ganglion_reatime_round_robin.py

we still need to figure out why the cyton daisy csv stops at 2923.

whwne there is a gap it seems like the playback also has a gap. in some files the gap is exactly inline with the gap in the csv.

trying to group epochs based on actual time stamp and not based on order in the stream. this accounts for gaps, data packet loss, etc. cyton_realtime_round_robin_accurate_time_buffer.py we are testing short gap file to make sure it detects the gap

we are testing the 30 second gap.

In cyton_realtime_round_robin_accurate_time_buffer.py, we're added handling data gaps: the DataAcquisition class detects and logs gaps in the raw data stream, while the BufferManager class handles gaps at the epoch level by resetting hidden states and skipping affected epochs. We decided any non-insiginificant gap should resett the hidden states to maintain prediction accuracy. In the future, we may implement interpolation at the data level, and maybe at the hidden states level for smaller gaps in data. Next we need to test this code to see if the gaps are handled correctly.

we are makign test files for testing the gap handling.

we creaetd a copy of the old cyton_realtime_round_robin_accurate_time_buffer.py and renamed it cyton_realtime_round_robin_accurate_time_buffer_old.py. this i because we had issues while editing the new one and then went back intime to an old semi working one. right now a gp is beting detected in the first buffer but not subsecent buffers...oh is it because we only correct the currect data? not every buffer should touch every peice of data and every gap

---

data:
@cyton_BrainFlow-gap_short.csv
We need to detect the gap in each buffer. right now we are only detecting the gap in the first buffer.
We need to debug this. What we should do is make data that at least 2 buffers can go through (more than 35 seconds) then we can check to see that both buffers detect the gap.

Where is the gap detection?

- detect_data_gap in get_new_data
- validate_and_group_data

Unclear last_processed_idx Usage (replaced some to last_epoch_start_idx):
The variable name and its meaning is unclear throughout the code
It's not immediately obvious that it represents the start index of the last processed epoch for each buffer
This lack of clarity makes the code harder to understand and maintain
Incorrect Assumption in \_validate_buffer_indices:
The function assumes all epochs are the first epoch of each buffer
It's not properly handling subsequent epochs
The validation logic doesn't account for the continuous nature of the round-robin system

refactoring \_process_single_epoch_on_buffer and making sure it makes sense. it does gap detection but probable shouldn't. removed

test that we find a gap in all buffers

seens we'r detecting gaps in both 0 and 1 buffer. next test longer data- how does the script handle continuing after

## 3-18-25

when streaming brainflow is it in milivolts? does gssc accept milivolts?

we should check that the data from the board is the same as the streamed data from the fif file. it looks like we made fif files using these two scripts

/Users/dashiellbarkhuss/Documents/openbci_and_python_playgound/kd-lucid-dream-lab/gssc-version-6990abd04c0fc72418500e73fd1bc16242dfc8d0/make_fif_out_of_set.py

/Users/dashiellbarkhuss/Documents/openbci_and_python_playgound/kd-lucid-dream-lab/gssc-version-6990abd04c0fc72418500e73fd1bc16242dfc8d0/sandbox/make_fif_file.py

we need a scored brainflow .csv file.

1. Stream and save to file.
1. Stream and save to file with mark.
1. Detect left and right eye movements in real time.
1. Use YASA https://github.com/raphaelvallat/yasa with streaming.
1. Detect REM but also within REM when is the best time to send sound or light?
1. Z3Score 70% accurate in real time https://github.com/amiyapatanaik/z3score-api https://github.com/amiyapatanaik/FASST-Z3Score
1. Newer than Z3 https://github.com/jshanna100/gssc/
