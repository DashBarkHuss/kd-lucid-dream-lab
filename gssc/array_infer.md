## ArrayInfer

ArrayInfer is a class that is used to infer the sleep stage of a given EEG signal. Unlike EEGInfer.mne_infer, ArrayInfer designed for streaming data.

### ArrayInfer Real-Time Inference

- **Designed for streaming data**: Processes one epoch at a time with low latency
- **Works directly with tensors**: Takes raw array/tensor input without requiring MNE objects
- **Maintains state between epochs**: Uses hidden state that can be passed from one epoch to the next
- **Lightweight processing**: Doesn't require loading full files or creating MNE objects
- **Well-suited for BCI applications**: Where you need immediate classification as data arrive

### ArrayInfer and EEGInfer.mne_infer similarites

- **Same Neural Networks**:
  - Both use the exact same trained models (sig_net_v1.pt and gru_net_v1.pt)
  - Same architecture and weights
- **Same Signal Processing Pipeline**:
  - Raw EEG/EOG → Feature extraction → Context modeling → Sleep stage prediction
- **Same mathematical operations**:
- **Same Channel Combination Strategy**:
  - Both try various EEG/EOG channel combinations
  - Both use loudest_vote to combine results
- **Same Underlying Algorithm**:
  - Same preprocessing (filtering, z-scoring)
  - Same classification approach

The differences are mainly in the API design and integration:

- **EEGInfer**: Wraps everything in MNE integration, handles file I/O, built for research workflows
- **ArrayInfer**: Exposes the core tensor processing directly, designed for streaming applications

It's like the difference between a car with automatic transmission (EEGInfer - easier to use, handles details for you) and manual transmission (ArrayInfer - more control, but you handle more details yourself). The engine underneath is the same!

### The flow of ArrayInfer

#### 1. Input

```python
input_dict = {
    'eeg': torch.tensor(shape=[1, 1, 2560]),  # One EEG channel
    'eog': torch.tensor(shape=[1, 1, 2560])   # One EOG channel
}
```

ArrayInfer.infer takes in:

- a dictionary with keys 'eeg' and/or 'eog'.
- a `hidden_state` tensor of shape [10, 1, 256]

Each valuein the dictionary (`eog` and `eeg`) is a tensor of shape [batch_size, 1, samples]

- `batch_size` = 1 (one epoch)
- `middle dimension` = 1 (channel dimension)
- `samples` = 2560 (30 seconds at 85.33Hz)

## Which channels get passed into ArrayInfer.infer?

You can only send one eeg and one eog channel at a time to ArrayInfer.infer. So how do you use more than just those 1 eog and 1 eeg channels? To use more than 1 eeg and 1 eog channels, you have to pass combinations of the eeg and eog channels you want to use through the ArrayInfer.infer function, calling ArrayInfer.infer multiple times. Then you combine all those resulting predictions to get a final prediction using a function called `loudest_vote`.

Let's say you want to use eeg F3, F4, O3 and eog L-EOG.

You have 4 channels but you can only send combinations of 2 channels at a time.

The possible combinations (2 at a time) of these 4 channels are:

- eeg F3 and eog L-EOG
- eeg F4 and L-EOG
- eeg O3 and L-EOG
- eeg F3 and no eog
- eeg F4 and no eog
- eeg O3 and no eog
- no eeg and L-EOG

So for each 30 second epoch, inorder to get a probability of that epoch's sleep stage using all 4 channels, you would have to call ArrayInfer.infer 7 times with the 7 different combinations.

## Channel Combinations Table

| Combination | EEG Channel | EOG Channel | ArrayInfer.infer returns |
| ----------- | ----------- | ----------- | ------------------------ |
| 1           | F3          | L-EOG       | Some logit tensor 1      |
| 2           | F4          | L-EOG       | Some logit tensor 2      |
| 3           | O3          | L-EOG       | Some logit tensor 3      |
| 4           | F3          | None        | Some logit tensor 4      |
| 5           | F4          | None        | Some logit tensor 5      |
| 6           | O3          | None        | Some logit tensor 6      |
| 7           | None        | L-EOG       | Some logit tensor 7      |

ArrayInfer.infer will return a tensor of logits shape [1, 1, 5] for each combination.

- 1: Batch size (processing one sample at a time)
- 1: Sequence length (one prediction)
- 5: Number of sleep stages/classes (typically Wake, N1, N2, N3, and REM)

Altogether, you'll get 7 logit tensors, 1 for each channel combination.

To get the final probability of the sleep stage for that 30 second epoch, you pass those 7 logit tensors to `loudest_vote`.

```python

# combine all the logits tensors into one numpy array
all_combo_logits = np.concatenate(logits_tensor_1, logits_tensor_2, logits_tensor_3, logits_tensor_4, logits_tensor_5, logits_tensor_6, logits_tensor_7, axis=0)

# get the final predicted class
final_predicted_class = loudest_vote(all_combo_logits)
```

## Hidden State

A hidden state is the "memory" or "internal representation" of a recurrent neural network (RNN). Think of it as the network's working memory that allows it to remember and use information from previous inputs.

### Simple Explanation

Imagine you're reading a book:

- **Without hidden state**: You'd read each sentence in isolation, forgetting everything before it
- **With hidden state**: You remember previous paragraphs, helping you understand the current one

### How It Works in Sleep Staging

1. **Initial state**: Starts as all zeros (no prior information)
2. **First epoch**: Network processes first 30s of EEG/EOG

- Updates hidden state based on what it sees

3. **Second epoch**: Network sees new data + previous hidden state

- Can now recognize patterns across epochs
- For example: "Previous two epochs were REM, this looks similar, likely still REM"

4. **Continues**: Each epoch builds on accumulated knowledge

This memory mechanism is why GSSC can recognize sleep patterns that span multiple epochs, like sleep cycles and stage transitions, rather than just classifying each 30-second chunk in isolation.

## Use in ArrayInfer

In EEGInfer.mne_infer, the hidden state is automatically handled in the background.

In ArrayInfer, the hidden state is passed in as a parameter to the infer function.

The first time you call ArrayInfer.infer, you can pass in a tensor of zeros.

```python
hidden_state = torch.zeros(10, 1, 256)
logits, res_logits, hiddens = ArrayInfer.infer(input_dict, hidden_state)
```

The second time and beyond you call ArrayInfer.infer, you pass in the hidden state returned from the previous call.

```python
logits, res_logits, hiddens = ArrayInfer.infer(input_dict, hiddens)
```

The hiddens tensor returned from the second call to ArrayInfer.infer will have a shape of [10, 1, 256].
