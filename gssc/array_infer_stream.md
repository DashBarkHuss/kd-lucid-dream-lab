```python
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
    #  for each epoch, get the logits, predicted classes, and class probabilities
    for i in range(len(filtered_eeg_tensor_epoched)):
        # this is hardcoded for eeg indices: 0, 1, 2 eeg: 4
        # prepare inpute downsized the eeg_tensor_epoched to 2560 samples and uses the set combinations
        input_dict_combo_1 = prepare_input(eeg_tensor_epoched[i], [0], [4])
        input_dict_combo_2 = prepare_input(eeg_tensor_epoched[i], [1], [4])
        input_dict_combo_3 = prepare_input(eeg_tensor_epoched[i], [2], [4])
        input_dict_combo_4 = prepare_input(eeg_tensor_epoched[i], [0], None)
        input_dict_combo_5 = prepare_input(eeg_tensor_epoched[i], [1], None)
        input_dict_combo_6 = prepare_input(eeg_tensor_epoched[i], [2], None)
        input_dict_combo_7 = prepare_input(eeg_tensor_epoched[i], None, [4])

        # get logits for each combo
        logits_1, res_logits_1, hidden_1 = infer.infer(input_dict_combo_1, hidden_1)
        logits_2, res_logits_2, hidden_2 = infer.infer(input_dict_combo_2, hidden_2)
        logits_3, res_logits_3, hidden_3 = infer.infer(input_dict_combo_3, hidden_3)
        logits_4, res_logits_4, hidden_4 = infer.infer(input_dict_combo_4, hidden_4)
        logits_5, res_logits_5, hidden_5 = infer.infer(input_dict_combo_5, hidden_5)
        logits_6, res_logits_6, hidden_6 = infer.infer(input_dict_combo_6, hidden_6)
        logits_7, res_logits_7, hidden_7 = infer.infer(input_dict_combo_7, hidden_7)

        # Convert logits to numpy arrays and combine them all combinations logit
        all_combo_logits = np.stack([
            logits_1.numpy(),
            logits_2.numpy(),
            logits_3.numpy(),
            logits_4.numpy(),
            logits_5.numpy(),
            logits_6.numpy(),
            logits_7.numpy()
        ])

        # list of logits for all epochs. This wouldn't be used if we were actually streaming the data.
        all_logits.append(all_combo_logits)

        predicted_classes = get_predicted_classes(all_combo_logits)

        print(f"Predicted classes for epoch {i+1}: {predicted_classes}")


        final_predicted_class = loudest_vote(all_combo_logits)
        print(f"Final predicted class: {final_predicted_class}")
```
