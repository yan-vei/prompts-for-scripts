def tokenize_and_align_labels(examples, tokenizer, task, label_all_tokens=False):
    """
    Taken from https://github.com/IS2AI/KazNERD/blob/main/bert/run_finetune_kaznerd.py#L16
    """

    tokenized_inputs = tokenizer(examples["tokens"], truncation=True,
                        is_split_into_words=True)

    labels = []
    for i, label in enumerate(examples[f"{task}_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are
            # automatically ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label or
                # -100, depending on the label_all_tokens flag.
            else:
                label_ids.append(label[word_idx] if label_all_tokens else -100)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs