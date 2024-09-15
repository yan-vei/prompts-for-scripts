def tokenize_and_align_labels_ner(examples, tags, token_type, tokenizer, padding_token=-100, labels_dict=None, str2int=False):
    """
    Tokenizing function for NER task for Kazakh and Turkish

    :param examples: a sequence of tokens
    :param tags: tagged sequence of tokens
    :param token_type: tokens or latinized tokens (for Kazakh)
    :param tokenizer: tokenizer object
    :param padding_token: padding token, default -100 in hydra config
    :param labels_dict: dict with all the NER labels
    :param str2int: flag to handle case with unindexed labels in the dataset (Turkish NER only)
    :return: tokenized_inputs
    """

    # The tokenizing function used for the datasets is based on the one
    # implemented by the authors of KazNERD paper, but modified for accommodating
    # multiple datasets' properties
    # Direct link: https://github.com/IS2AI/KazNERD/blob/main/bert/run_finetune_kaznerd.py

    tokenized_inputs = tokenizer(examples[token_type], truncation=True,
                                 is_split_into_words=True, padding=True)

    tokenized_labels = examples[tags]

    # Handle the case when labels in the dataset haven't been indexed
    if str2int:
        tokenized_labels = [[labels_dict[tag] for tag in seq] for seq in examples[tags]]

    labels = []
    for i, label in enumerate(tokenized_labels):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []

        # Only the first token gets assigned the NER tag,
        # and the rest get assigned the padding token of "-100"
        # which safely gets ignored by the loss function
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(padding_token)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(padding_token)
            previous_word_idx = word_idx

        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs
