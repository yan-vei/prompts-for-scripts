# The tokenizing function used for the datasets is based on the one
# implemented by the authors of KazNERD paper, but modified for accommodating
# multiple datasets' properties
# Direct link: https://github.com/IS2AI/KazNERD/blob/main/bert/run_finetune_kaznerd.py

# Padding is not essential since we are doing mini-batches of size 1, however,
# the feature is kept for compatibility reasons


def tokenize_and_align_labels(examples, tags, tokenizer, padding_token=-100, labels_dict=None, str2int=False):
    """
      Tokenizing, padding, and aligning labels in the example sentences

      Input:
      examples - dict, dictionaries with word tokens and assigned NER tags
      tags - str, what is the key for tags in the examples object
      labels_dict - None | dict, dictionary object for label conversion in case labels are strings
      str2int - None | bool, flag to check whether label conversion is necessary

      Output:
      tokenized_inputs - dict, processed sentences with aligned labels and padded sentences
    """

    tokenized_inputs = tokenizer(examples['tokens'], truncation=True,
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