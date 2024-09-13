import random


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


def tokenize_and_align_retrieval_pairs(batch, tokenizer):
    """
    Tokenize queries, positive and negative passages for QAD
    :param batch: queries, positive and negative passages
    :param tokenizer: e.g. mBERT tokenizer
    :return: dict with tokenized queries, positive and negative passages
    """

    query_encodings = tokenizer(
        batch['query'],
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )

    positive_encodings = tokenizer(
        batch['positive_passage'],
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )

    negative_encodings = tokenizer(
        batch['negative_passage'],
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )

    return {
        'query_input_ids': query_encodings['input_ids'].squeeze(),
        'query_attention_mask': query_encodings['attention_mask'].squeeze(),
        'positive_input_ids': positive_encodings['input_ids'].squeeze(),
        'positive_attention_mask': positive_encodings['attention_mask'].squeeze(),
        'negative_input_ids': negative_encodings['input_ids'].squeeze(),
        'negative_attention_mask': negative_encodings['attention_mask'].squeeze()
    }


def get_retrieval_pairs(example, dataset):
    """
    Organize retrieval pairs from QAD. Make sure that negative contexts are taken
    from different samples in the same dataset.
    :param example: current example
    :param dataset: Dataset object
    :return: dict with query, positive and negative passages, and answer
    """

    query = example['question']
    positive_passage = example['context']

    # Select a random context from another example for a negative passage
    random_idx = random.randint(0, len(dataset) - 1)
    while dataset[random_idx]['id'] == example['id']:
        random_idx = random.randint(0, len(dataset) - 1)

    negative_passage = dataset[random_idx]['context']

    return {
        'query': query,
        'positive_passage': positive_passage,
        'negative_passage': negative_passage,
        'answer': example['answers']
    }