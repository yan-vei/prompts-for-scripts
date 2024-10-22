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


def tokenize_and_align_turkish_qa(examples, tokenizer, max_length=384, doc_stride=128):
    """
    Tokenize and align the THQuAD dataset.
    :param examples: a sequence of tokens
    :param tokenizer: tokenizer object, e.g. BertTokenizer
    :param max_length: maximum length of tokenized sequence (after is truncated)
    :param doc_stride: stride of the sequence

    :return: tokenized_inputs
    """

    tokenized_examples = tokenizer(
        examples['question'],
        examples['context'],
        max_length=max_length,
        truncation=True,
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Initialize lists for start and end positions as well as answers
    start_positions = []
    end_positions = []
    answers_list = []

    # Remap the features to the new indices
    for i, offsets in enumerate(offset_mapping):
        sample_index = sample_mapping[i]
        answers = examples["answers"][sample_index]
        answers_list.append(answers)

        answer_start = answers["answer_start"][0]
        answer_end = answer_start + len(answers["text"][0])

        sequence_ids = tokenized_examples.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if not (offsets[context_start][0] <= answer_start and offsets[context_end][1] >= answer_end):
            start_positions.append(0)
            end_positions.append(0)
        else:
            token_start_index = context_start
            token_end_index = context_end

            while token_start_index <= context_end and offsets[token_start_index][0] <= answer_start:
                token_start_index += 1
            start_positions.append(token_start_index - 1)

            while token_end_index >= context_start and offsets[token_end_index][1] >= answer_end:
                token_end_index -= 1
            end_positions.append(token_end_index + 1)

    # Add the new fields to tokenized_examples
    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    tokenized_examples["answers"] = answers_list

    return tokenized_examples


def tokenize_and_align_kazakh_qa(examples, tokenizer, max_length=512, doc_stride=128, token_type='tokens'):
    """
    Tokenize and align the Kazakh QA dataset.
    :param examples: a sequence of tokens
    :param tokenizer: tokenizer object, e.g. BertTokenizer
    :param max_length: maximum length of tokenized sequence (after is truncated)
    :param doc_stride: stride of the sequence
    :param token_type: tokens or latinized tokens (for Kazakh)

    :return: tokenized_inputs
    """

    # Select the appropriate keys based on token_type
    if token_type == 'latinized':
        question_key = 'latinized_question'
        context_key = 'latinized_context'
        answers_key = 'latinized_answers'
    else:
        question_key = 'question'
        context_key = 'context'
        answers_key = 'answers'


    tokenized_examples = tokenizer(
        examples[question_key],
        examples[context_key],
        max_length=max_length,
        truncation="only_second",
        stride=doc_stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    # Map from tokenized examples to original examples
    sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized_examples.pop("offset_mapping")

    # Initialize lists for start and end positions as well as answers
    start_positions = []
    end_positions = []
    answers_list = []

    # Remap features to the new indices
    for i, offsets in enumerate(offset_mapping):
        sample_index = sample_mapping[i]
        answers = examples[answers_key][sample_index]
        answers_list.append(answers)

        # Check if there are any answers
        if 'text' in answers and len(answers['text']) > 0:
            answer_text = answers['text'][0]
            answer_start = answers['answer_start'][0]
        else:
            # If no answers are present, set default values
            answer_text = ''
            answer_start = 0

        # Handle cases with empty answers
        if len(answer_text) == 0:
            start_positions.append(0)
            end_positions.append(0)
            continue

        answer_end = answer_start + len(answer_text)
        sequence_ids = tokenized_examples.sequence_ids(i)

        # Find the start and end of the context in the tokenized sequence
        idx = 0
        while idx < len(sequence_ids) and sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while idx < len(sequence_ids) and sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        if not (offsets[context_start][0] <= answer_start and offsets[context_end][1] >= answer_end):
            start_positions.append(0)
            end_positions.append(0)
        else:
            token_start_index = context_start
            while token_start_index <= context_end and offsets[token_start_index][0] <= answer_start:
                token_start_index += 1
            start_positions.append(token_start_index - 1)

            token_end_index = context_end
            while token_end_index >= context_start and offsets[token_end_index][1] >= answer_end:
                token_end_index -= 1
            end_positions.append(token_end_index + 1)

    # Form the new tokenized examples objects
    tokenized_examples["start_positions"] = start_positions
    tokenized_examples["end_positions"] = end_positions
    tokenized_examples["answers"] = answers_list

    return tokenized_examples
