from datasets import load_from_disk
import torch
from torch.utils.data import DataLoader
from .tokenize import tokenize_and_align_labels_ner, tokenize_and_align_turkish_qa, tokenize_and_align_kazakh_qa


def create_turkish_ner_dataloader(tokenizer, token_type, train_path, test_path, padding_token, batch_size):
    """
    Load and tokenize the TurkishNER dataset
    :param tokenizer: tokenizer object
    :param train_path: path to train dataset
    :param test_path: path to test dataset
    :param padding_token: padding token
    :param token_type: latin or cyrillic tokens
    :param batch_size: specified in config
    :return: train dataloader, test dataloader, num classes
    """

    turkish_ner_train = load_from_disk(train_path)
    turkish_ner_test = load_from_disk(test_path)

    # Feature names are not directly available in Turkish NER dataset,
    # so we manually extract them
    tr_labels_list = {tag for seq in turkish_ner_train for tag in seq['tags']}
    tr_labels_dict = {label: index for index, label in enumerate(tr_labels_list)}

    # Tokenize and create dataloaders for Turkish NER dataset
    tr_tokenized_train = turkish_ner_train.map(
        lambda e: tokenize_and_align_labels_ner(e, tags='tags', token_type=token_type, tokenizer=tokenizer,
                                                padding_token=padding_token, labels_dict=tr_labels_dict, str2int=True),
        batch_size=batch_size, batched=True)
    tr_tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    tr_tokenized_test = turkish_ner_test.map(
        lambda e: tokenize_and_align_labels_ner(e, tags='tags', token_type=token_type, tokenizer=tokenizer,
                                                padding_token=padding_token, labels_dict=tr_labels_dict, str2int=True),
        batch_size=batch_size, batched=True)
    tr_tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    tr_train_dataloder = DataLoader(tr_tokenized_train, batch_size=batch_size)
    tr_test_dataloader = DataLoader(tr_tokenized_test, batch_size=batch_size)

    return tr_train_dataloder, tr_test_dataloader, len(tr_labels_list)


def create_kaznerd_dataloader(tokenizer, token_type, train_path, test_path, padding_token, batch_size, is_subset=True):
    """
    Load and tokenize the KazNERD dataset
    :param tokenizer: tokenizer object
    :param batch_size: specified in config
    :param train_path: path to train dataset
    :param test_path: path to test dataset
    :param token_type: type of the tokens (latinized/not)
    :param padding_token: padding token
    :return: train dataloader, test dataloader, num classes
    """
    kaznerd_train = load_from_disk(train_path)
    kaznerd_test = load_from_disk(test_path)

    # List of labels is necessary to fix the amount of classes
    kz_labels_list = kaznerd_train.features["ner_tags"].feature.names

    # Select a random subset of KAZnerd to compare to Turkish wiki NER
    # There are roughly 18k training examples and 1k validation and test examples in Turkish NER
    if is_subset is True:
        kaznerd_train = kaznerd_train.shuffle(seed=42).select(range(18000))
        kaznerd_test = kaznerd_test.shuffle(seed=42).select(range(1000))

    # Tokenize and create dataloaders for KazNERD dataset
    kz_tokenized_train = kaznerd_train.map(lambda e: tokenize_and_align_labels_ner(e, tags='ner_tags',
                                                                                   token_type=token_type,
                                                                                   tokenizer=tokenizer,
                                                                                   padding_token=padding_token), batched=True,
                                           batch_size=batch_size)
    kz_tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


    kz_tokenized_test = kaznerd_test.map(lambda e: tokenize_and_align_labels_ner(e, tags='ner_tags',
                                                                                 token_type=token_type,
                                                                                 tokenizer=tokenizer,
                                                                                 padding_token=padding_token), batched=True,
                                         batch_size=batch_size)
    kz_tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    kz_train_dataloader = DataLoader(kz_tokenized_train, batch_size=batch_size)
    kz_test_dataloader = DataLoader(kz_tokenized_test, batch_size=batch_size)

    return kz_train_dataloader, kz_test_dataloader, len(kz_labels_list)


def create_turkish_qa_dataloader(tokenizer, train_path, test_path, batch_size, max_length=384, doc_stride=128):
    """
    Load and tokenize the TurkishNER dataset
    :param tokenizer: tokenizer object
    :param train_path: path to train dataset
    :param test_path: path to test dataset
    :param max_length: maximum length of sequence
    :param doc_stride: document stride
    :param batch_size: specified in config
    :return: train dataloader, test dataloader, num classes
    """

    turkish_qa_train = load_from_disk(train_path)
    turkish_qa_test = load_from_disk(test_path)

    # Tokenize and create dataloaders for Turkish QA dataset
    tr_tokenized_train = turkish_qa_train.map(
        lambda e: tokenize_and_align_turkish_qa(e, tokenizer=tokenizer, max_length=max_length, doc_stride=doc_stride),
        batched=True,
        remove_columns=turkish_qa_train.column_names,
    )
    tr_tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])

    tr_tokenized_test = turkish_qa_test.map(
        lambda e: tokenize_and_align_turkish_qa(e, tokenizer=tokenizer, max_length=max_length, doc_stride=doc_stride),
        batched=True, batch_size=batch_size,
        remove_columns=['question', 'id', 'context', 'title'])
    tr_tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions', 'answers'])

    tr_train_dataloader = DataLoader(tr_tokenized_train, batch_size=batch_size)
    tr_test_dataloader = DataLoader(tr_tokenized_test, batch_size=batch_size)

    return tr_train_dataloader, tr_test_dataloader


def create_kazakh_qa_dataloader(tokenizer, train_path, test_path, batch_size, token_type="tokens", max_length=384, doc_stride=128):
    """
    Load and tokenize the TurkishNER dataset
    :param tokenizer: tokenizer object
    :param train_path: path to train dataset
    :param test_path: path to test dataset
    :param max_length: maximum length of sequence
    :param doc_stride: document stride
    :param batch_size: specified in config
    :return: train dataloader, test dataloader, num classes
    """

    kazakh_qa_train = load_from_disk(train_path)
    kazakh_qa_test = load_from_disk(test_path)

    # Tokenize and create dataloaders for Turkish QA dataset
    kz_tokenized_train = kazakh_qa_train.map(
        lambda e: tokenize_and_align_kazakh_qa(e, tokenizer=tokenizer, max_length=max_length, doc_stride=doc_stride, token_type=token_type),
        batched=True,
        remove_columns=kazakh_qa_train.column_names,
    )
    kz_tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'])

    kz_tokenized_test = kazakh_qa_test.map(
        lambda examples: tokenize_and_align_kazakh_qa(examples, tokenizer, token_type=token_type),
        batched=True,
        remove_columns=kazakh_qa_test.column_names
    )
    kz_tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'start_positions', 'end_positions'], output_all_columns=True)

    kz_train_dataloader = DataLoader(kz_tokenized_train, shuffle=False, drop_last=True, batch_size=batch_size)
    kz_test_dataloader = DataLoader(kz_tokenized_test, shuffle=False, drop_last=True, batch_size=batch_size, collate_fn=custom_collate_fn)

    return kz_train_dataloader, kz_test_dataloader


def custom_collate_fn(batch):
    """
        Custom collate function to make sure answers are tokenized properly.
        :param batch: list of examples
        :return: batched examples
    """

    # Stack tensor fields
    tensor_keys = ['input_ids', 'attention_mask', 'start_positions', 'end_positions']
    batch_tensors = {key: torch.stack([item[key] for item in batch]) for key in tensor_keys}

    # Keep 'answers' as a list
    batch_tensors['answers'] = [item['answers'] for item in batch]

    return batch_tensors
