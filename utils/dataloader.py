from datasets import load_from_disk
from torch.utils.data import DataLoader
import os
from .tokenize import tokenize_and_align_labels


def create_kaznerd_dataloader(tokenizer, train_path, test_path, batch_size):
    """
    Load and tokenize the KazNERD dataset
    :param tokenizer: tokenizer object
    :param batch_size: specified in config
    :return: train dataloder, test dataloader, num classes
    """
    kaznerd_train = load_from_disk(train_path)
    kaznerd_test = load_from_disk(test_path)

    kz_labels_list = kaznerd_train.features["ner_tags"].feature.names

    kz_tokenized_train = kaznerd_train.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer":tokenizer,
                                                                                               "task": "ner"})
    kz_tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    kz_tokenized_test = kaznerd_test.map(tokenize_and_align_labels, batched=True, fn_kwargs={"tokenizer":tokenizer,
                                                                                               "task": "ner"})
    kz_tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    kz_train_dataloader = DataLoader(kz_tokenized_train, batch_size=batch_size)
    kz_test_dataloader = DataLoader(kz_tokenized_test, batch_size=batch_size)

    return kz_train_dataloader, kz_test_dataloader, len(kz_labels_list)
