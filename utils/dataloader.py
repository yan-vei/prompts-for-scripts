from datasets import load_from_disk
from torch.utils.data import DataLoader
from .tokenize import tokenize_and_align_labels


def create_kaznerd_dataloader(tokenizer, train_path, test_path, padding_token, batch_size):
    """
    Load and tokenize the KazNERD dataset
    :param tokenizer: tokenizer object
    :param batch_size: specified in config
    :param train_path: path to train dataset
    :param test_path: path to test dataset
    :param padding_token: padding token
    :return: train dataloder, test dataloader, num classes
    """
    kaznerd_train = load_from_disk(train_path)
    kaznerd_test = load_from_disk(test_path)

    kz_labels_list = kaznerd_train.features["ner_tags"].feature.names

    kz_tokenized_train = kaznerd_train.map(lambda e: tokenize_and_align_labels(e, tokenizer=tokenizer,
                                                                               padding_token=padding_token,
                                                                               tags='ner_tags'), batched=True,
                                            batch_size=batch_size)
    kz_tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


    kz_tokenized_test = kaznerd_test.map(lambda e: tokenize_and_align_labels(e, tokenizer=tokenizer,
                                                                               padding_token=padding_token,
                                                                               tags='ner_tags'), batched=True,
                                            batch_size=batch_size)
    kz_tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    kz_train_dataloader = DataLoader(kz_tokenized_train, batch_size=batch_size)
    kz_test_dataloader = DataLoader(kz_tokenized_test, batch_size=batch_size)

    return kz_train_dataloader, kz_test_dataloader, len(kz_labels_list)
