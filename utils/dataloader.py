from datasets import load_from_disk
from torch.utils.data import DataLoader
from .tokenize import tokenize_and_align_labels


def create_turkish_ner_dataloader(tokenizer, token_type, train_path, test_path, padding_token, batch_size):
    """
    Load and tokenize the TurkishNER dataset
    :param tokenizer: tokenizer object
    :param train_path: path to train dataset
    :param test_path: path to test dataset
    :param padding_token: padding token
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
        lambda e: tokenize_and_align_labels(e, token_type=token_type, tokenizer=tokenizer, padding_token=padding_token, tags='tags', labels_dict=tr_labels_dict, str2int=True),
        batch_size=batch_size, batched=True)
    tr_tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    tr_tokenized_test = turkish_ner_test.map(
        lambda e: tokenize_and_align_labels(e, tokenizer=tokenizer, token_type=token_type, padding_token=padding_token, tags='tags', labels_dict=tr_labels_dict, str2int=True),
        batch_size=batch_size, batched=True)
    tr_tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    tr_train_dataloder = DataLoader(tr_tokenized_train, batch_size=batch_size)
    tr_test_dataloader = DataLoader(tr_tokenized_test, batch_size=batch_size)

    return tr_train_dataloder, tr_test_dataloader, len(tr_labels_list)


def create_kaznerd_dataloader(tokenizer, token_type, train_path, test_path, padding_token, batch_size):
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

    # List of labels is necessary to fix the amount of classes
    kz_labels_list = kaznerd_train.features["ner_tags"].feature.names

    # Tokenize and create dataloaders for KazNERD dataset
    kz_tokenized_train = kaznerd_train.map(lambda e: tokenize_and_align_labels(e, tokenizer=tokenizer,
                                                                               padding_token=padding_token,
                                                                               token_type=token_type,
                                                                               tags='ner_tags'), batched=True,
                                            batch_size=batch_size)
    kz_tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


    kz_tokenized_test = kaznerd_test.map(lambda e: tokenize_and_align_labels(e, tokenizer=tokenizer,
                                                                             token_type=token_type,
                                                                               padding_token=padding_token,
                                                                               tags='ner_tags'), batched=True,
                                            batch_size=batch_size)
    kz_tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    kz_train_dataloader = DataLoader(kz_tokenized_train, batch_size=batch_size)
    kz_test_dataloader = DataLoader(kz_tokenized_test, batch_size=batch_size)

    return kz_train_dataloader, kz_test_dataloader, len(kz_labels_list)
