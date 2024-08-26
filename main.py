import torch
import torch.nn as nn
from datasets import load_from_disk
from torch.utils.data import DataLoader
from utils.tokenizer import NERTokenizer
from utils.train import train_ner
from models.base_model import BertNerd

config = {
    'PADDING_TOKEN': -100,
    'LEARNING_RATE': 0.001,
    'NUM_EPOCHS': 10,
    'BATCH_SIZE': 16,
    'RANDOM_SEED': 42,
    'CHUNK_SIZE': 100,
    'HIDDEN_SIZE': 768
}

# Load datasets
kaznerd_train = load_from_disk('datasets/kaznerd-train.hf')
kaznerd_test = load_from_disk('datasets/kaznerd-test.hf')

kz_labels_list = kaznerd_train.features["ner_tags"].feature.names
config['NUM_CLASSES'] = len(kz_labels_list)
config['DEVICE'] = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize tokenizer
tokenizer = NERTokenizer("bert-base-uncased")

# Tokenize and create dataloaders for Kazakh NER dataset
kz_tokenized_train = kaznerd_train.map(lambda e: tokenizer.tokenize_and_align_labels(e, tags='ner_tags'), batched=True, batch_size=config['BATCH_SIZE'])
kz_tokenized_train.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

kz_train_dataloader = DataLoader(kz_tokenized_train, batch_size=config['BATCH_SIZE'])

# Define model, loss function, optimizer
kaznerd_model = BertNerd(config)
loss_func = nn.CrossEntropyLoss(ignore_index=config['PADDING_TOKEN'])
optimizer = torch.optim.Adam(kaznerd_model.get_params(), lr=config['LEARNING_RATE'])

train_ner(model=kaznerd_model, optimizer=optimizer, loss_func=loss_func, train_dataloader=kz_train_dataloader,
          config=config)
