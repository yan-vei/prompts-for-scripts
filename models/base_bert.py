import torch
from transformers import BertModel


class BertNerd(torch.nn.Module):
    """
        mBERT-based model for performing NER tasks
        on Kazakh and Turkish languages.
    """

    def __init__(self, name, device, hidden_size, num_classes, freeze=True):
        super(BertNerd, self).__init__()
        self.device = device
        self.mbert = BertModel.from_pretrained(name)
        self.linear = torch.nn.Linear(hidden_size, num_classes)

        if freeze:
            self.freeze_params()

        print(f"\tModel {name} for NER with hidden size {hidden_size} initialized.")

    def forward(self, input_seq, attention_mask):
        """
        Define the model's forward pass.

        :param input_seq: sequence of input tokens
        :param attention_mask: attention mask
        :return: predicted logits
        """
        output = self.mbert(input_seq, attention_mask).last_hidden_state
        logits = self.linear(output)

        return logits

    def freeze_params(self):
        """
        Only train the soft prompts, don't train any model parameters.

        :return: void
        """

        for param in self.mbert.parameters():
            param.requires_grad = False


class BertQA(torch.nn.Module):
    """
        mBERT-based model for extractive QA in Kazakh and Turkish languages.
    """

    def __init__(self, name, device, hidden_size, freeze=True):
        super(BertQA, self).__init__()
        self.device = device
        self.mbert = BertModel.from_pretrained(name)

        # Initialize 2 linear layers for classification of the starting
        # and ending position for each question in each context
        self.linear_start = torch.nn.Linear(hidden_size, 1)
        self.linear_end = torch.nn.Linear(hidden_size, 1)

        if freeze:
            self.freeze_params()

        print(f"\tModel {name} for extractive QA with hidden size {hidden_size} initialized.")

    def forward(self, input_seq, attention_mask):
        """
        Define the model's forward pass.

        :param input_seq: sequence of input tokens
        :param attention_mask: attention mask
        :return: predicted logits
        """
        output = self.mbert(input_seq, attention_mask).last_hidden_state

        # Detect the beginning and the end of the answer
        # in the provided context
        start_logits = self.linear_start(output).squeeze(-1)
        end_logits = self.linear_end(output).squeeze(-1)

        return start_logits, end_logits

    def freeze_params(self):
        """
        Only train the soft prompts, don't train any model parameters.

        :return: void
        """

        for param in self.mbert.parameters():
            param.requires_grad = False
