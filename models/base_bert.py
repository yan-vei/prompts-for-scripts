import torch
from transformers import BertModel


class BertNerd(torch.nn.Module):
    """
        MBert-based model for performing NER tasks
        on Kazakh and Turkish languages.
    """

    def __init__(self, name, device, hidden_size, num_classes, freeze=False):
        super(BertNerd, self).__init__()
        self.device = device
        self.mbert = BertModel.from_pretrained(name)
        self.linear = torch.nn.Linear(hidden_size, num_classes)

        if freeze:
            self.freeze_params()

        print(f"\tModel {name} with hidden size {hidden_size} initialized.")

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
