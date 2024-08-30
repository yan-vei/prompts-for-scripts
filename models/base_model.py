import torch
from transformers import BertModel

class BertNerd(torch.nn.Module):
    """
        MBert-based model for performing NER tasks
        on Kazakh and Turkish languages.
    """

    def __init__(self, config, freeze=True):
        super(BertNerd, self).__init__()
        self.device = config['DEVICE']
        self.mbert = BertModel.from_pretrained("google-bert/bert-base-multilingual-uncased").to(self.device)
        self.linear = torch.nn.Linear(config['HIDDEN_SIZE'], config['NUM_CLASSES'])

        if freeze:
            self.freeze_params()

        print("\tModel initialized.")

    def forward(self, input_seq, attention_mask):
        """
        Define the model's forward pass.

        :param input_seq: sequence of input tokens
        :param attention_mask: attention mask
        :return: predicted logits
        """
        input_seq = self.mbert(input_seq, attention_mask).last_hidden_state.to(self.device)
        logits = self.linear(input_seq)

        return logits

    def freeze_params(self):
        """
        Only train the soft prompts, don't train any model parameters.

        :return: void
        """

        for param in self.mbert.parameters():
            param.requires_grad = False

    def get_params(self):
        """
        Return tunable parameters of the model.

        :return: list of tunable params
        """

        params = []

        for param in self.parameters():
            if param.requires_grad:
                params.append(param)

        return params