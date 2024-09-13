import torch
from transformers import BertModel
from peft import PeftModel, PeftConfig


class BertNerd(torch.nn.Module):
    """
        MBert-based model for performing NER tasks
        on Kazakh and Turkish languages.
    """

    def __init__(self, name, device, hidden_size, num_classes, soft_prompts_path=None, freeze=True):
        super(BertNerd, self).__init__()
        self.device = device
        self.mbert = BertModel.from_pretrained(name)

        # If model is to be used with soft prompts, attach the matrix to the model
        if soft_prompts_path is not None:
            self.mbert = PeftModel.from_pretrained(self.mbert, soft_prompts_path)

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