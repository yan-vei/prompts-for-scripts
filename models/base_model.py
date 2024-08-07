import torch
from transformers import AutoModelForMaskedLM

class BertNerd(torch.nn.Module):
    """
        MBert-based model for performing NER tasks w/ and w/o
        soft prompts on Kazakh and Turkish languages.
    """

    def __init__(self, config, device, freeze=True):
        super(BertNerd, self).__init__()
        self.mbert = AutoModelForMaskedLM("google-bert/bert-base-multilingual-cased")
        self.linear = torch.nn.Linear(config.hidden_size, config.num_classes)
        self.device = device

        if freeze:
            self.freeze_params()

    def forward(self, input_seq, attention_mask):
        """
        Define the model's forward pass.
        :param input_seq: sequence of input tokens
        :param attention_mask: attention mask
        :return: logits
        """

        input_seq = self.mbert(input_seq, attention_mask).last_hidden_state.to(self.device)
        logits = self.linear(input_seq)

        return logits

    def get_loss(self, loss_fn, logits, labels, ignore_index=None):
        """
        Get loss for the forward pass of the current batch.

        :param loss_fn: e.g. nn.CrossEntropyLoss
        :param logits: predicted labels
        :param labels: actual labels from the dataset
        :param ignore_index: padding index to ignore
        :return: loss per batch
        """

        loss_func = loss_fn(ignore_index=ignore_index)

        # ToDo
        # Logits/labels should probably be flattened, so we get the right dimension

        return loss_func(logits, labels).item()


    def freeze_params(self):
        """
        Only train the soft prompts, don't train any model parameters.
        :return: void
        """

        # ToDo
        # Freeze MBert parameters, only train soft embeddings
        # Check "mbert" params in the model and freeze them

        pass