import torch
from transformers import BertModel

class BertNerd(torch.nn.Module):
    """
        MBert-based model for performing NER tasks
        on Kazakh and Turkish languages.
    """

    def __init__(self, name, device, hidden_size, num_classes, freeze=True):
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
        if torch.isnan(input_seq).any() or torch.isinf(input_seq).any():
            print("NaN or Inf detected in input_seq")
        if torch.isnan(attention_mask).any() or torch.isinf(attention_mask).any():
            print("NaN or Inf detected in attention_mask")

        print(f"input_seq device: {input_seq.device}, attention_mask device: {attention_mask.device}")
        print(f"input_seq shape: {input_seq.shape}, attention_mask shape: {attention_mask.shape}")
        print(f"mbert device: {next(self.mbert.parameters()).device}")

        torch.cuda.synchronize()
        output = self.mbert(input_seq, attention_mask)
        print(f"Output shape: {output.shape}")
        last_hidden_state = output.last_hidden_state
        print(
            f"Last hidden state device: {last_hidden_state.device}, dtype: {last_hidden_state.dtype}, min: {last_hidden_state.min().item()}, max: {last_hidden_state.max().item()}")
        logits = self.linear(last_hidden_state)
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