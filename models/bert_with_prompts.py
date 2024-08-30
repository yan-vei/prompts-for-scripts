import torch
from transformers import BertModel
from prompts.soft_prompts import SoftPrompts


class SoftPromptedBertNerd(torch.nn.Module):
    """
        MBert-based model with soft prompts for Kazakh
        and Turkish language tasks.
    """

    def __init__(self, config, freeze=True):
        super(SoftPromptedBertNerd, self).__init__()
        self.device = config['DEVICE']
        self.prompt_length = config['PROMPT_LENGTH']

        self.mbert = BertModel.from_pretrained("google-bert/bert-base-multilingual-uncased").to(self.device)
        self.soft_prompts = SoftPrompts(self.prompt_length, self.mbert.config.hidden_size)

        print("\tModel initialized.")

    def forward(self, input_ids, attention_mask):
        """
        Define the model's forward pass.

        :param input_ids: sequence of input tokens
        :param attention_mask: attention mask
        :return: predicted outputs
        """
        input_embeddings = self.mbert.embeddings(input_ids)
        extended_embeddings = self.soft_prompts(input_embeddings)
        extended_attention_mask = torch.cat([torch.ones(input_ids.size(0), self.prompt_length).to(self.config['DEVICE']),
                                             attention_mask], dim=1)
        outputs = self.mbert(inputs_embeds=extended_embeddings, attention_mask=extended_attention_mask)

        return outputs