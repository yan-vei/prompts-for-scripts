import torch
from torch import nn


class SoftPrompts(nn.Module):
    """
        Soft Prompts wrapper class.
    """

    def __init__(self, prompt_length, hidden_size):
        super(SoftPrompts, self).__init__()
        self.prompt_length = prompt_length
        self.prompt_embedding = nn.Parameter(torch.randn(prompt_length, hidden_size))

    def forward(self, input_embeddings):
        """
            Concatenate prompt embeddings with the input embeddings.
        :param input_embeddings: embeddings of the input tokens
        :return: concatenated embeddings matrix
        """

        prompt_embeddings = self.prompt_embedding.unsqueeze(0).expand(input_embeddings.size(0), -1, -1)
        return torch.cat([prompt_embeddings, input_embeddings], dim=1)
