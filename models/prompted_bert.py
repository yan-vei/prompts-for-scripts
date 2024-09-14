import torch
from transformers import BertForTokenClassification
from peft import PeftModel


class PromptedBertNerd(torch.nn.Module):
    """
        MBert-based model for performing NER tasks
        on Kazakh and Turkish languages.
    """

    def __init__(self, name, device, hidden_size, num_classes, soft_prompts_path, freeze=True):
        super(PromptedBertNerd, self).__init__()
        self.device = device
        self.mbert = BertForTokenClassification.from_pretrained(name, num_labels=39)
        self.peft_model = PeftModel.from_pretrained(self.mbert, soft_prompts_path)

        for param in self.peft_model.bert.parameters():
            param.requires_grad = False

        self.peft_model.classifier = torch.nn.Linear(self.peft_model.config.hidden_size, num_classes)

        # Optional: Initialize the new classifier weights with the pretrained BERT model weights for a smoother start
        self.peft_model.classifier.weight.data = self.peft_model.bert.embeddings.word_embeddings.weight.data[:num_classes]
        print(self.peft_model)
        print(f"\tPEFT Model {name} with hidden size {hidden_size} initialized.")

    def forward(self, input_ids, attention_mask, labels=None):
        """
        Define the model's forward pass.
        :param input_ids: Tokenized input sequence
        :param attention_mask: Attention mask for input
        :param labels: Optional labels for computing loss (during training)
        :return: logits or (logits, loss) based on input arguments
        """
        return self.peft_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

