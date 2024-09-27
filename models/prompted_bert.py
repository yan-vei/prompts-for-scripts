import torch
from transformers import BertModel, BertForTokenClassification, BertForQuestionAnswering
from peft import PeftModel


class PromptedBert(torch.nn.Module):
    def __init__(self, device, name, num_classes, soft_prompts_path, num_orig_ner_labels, hidden_size=768,
                 freeze=True):
        super().__init__()
        self.device = device
        self.name = name

        # Model layers
        self.mbert = BertModel.from_pretrained(self.name)
        self.linear = torch.nn.Linear(hidden_size, num_classes)

        # Soft prompts
        self.soft_prompts_path = soft_prompts_path
        self.num_orig_ner_labels = num_orig_ner_labels # original number of labels in Turkish NER
        self.soft_prompts = self.extract_soft_prompts().to(self.device)
        self.num_tokens = self.soft_prompts.size(0)

        if freeze:
            self.freeze_params()

    def extract_soft_prompts(self):
        """
        Extract soft prompts form the trained PEFT model.
        :return: soft prompts form the trained PEFT model
        """

        # Because we trained on Turkish NER which has a different amount of labels,
        # We cannot use the classifier layer from that model directly
        # So we do a workaround to only extract the soft prompts matrix
        # And then append it to our inputs

        model = BertForTokenClassification.from_pretrained(self.name, num_labels=self.num_orig_ner_labels)
        peft_model = PeftModel.from_pretrained(model, self.soft_prompts_path)
        soft_prompts = peft_model.prompt_encoder['default'].embedding.weight.data

        return soft_prompts

    def freeze_params(self):
        """
        Only train the soft prompts, don't train any model parameters.
        :return: void
        """

        for param in self.mbert.parameters():
            param.requires_grad = False

    def forward(self, input_seq, attention_mask):
        """
        Forward pass for the model with preprending soft prompts.
        :param input_seq: input_ids 
        :param attention_mask: attention_mask
        :return: logits
        """""

        # Expand soft prompts to match the batch size
        batch_size = input_seq.size(0)
        expanded_prompts = self.soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)

        # Embedd the inputs with the model to match the dimensions of the soft prompts
        with torch.no_grad():
            embedded_inputs = self.mbert.embeddings(input_seq)

        # Concatenate soft prompts with the input embeddings
        input_with_prompts = torch.cat([expanded_prompts, embedded_inputs], dim=1)

        # Expand the attention mask to match soft prompts dimensions
        prompt_mask = torch.ones((batch_size, self.num_tokens), device=attention_mask.device)
        attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # Make a forward pass
        outputs = self.mbert(
            inputs_embeds=input_with_prompts,
            attention_mask=attention_mask
        )

        sequence_output = outputs.last_hidden_state
        logits = self.linear(sequence_output)

        return logits
