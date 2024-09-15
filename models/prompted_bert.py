import torch
from transformers import BertModel, BertForTokenClassification
from peft import PeftModel


class PromptedBert(torch.nn.Module):
    def __init__(self, device, name, num_classes, soft_prompts_path, hidden_size=768):
        super().__init__()
        self.device = device
        self.model = BertModel.from_pretrained(name)
        self.linear = torch.nn.Linear(hidden_size, num_classes)
        self.soft_prompts = self.extract_soft_prompts(name, soft_prompts_path).to(self.device)
        self.num_prompts = self.soft_prompts.size(0)

        for param in self.model.parameters():
            param.requires_grad = False

    def extract_soft_prompts(self, name, soft_prompts_path):
        model = BertForTokenClassification.from_pretrained(name, num_labels=39)
        peft_model = PeftModel.from_pretrained(model, soft_prompts_path)
        soft_prompts = peft_model.prompt_encoder['default'].embedding.weight.data

        return soft_prompts

    def forward(self, input_seq, attention_mask):
        """
        Forward pass for the model. Soft prompts are prepended to the input embeddings.

        Args:
            input_ids (torch.Tensor): Token IDs from the input sequences.
            attention_mask (torch.Tensor, optional): Attention mask.
            token_type_ids (torch.Tensor, optional): Token type IDs.

        Returns:
            torch.Tensor: Output logits from the classifier.
        """

        # Expand soft prompts to match the batch size
        batch_size = input_seq.size(0)
        expanded_prompts = self.soft_prompts.unsqueeze(0).expand(batch_size, -1, -1)
        with torch.no_grad():
            embedded_inputs = self.model.embeddings(input_seq)

        # Concatenate soft prompts with the input embeddings
        input_with_prompts = torch.cat([expanded_prompts, embedded_inputs], dim=1)


        # Create a new attention mask to include the soft prompts
        if attention_mask is not None:
            prompt_mask = torch.ones((batch_size, self.num_prompts), device=attention_mask.device)
            attention_mask = torch.cat([prompt_mask, attention_mask], dim=1)

        # Pass the modified embeddings through BERT's encoder
        outputs = self.model(
            inputs_embeds=input_with_prompts,
            attention_mask=attention_mask
        )

        # Take the last hidden states
        sequence_output = outputs.last_hidden_state

        # Apply the classifier to the last hidden state
        logits = self.linear(sequence_output)
        return logits