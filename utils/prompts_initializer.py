from peft import PromptTuningConfig
import torch


def initialize_randomly(num_tokens):
    """
    Randomly initialize the soft prompt embeddings.
    :param num_tokens: int, number of tokens in the soft prompt
    :return: PromptTuningConfig
    """

    return PromptTuningConfig(num_virtual_tokens=num_tokens)


def initialize_with_task(num_tokens, task, model, tokenizer):
    """
    Initialize the soft prompt embeddings with keywords for a given task.
    :param num_tokens: int, number of tokens in the soft prompt
    :param task: NER or QAD
    :param model: e.g. mBERT
    :param tokenizer: e.g. BertTokenizer
    :return: PromptTuningConfig
    """

    if task == 'NER':
        init_tokens = ['entity', 'label', 'tag', 'identify', 'recognize']

    token_ids = tokenizer(init_tokens, add_special_tokens=False).input_ids
    token_embeddings = model.get_input_embeddings()(torch.tensor(token_ids))

    return PromptTuningConfig(
        num_virtual_tokens=num_tokens,
        prompt_init_embeddings=token_embeddings
    )


def initialize_normal(num_tokens, hidden_size):
    """
    Initialize the soft prompt embeddings with normal distribution.
    :param num_tokens: int, number of tokens in the soft prompt
    :param hidden_size: hidden dim of the model, e.g. 768
    :return: PromptTuningConfig
    """

    init_embeddings = torch.randn(num_tokens, hidden_size)

    return PromptTuningConfig(
        num_virtual_tokens=num_tokens,
        prompt_init_embeddings=init_embeddings
    )
