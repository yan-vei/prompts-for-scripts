from peft import PromptTuningConfig, TaskType, get_peft_model, PromptTuningInit
import torch


def initialize_randomly(num_tokens, task, model):
    """
    Randomly initialize the soft prompt embeddings.
    :param num_tokens: int, number of tokens in the soft prompt
    :param task: str, name of the task
    :param model: str, e.g. mBERT
    :return: PromptTuningConfig
    """
    if task == 'NER':
        peft_config = PromptTuningConfig(num_virtual_tokens=num_tokens, task_type=TaskType.TOKEN_CLS)

    return get_peft_model(model, peft_config)


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
        task_type = TaskType.TOKEN_CLS

    peft_config = PromptTuningConfig(
        num_virtual_tokens=num_tokens,
        prompt_tuning_init=PromptTuningInit.TEXT,
        prompt_tuning_init_text=" ".join(init_tokens),
        task_type=task_type,
        tokenizer_name_or_path=tokenizer
    )

    return get_peft_model(model, peft_config)


def initialize_normal(num_tokens, hidden_size, task, model):
    """
    Initialize the soft prompt embeddings with normal distribution.
    :param num_tokens: int, number of tokens in the soft prompt
    :param hidden_size: hidden dim of the model, e.g. 768
    :return: PromptTuningConfig
    """

    init_embeddings = torch.randn(num_tokens, hidden_size)

    if task == 'NER':
        peft_config = PromptTuningConfig(
            num_virtual_tokens=num_tokens,
            task_type=TaskType.TOKEN_CLS
        )

    peft_model = get_peft_model(model, peft_config)

    # Replace the soft embeddings random init with normal
    with torch.no_grad():
        peft_model.prompt_encoder['default'].embedding.weight.data.copy_(init_embeddings)

    return peft_model
