import torch
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit
from utils.dataloader import create_kaznerd_dataloader
from logging_config import settings
from utils.train import train_ner, train_ner_soft_prompts
from models.base_bert import BertNerd


# Define model, loss function, optimizer
#kaznerd_model = SoftPromptedBertNerd(config).to(config['DEVICE'])
#loss_func = nn.CrossEntropyLoss(ignore_index=config['PADDING_TOKEN'])
#optimizer = torch.optim.Adam(list(kaznerd_model.soft_prompts.parameters()) + list(kaznerd_model.linear.parameters()), lr=config['LEARNING_RATE'])

#train_ner_soft_prompts(model=kaznerd_model, optimizer=optimizer, loss_func=loss_func, train_dataloader=kz_train_dataloader,
          #config=config)

#model = AutoModelForCausalLM.from_pretrained('bert-base-multilingual-uncased')
#model = get_peft_model(model, peft_config).to(device)
#print(model.print_trainable_parameters())

#optimizer = torch.optim.AdamW(model.parameters(), lr=config['LEARNING_RATE'])

@hydra.main(config_path='configs', config_name='defaults', version_base=None)
def run_pipeline(cfg: DictConfig):
    """
    Define the run managed by hydra configs
    :param cfg:
    :return: void
    """

    # Log into wandb if required
    use_wandb = cfg.basic.use_wandb
    if use_wandb:
        cfg_copy = OmegaConf.to_container(cfg, resolve=True)
        wandb.login(key=settings.WANDB_API_KEY)
        wandb.init(project=cfg.dataset.name, name=cfg.basic.wandb_run,
                   config=cfg_copy)

    # Initialize the loss function
    lossfn = hydra.utils.instantiate(cfg.loss)

    # Initialize a device (cpu, gpu or mps (Apple Sillicon))
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    print(f'\tConfigured loss function: {str(lossfn)} and device: {device}.')

    if cfg.basic.task == 'NER':
        run_ner_pipeline(cfg, lossfn, device, cfg.basic.use_wandb)

    if use_wandb:
        wandb.finish()


def run_ner_pipeline(cfg: DictConfig, lossfn, device, use_wandb=False):

    # Instantiate MBert model
    model = AutoModelForCausalLM.from_pretrained(cfg.model.name)

    # Instantiate tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)

    if cfg.basic.lang == 'KZ':
        train_dataloder, test_dataloader, num_classes = create_kaznerd_dataloader(tokenizer, cfg.dataset.train_path,
                                                                                  cfg.dataset.test_path,
                                                                                  cfg.dataset.batch_size)

    if cfg.basic.with_soft_prompts is False:
        pass
    elif cfg.basic.with_soft_prompts is True:
        peft_config = PromptTuningConfig(
            task_type=TaskType.CAUSAL_LM,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=cfg.soft_prompts.num_virtual_tokens,
            prompt_tuning_init_text="Classify NER tokens",
            tokenizer_name_or_path=cfg.tokenizer.name,
        )
        model = get_peft_model(model, peft_config).to(device)


if __name__ == "__main__":
    run_pipeline()