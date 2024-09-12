import torch
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit
from utils.dataloader import create_kaznerd_dataloader, create_turkish_ner_dataloader
from logging_config import settings
from utils.train import train_ner, evaluate_ner, train_ner_with_soft_prompts
from models.base_bert import BertNerd
from models.prompted_bert import PromptedBertNerd


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

        # Uncomment for sweeps
        project_name = str(cfg.dataset.name) + "_" + str(cfg.lang) + "_" + str(cfg.soft_prompts.num_virtual_tokens)
        #project_name = cfg.dataset.name

        wandb.init(project=project_name, name=cfg.basic.wandb_run,
                   config=cfg_copy)

    # Initialize the loss function
    lossfn = hydra.utils.instantiate(cfg.loss)

    # Initialize a device (cpu, gpu or mps (Apple Sillicon))
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    print(f'\tConfigured loss function: {str(lossfn)} and device: {device}.')

    if cfg.basic.task == 'NER':
        run_ner_pipeline(cfg, lossfn, device)

    if use_wandb:
        wandb.finish()


def run_ner_pipeline(cfg: DictConfig, lossfn, device):

    # Instantiate tokenizer and create dataloaders for the respective language
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)
    if cfg.basic.lang == 'KZ':
        train_dataloder, test_dataloader, num_classes = create_kaznerd_dataloader(tokenizer, cfg.basic.token_type,
                                                                                  cfg.dataset.train_path,
                                                                                  cfg.dataset.test_path,
                                                                                  cfg.basic.padding_token,
                                                                                  cfg.dataset.batch_size)
    elif cfg.basic.lang == 'TR':
        train_dataloder, test_dataloader, num_classes = create_turkish_ner_dataloader(tokenizer, cfg.basic.token_type,
                                                                                cfg.dataset.train_path,
                                                                                  cfg.dataset.test_path,
                                                                                  cfg.basic.padding_token,
                                                                                  cfg.dataset.batch_size)

    if cfg.basic.with_soft_prompts is False:
        # Initialize a MBert model with a linear layer on top
        model = BertNerd(cfg.model.name, device, cfg.basic.hidden_size, num_classes).to(device)

        # Initialize the optimizer
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

        print(f"\t Training MBert on NER task without soft prompts with tokens of type {cfg.basic.token_type}")
        # Train MBert on NER task
        train_ner(model=model, train_dataloader=train_dataloder, loss_func=lossfn,
                  optimizer=optimizer, num_epochs=cfg.train.num_epochs, device=device,
                  use_wandb=cfg.basic.use_wandb)

        # Evaluate the model
        print("\t Training finished. Starting evaluation of MBert on NER task.")
        evaluate_ner(model=model, val_dataloader=test_dataloader, device=device,
                     use_wandb=cfg.basic.use_wandb)

    elif cfg.basic.with_soft_prompts is True:

        if cfg.soft_prompts.evaluate is True: # Zero-shot evaluation of soft prompts
            # Initialize the prompted mBERT model
            model = PromptedBertNerd(cfg.model.name, device, cfg.basic.hidden_size, num_classes,
                                     "soft_prompts/ner/" + str(cfg.soft_prompts.num_virtual_tokens)).to(device)

            print(f"\t Training mBERT on NER task with soft prompts with tokens of type {cfg.basic.token_type}")

            # Evaluate the model
            print("\t Training finished. Starting evaluation of mBERT on NER task.")
            evaluate_ner(model=model, val_dataloader=test_dataloader, device=device,
                         use_wandb=cfg.basic.use_wandb)

        else: # Train soft prompts
            # Initialize the mBERT model
            model = AutoModelForCausalLM.from_pretrained(cfg.model.name)

            peft_config = PromptTuningConfig(
                task_type=TaskType.CAUSAL_LM,
                prompt_tuning_init=PromptTuningInit.TEXT,
                num_virtual_tokens=cfg.soft_prompts.num_virtual_tokens,
                prompt_tuning_init_text="Classify NER tokens",
                tokenizer_name_or_path=cfg.tokenizer.name,
            )
            model = get_peft_model(model, peft_config).to(device)
            model.print_trainable_parameters()

            # Initialize the optimizer
            optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

            print("\t Training MBert on NER task with soft prompts.")
            # Train MBert on NER task
            train_ner_with_soft_prompts(model=model, tokenizer=tokenizer, train_dataloader=train_dataloder, test_dataloader=test_dataloader,
                                        optimizer=optimizer, num_epochs=cfg.train.num_epochs, device=device)

            print("\t Soft prompts trained. Saving model...")
            model.save_pretrained("soft_prompts/ner/" + str(cfg.soft_prompts.num_virtual_tokens) + "/" + str(
                cfg.train.num_epochs
            ))


if __name__ == "__main__":
    run_pipeline()