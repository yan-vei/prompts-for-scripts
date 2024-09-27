import torch
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
from transformers import AutoModelForTokenClassification, AutoModelForQuestionAnswering, AutoTokenizer, get_linear_schedule_with_warmup
from utils.dataloader import create_kaznerd_dataloader, create_turkish_ner_dataloader, create_turkish_qa_dataloader, create_kazakh_qa_dataloader
from logging_config import settings
from utils.train import train_ner, evaluate_ner, train_ner_with_soft_prompts, train_qa, train_qa_with_soft_prompts, evaluate_qa
from utils.prompts_initializer import initialize_randomly, initialize_with_task, initialize_normal
from models.base_bert import BertNerd, BertQA
from models.prompted_bert import PromptedBertNER, PromptedBertQA


@hydra.main(config_path='configs', config_name='defaults', version_base=None)
def run_pipeline(cfg: DictConfig):
    """
    Define the run managed by hydra configs
    :param cfg: hydra config
    :return: void
    """

    # Log into wandb if required
    use_wandb = cfg.basic.use_wandb
    if use_wandb:
        cfg_copy = OmegaConf.to_container(cfg, resolve=True)
        wandb.login(key=settings.WANDB_API_KEY)

        project_name = cfg.dataset.name

        wandb.init(project=project_name, name=cfg.basic.wandb_run,
                   config=cfg_copy)

    # Initialize the loss function
    lossfn = hydra.utils.instantiate(cfg.loss)

    # Instantiate the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(cfg.tokenizer.name)

    # Initialize a device (cpu, gpu or mps (Apple Sillicon))
    device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

    print(f'\tConfigured loss function: {str(lossfn)} and device: {device}.')

    if cfg.basic.task == 'NER':
        run_ner_pipeline(cfg, lossfn, device, tokenizer)
    elif cfg.basic.task == 'QA':
        run_qa_pipeline(cfg, lossfn, device, tokenizer)

    if use_wandb:
        wandb.finish()


def run_qa_pipeline(cfg: DictConfig, lossfn, device, tokenizer):
    """
    Run the pipeline for QA baseline and soft prompts.
    :param cfg: hydra config
    :param lossfn: CrossEntropy
    :param device: GPU/CPU/MPS
    :return: void
    """

    if cfg.basic.lang == 'TR':
        train_dataloader, test_dataloader = create_turkish_qa_dataloader(tokenizer=tokenizer, train_path=cfg.dataset.train_path,
                                                                         test_path=cfg.dataset.test_path, batch_size=cfg.dataset.batch_size,
                                                                         max_length=cfg.dataset.max_length, doc_stride=cfg.dataset.doc_stride)
    elif cfg.basic.lang == 'KZ':
        train_dataloader, test_dataloader = create_kazakh_qa_dataloader(tokenizer=tokenizer, train_path=cfg.dataset.train_path,
                                                                         test_path=cfg.dataset.test_path, batch_size=cfg.dataset.batch_size,
                                                                         max_length=cfg.dataset.max_length, doc_stride=cfg.dataset.doc_stride)

    if cfg.basic.with_soft_prompts is False:
        # Initialize a mBERT model with a  2 linear layers on top
        # To detect start and end positions of the answers
        model = BertQA(cfg.model.name, device, cfg.basic.hidden_size).to(device)

        # Initialize the optimizer
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

        # Initialize the linear scheduler for LR warmup
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=cfg.scheduler.warmup_steps,
                                                    num_training_steps=len(train_dataloader)*cfg.train.num_epochs)

        print(f"\t Training mBERT on QA task without soft prompts with tokens of type {cfg.basic.token_type}")
        # Train mBERT on extractive QA task
        train_qa(model=model, train_dataloader=train_dataloader, loss_func=lossfn,
                  optimizer=optimizer, num_epochs=cfg.train.num_epochs, device=device,
                  use_wandb=cfg.basic.use_wandb, scheduler=scheduler)

        # Evaluate the model
        print("\t Training finished. Starting evaluation of mBERT on QA task.")
        evaluate_qa(model=model, val_dataloader=test_dataloader,
                    device=device,
                     use_wandb=cfg.basic.use_wandb, tokenizer=tokenizer)

    # Training or evaluating soft prompts
    elif cfg.basic.with_soft_prompts is True:

        if cfg.soft_prompts.evaluate is True:

            soft_prompts_path = ("soft_prompts/qa/" + str(cfg.soft_prompts.num_virtual_tokens) + "/" +
                                 str(cfg.soft_prompts.init_strategy) + "/" + str(cfg.train.num_epochs))
            print(f'Loading soft prompts from {soft_prompts_path}...')

            # Initialize the model with prompts
            model = PromptedBertQA(name=cfg.model.name, device=device, hidden_size=cfg.basic.hidden_size,
                                    soft_prompts_path=soft_prompts_path).to(device)

            # Initialize the optimizer
            optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

            # Initialize the linear scheduler for LR warmup
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=cfg.scheduler.warmup_steps,
                                                        num_training_steps=len(train_dataloader) * cfg.train.num_epochs)

            # Train model
            print(f"\t Training mBERT on NER task with soft prompts with tokens of type {cfg.basic.token_type}")
            train_ner(model=model, train_dataloader=train_dataloader, loss_func=lossfn, with_soft_prompts=True,
                      num_tokens=cfg.soft_prompts.num_virtual_tokens, optimizer=optimizer,
                      num_epochs=cfg.train.num_epochs, device=device, scheduler=scheduler, use_wandb=cfg.basic.use_wandb)

            # Evaluate the model
            print("\t Training finished. Starting evaluation of mBERT on NER task.")
            evaluate_ner(model=model, val_dataloader=test_dataloader, device=device,
                         use_wandb=cfg.basic.use_wandb, with_soft_prompts=True,
                         num_tokens=cfg.soft_prompts.num_virtual_tokens)

        else:
            # Initialize the mBERT model
            model = AutoModelForQuestionAnswering.from_pretrained(cfg.model.name)

            # Initialize prompts according to the declared strategy
            if cfg.soft_prompts.init_strategy == 'random':
                model = initialize_randomly(cfg.soft_prompts.num_virtual_tokens,
                                            cfg.basic.task, model)
            elif cfg.soft_prompts.init_strategy == 'task':
                model = initialize_with_task(cfg.soft_prompts.num_virtual_tokens,
                                             cfg.basic.task, model, cfg.tokenizer.name)
            elif cfg.soft_prompts.init_strategy == 'normal':
                model = initialize_normal(cfg.soft_prompts.num_virtual_tokens, cfg.basic.hidden_size,
                                          cfg.basic.task, model)

            model.to(device)

            # Check that the soft prompts have been correctly initialized
            model.print_trainable_parameters()

            # Initialize the optimizer
            optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

            print("\t Training mBERT on QA task with soft prompts.")
            train_qa_with_soft_prompts(model=model, tokenizer=tokenizer, train_dataloader=train_dataloader,
                                        test_dataloader=test_dataloader,
                                        optimizer=optimizer, num_epochs=cfg.train.num_epochs, device=device,
                                        num_tokens=cfg.soft_prompts.num_virtual_tokens)

            # Save the trained model
            print("\t Soft prompts trained. Saving model...")
            model.save_pretrained("soft_prompts/qa/" + str(cfg.soft_prompts.num_virtual_tokens) + "/" + str(
                cfg.soft_prompts.init_strategy) + "/" + str(cfg.train.num_epochs)
                              )

def run_ner_pipeline(cfg: DictConfig, lossfn, device, tokenizer):
    """
    Run the pipeline for NER baseline and soft prompts.
    :param cfg: hydra config
    :param lossfn: CrossEntropy
    :param device: GPU/CPU/MPS
    :param tokenizer: tokenizer, e.g. BertTokenizerFast
    :return: void
    """

    if cfg.basic.lang == 'KZ':
        train_dataloader, test_dataloader, num_classes = create_kaznerd_dataloader(tokenizer, cfg.basic.token_type,
                                                                                  cfg.dataset.train_path,
                                                                                  cfg.dataset.test_path,
                                                                                  cfg.basic.padding_token,
                                                                                  cfg.dataset.batch_size)
    elif cfg.basic.lang == 'TR':
        train_dataloader, test_dataloader, num_classes = create_turkish_ner_dataloader(tokenizer, cfg.basic.token_type,
                                                                                cfg.dataset.train_path,
                                                                                  cfg.dataset.test_path,
                                                                                  cfg.basic.padding_token,
                                                                                  cfg.dataset.batch_size)
    # Baseline runs
    if cfg.basic.with_soft_prompts is False:
        # Initialize a mBERT model with a linear layer on top
        model = BertNerd(cfg.model.name, device, cfg.basic.hidden_size, num_classes).to(device)

        # Initialize the optimizer
        optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

        # Initialize the linear scheduler for LR warmup
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=cfg.scheduler.warmup_steps,
                                                    num_training_steps=len(train_dataloader)*cfg.train.num_epochs)

        print(f"\t Training mBERT on NER task without soft prompts with tokens of type {cfg.basic.token_type}")
        # Train mBERT on NER task
        train_ner(model=model, train_dataloader=train_dataloader, loss_func=lossfn,
                  optimizer=optimizer, num_epochs=cfg.train.num_epochs, device=device,
                  use_wandb=cfg.basic.use_wandb, scheduler=scheduler)

        # Evaluate the model
        print("\t Training finished. Starting evaluation of MBert on NER task.")
        evaluate_ner(model=model, val_dataloader=test_dataloader, device=device,
                     use_wandb=cfg.basic.use_wandb)

    # Training or evaluating soft prompts
    elif cfg.basic.with_soft_prompts is True:

        if cfg.soft_prompts.evaluate is True:

            soft_prompts_path = ("soft_prompts/ner/" + str(cfg.soft_prompts.num_virtual_tokens) + "/" +
                                 str(cfg.soft_prompts.init_strategy) + "/" + str(cfg.train.num_epochs))
            print(f'Loading soft prompts from {soft_prompts_path}...')

            # Initialize the model with prompts
            model = PromptedBertNER(name=cfg.model.name, device=device, hidden_size=cfg.basic.hidden_size,
                                    num_classes=num_classes, soft_prompts_path=soft_prompts_path,
                                    num_orig_ner_labels=cfg.basic.num_orig_ner_labels).to(device)

            # Initialize the optimizer
            optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

            # Initialize the linear scheduler for LR warmup
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                        num_warmup_steps=cfg.scheduler.warmup_steps,
                                                        num_training_steps=len(train_dataloader) * cfg.train.num_epochs)

            # Train model
            print(f"\t Training mBERT on NER task with soft prompts with tokens of type {cfg.basic.token_type}")
            train_ner(model=model, train_dataloader=train_dataloader, loss_func=lossfn, with_soft_prompts=True,
                      num_tokens=cfg.soft_prompts.num_virtual_tokens, optimizer=optimizer,
                      num_epochs=cfg.train.num_epochs, device=device, scheduler=scheduler, use_wandb=cfg.basic.use_wandb)

            # Evaluate the model
            print("\t Training finished. Starting evaluation of mBERT on NER task.")
            evaluate_ner(model=model, val_dataloader=test_dataloader, device=device,
                         use_wandb=cfg.basic.use_wandb, with_soft_prompts=True,
                         num_tokens=cfg.soft_prompts.num_virtual_tokens)

        else:
            # Initialize the mBERT model
            model = AutoModelForTokenClassification.from_pretrained(cfg.model.name, num_labels=num_classes)

            # Initialize prompts according to the declared strategy
            if cfg.soft_prompts.init_strategy == 'random':
                model = initialize_randomly(cfg.soft_prompts.num_virtual_tokens,
                                            cfg.basic.task, model)
            elif cfg.soft_prompts.init_strategy == 'task':
                model = initialize_with_task(cfg.soft_prompts.num_virtual_tokens,
                                             cfg.basic.task, model, cfg.tokenizer.name)
            elif cfg.soft_prompts.init_strategy == 'normal':
                model = initialize_normal(cfg.soft_prompts.num_virtual_tokens, cfg.basic.hidden_size,
                                          cfg.basic.task, model)

            model.to(device)

            # Check that the soft prompts have been correctly initialized
            model.print_trainable_parameters()

            # Initialize the optimizer
            optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())

            print("\t Training mBERT on NER task with soft prompts.")
            # Train mBERT on NER task
            train_ner_with_soft_prompts(model=model, tokenizer=tokenizer, train_dataloader=train_dataloader, test_dataloader=test_dataloader,
                                        optimizer=optimizer, num_epochs=cfg.train.num_epochs, device=device, num_tokens=cfg.soft_prompts.num_virtual_tokens)

            # Save the trained model
            print("\t Soft prompts trained. Saving model...")
            model.save_pretrained("soft_prompts/ner/" + str(cfg.soft_prompts.num_virtual_tokens) + "/" + str(
                cfg.soft_prompts.init_strategy) + "/" + str(cfg.train.num_epochs)
            )


if __name__ == "__main__":
    run_pipeline()