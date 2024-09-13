import torch
import wandb
from .metrics import get_accuracy


def train_ner_with_soft_prompts(model, tokenizer, train_dataloader, test_dataloader, optimizer, num_epochs, device):
    """
    Train soft prompts on Turkish for NER.
    :param model: mBERT
    :param tokenizer: mBERT tokenizer
    :param train_dataloader: train data
    :param test_dataloader: test data
    :param optimizer: e.g. AdamW
    :param num_epochs: int, specified in hydra config
    :param device: CPU/GPU/MPS
    :return: void
    """
    for epoch in range(num_epochs):
        print(f'Training epoch {epoch + 1}/{num_epochs} started.')

        # Set model in the training mode
        model.train()

        loss_per_epoch = 0

        for idx, batch in enumerate(train_dataloader):
            print(f'Training batch {idx+1} of {len(train_dataloader)}...')

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss_per_epoch += loss.detach().float()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        eval_loss = 0
        eval_preds = []
        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model(**batch)
            loss = outputs.loss
            eval_loss += loss.detach().float()
            eval_preds.extend(
                tokenizer.batch_decode(torch.argmax(outputs.logits, -1).detach().cpu().numpy(),
                                       skip_special_tokens=True)
            )

        eval_epoch_loss = eval_loss / len(test_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = loss_per_epoch / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")


def train_ner(model, train_dataloader, loss_func, optimizer, scheduler, num_epochs, device, use_wandb=False):
    """
    Baseline training for NER task.
    :param model: mBERT
    :param train_dataloader: train data
    :param loss_func: e.g. CrossEntropyLoss
    :param optimizer: e.g. AdamW
    :param scheduler: use for warmup
    :param num_epochs: int, specified in hydra config
    :param device: CPU/GPU/MPS
    :param use_wandb: bool, needed for logging
    :return: void
    """
    accuracies = []

    for epoch in range(num_epochs):

        # Metrics to be logged into wandb
        logging_dict = {}

        print(f'Training epoch {epoch+1}/{num_epochs} started.')

        loss_per_epoch = 0
        correct = 0
        total = 0

        # Set the model to train
        model.train()

        for idx, batch in enumerate(train_dataloader):
            print(f'Training batch {idx+1} of {len(train_dataloader)}...')

            inputs, attention_mask, labels = (batch["input_ids"].to(device), batch["attention_mask"].to(device),
                                              batch["labels"].to(device))
            # Make prediction
            logits = model(inputs, attention_mask)

            # Calculate loss
            batch_loss = loss_func(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss_per_epoch += batch_loss.detach().item()

            # Get ids corresponding to the most probably NER tags
            tag_ids = torch.max(logits, dim=2).indices

            # Calculate accuracy for this batch
            correct_in_batch, total_in_batch = get_accuracy(tag_ids, labels)

            correct += correct_in_batch
            total += total_in_batch

            acc = correct / total
            accuracies.append(acc)

            # Backpropagate
            optimizer.zero_grad()
            batch_loss.backward()

            # Clip gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

        # Log metrics into wandb
        logging_dict["loss"] = loss_per_epoch
        logging_dict["accuracy"] = acc

        if use_wandb is True:
            wandb.log(logging_dict)

        # Display additional information for debugging purposes
        print(f"\tEpoch: {epoch+1}\nLoss: {loss_per_epoch}   ---  Accuracy on train: {acc}")

    return model, accuracies


def evaluate_ner(model, val_dataloader, device, use_wandb=False):
    """
    Baseline evaluation on NER task.
    :param model: mBERT
    :param val_dataloader: evaluation data
    :param device: CPU/GPU/MPS
    :param use_wandb: bool, needed for logging
    :return: void
    """

    accuracies = []
    correct = 0
    total = 0

    # Disable backpropagation
    with torch.no_grad():
        logging_dict = {}

        # Set the model into the evaluation mode
        model.eval()

        for idx, batch in enumerate(val_dataloader):
            print(f'Evaluating batch {idx+1} of {len(val_dataloader)}...')

            inputs, attention_mask, labels = batch["input_ids"].to(device), batch["attention_mask"].to(device), batch["labels"].to(device)

            # Make predictions and get most probable NER tags
            logits = model(inputs, attention_mask)
            permutated_logits = logits.permute(0, 2, 1)
            predicted_tags = torch.max(permutated_logits, dim=1).indices

            # Calculate validation accuracy
            correct_in_batch, total_in_batch = get_accuracy(predicted_tags, labels)
            correct += correct_in_batch
            total += total_in_batch

        acc = correct / total
        accuracies.append(acc)

        # Log metrics into wandb
        logging_dict["accuracy"] = acc

        if use_wandb is True:
            wandb.log(logging_dict)

        print(f"\tAccuracy on validation: {acc}")

    return accuracies
