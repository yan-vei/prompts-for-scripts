import torch
from metrics import get_accuracy


def train_ner(model, train_dataloader, optimizer, config):
    """
        Define the training loop for NER.
    :param model: corresponding model class
    :param train_loader: train data
    :param config: config file with hyperparameters
    :return: model, metrics
    """
    accuracies = []

    for epoch in range(config.epochs):
        loss_per_epoch = 0
        correct = 0
        total = 0

        # Set the model to train
        model.train()

        for batch in train_dataloader:
            inputs, attention_mask, labels = batch["input_ids"].to(config.device), batch["attention_mask"].to(config.device), batch["labels"].to(config.device)

            # Make prediction
            logits = model(inputs, attention_mask)

            # Calculate loss
            batch_loss = model.get_loss(logits, labels)
            loss_per_epoch += batch_loss

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
            optimizer.step()

        # Display additional information for debugging purposes
        print(f"Epoch: {epoch}\nLoss: {loss_per_epoch}   ---  Accuracy on train: {acc}")

        return model, accuracies


def evaluate_ner(model, val_dataloader, config):
    """
        Define the evaluation loop for NER.
    :param model: corresponding model class
    :param val_dataloader: vaidation data
    :param config: config file with hyperparameters
    :return: validation accuracies
    """

    accuracies = []
    correct = 0
    total = 0

    # Disable backpropagation
    with torch.no_grad():

        # Set the model in correct mode
        model.eval()

        for batch in val_dataloader:
            inputs, attention_mask, labels = batch["input_ids"].to(config.device), batch["attention_mask"].to(config.device), batch["labels"].to(config.device)

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

        print(f"Accuracy on validation: {acc}")

        return accuracies
