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