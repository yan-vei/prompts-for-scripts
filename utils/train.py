import torch
from .metrics import get_accuracy

def train_ner_soft_prompts(model, train_dataloader, loss_func, optimizer, device, num_epochs):

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for idx, batch in enumerate(train_dataloader):

            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            total_loss += loss.detach().float()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()


def train_ner(model, train_dataloader, loss_func, optimizer, num_epochs, device):
    accuracies = []

    for epoch in range(num_epochs):
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

        # Display additional information for debugging purposes
        print(f"\tEpoch: {epoch}\nLoss: {loss_per_epoch}   ---  Accuracy on train: {acc}")

    return model, accuracies


def evaluate_ner(model, val_dataloader, device):

    accuracies = []
    correct = 0
    total = 0

    # Disable backpropagation
    with torch.no_grad():

        # Set the model in correct mode
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

        print(f"\tAccuracy on validation: {acc}")

    return accuracies
