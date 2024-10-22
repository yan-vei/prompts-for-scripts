import torch
import wandb
from .metrics import get_accuracy, compute_exact_match


def train_qa_with_soft_prompts(model, tokenizer, train_dataloader, test_dataloader, optimizer, num_epochs, device, num_tokens):
    """
    Train soft prompts on Turkish for extractive QA.
    :param model: mBERT
    :param tokenizer: mBERT tokenizer
    :param train_dataloader: train data
    :param test_dataloader: test data
    :param optimizer: e.g., AdamW
    :param num_epochs: int, specified in the configuration
    :param device: CPU/GPU/MPS
    :param num_tokens: number of soft prompt tokens
    :return: void
    """
    for epoch in range(num_epochs):
        print(f'Training epoch {epoch + 1}/{num_epochs} started.')

        # Set model to training mode
        model.train()

        loss_per_epoch = 0

        for idx, batch in enumerate(train_dataloader):
            print(f'Training batch {idx+1} of {len(train_dataloader)}...')

            batch = {k: v.to(device) for k, v in batch.items()}

            # Adjust start_positions and end_positions by num_tokens
            batch['start_positions'] = batch['start_positions'] + num_tokens
            batch['end_positions'] = batch['end_positions'] + num_tokens

            # Ensure positions do not exceed sequence length
            max_len = batch['input_ids'].size(1)
            batch['start_positions'] = torch.clamp(batch['start_positions'], max=max_len - 1)
            batch['end_positions'] = torch.clamp(batch['end_positions'], max=max_len - 1)

            outputs = model(**batch)
            loss = outputs.loss
            loss_per_epoch += loss.detach().float()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Model evaluation
        model.eval()
        eval_loss = 0

        for idx, batch in enumerate(test_dataloader):
            print(f'Evaluating batch {idx+1} of {len(test_dataloader)}...')

            batch.pop('answers', None)
            batch = {k: v.to(device) for k, v in batch.items()}

            # Adjust start_positions and end_positions
            batch['start_positions'] = batch['start_positions'] + num_tokens
            batch['end_positions'] = batch['end_positions'] + num_tokens

            # Ensure positions do not exceed sequence length
            max_len = batch['input_ids'].size(1)
            batch['start_positions'] = torch.clamp(batch['start_positions'], max=max_len - 1)
            batch['end_positions'] = torch.clamp(batch['end_positions'], max=max_len - 1)

            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            eval_loss += loss.detach().float()

            eval_epoch_loss = eval_loss / len(test_dataloader)
            train_epoch_loss = loss_per_epoch / len(train_dataloader)

        print(f"{epoch=}: {train_epoch_loss=} {eval_epoch_loss=}")


def train_ner_with_soft_prompts(model, tokenizer, train_dataloader, test_dataloader, optimizer, num_epochs, device, num_tokens):
    """
    Train soft prompts on Turkish for NER.
    :param model: mBERT
    :param tokenizer: mBERT tokenizer
    :param train_dataloader: train data
    :param test_dataloader: test data
    :param optimizer: e.g. AdamW
    :param num_epochs: int, specified in hydra config
    :param device: CPU/GPU/MPS
    :param num_tokens: number of tokens to train
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

            # Pad labels to the length of the tokens
            labels = batch['labels']
            padded_labels = torch.nn.functional.pad(labels, (num_tokens, 0), value=-100)
            batch['labels'] = padded_labels

            outputs = model(**batch)
            loss = outputs.loss
            loss_per_epoch += loss.detach().float()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        # Evaluate model's performance
        model.eval()
        eval_loss = 0

        for batch in test_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}

            # Pad labels again (to match num of soft prompt tokens):
            labels = batch['labels']
            padded_labels = torch.nn.functional.pad(labels, (num_tokens, 0), value=-100)
            batch['labels'] = padded_labels

            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            eval_loss += loss.detach().float()

        eval_epoch_loss = eval_loss / len(test_dataloader)
        eval_ppl = torch.exp(eval_epoch_loss)
        train_epoch_loss = loss_per_epoch / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)

        print(f"{epoch=}: {train_ppl=} {train_epoch_loss=} {eval_ppl=} {eval_epoch_loss=}")


def train_ner(model, train_dataloader, loss_func, optimizer, scheduler, num_epochs, device, with_soft_prompts=False,
              num_tokens=None, use_wandb=False):
    """
    Baseline training for NER task.
    :param model: mBERT
    :param train_dataloader: train data
    :param loss_func: e.g. CrossEntropyLoss
    :param optimizer: e.g. AdamW
    :param scheduler: use for warmup
    :param num_epochs: int, specified in hydra config
    :param with_soft_prompts: whether to use soft prompts or not
    :param num_tokens: int, number of soft prompt tokens
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

            # Pad labels in case we are training with soft prompts
            if with_soft_prompts is True:
                labels = torch.nn.functional.pad(batch['labels'], (num_tokens, 0), value=-100).to(device)

            loss_logits = logits.view(-1, logits.size(-1))

            batch_loss = loss_func(loss_logits, labels.view(-1))
            loss_per_epoch += batch_loss.detach().item()

            # Get ids corresponding to the most probably NER tags
            tag_ids = torch.max(logits, dim=2).indices

            # Calculate accuracy for this batch
            correct_in_batch, total_in_batch = get_accuracy(tag_ids, labels)

            correct += correct_in_batch
            total += total_in_batch

            acc = correct / total
            accuracies.append(acc)

            optimizer.zero_grad()
            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

        # Log metrics into wandb
        logging_dict["loss"] = loss_per_epoch
        logging_dict["accuracy"] = acc

        if use_wandb is True:
            wandb.log(logging_dict)

        print(f"\tEpoch: {epoch+1}\nLoss: {loss_per_epoch}   ---  Accuracy on train: {acc}")

    return model, accuracies


def evaluate_ner(model, val_dataloader, device, num_tokens=None, with_soft_prompts=False, use_wandb=False):
    """
    Baseline evaluation on NER task.
    :param model: mBERT
    :param val_dataloader: evaluation data
    :param device: CPU/GPU/MPS
    :param num_tokens: number of soft prompt tokens
    :param with_soft_prompts: whether to use soft prompts or not
    :param use_wandb: bool, needed for logging
    :return: void
    """

    accuracies = []
    correct = 0
    total = 0

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

            # Pad labels in case we are training with soft prompts
            if with_soft_prompts is True:
                labels = torch.nn.functional.pad(batch['labels'], (num_tokens, 0), value=-100).to(device)

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


def train_qa(model, train_dataloader, loss_func, optimizer, scheduler, num_epochs, device, use_wandb=False,
             with_soft_prompts=False, num_tokens=None):
    """
    Training function for extractive QA task using mBERT.
    :param model: mBERT model for question answering
    :param train_dataloader: DataLoader for training data
    :param loss_func: loss function, e.g., CrossEntropyLoss
    :param optimizer: optimizer, e.g., AdamW
    :param scheduler: learning rate scheduler
    :param num_epochs: number of epochs to train
    :param device: device to train on (CPU/GPU)
    :param use_wandb: whether to use wandb for logging
    :return: trained model, list of losses, list of accuracies
    """
    losses = []
    accuracies = []

    for epoch in range(num_epochs):
        # Metrics to be logged into wandb
        logging_dict = {}

        print(f'Training epoch {epoch+1}/{num_epochs} started.')

        loss_per_epoch = 0
        total_correct_start = 0
        total_correct_end = 0
        total_correct_span = 0
        total_examples = 0

        # Set the model to train mode
        model.train()

        for idx, batch in enumerate(train_dataloader):
            print(f'Training batch {idx+1} of {len(train_dataloader)}...')

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            if with_soft_prompts and num_tokens is not None:
                # Adjust start_positions and end_positions
                start_positions = start_positions + num_tokens
                end_positions = end_positions + num_tokens

            optimizer.zero_grad()

            outputs = model(input_seq=input_ids, attention_mask=attention_mask)

            start_logits = outputs[0]
            end_logits = outputs[1]

            loss_start = loss_func(start_logits, start_positions)
            loss_end = loss_func(end_logits, end_positions)
            batch_loss = (loss_start + loss_end) / 2

            loss_per_epoch += batch_loss.detach().item()
            losses.append(batch_loss.detach().item())

            batch_loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                pred_start_positions = torch.argmax(start_logits, dim=1)
                pred_end_positions = torch.argmax(end_logits, dim=1)

                correct_start = (pred_start_positions == start_positions).sum().item()
                correct_end = (pred_end_positions == end_positions).sum().item()
                correct_span = ((pred_start_positions == start_positions) & (pred_end_positions == end_positions)).sum().item()

                total_correct_start += correct_start
                total_correct_end += correct_end
                total_correct_span += correct_span
                total_examples += start_positions.size(0)

        # Calculate epoch metrics
        epoch_loss = loss_per_epoch / len(train_dataloader)
        accuracy_start = 100.0 * total_correct_start / total_examples
        accuracy_end = 100.0 * total_correct_end / total_examples
        accuracy_span = 100.0 * total_correct_span / total_examples
        accuracies.append(accuracy_span)

        # Log metrics into wandb
        logging_dict["loss"] = epoch_loss
        logging_dict["accuracy_start"] = accuracy_start
        logging_dict["accuracy_end"] = accuracy_end
        logging_dict["accuracy_span"] = accuracy_span

        if use_wandb:
            wandb.log(logging_dict)

        print(f"\tEpoch: {epoch+1}")
        print(f"Loss: {epoch_loss:.4f}   ---  Start Accuracy: {accuracy_start:.2f}%  End Accuracy: {accuracy_end:.2f}%  Span Accuracy: {accuracy_span:.2f}%")

    return model, losses, accuracies


def evaluate_qa(model, val_dataloader, device, tokenizer, with_soft_prompts=False, num_tokens=None, use_wandb=False, lang='KZ'):
    """
    Evaluate the model on the validation set and compute EM, F1 scores, and start/end/span accuracies, with logging to Wandb.
    :param model: Trained mBERT model for question answering
    :param val_dataloader: DataLoader for validation data
    :param device: Device to run evaluation on (CPU/GPU)
    :param tokenizer: Tokenizer used to decode the tokens back to text
    :param use_wandb: Whether to log metrics to Wandb
    :return: EM score, start accuracy, end accuracy, span accuracy
    """
    model.eval()
    total_loss = 0
    total_examples = 0

    total_em = 0

    total_correct_start = 0
    total_correct_end = 0
    total_correct_span = 0

    with torch.no_grad():
        for idx, batch in enumerate(val_dataloader):
            print(f"Evaluating batch {idx+1} of {len(val_dataloader)}...")

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            start_positions = batch["start_positions"].to(device)
            end_positions = batch["end_positions"].to(device)

            if lang == 'KZ':
                answers = batch["answers"]  # Ground truth answers (list of acceptable answers)
            else:
                answers = batch["answers"]["text"]

            if with_soft_prompts and num_tokens is not None:
                # Adjust start_positions and end_positions
                start_positions = start_positions + num_tokens
                end_positions = end_positions + num_tokens

            outputs = model(input_seq=input_ids, attention_mask=attention_mask)
            start_logits = outputs[0]
            end_logits = outputs[1]

            loss_func = torch.nn.CrossEntropyLoss()
            loss_start = loss_func(start_logits, start_positions)
            loss_end = loss_func(end_logits, end_positions)
            batch_loss = (loss_start + loss_end) / 2
            total_loss += batch_loss.item()
            batch_size = input_ids.size(0)
            total_examples += batch_size

            pred_start_positions = torch.argmax(start_logits, dim=1)
            pred_end_positions = torch.argmax(end_logits, dim=1)

            # Compute position accuracies
            correct_start = (pred_start_positions == start_positions).sum().item()
            correct_end = (pred_end_positions == end_positions).sum().item()
            correct_span = ((pred_start_positions == start_positions) & (pred_end_positions == end_positions)).sum().item()

            total_correct_start += correct_start
            total_correct_end += correct_end
            total_correct_span += correct_span


            for i in range(batch_size):
                # Extract predicted answer text
                input_id = input_ids[i]
                tokens = input_id.cpu().tolist()

                if with_soft_prompts and num_tokens is not None:
                    tokens = tokens[num_tokens:]  # Exclude soft prompt tokens
                    pred_start = pred_start_positions[i].item() - num_tokens
                    pred_end = pred_end_positions[i].item() - num_tokens
                else:
                    pred_start = pred_start_positions[i].item()
                    pred_end = pred_end_positions[i].item()

                if pred_start > pred_end:
                    pred_end = pred_start

                # Ensure indices are within bounds
                pred_start = max(0, min(pred_start, len(tokens) - 1))
                pred_end = max(0, min(pred_end, len(tokens) - 1))

                pred_answer_tokens = tokens[pred_start:pred_end + 1]
                pred_answer = tokenizer.decode(pred_answer_tokens, skip_special_tokens=True)

                # Get ground truth answer texts based on the language selected
                if lang == 'KZ':
                    ground_truth_answers = answers[i][0]  # List of acceptable answers
                else:
                    ground_truth_answers = answers[0][i]

                em_for_example = max(compute_exact_match(pred_answer, gt_answer) for gt_answer in ground_truth_answers)
                total_em += em_for_example

    accuracy_start = 100.0 * total_correct_start / total_examples
    accuracy_end = 100.0 * total_correct_end / total_examples
    accuracy_span = 100.0 * total_correct_span / total_examples

    em_score = 100.0 * total_em / total_examples

    avg_loss = total_loss / len(val_dataloader)

    # Log metrics to Wandb
    if use_wandb:
        wandb.log({
            'Validation Loss': avg_loss,
            'EM': em_score,
            'Accuracy Start': accuracy_start,
            'Accuracy End': accuracy_end,
            'Accuracy Span': accuracy_span
        })

    print(f"Validation Loss: {avg_loss:.4f} | EM: {em_score:.2f}% ")
    print(f"Start Accuracy: {accuracy_start:.2f}% | End Accuracy: {accuracy_end:.2f}% | Span Accuracy: {accuracy_span:.2f}%")

    return em_score, accuracy_start, accuracy_end, accuracy_span

