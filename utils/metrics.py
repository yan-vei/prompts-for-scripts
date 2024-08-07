def accuracy(batch, gold_batch, padding_index=-100):
    """
    Computes the accuracy of the predicted batch

    :param batch: labels predicted by the model
    :param gold_batch: correct labels from the dataset
    :param padding_index
    :return: number of correctly predicted and total labels (int, int)
    """
    results = [(p == g) for sent, gold in zip(batch, gold_batch) for p, g in zip(sent, gold) if g != padding_index]

    return sum(results), len(results)
