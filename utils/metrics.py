import re
import string


def normalize_answer(s):
    """
    Se of functions to lower text and remove punctuation, articles, and extra whitespace.
    """

    def lower(text):
        return text.lower()

    def remove_punctuation(text):
        return ''.join(ch for ch in text if ch not in set(string.punctuation))

    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    return white_space_fix(remove_articles(remove_punctuation(lower(s))))

def compute_exact_match(prediction, ground_truth):
    """
    Compute the Exact Match score between the prediction and ground truth answers.
    :param prediction: prediction answer from the span
    :param ground_truth: ground truth answer from the span
    :return: Exact Match score - accuracy in %
    """
    return int(normalize_answer(prediction) == normalize_answer(ground_truth))


def get_accuracy(batch, gold_batch, padding_index=-100):
    """
    Computes the accuracy of the predicted batch

    :param batch: labels predicted by the model
    :param gold_batch: correct labels from the dataset
    :param padding_index
    :return: number of correctly predicted and total labels (int, int)
    """
    results = [(p == g) for sent, gold in zip(batch, gold_batch) for p, g in zip(sent, gold) if g != padding_index]

    return sum(results), len(results)
