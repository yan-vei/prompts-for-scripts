from transformers import AutoTokenizer


class NERTokenizer:
    """
        Tokenizer class for the NER task in Kazakh and Turkish.
    """

    def __init__(self, tokenizer_name):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    def tokenize_and_align_labels(self, examples, tags, labels_dict=None, str2int=False,
                                  padding_token=-100):
        """
        Tokenize and align labels for the NER task.
        Largely adopted from: https://github.com/IS2AI/KazNERD/blob/main/bert/kaznerd.py

        :param examples: dicts with word tokens and NER tags
        :param tags: key for tags in the examples object
        :param labels_dict: dict with labels for the NER task
        :param str2int: convert labels to int if necessary
        :return: tokenized_inputs
        """

        tokenized_inputs = self.tokenizer(examples['tokens'], truncation=True,
                                     is_split_into_words=True, padding=True)

        tokenized_labels = examples[tags]

        # Handle the case when labels in the dataset haven't been indexed
        if str2int:
            tokenized_labels = [[labels_dict[tag] for tag in seq] for seq in examples[tags]]

        labels = []
        for i, label in enumerate(tokenized_labels):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []

            # Only the first token gets assigned the NER tag,
            # and the rest get assigned the padding token of "-100"
            # which safely gets ignored by the loss function
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(padding_token)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(padding_token)
                previous_word_idx = word_idx

            labels.append(label_ids)

        tokenized_inputs["labels"] = labels
        return tokenized_inputs