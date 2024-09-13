from datasets import load_from_disk

kaznerd_ner_tr = load_from_disk("datasets/kaznerd-ner-train.hf")
kaznerd_ner_val = load_from_disk("datasets/kaznerd-ner-val.hf")
