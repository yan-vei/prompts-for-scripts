## Do Soft Prompts Know about Language Script
This is a repository with the experiments for the paper "Do Soft Prompts Know about Language Script?".
Additionally, all the trained soft prompts (their adapter configs for the PEFT library) are also available under the ./soft_prompts directory.
## Installation
Make sure you have Python >= 3.10 and pip >= 20.1 installed, then run:
```
pip install -r requirements.txt
```
Or, alternatively, if you use conda:
```
conda env create --name prompts --file=environment.yaml
```

If you wish to log results into W&B, make sure to create a .env file
which contains your W&B API key.

## Experiments
It is best to run the experiments by making alternations in the config files under the /config folder.
These config files are managed by hydra.

For example, to run mBERT on Kazakh (Cyrillic) NER task with 20 soft prompts tokens, initialized with the task embeddings and trained
for 50 epochs, run the following command:
```
python main.py token_type=tokens task=NER lang=KZ basic.model=mbert basic.tokenizer=mbert basic.dataset=kaznerd basic.soft_prompts.num_epochs_trained=50 basic.soft_prompts.num_virtual_tokens=20 basic.soft_prompts.evaluate=True
```

This will first only train the linear layer of the model, and then will perform evaluation of the results.