import pandas as pd
import numpy as np
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import Dataset

import sys
sys.path.append('../pipeline')
from nlp_tasks import NLP_Tasks

### PARAMETERS ###

# These are the parameters to change 
path_to_data = '/Users/bea/Documents/AI4CI/projects/comment_summariser/comment_crunch/outputs/train_comments.csv'
model_output_folder = '/Users/bea/Documents/AI4CI/projects/comment_summariser/comment_crunch/outputs/nlp_fine_tuning_practise'
model_name = 'practise_model'

### LOAD THE INITIAL MODEL ###
model_checkpoint = "distilbert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained(model_checkpoint)
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Load the NLP_Tasks class
nlp_t = NLP_Tasks(model_path=model_checkpoint)

### LOAD THE DATA ###
train_df = pd.read_csv(path_to_data)
# Convert the DataFrame to a Dataset
train_dataset = Dataset.from_pandas(train_df)

# # Tokenization function
# def tokenize_function(examples):
#     result = tokenizer(examples["comment_text"])
#     if tokenizer.is_fast:
#         result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
#     return result

### TOKENIZATION ###

# batched=True activates fast multithreading!
tokenized_datasets = train_dataset.map(nlp_t.tokenize_func, batched=True, remove_columns=["address", "stance", "date", "comment_text"])

# # Set maximum chunk size 
# chunk_size=128

# def group_texts(examples):
#     # Concatenate all the comment texts 
#     concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
#     # Compute length of concatenated texts 
#     total_length = len(concatenated_examples[list(examples.keys())[0]])
#     # We drop the last chunk if it's smaller than chunk_size
#     total_length = (total_length // chunk_size) * chunk_size
#     # Split by chunks of max_len 
#     result = {
#         k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
#         for k, t in concatenated_examples.items()
#     }
#     result["labels"] = result["input_ids"].copy()
#     return result

lm_datasets = tokenized_datasets.map(nlp_t.group_chunk_func, batched=True)

mlm_prob = 0.15 # mlm is the fraction of tokens to mask - 15% is popular in the literature. 

# mask the tokens
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=mlm_prob)

# Downsample the dataset to 20% of the original size
downsampled_dataset = lm_datasets.train_test_split(test_size=0.2)

print(f'The number of items in the training dataset: {len(downsampled_dataset["train"])}')

### TRAINING ###
batch_size = 8

logging_steps = len(downsampled_dataset["train"]) // batch_size
training_args = TrainingArguments(
    output_dir="../outputs",
    overwrite_output_dir=True,
    eval_strategy="epoch",
    learning_rate=2e-5,
    weight_decay=0.01,
    per_device_eval_batch_size=batch_size,
    per_device_train_batch_size=batch_size,
    logging_steps=logging_steps,
    fp16=False,
    bf16=True # Note this enables bfloat16 conversion which is supported by the Apple MPD backend
)

trainer = Trainer(
    model=model, 
    args=training_args,
    train_dataset=downsampled_dataset["train"],
    eval_dataset=downsampled_dataset["test"],
    data_collator=data_collator,
    tokenizer=tokenizer 
)

eval_results = trainer.evaluate()
trainer.train()

### SAVE THE MODEL ###
model.save_pretrained(model_output_folder+"/"+model_name)
tokenizer.save_pretrained(model_output_folder+"/"+model_name)