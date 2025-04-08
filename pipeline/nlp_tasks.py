import pandas as pd
import numpy as np
import math
import collections
from transformers import AutoModelForMaskedLM, AutoTokenizer, DataCollatorForLanguageModeling, default_data_collator, TrainingArguments, Trainer, pipeline
import torch
from datasets import Dataset
import re
import string 


class NLP_Tasks:
    def __init__(self, model_path):
        self.model_checkpoint = model_path
        self.model = AutoModelForMaskedLM.from_pretrained(self.model_checkpoint)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)

        self.ner_pipeline = pipeline(
            task="ner",
            model="cjber/reddit-ner-place_names",
            tokenizer="cjber/reddit-ner-place_names",
            aggregation_strategy="first",
        )

        # self.model_checkpoint = model_path
        # self.data = data
        # self.train_data = self.data["train"]
        # self.test_data = self.data["test"]
        # self.model = AutoModelForMaskedLM.from_pretrained(self.model_checkpoint)
        # self.tokenizer = AutoTokenizer.from_pretrained(self.model_checkpoint)
        
        # self.batch_size = 8
        # self.logging_steps = len(self.data["train"]) // self.batch_size

        # # mlm is the fraction of tokens to mask - 15% is popular in the literature. 
        # self.mlm_prob = 0.15

        # # mask the tokens
        # self.data_collator = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm_probability=self.mlm_prob)

        # self.training_args = TrainingArguments(
        # output_dir="../outputs",
        # overwrite_output_dir=True,
        # eval_strategy="epoch",
        # learning_rate=2e-5,
        # weight_decay=0.01,
        # per_device_eval_batch_size=self.batch_size,
        # per_device_train_batch_size=self.batch_size,
        # logging_steps=self.logging_steps,
        # fp16=False,
        # bf16=True # Note this enables bfloat16 conversion which is supported by the Apple MPD backend
        # )

        # self.trainer = Trainer(
        # model=self.model, 
        # args=self.training_args,
        # train_dataset=self.train_data,
        # eval_dataset=self.test_data,
        # data_collator=self.data_collator,
        # tokenizer=self.tokenizer 
        # )


    def tokenizing_function(self, examples):
        result = self.tokenizer(examples["comment_text"])
        if self.tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result
    

    # def grouping_chunking_function(self, examples):
    #     chunk_size = 128
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
    

    def grouping_chunking_function(self, examples):
        chunk_size = 128
        
        # Ensure concatenation works regardless of whether input is a list of lists or a flat list
        concatenated_examples = {k: [] for k in examples.keys()}
        for k in examples.keys():
            for item in examples[k]:  
                if isinstance(item, list):  
                    concatenated_examples[k].extend(item)  # Flatten properly
                else:
                    concatenated_examples[k].append(item)  # Directly add if already flat

        # Compute total length, dropping the last chunk if it's smaller than chunk_size
        total_length = (len(concatenated_examples[list(examples.keys())[0]]) // chunk_size) * chunk_size

        # Chunking
        result = {
            k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
            for k, t in concatenated_examples.items()
        }

        # Copy "input_ids" as labels for model training
        if "input_ids" in result:
            result["labels"] = result["input_ids"].copy()

        return result
    

    def process_data(self, data, test_size=0.2):
        tokenized_data = data.map(self.tokenizing_function, batched=True, remove_columns=["address", "stance", "date", "comment_text"])
        chunked_data = tokenized_data.map(self.grouping_chunking_function, batched=True)
        split_data = chunked_data.train_test_split(test_size=test_size)
        return split_data
    

    # def ner_locations_function(self, text):
    #     ner_results = self.ner_pipeline(text)

    #     # Sort entities by their start index in descending order to avoid index shifting
    #     ner_results = sorted(ner_results, key=lambda x: x["start"], reverse=True)
            
    #     # Remove locations by replacing them with an empty string
    #     for entity in ner_results:
    #         text = text[:entity["start"]] + text[entity["end"]:]  # Slice out the location

    #     # Remove extra spaces caused by deletion
    #     text = re.sub(r'\s+', ' ', text).strip()

    #     return text

    def remove_place_names(self, df, column='text', batch_size=64):
        df = df.copy()
        texts = df[column].fillna('').astype(str).tolist()
        cleaned_texts = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = self.ner_pipeline(batch)

            for text, entities in zip(batch, batch_results):
                # Remove location entities (optional: filter for 'LOC' only if needed)
                sorted_entities = sorted(entities, key=lambda x: x["start"], reverse=True)

                for entity in sorted_entities:
                    text = text[:entity["start"]] + text[entity["end"]:]

                # Preserve newlines, just collapse tabs/spaces
                text = re.sub(r'[ \t]+', ' ', text).strip()
                cleaned_texts.append(text)

        # Add to DataFrame as a new column
        df[f'cleaned_{column}'] = cleaned_texts
        return df
    


    def remove_numbers(self, df, column='cleaned_text'):
        df = df.copy()

        # Ensure column is string
        df[column] = df[column].fillna('').astype(str)

        # Remove numbers and extra spaces
        df[column] = df[column].apply(lambda x: re.sub(r'\d+', '', x).strip())

        return df
    


    def remove_non_ascii(self, df, column='cleaned_text'):
        # Remove non-ASCII characters directly in the specified column
        df = df.copy()

        # Apply the transformation to the specified column
        df[column] = df[column].fillna('').astype(str).apply(
            lambda text: ''.join([char for char in text if char in string.printable])
        )

        return df
    


    def split_text_on_newline(self, df, column='cleaned_text', filter_empty=True, filter_short=True, min_length=5):
        # Split the text column on '\n' and explode it into new rows
        df = df.copy()
        df[column] = df[column].fillna('').astype(str)  # Ensure it's all strings
        df[column] = df[column].str.split('\n')

        # Explode the DataFrame
        df = df.explode(column, ignore_index=True)

        # Strip whitespace from resulting chunks
        df[column] = df[column].str.strip()

        # If filter_empty is True, drop any rows where the split chunk is empty
        if filter_empty:
            df = df[df[column] != '']

        # If filter_short is True, drop any rows where the split chunk is shorter than min_length
        if filter_short:
            df = df[df[column].str.len() >= min_length]

        # Reset index after exploding
        df.reset_index(drop=True, inplace=True)

        return df



    # def filter_out_short_texts(self, df, column='cleaned_text', min_length=5):
    #     df = df.copy()

    #     # Ensure column is string and fill NaNs
    #     df[column] = df[column].fillna('').astype(str)

    #     return df[df[column].str.len() >= min_length].reset_index(drop=True)



    # def remove_ner_locations(self, text):
    #     """Removes locations from a single string using NER."""
    #     if not text:  # Handle empty strings
    #         return text

    #     ner_results = self.ner_pipeline(text)

    #     # Sort entities in reverse order (to prevent index shifting)
    #     ner_results = sorted(ner_results, key=lambda x: x["start"], reverse=True)

    #     # Remove detected location entities
    #     for entity in ner_results:
    #         text = text[:entity["start"]] + text[entity["end"]:]

    #     # Clean up extra spaces
    #     return re.sub(r'\s+', ' ', text).strip()
    

    # def remove_numbers(self, text):
    #     text = [re.sub(r'\d+', '', single_text) for single_text in text]
    #     # remove extra spaces
    #     text = [re.sub(r'\s+', ' ', single_text).strip() for single_text in text]

    #     return text


    # def clean_and_remove_locations(self, examples):
    #     """Processes a batch of text, splitting and applying NER removal."""
    #     text_list = examples["comment_text"]  # List of text strings

    #     processed_texts = []
    #     for text in text_list:
    #         if not isinstance(text, str):  # Ensure it's a string
    #             continue

    #         # Split into separate sentences if there are newlines
    #         sentences = text.split("\n")
    #         sentences = [s.strip() for s in sentences if s.strip()]  # Remove empty strings

    #         # Apply NER cleaning to each sentence
    #         cleaned_sentences = [self.ner_locations_function(sentence) for sentence in sentences]

    #         # Store each sentence as a new row
    #         processed_texts.extend(cleaned_sentences)

    #     return {"comment_text": processed_texts} 
    
    
    
    # def ner_process_dataset(self, dataset):
    #     """Applies the cleaning function using map."""
    #     return dataset.map(self.clean_and_remove_locations, batched=True)
    

    # def clean_and_remove_locations(self, text):
    #     text = [sentence.split('\n') for sentence in text]
    #     text = [item for sublist in text for item in sublist]
    #     text = [sentence.replace('\n', '') for sentence in text]

    #     # remove empty strings from the list
    #     text = list(filter(None, text))

    #     cleaned_text = [self.ner_locations_function(short_text) for short_text in text]

    #     return cleaned_text
