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

    ## Functions to pre-process the data ahead of fine-tuning the model to have domain specific knowledge. 

    def tokenizing_function(self, examples):
        result = self.tokenizer(examples["comment_text"])
        if self.tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result
    


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
    


    ## Functions to pre-process the data ahead of topic modelling 

    def remove_place_names(self, df, column='text', batch_size=64):
        """Remove place names from the text column of a DataFrame using a NER pipeline.

        Args:
            df (dataframe): The DataFrame containing the text data.
            column (str, optional): Name of column with text. Defaults to 'text'.
            batch_size (int, optional): Size of batch. Defaults to 64.

        Returns:
            dataframe: The DataFrame with a new column containing the cleaned text.
        """


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
        """Remove numbers from the specified column in the DataFrame.

        Args:
            df (dataframe): The DataFrame containing the text data.
            column (str, optional): Name of column with text. Defaults to 'cleaned_text'.

        Returns:
            dataframe: The DataFrame with the specified column cleaned of numbers.
        """


        df = df.copy()

        # Ensure column is string
        df[column] = df[column].fillna('').astype(str)

        # Remove numbers and extra spaces
        df[column] = df[column].apply(lambda x: re.sub(r'\d+', '', x).strip())

        return df
    


    def remove_non_ascii(self, df, column='cleaned_text'):
        """Remove non-ASCII characters from the specified column in the DataFrame.

        Args:
            df (dataframe): The DataFrame containing the text data.
            column (str, optional): Name of column with text. Defaults to 'cleaned_text'.

        Returns:
            dataframe: The DataFrame with the specified column cleaned of non-ASCII characters.
        """


        # Remove non-ASCII characters directly in the specified column
        df = df.copy()

        # Apply the transformation to the specified column
        df[column] = df[column].fillna('').astype(str).apply(
            lambda text: ''.join([char for char in text if char in string.printable])
        )

        return df
    


    def split_text_on_newline(self, df, column='cleaned_text', filter_empty=True, filter_short=True, min_length=5):
        """Split the text in the specified column of a DataFrame on newline characters.
        This function also filters out empty strings and short strings based on the provided criteria.

        Args:
            df (dataframe): The DataFrame containing the text data.
            column (str, optional): Name of column with text. Defaults to 'cleaned_text'.
            filter_empty (bool, optional): Indicates whether to remove empty strings. Defaults to True.
            filter_short (bool, optional): Indicates whether to remove strings shorter thhan min_length. Defaults to True.
            min_length (int, optional): Minimum length of string to keep. Defaults to 5.

        Returns:
            dataframe: The DataFrame with the specified column split into multiple rows based on newline characters. Each row will contain a single chunk of text.
        """


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




