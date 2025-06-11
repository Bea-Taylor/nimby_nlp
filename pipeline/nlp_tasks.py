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
    def __init__(self):

        self.place_ner_pipeline = pipeline(
            task="ner",
            model="cjber/reddit-ner-place_names",
            tokenizer="cjber/reddit-ner-place_names",
            aggregation_strategy="first",
        )

        self.people_ner_pipeline = pipeline(
            task="ner",
            model="dslim/bert-base-NER",
            tokenizer="dslim/bert-base-NER",
            aggregation_strategy="first"
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


    def tokenize_func(self, examples):
        result = self.tokenizer(examples["comment_text"])
        if self.tokenizer.is_fast:
            result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
        return result
    


    def group_chunk_func(self, examples, chunk_size=128):
        
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

    def remove_place_names(self, df, column='text', new_column_name='cleaned_text', batch_size=64):
        """Remove place names from the text column of a DataFrame using a NER pipeline.

        Args:
            df (dataframe): The DataFrame containing the text data.
            column (str, optional): Name of column with text. Defaults to 'text'.
            new_column_name (str, optional): Name of the new column to store cleaned text. Defaults to 'cleaned_text'.
            batch_size (int, optional): Size of batch. Defaults to 64.

        Returns:
            dataframe: The DataFrame with a new column containing the cleaned text.
        """


        df = df.copy()
        texts = df[column].fillna('').astype(str).tolist()
        cleaned_texts = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = self.place_ner_pipeline(batch)

            for text, entities in zip(batch, batch_results):
                # Remove location entities (optional: filter for 'LOC' only if needed)
                sorted_entities = sorted(entities, key=lambda x: x["start"], reverse=True)

                for entity in sorted_entities:
                    text = text[:entity["start"]] + text[entity["end"]:]

                # Preserve newlines, just collapse tabs/spaces
                text = re.sub(r'[ \t]+', ' ', text).strip()
                cleaned_texts.append(text)

        # Add to DataFrame as a new column
        df[new_column_name] = cleaned_texts
        return df



    def remove_person_names(self, df, column='text', new_column_name='cleaned_text', batch_size=64, replace=False):
        """Remove or replace person names in the text column of a DataFrame using a NER pipeline.

        Args:
            df (dataframe): The DataFrame containing the text data.
            column (str, optional): Name of column with text. Defaults to 'text'.
            new_column_name (str, optional): Name of the new column to store cleaned text. Defaults to 'cleaned_text'.
            batch_size (int, optional): Size of batch. Defaults to 64.
            replace (bool, optional): If True, replaces names with '[NAME]'; if False, removes them.

        Returns:
            dataframe: The DataFrame with a new column containing the cleaned text.
        """
        df = df.copy()
        texts = df[column].fillna('').astype(str).tolist()
        cleaned_texts = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_results = self.people_ner_pipeline(batch)

            for text, entities in zip(batch, batch_results):
                # Filter for person entities only
                person_entities = [e for e in entities if e['entity_group'] == 'PER']
                sorted_entities = sorted(person_entities, key=lambda x: x["start"], reverse=True)

                for entity in sorted_entities:
                    if replace:
                        text = text[:entity["start"]] + '[NAME]' + text[entity["end"]:]
                    else:
                        text = text[:entity["start"]] + text[entity["end"]:]

                # Preserve newlines, just collapse tabs/spaces
                text = re.sub(r'[ \t]+', ' ', text).strip()
                cleaned_texts.append(text)

        # Add to DataFrame as a new column
        df[new_column_name] = cleaned_texts
        return df
    


    def remove_numbers(self, df, column='text', new_column_name='cleaned_text',):
        """Remove numbers from the specified column in the DataFrame.

        Args:
            df (dataframe): The DataFrame containing the text data.
            column (str, optional): Name of column with text. Defaults to 'cleaned_text'.
            new_column_name (str, optional): Name of the new column to store cleaned text. Defaults to 'cleaned_text'.

        Returns:
            dataframe: The DataFrame with the specified column cleaned of numbers.
        """


        df = df.copy()

        # Ensure column is string
        df[new_column_name] = df[column].fillna('').astype(str)

        # Remove numbers and extra spaces
        df[new_column_name] = df[new_column_name].apply(lambda x: re.sub(r'\d+', '', x).strip())

        return df
    


    def remove_non_ascii(self, df, column='text', new_column_name='cleaned_text'):
        """Remove non-ASCII characters from the specified column in the DataFrame.

        Args:
            df (dataframe): The DataFrame containing the text data.
            column (str, optional): Name of column with text. Defaults to 'text'.
            new_column_name (str, optional): Name of the new column to store cleaned text. Defaults to 'cleaned_text'.

        Returns:
            dataframe: The DataFrame with the specified column cleaned of non-ASCII characters.
        """


        # Remove non-ASCII characters directly in the specified column
        df = df.copy()

        # Apply the transformation to the specified column
        df[new_column_name] = df[column].fillna('').astype(str).apply(
            lambda text: ''.join([char for char in text if char in string.printable])
        )

        return df
    


    # def split_text_on_newline(self, df, column='text', filter_empty=True, filter_short=True, min_length=5):
    #     """Split the text in the specified column of a DataFrame on newline characters.
    #     This function also filters out empty strings and short strings based on the provided criteria.

    #     Args:
    #         df (dataframe): The DataFrame containing the text data.
    #         column (str, optional): Name of column with text. Defaults to 'text'.
    #         filter_empty (bool, optional): Indicates whether to remove empty strings. Defaults to True.
    #         filter_short (bool, optional): Indicates whether to remove strings shorter than min_length. Defaults to True.
    #         min_length (int, optional): Minimum length of string to keep. Defaults to 5.

    #     Returns:
    #         dataframe: The DataFrame with the specified column split into multiple rows based on newline characters. Each row will contain a single chunk of text.
    #     """

    #     df_copy = df.copy()

    #     # Store the original index as a new column before exploding
    #     df_copy['original_comment_id'] = df_copy.index

    #     # Ensure the text column is string type and handle NaNs
    #     df_copy[column] = df_copy[column].fillna('').astype(str)

    #     # Split the text column on '\n'
    #     df_copy[column] = df_copy[column].str.split('\n')

    #     # Explode the DataFrame, maintaining the original_comment_id
    #     # We explicitly drop the original index here, as we are creating a new row for each sentence.
    #     # The `original_comment_id` column now links back to the original comment.
    #     df_exploded = df_copy.explode(column, ignore_index=True)

    #     # Strip whitespace from resulting chunks
    #     df_exploded[column] = df_exploded[column].str.strip()

    #     # If filter_empty is True, drop any rows where the split chunk is empty
    #     if filter_empty:
    #         df_exploded = df_exploded[df_exploded[column] != '']

    #     # If filter_short is True, drop any rows where the split chunk is shorter than min_length
    #     if filter_short:
    #         df_exploded = df_exploded[df_exploded[column].str.len() >= min_length]

    #     # Reset index after exploding and filtering
    #     df_exploded.reset_index(drop=True, inplace=True)

    #     return df


    
    def split_text_on_newline(self, df, column='text', filter_empty=True, filter_short=True, min_length=5):

        """Split the text in the specified column of a DataFrame on newline characters.
        This function also filters out empty strings and short strings based on the provided criteria.

        Args:
        df (dataframe): The DataFrame containing the text data.
        column (str, optional): Name of column with text. Defaults to 'text'.
        filter_empty (bool, optional): Indicates whether to remove empty strings. Defaults to True.
        filter_short (bool, optional): Indicates whether to remove strings shorter than min_length. Defaults to True.
        min_length (int, optional): Minimum length of string to keep. Defaults to 5.

        Returns:
        dataframe: The DataFrame with the specified column split into multiple rows based on newline characters. Each row will contain a single chunk of text.

        """
        # Split the text column on '\n' and explode it into new rows
        df = df.copy()
        
        df['original_comment_id'] = df.index

        df[column] = df[column].fillna('').astype(str) # Ensure it's all strings

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
    

    def split_text_on_period(self, df, column='text', filter_empty=True, filter_short=True, min_length=5):
        """Split the text in the specified column of a DataFrame on full stop characters.
        This function also filters out empty strings and short strings based on the provided criteria.

        Args:
            df (dataframe): The DataFrame containing the text data.
            column (str, optional): Name of column with text. Defaults to 'text'.
            filter_empty (bool, optional): Indicates whether to remove empty strings. Defaults to True.
            filter_short (bool, optional): Indicates whether to remove strings shorter than min_length. Defaults to True.
            min_length (int, optional): Minimum length of string to keep. Defaults to 5.

        Returns:
            dataframe: The DataFrame with the specified column split into multiple rows based on newline characters. Each row will contain a single chunk of text.
        """


        # Split the text column on '\n' and explode it into new rows
        df = df.copy()
        df[column] = df[column].fillna('').astype(str)  # Ensure it's all strings
        df[column] = df[column].str.split('.')

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
    

    def remove_empty_rows(self, df, column='text'):
        """Remove empty rows from the specified column in the DataFrame.

        Args:
            df (dataframe): The DataFrame containing the text data.
            column (str, optional): Name of column with text. Defaults to 'cleaned_text'.

        Returns:
            dataframe: The DataFrame with the specified column cleaned of empty rows.
        """

        df = df.copy()
        df[column] = df[column].fillna('').astype(str)
        df = df[df[column] != '']
        df = df[df[column].str.strip() != 'Not Available']
        # remove rows with empty strings
        df = df[df[column].str.strip() != 'not available']

        return df



    def chunk_with_overlap(self, text, max_length=512, overlap=20):
        """Chunk the text into smaller segments with a specified overlap.

        Args:
            text (str): The text to be chunked.
            max_length (int, optional): The max token length of the chunk. Defaults to 512.
            overlap (int, optional): The overlap of tokens between chunks. Defaults to 20.

        Returns:
            list: List of strings, each representing a chunk of the original text.
        """


        tokens = self.tokenizer.tokenize(text)

        chunks = []
        for i in range(0, len(tokens), max_length - overlap):
            chunk = tokens[i:i + max_length]
            chunks.append(self.tokenizer.convert_tokens_to_string(chunk))
        return chunks
    


    def split_text_by_length(self, df, column='text', max_length=512, overlap=20, filter_empty=True, filter_short=True, min_length=5):
        """Split the text in the specified column of a DataFrame into smaller chunks of a specified maximum length.
        This function also filters out empty strings and short strings based on the provided criteria.

        Args:
            df (dataframe): The DataFrame containing the text data.
            column (str, optional): Name of column with text. Defaults to 'text'.
            max_length (int, optional): The max token length of the chunk. Defaults to 512.
            overlap (int, optional): The overlap of tokens between chunks. Defaults to 20.
            filter_empty (bool, optional): Indicates whether to remove empty strings. Defaults to True.
            filter_short (bool, optional): Indicates whether to remove strings shorter than min_length. Defaults to True.
            min_length (int, optional): Minimum length of string to keep. Defaults to 5.

        Returns:
            dataframe: The DataFrame with the specified column split into multiple rows based on the specified chunking criteria. Each row will contain a single chunk of text.
        """


        df = df.copy()
        df[column] = df[column].fillna('').astype(str)
        df[column] = df[column].apply(lambda x: NLP_Tasks.chunk_with_overlap(x, max_length, overlap))

        # Explode the DataFrame to have one row per chunk
        df = df.explode(column, ignore_index=True)

        # Strip whitespace from the resulting chunks
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



    def process_string(self, text, replace_person_names=False):
        """
        Process a single string by:
        - Removing place names
        - Removing or replacing person names
        - Removing non-ASCII characters

        Args:
            text (str): The input text string.
            replace_person_names (bool): If True, replaces person names with '[NAME]'; otherwise removes them.

        Returns:
            str: The cleaned text.
        """
        if not text:
            return ""

        # Remove non-ASCII characters early
        text = ''.join([char for char in text if char in string.printable])

        # Run NER pipelines
        place_entities = self.place_ner_pipeline([text])[0]
        person_entities = self.people_ner_pipeline([text])[0]

        # Filter only person entities
        person_entities = [e for e in person_entities if e.get("entity_group") == "PER"]

        # Combine all entities
        all_entities = place_entities + person_entities

        # Sort in reverse order of start index to avoid offset issues during string manipulation
        all_entities = sorted(all_entities, key=lambda x: x["start"], reverse=True)

        for entity in all_entities:
            if entity in person_entities and replace_person_names:
                text = text[:entity["start"]] + "[NAME]" + text[entity["end"]:]
            else:
                text = text[:entity["start"]] + text[entity["end"]:]

        # Clean up extra spaces and tabs (preserve newlines)
        text = re.sub(r'[ \t]+', ' ', text).strip()

        return text
    


    def merge_sentences_back_to_comments(self, df, text_column = 'text', topic_column = 'topic', topic_probability_column = None, min_topic_probability = 0.02, original_id_column = 'original_comment_id'):
        """
        Merges sentences back into their original comments and collects the set of unique topics
        associated with each original comment, optionally filtering by topic assignment probability.

        Assumes the input DataFrame has an 'original_comment_id' column (created by the updated
        split_text_on_newline function), a 'text_column' (containing sentences),
        and a 'topic_column' (assigned by bertopic or similar).
        If topic_probability_column is provided, topics will be filtered based on min_topic_probability.

        Args:
            df (pd.DataFrame): The DataFrame containing exploded sentences, original IDs, and topics.
            text_column (str, optional): The name of the column containing the split sentences. Defaults to 'text'.
            topic_column (str, optional): The name of the column containing the assigned topics. Defaults to 'topic'.
            topic_probability_column (str, optional): The name of the column containing the topic assignment probabilities.
                                                      If None, all unique topics are collected without probability filtering. Defaults to None.
            min_topic_probability (float, optional): The minimum probability a topic assignment must have to be included.
                                                     Only applies if topic_probability_column is provided. Defaults to 0.02.
            original_id_column (str, optional): The name of the column linking back to the original comment ID.
                                                 Defaults to 'original_comment_id'.

        Returns:
            pd.DataFrame: A DataFrame where each row represents an original comment,
                          with the merged comment text and a list of its unique (and possibly filtered) topics.
        """
        # Group by the original comment ID
        grouped = df.groupby(original_id_column)

        def aggregate_group_data(group_df):
            """
            Helper function to aggregate text and topics for a single group (original comment).
            This function is applied to each sub-DataFrame of the grouped object.
            """
            merged_comment_text = '. '.join(group_df[text_column].astype(str))

            all_topics_for_group = set()

            # Iterate through each row (representing a sentence) in the current group_df
            for _, row in group_df.iterrows():
                topics_for_sentence = row[topic_column]
                
                # Check for topic_probability_column existence and validity
                prob_for_sentence = None
                if topic_probability_column and topic_probability_column in group_df.columns and pd.notna(row[topic_probability_column]):
                    prob_for_sentence = row[topic_probability_column]

                # Ensure topics_for_sentence is iterable (can be a single topic or a list of topics)
                # This handles cases where a sentence might have a single topic or multiple topics.
                if not isinstance(topics_for_sentence, list):
                    topics_for_sentence = [topics_for_sentence] # Convert single topic to a list for uniform processing

                # Apply probability filtering if specified and probability exists
                if topic_probability_column and prob_for_sentence is not None:
                    if prob_for_sentence > min_topic_probability:
                        # If probability threshold is met for the sentence, add all its topics
                        for topic in topics_for_sentence:
                            all_topics_for_group.add(topic)
                else:
                    # If no probability filtering or column not found, add all topics from this sentence
                    for topic in topics_for_sentence:
                        all_topics_for_group.add(topic)
            
            # Return a Series with the aggregated results for this group
            return pd.Series({
                'merged_comment': merged_comment_text,
                'topics': sorted(list(all_topics_for_group))
            })

        # Apply the custom aggregation function to each group.
        # This will return a DataFrame where the index is original_comment_id
        # and columns are 'merged_comment' and 'topics'.
        merged_comments_df = grouped.apply(aggregate_group_data).reset_index()

        return merged_comments_df