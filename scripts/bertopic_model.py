# Import necessary packages 
import pandas as pd
import numpy as np

from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired
from sentence_transformers import SentenceTransformer

from bertopic.representation import MaximalMarginalRelevance, KeyBERTInspired

from hdbscan import HDBSCAN
from umap import UMAP

from database.comments import Comments
from database.topics import Topics

import sys
sys.path.append('/root/comment_crunch/pipeline')
from nlp_tasks import NLP_Tasks


# Redirect stdout to a file
orig_stdout = sys.stdout
stdout_filepath = 'topic_stdout/bertopic_model_stdout.txt'
f = open(stdout_filepath, 'w')
sys.stdout = f

# define filepath for model outputs 
output_filepath = '/root/comment_crunch/outputs/topic_model/full_bertopic/'

# instantiate the Comments and Topics classes
cs = Comments(env="dev")
tp = Topics(env="dev")
nlp_tasks = NLP_Tasks()


# Load the fine-tuned SentenceTransformer model
# This model is fine-tuned for objection classification
sentence_model = SentenceTransformer("Bea-Taylor/objection_fine_tuned_4")
tokenizer = sentence_model.tokenizer


### DATA ### 

# Load the comments from the database
df = cs.read_all()

df_objects = df[df['stance']=='Objects']
df_neutral = df[df['stance']=='Neutral']
df_supports = df[df['stance']=='Supports']

### Preprocess the text data

# split text on newlines, this function preserves the metadata by exploding the dataframe
df_objects_split = nlp_tasks.split_text_on_newline(df=df_objects, column='cleaned_comment_text')
df_neutral_split = nlp_tasks.split_text_on_newline(df=df_neutral, column='cleaned_comment_text')
df_supports_split = nlp_tasks.split_text_on_newline(df=df_supports, column='cleaned_comment_text')

print(f'\n Length after splitting data')
print('Objects:', len(df_objects_split))
print('Neutral:', len(df_neutral_split))
print('Supports:', len(df_supports_split))

# split the text by chunks of a maximum length, this function preserves the metadata by exploding the dataframe

max_length_tokens = sentence_model.get_max_seq_length()

df_objects_split = nlp_tasks.split_text_by_length(df=df_objects_split, column='cleaned_comment_text', max_length=max_length_tokens, overlap=40, filter_empty=False, filter_short=False)
df_neutral_split = nlp_tasks.split_text_by_length(df=df_neutral_split, column='cleaned_comment_text', max_length=max_length_tokens, overlap=40, filter_empty=False, filter_short=False)
df_supports_split = nlp_tasks.split_text_by_length(df=df_supports_split, column='cleaned_comment_text', max_length=max_length_tokens, overlap=40, filter_empty=False, filter_short=False)

print(f'\n Length after chunking data')
print('Objects:', len(df_objects_split))
print('Neutral:', len(df_neutral_split))
print('Supports:', len(df_supports_split))

cleaned_object_text = df_objects_split['cleaned_comment_text'].tolist()
cleaned_neutral_text = df_neutral_split['cleaned_comment_text'].tolist()
cleaned_supports_text = df_supports_split['cleaned_comment_text'].tolist()


### MODEL HYPERPARAMETERS ###

# generate embeddings usign the fine-tuned SentenceTransformer model
object_embeddings = sentence_model.encode(cleaned_object_text, show_progress_bar=True)
neutral_embeddings = sentence_model.encode(cleaned_neutral_text, show_progress_bar=True)
supports_embeddings = sentence_model.encode(cleaned_supports_text, show_progress_bar=True)

# this controls the seed - allowing for reproducible maps 
umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0, metric='cosine', random_state=43)

# this controls the topic parameters
hdbscan_model = HDBSCAN(min_cluster_size=10, metric='euclidean', cluster_selection_method='eom', prediction_data=True)

# this controls the topic representation
rm_MMR = MaximalMarginalRelevance(diversity=0.3)
rm_KBERT = KeyBERTInspired()
representation_model = {
    "MaximalMarginalRelevance": rm_MMR,
    "KeyBERTInspired": rm_KBERT,
}

# minimum probability for a topic to be considered relevant when recording to db 
min_topic_prob = 0.02  

### INSTANTIATE AND FIT THE TOPIC MODEL ###

### OBJECT TOPICS ###

# define the topic model with parameters 
topic_model = BERTopic(embedding_model=sentence_model,
                       hdbscan_model=hdbscan_model, 
                       umap_model=umap_model, 
                       representation_model=representation_model,
                       verbose=True, 
                       calculate_probabilities=True)

# fit the model to the cleaned text
print('Fitting the topic model to the object text...')
object_topics, object_probs = topic_model.fit_transform(cleaned_object_text, object_embeddings)

# save the topics with representative documents to a .csv
object_topic_df = topic_model.get_topic_info()
object_topic_df[['doc_1', 'doc_2', 'doc_3']] = pd.DataFrame(object_topic_df['Representative_Docs'].tolist(), index=object_topic_df.index)
object_topic_path = output_filepath+'object_topics.csv'
object_topic_df.to_csv(object_topic_path, index=False)

print(f'Object topics saved to: {object_topic_path}')
print(f'Number of object topics: {len(object_topic_df)}')

# save the topic embeddings to a .csv 
object_topic_embeddings = topic_model.topic_embeddings_
object_topic_embeddings_df = pd.DataFrame(object_topic_embeddings)
object_topic_embeddings_path = output_filepath+'object_topic_embeddings.csv'
object_topic_embeddings_df.to_csv(object_topic_embeddings_path, index=False)

print(f'Object topic embeddings saved to: {object_topic_embeddings_path}')

# merge topics back to comments 
# and save to the remote db 
nlp_tasks.merge_topics_to_comments(df_objects_split, object_probs, min_prob=min_topic_prob, insert_db=True)


### NEUTRAL TOPICS ###

# define the topic model with parameters
topic_model = BERTopic(embedding_model=sentence_model,
                       hdbscan_model=hdbscan_model, 
                       umap_model=umap_model, 
                       representation_model=representation_model,
                       verbose=True, 
                       calculate_probabilities=True)

# fit the model to the cleaned text
print('Fitting the topic model to the neutral text...')
neutral_topics, neutral_probs = topic_model.fit_transform(cleaned_neutral_text, neutral_embeddings)

# save the topics with representative documents to a .csv
neutral_topic_df = topic_model.get_topic_info()
neutral_topic_df[['doc_1', 'doc_2', 'doc_3']] = pd.DataFrame(neutral_topic_df['Representative_Docs'].tolist(), index=neutral_topic_df.index)
neutral_topic_path = output_filepath+'neutral_topics.csv'
neutral_topic_df.to_csv(neutral_topic_path, index=False)

print(f'Neutral topics saved to: {neutral_topic_path}')
print(f'Number of neutral topics: {len(neutral_topic_df)}')

# save the topic embeddings to a .csv
neutral_topic_embeddings = topic_model.topic_embeddings_
neutral_topic_embeddings_df = pd.DataFrame(neutral_topic_embeddings)
neutral_topic_embeddings_path = output_filepath+'neutral_topic_embeddings.csv'
neutral_topic_embeddings_df.to_csv(neutral_topic_embeddings_path, index=False)

print(f'Neutral topic embeddings saved to: {neutral_topic_embeddings_path}')

# merge topics back to comments
# and save to the remote db
nlp_tasks.merge_topics_to_comments(df_neutral_split, neutral_probs, min_prob=min_topic_prob, insert_db=True)


### SUPPORTS TOPICS ###

# define the topic model with parameters
topic_model = BERTopic(embedding_model=sentence_model,
                       hdbscan_model=hdbscan_model, 
                       umap_model=umap_model, 
                       representation_model=representation_model,
                       verbose=True, 
                       calculate_probabilities=True)

# fit the model to the cleaned text
print('Fitting the topic model to the supports text...')
supports_topics, supports_probs = topic_model.fit_transform(cleaned_supports_text, supports_embeddings)

# save the topics with representative documents to a .csv
supports_topic_df = topic_model.get_topic_info()
supports_topic_df[['doc_1', 'doc_2', 'doc_3']] = pd.DataFrame(supports_topic_df['Representative_Docs'].tolist(), index=supports_topic_df.index)
supports_topic_path = output_filepath+'supports_topics.csv'
supports_topic_df.to_csv(supports_topic_path, index=False)

print(f'Supports topics saved to: {supports_topic_path}')
print(f'Number of supports topics: {len(supports_topic_df)}')

# save the topic embeddings to a .csv
supports_topic_embeddings = topic_model.topic_embeddings_
supports_topic_embeddings_df = pd.DataFrame(supports_topic_embeddings)
supports_topic_embeddings_path = output_filepath+'supports_topic_embeddings.csv'
supports_topic_embeddings_df.to_csv(supports_topic_embeddings_path, index=False)

print(f'Supports topic embeddings saved to: {supports_topic_embeddings_path}')

# merge topics back to comments
# and save to the remote db
nlp_tasks.merge_topics_to_comments(df_supports_split, supports_probs, min_prob=min_topic_prob, insert_db=True)

# ### Add the topics back to the original dataframe

# # add the topics and probabilities to the train_df_split dataframe
# all_topic_list =[]
# all_prob_list = []

# for i in range(len(train_df_split)):
#     high_prob_indices = np.where(probs[i] > 0.02)[0]
#     high_prob_topics = [i for i in high_prob_indices]
#     high_prob_probs = [probs[i][j] for j in high_prob_indices]
#     topic_list = []
#     prob_list = []
#     for topic, prob in zip(high_prob_topics, high_prob_probs):
#         topic_list.append(topic)
#         prob_list.append(prob)
#     all_topic_list.append(topic_list)
#     all_prob_list.append(prob_list)

# train_df_split['topics'] = all_topic_list
# train_df_split['probs'] = all_prob_list


# # Group by original_comment_id
# grouped = train_df_split.groupby('original_comment_id')

# def ensure_list(x):
#     return [x] if not isinstance(x, (list, tuple)) else list(x)

# for original_comment_id, group in grouped:

#     # Flatten topics and probs, maintaining their pairings
#     flat_pairs = [
#         (int(topic), prob)
#         for topics_sublist, probs_sublist in zip(group['topics'], group['probs'])
#         for topic, prob in zip(topics_sublist, probs_sublist)
#     ]

#     # Use a dictionary to deduplicate by topic while preserving the first associated prob
#     seen = {}
#     for topic, prob in flat_pairs:
#         if topic not in seen:
#             seen[topic] = prob

#     # Extract deduplicated topics and corresponding probs
#     grouped_topics = list(seen.keys())
#     grouped_probs = list(seen.values())

#     if len(grouped_topics) != len(grouped_probs):
#         print(f"Length mismatch for original_comment_id {original_comment_id}: {len(grouped_topics)} vs {len(grouped_probs)}")

#     comment_id = group['comment_id'].iloc[0]

#     tp.insert_topic(comment_id=comment_id,
#         topic_number=grouped_topics,
#         probability=grouped_probs
#     )


# new_df = nlp_tasks.merge_sentences_back_to_comments(train_df_split, text_column='cleaned_comment_text',
#                                            topic_column='topics')

# # save the new dataframe with topics and probabilities to the database
# new_df.to_csv('/root/comment_crunch/outputs/comments_with_topics_100.csv')
print('All done!')
# print('Comments with topics and probabilities saved to /root/comment_crunch/outputs/comments_with_topics_100.csv')

sys.stdout = orig_stdout
f.close()