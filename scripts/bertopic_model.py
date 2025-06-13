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

cs = Comments(env="dev")
tp = Topics(env="dev")
nlp_tasks = NLP_Tasks()

df = cs.read_all()

# select only the comments that are classified as 'Objects'
df = df[df['stance']=='Objects']

### Preprocess the text data

# split text on newlines, this function preserves the metadata by exploding the dataframe
train_df_split = nlp_tasks.split_text_on_newline(df=df, column='cleaned_comment_text')
print(f'Dataset size after splitting on newlines: {len(train_df_split)}')

cleaned_text = train_df_split['cleaned_comment_text'].tolist()

### Model hyperparameters

# Load the fine-tuned SentenceTransformer model
# This model is fine-tuned for objection classification
model = SentenceTransformer("Bea-Taylor/objection_fine_tuned_4")
tokenizer = model.tokenizer

# this controls the embedding model
sentence_model = SentenceTransformer("Bea-Taylor/objection_fine_tuned_4")
embeddings = sentence_model.encode(cleaned_text, show_progress_bar=True)

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

### Instantiate the BERTopic model

# define the topic model with parameters 
topic_model = BERTopic(embedding_model=sentence_model,
                       hdbscan_model=hdbscan_model, 
                       umap_model=umap_model, 
                       representation_model=representation_model,
                       verbose=True, 
                       calculate_probabilities=True)

# fit the model to the cleaned text
topics, probs = topic_model.fit_transform(cleaned_text, embeddings)

# save the topics to the database
topic_df = topic_model.get_topic_info()
topic_df[['doc_1', 'doc_2', 'doc_3']] = pd.DataFrame(topic_df['Representative_Docs'].tolist(), index=topic_df.index)
topic_df.to_csv('/root/comment_crunch/outputs/topics_all_comments.csv', index=False)

print('Topics extracted successfully! Saved to /root/comment_crunch/outputs/topics_all_comments.csv')

print(f'Number of topics: {len(topic_df)}')

### Add the topics back to the original dataframe

# add the topics and probabilities to the train_df_split dataframe
all_topic_list =[]
all_prob_list = []

for i in range(len(train_df_split)):
    high_prob_indices = np.where(probs[i] > 0.02)[0]
    high_prob_topics = [i for i in high_prob_indices]
    high_prob_probs = [probs[i][j] for j in high_prob_indices]
    topic_list = []
    prob_list = []
    for topic, prob in zip(high_prob_topics, high_prob_probs):
        topic_list.append(topic)
        prob_list.append(prob)
    all_topic_list.append(topic_list)
    all_prob_list.append(prob_list)

train_df_split['topics'] = all_topic_list
train_df_split['probs'] = all_prob_list


# Group by original_comment_id
grouped = train_df_split.groupby('original_comment_id')

def ensure_list(x):
    return [x] if not isinstance(x, (list, tuple)) else list(x)

for original_comment_id, group in grouped:
    # Flatten and deduplicate topics
    grouped_topics = list(set(
        int(item) for sublist in group['topics'] for item in sublist
    ))
    grouped_topics = ensure_list(grouped_topics)

    # Flatten and deduplicate probs
    grouped_probs = list(set(
        prob for sublist in group['probs']
        for prob in (sublist if isinstance(sublist, list) else [sublist])
    ))
    grouped_probs = ensure_list(grouped_probs)

    # Optional: get a representative comment_id (e.g., first one)
    comment_id = group['comment_id'].iloc[0]

    tp.insert_topic(comment_id=comment_id,
        topic_number=grouped_topics,
        probability=grouped_probs
    )


# new_df = nlp_tasks.merge_sentences_back_to_comments(train_df_split, text_column='cleaned_comment_text',
#                                            topic_column='topics')

# # save the new dataframe with topics and probabilities to the database
# new_df.to_csv('/root/comment_crunch/outputs/comments_with_topics_100.csv')
print('All done!')
# print('Comments with topics and probabilities saved to /root/comment_crunch/outputs/comments_with_topics_100.csv')