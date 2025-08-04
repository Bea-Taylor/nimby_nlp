import numpy as np
import regex

from transformers import BertTokenizerFast, pipeline

from database.comments import Comments

import sys
sys.path.append('../pipeline')
from nlp_tasks import NLP_Tasks

cs = Comments(env='dev')
nlp = NLP_Tasks()

df = cs.read_all()

# to ensure that the sentiment scoring is only done on comments that do not already have a sentiment score
df = df[df['sentiment_score'].isna()].reset_index(drop=True)

sentiment_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")

for i in range(len(df)):
    id = df['comment_id'][i]
    comment = str(df['cleaned_comment_text'][i])
    stance = df['stance'][i]
    
    score = nlp.sentiment_score(comment, stance, sentiment_model)
    
    cs.update_sentiment_score_by_comment_id(id, score)