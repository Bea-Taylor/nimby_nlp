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

sentiment_model = pipeline(model="finiteautomata/bertweet-base-sentiment-analysis")

for i in range(len(df)):
    id = df['comment_id'][i]
    comment = df['cleaned_comment_text'][i]
    stance = df['stance'][i]
    
    score = nlp.sentiment_score(comment, stance, sentiment_model)
    
    cs.update_sentiment_score_by_comment_id(id, score)