import pandas as pd
from database.comments import Comments

import sys
import os

sys.path.append('/Users/bea/Documents/AI4CI/projects/comment_summariser/comment_crunch/pipeline')
from nlp_tasks import NLP_Tasks
# from comments_saver import CommentsSaver

# Initialize saver and NLP class
cs = Comments(env="dev")
# cs = CommentsSaver(env='dev')
df = cs.read_all()
n = len(df)

nlp = NLP_Tasks()

for i in range(n):
    row = df.iloc[i]
    text_id = row['comment_id']
    text_string = row['comment_text']
    cleaned_text_existing = row.get('cleaned_comment_text')

    # Skip if already processed
    if pd.notna(cleaned_text_existing) and cleaned_text_existing.strip() != '':
        print(f'Skipping comment {i+1} with id {text_id} (already processed).')
        continue

    # Process and update
    print(f'Processing comment {i+1} with id {text_id}')
    # print(f'Original text: {text_string}')
    # print('Cleaning text...')
    cleaned_text_string = nlp.process_string(text=text_string)
    # print(f'Cleaned text: {cleaned_text_string}')
    cs.update_cleaned_comment_text(
        comment_id=text_id,
        cleaned_text=cleaned_text_string
    )

print('All unprocessed comments handled successfully.')
