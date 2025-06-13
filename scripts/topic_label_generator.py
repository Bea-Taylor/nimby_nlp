# Run this script to generate topic labels based on the topics extracted from the comments.
# This script will prompt the user to provide a name for each topic and save the results in a CSV file.

import pandas as pd

# Load the topics DataFrame
topic_df = pd.read_csv('/root/comment_crunch/outputs/topics.csv')

data = []
for i in range(len(topic_df)):
    topic = topic_df.iloc[i]
    mmr_topic = topic['MaximalMarginalRelevance']
    kBERT_topic = topic['KeyBERTInspired']
    rep_doc = topic['doc_1']
    print("-" * 50)
    print(f"Topic: {i}")
    print("-" * 50)
    print(f"""MMR representation: {mmr_topic}\n
          KeyBERT representation: {kBERT_topic}""")
    print("Representative document")
    print(rep_doc)
    print("-" * 50)

    # Prompt user for topic label
    while True:
        topic_label_response = input("Please provide a name for this topic: ").strip()
        if topic_label_response:
            break
        else:
            print("Topic name cannot be empty. Please try again.")
    topic_label = topic_label_response

    # Prepare data for DataFrame
    data.append({
        'Topic': topic['Topic'],
        'Name': topic['Name'],
        'Count': topic['Count'],
        'MaximalMarginalRelevance': mmr_topic,
        'KeyBERTInspired': kBERT_topic,
        'Representative_Docs': rep_doc,
        'Label': topic_label
    })

    # Create DataFrame
    df = pd.DataFrame(data)
    # Save the DataFrame to a CSV file
    df.to_csv('/root/comment_crunch/outputs/named_n_labelled_topics.csv', index=False)

print("\nAll responses collected! The topics have been saved to 'named_n_labelled_topics.csv'.")

