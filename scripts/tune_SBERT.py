# NOTE: this is not the script used to train the model - I actually used '1_0_6_SBERT_fine_tune.ipynb' notebook to train the model

import pickle
import logging
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader

# Load training examples from a pickle file
with open('outputs/train_examples_downsample.pkl', 'rb') as f:
    train_examples = pickle.load(f)

# Summary stats of train_examples
print(f"Number of training examples: {len(train_examples)}")

# Configure logging
logging.basicConfig(format='%(asctime)s - %(message)s',
                    level=logging.INFO)

# Initialize the SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define the training dataloader and loss function
# Use CosineSimilarityLoss as have soft-labelled pairs in the training data
train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=16)
train_loss = losses.CosineSimilarityLoss(model=model) 

# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)], epochs=10)

model.save_to_hub(
    "objection_fine_tuned"
    )
