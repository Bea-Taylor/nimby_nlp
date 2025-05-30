import pickle
import logging
from sentence_transformers import SentenceTransformer, losses
from torch.utils.data import DataLoader

# Load training examples from a pickle file
with open('outputs/SBERT_pairs/train_examples.pkl', 'rb') as f:
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
