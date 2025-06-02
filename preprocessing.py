import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from sklearn.model_selection import train_test_split
import re
import os
from google.colab import drive



# Define paths
DATA_PATH = '/content/drive/MyDrive/Persona/New-Persona-New-Conversations.csv'  
MODEL_SAVE_PATH = '/content/drive/MyDrive/Persona/bert_persona_chatbot'

# Create directory for saving model if it doesn't exist
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)

# Load the dataset
def load_dataset(file_path):
    df = pd.read_csv(file_path)
    print(f"Dataset loaded with shape: {df.shape}")
    print(df.columns)
    return df

df = load_dataset(DATA_PATH)

# Define functions to clean and process the conversations
def clean_text(text):
    if isinstance(text, str):
        # Remove special characters and extra spaces
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
    return ""

def extract_conversations(conv_text):
    """Extract turns from conversation text."""
    if not isinstance(conv_text, str):
        return []
    
    # Split by newline and "User 1:" or "User 2:"
    turns = []
    lines = conv_text.strip().split('\n')
    
    for line in lines:
        line = line.strip()
        if line.startswith("User 1:") or line.startswith("User 2:"):
            # Separate speaker from message
            parts = line.split(':', 1)
            if len(parts) == 2:
                speaker = parts[0].strip()
                message = parts[1].strip()
                turns.append((speaker, message))
    
    return turns

def create_conversation_pairs(df):
    """Create input-response pairs from conversations."""
    all_pairs = []
    
    for _, row in df.iterrows():
        user1_persona = clean_text(row['user 1 personas'])
        user2_persona = clean_text(row['user 2 personas'])
        conversation = row['Best Generated Conversation']
        
        turns = extract_conversations(conversation)
        
        # Create pairs of (context + input, response)
        for i in range(1, len(turns)):
            # Check if consecutive turns are from different speakers
            if turns[i-1][0] != turns[i][0]:
                context = []
                # Include up to 3 previous turns as context
                for j in range(max(0, i-3), i):
                    context.append(turns[j][1])
                
                input_text = turns[i-1][1]
                response = turns[i][1]
                
                # Add persona information based on speaker
                if turns[i][0] == "User 1:":
                    persona = user1_persona
                else:
                    persona = user2_persona
                
                # Combine context, current input, and persona
                context_str = " ".join(context)
                combined_input = f"Context: {context_str} Input: {input_text} Persona: {persona}"
                
                all_pairs.append({
                    'input': combined_input,
                    'response': response,
                    'persona': persona
                })
    
    return pd.DataFrame(all_pairs)

# Process the dataset to create conversation pairs
conversation_pairs = create_conversation_pairs(df)
print(f"Created {len(conversation_pairs)} conversation pairs")

# Split the data into training and validation sets
train_df, val_df = train_test_split(conversation_pairs, test_size=0.1, random_state=42)
print(f"Training data: {len(train_df)}, Validation data: {len(val_df)}")

# Save processed data
train_df.to_csv(f"{MODEL_SAVE_PATH}/train_pairs.csv", index=False)
val_df.to_csv(f"{MODEL_SAVE_PATH}/val_pairs.csv", index=False)

print("Preprocessing complete. Data saved to drive.")