import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import pandas as pd

# Define paths
MODEL_SAVE_PATH = '/content/drive/MyDrive/Persona/bert_persona_chatbot'

# Load the processed data
train_df = pd.read_csv(f"{MODEL_SAVE_PATH}/train_pairs.csv")
val_df = pd.read_csv(f"{MODEL_SAVE_PATH}/val_pairs.csv")

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Save the tokenizer for later use
tokenizer.save_pretrained(f"{MODEL_SAVE_PATH}/tokenizer")

# Define the dataset class
class PersonaChatDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=512):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        input_text = self.data.iloc[idx]['input']
        response_text = self.data.iloc[idx]['response']
        
        # Tokenize input and response
        input_encoding = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        response_encoding = self.tokenizer(
            response_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': input_encoding['input_ids'].squeeze(),
            'attention_mask': input_encoding['attention_mask'].squeeze(),
            'response_ids': response_encoding['input_ids'].squeeze(),
            'response_mask': response_encoding['attention_mask'].squeeze(),
            'input_text': input_text,
            'response_text': response_text
        }

# Create dataset objects
train_dataset = PersonaChatDataset(train_df, tokenizer)
val_dataset = PersonaChatDataset(val_df, tokenizer)

# Create data loaders
batch_size = 8
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Number of batches in train loader: {len(train_loader)}")
print(f"Number of batches in validation loader: {len(val_loader)}")

# Check a sample from the dataset
sample = next(iter(train_loader))
print("\nSample input text:")
print(sample['input_text'][0])
print("\nSample response text:")
print(sample['response_text'][0])
print("\nInput tensor shape:", sample['input_ids'].shape)
print("Response tensor shape:", sample['response_ids'].shape)