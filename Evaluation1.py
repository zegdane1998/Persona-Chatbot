import torch
import numpy as np
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
import nltk
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
import os
from transformers import BertTokenizer
from Encoder import initialize_model  # Import your model architecture

# Configuration
MODEL_PATH = '/content/drive/MyDrive/Persona/bert_persona_chatbot/best_model.pt'
DATA_PATH = '/content/drive/MyDrive/Persona/bert_persona_chatbot/val_pairs.csv'
MAX_LENGTH = 128
BATCH_SIZE = 8

def download_nltk_resources():
    try:
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('omw-1.4')  # Required for METEOR
        nltk.download('punkt_tab')  # Specifically for the error you encountered
    except Exception as e:
        print(f"Error downloading NLTK resources: {e}")

# Call this at the start of your script
download_nltk_resources()

def load_model():
    """Load model matching your training setup"""
    # 1. Load tokenizer (same as training)
    tokenizer = BertTokenizer.from_pretrained(f"{os.path.dirname(MODEL_PATH)}/tokenizer")
    
    # 2. Initialize model (same architecture as training)
    model = initialize_model(tokenizer)  # From your Encoder.py
    
    # 3. Load weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    
    # Handle potential DataParallel wrapping
    if all(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k,v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

class TestDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer):
        self.data = df
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        input_text = row['input'].split("Input:")[-1].split("Persona:")[0].strip()
        persona = row['input'].split("Persona:")[-1].strip() if "Persona:" in row['input'] else ""
        
        # Format matching your training data
        full_input = f"Input: {input_text}"
        if persona:
            full_input += f" Persona: {persona}"
            
        # Tokenize like in training
        inputs = self.tokenizer(
            full_input,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            row['response'],
            max_length=MAX_LENGTH,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'response_ids': targets['input_ids'].squeeze(),
            'response_mask': targets['attention_mask'].squeeze(),
            'input_text': full_input,
            'response_text': row['response']
        }

def generate_response(model, tokenizer, input_text, persona, device):
    """Generation matching your training setup"""
    combined_input = f"Input: {input_text} Persona: {persona}"
    
    # Tokenize input
    inputs = tokenizer(
        combined_input,
        return_tensors="pt",
        max_length=512,
        padding='max_length',
        truncation=True
    ).to(device)
    
    # Initialize response with [CLS]
    response_ids = torch.tensor([[tokenizer.cls_token_id]]).to(device)
    
    # Generate tokens autoregressively
    for _ in range(MAX_LENGTH):
        with torch.no_grad():
            outputs = model(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                response_ids=response_ids,
                response_mask=torch.ones_like(response_ids).to(device),
                teacher_forcing_ratio=0.0
            )
        
        # Get next token (greedy)
        next_token = outputs[0, -1].argmax()
        response_ids = torch.cat([response_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        
        # Stop at [SEP]
        if next_token == tokenizer.sep_token_id:
            break
    
    return tokenizer.decode(response_ids[0], skip_special_tokens=True)

def evaluate(model, tokenizer, device):
    """Main evaluation function with fixed tokenization"""
    df = pd.read_csv(DATA_PATH)
    dataset = TestDataset(df, tokenizer)
    loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE)
    
    predictions = []
    references = []
    
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_texts = batch['input_text']
            response_texts = batch['response_text']
            
            # Generate responses
            batch_preds = []
            for text in input_texts:
                parts = text.split("Persona:")
                input_part = parts[0].replace("Input:", "").strip()
                persona = parts[1].strip() if len(parts) > 1 else ""
                pred = generate_response(model, tokenizer, input_part, persona, device)
                batch_preds.append(pred)
            
            predictions.extend(batch_preds)
            references.extend(response_texts)
    
    # Tokenize all texts for METEOR
    ref_tokens = [nltk.word_tokenize(ref.lower()) for ref in references]
    pred_tokens = [nltk.word_tokenize(pred.lower()) for pred in predictions]
    
    # Calculate metrics
    results = {
        'bleu_1': corpus_bleu([[r] for r in ref_tokens], pred_tokens, weights=(1,0,0,0)),
        'bleu_2': corpus_bleu([[r] for r in ref_tokens], pred_tokens, weights=(0.5,0.5,0,0)),
        'bleu_3': corpus_bleu([[r] for r in ref_tokens], pred_tokens, weights=(0.33,0.33,0.33,0)),
        'bleu_4': corpus_bleu([[r] for r in ref_tokens], pred_tokens),
        'meteor': np.mean([meteor_score([ref], pred) for ref, pred in zip(ref_tokens, pred_tokens)]),
        'exact_match': np.mean([int(p.strip().lower() == r.strip().lower()) for r,p in zip(references, predictions)]),
        'avg_length': np.mean([len(p.split()) for p in predictions])
    }
    
    # Save results
    results_df = pd.DataFrame([results])
    results_df.to_csv(f"{os.path.dirname(MODEL_PATH)}/evaluation_results.csv", index=False)
    
    # Print results
    print("\nEvaluation Results:")
    for metric, value in results.items():
        print(f"{metric:>15}: {value:.4f}")
    
    return results

if __name__ == "__main__":
    model, tokenizer, device = load_model()
    results = evaluate(model, tokenizer, device)
    
    # Plot results
    plt.figure(figsize=(12, 5))
    plt.subplot(1,2,1)
    plt.bar(range(1,5), [results[f'bleu_{n}'] for n in range(1,5)])
    plt.title('BLEU Scores')
    plt.xlabel('N-gram')
    
    plt.subplot(1,2,2)
    plt.bar(['METEOR', 'Exact Match'], [results['meteor'], results['exact_match']])
    plt.title('Other Metrics')
    plt.tight_layout()
    plt.savefig(f"{os.path.dirname(MODEL_PATH)}/metrics.png")
    plt.show()