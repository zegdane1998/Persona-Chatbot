import torch
import torch.nn.functional as F
import numpy as np
import random
import re
from Encoder import *

# Define paths
MODEL_SAVE_PATH = '/content/drive/MyDrive/Persona/bert_persona_chatbot'

# Load tokenizer and model
def load_model_for_inference():
    from transformers import BertTokenizer
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained(f"{MODEL_SAVE_PATH}/tokenizer")
    
    # Initialize model
    model = initialize_model(tokenizer)
    
    # Load saved model weights
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load(f"{MODEL_SAVE_PATH}/best_model.pt", map_location=device))
    model.to(device)
    model.eval()
    
    return model, tokenizer, device

# Function to generate a response
def generate_response(model, tokenizer, input_text, persona, device, max_length=50, temperature=0.7):
    # Combine input with persona
    combined_input = f"Input: {input_text} Persona: {persona}"
    
    # Tokenize input
    inputs = tokenizer(
        combined_input,
        max_length=512,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    
    # Create initial decoder input (start token)
    decoder_input = torch.tensor([[tokenizer.cls_token_id]]).to(device)
    
    # Initialize decoder hidden state from encoder
    encoder_outputs = model.encoder(
        input_ids=input_ids,
        attention_mask=attention_mask,
        return_dict=True
    )
    
    decoder_hidden = encoder_outputs.pooler_output.unsqueeze(0)
    decoder_cell = torch.zeros_like(decoder_hidden)
    
    # Store generated tokens
    generated_tokens = [tokenizer.cls_token_id]
    
    # Autoregressive generation
    for _ in range(max_length):
        # Embed the current token
        embedded = model.decoder_embedding(decoder_input)
        
        # Encoder hidden states
        encoder_hidden_states = encoder_outputs.last_hidden_state
        
        # Attention mechanism
        hidden_expanded = decoder_hidden.permute(1, 0, 2).expand(-1, encoder_hidden_states.size(1), -1)
        attention_input = torch.cat((hidden_expanded, encoder_hidden_states), dim=2)
        attention_scores = model.attention(attention_input).squeeze(2)
        attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
        attention_weights = F.softmax(attention_scores, dim=1).unsqueeze(1)
        context = torch.bmm(attention_weights, encoder_hidden_states)
        
        # Combine context vector with decoder input
        lstm_input = torch.cat((embedded, context), dim=2)
        lstm_input = model.lstm_input_projection(lstm_input)  # <- Fix mismatch
        
        # Pass through LSTM
        output, (decoder_hidden, decoder_cell) = model.decoder_lstm(
            lstm_input, 
            (decoder_hidden, decoder_cell)
        )
        
        # Generate logits
        logits = model.fc_out(output).squeeze(1)
        
        # Apply temperature to control randomness
        logits = logits / temperature
        
        # Sample from the distribution
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Add to generated tokens
        generated_tokens.append(next_token.item())
        
        # Stop if end token is generated
        if next_token.item() == tokenizer.sep_token_id:
            break
        
        # Update decoder input
        decoder_input = next_token
    
    # Convert tokens to text
    response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    
    return response

# Interactive chatbot function
def interactive_chatbot():
    # Load model, tokenizer and device
    model, tokenizer, device = load_model_for_inference()
    
    # Load some sample personas from the dataset
    import pandas as pd
    
    try:
        df = pd.read_csv('/content/drive/MyDrive/Persona/New-Persona-New-Conversations.csv')
        personas = df['user 2 personas'].tolist()[:10]  # Take first 10 personas as examples
    except:
        # Fallback personas if the dataset isn't accessible
        personas = [
            "I am comfortable with the weather, and enjoy spending time outdoors. I like listening to blues music. I dance for an hour every day to Prince songs.",
            "I like to run and do yoga in my spare time. I love listening to punk music to help me cope with the stress of my life. I take care of my health and try to stay active.",
            "I work as a software engineer. I enjoy playing video games in my free time. I have a pet cat named Whiskers."
        ]
    
    # Clean personas
    cleaned_personas = []
    for persona in personas:
        if isinstance(persona, str):
            cleaned = re.sub(r'\s+', ' ', persona).strip()
            cleaned_personas.append(cleaned)
    
    if not cleaned_personas:
        cleaned_personas = personas
    
    # Select a random persona for the chatbot
    selected_persona = random.choice(cleaned_personas)
    print("Chatbot initialized with the following persona:")
    print(selected_persona)
    print("\nType 'quit' to exit the chat.")
    
    # Chat loop
    chat_history = []
    while True:
        # Get user input
        user_input = input("\nYou: ")
        
        if user_input.lower() == 'quit':
            print("Ending chat session.")
            break
        
        # Add input to chat history
        chat_history.append(f"User: {user_input}")
        
        # Generate context from recent history (last 3 turns)
        context = " ".join(chat_history[-3:]) if len(chat_history) > 0 else ""
        
        # Generate response
        response = generate_response(
            model, 
            tokenizer, 
            user_input if not context else f"{context} {user_input}", 
            selected_persona,
            device
        )
        
        # Print response
        print(f"Chatbot: {response}")
        
        # Add response to chat history
        chat_history.append(f"Chatbot: {response}")

# Run the chatbot
interactive_chatbot()

# Alternative: Single response generation for testing
def test_response_generation():
    model, tokenizer, device = load_model_for_inference()
    
    test_input = "Hi! How are you today?"
    test_persona = "I like to run and do yoga in my spare time. I love listening to punk music to help me cope with the stress of my life."
    
    response = generate_response(model, tokenizer, test_input, test_persona, device)
    print(f"Input: {test_input}")
    print(f"Persona: {test_persona}")
    print(f"Response: {response}")

# Run test
test_response_generation()