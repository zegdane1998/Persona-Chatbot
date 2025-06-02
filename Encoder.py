import torch
import torch.nn as nn
from transformers import BertModel, BertConfig

class BERTPersonaChatbot(nn.Module):
    def __init__(self, vocab_size, hidden_size=768):
        super(BERTPersonaChatbot, self).__init__()
        
        # Encoder (BERT model)
        self.encoder = BertModel.from_pretrained('bert-base-uncased')
        
        self.lstm_input_projection = nn.Linear(hidden_size * 2, hidden_size)
        
        # Decoder components
        self.decoder_embedding = nn.Embedding(vocab_size, hidden_size)
        self.decoder_lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True
        )
        self.fc_out = nn.Linear(hidden_size, vocab_size)
        
        # For attention mechanism
        self.attention = nn.Linear(hidden_size * 2, 1)
        
    def forward(self, input_ids, attention_mask, response_ids=None, response_mask=None, teacher_forcing_ratio=0.5):
        batch_size = input_ids.size(0)
        
        # Encode the input sequence
        encoder_outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Get encoder hidden states
        encoder_hidden_states = encoder_outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Use the [CLS] token representation as the initial hidden state for the decoder
        decoder_hidden = encoder_outputs.pooler_output.unsqueeze(0)  # [1, batch_size, hidden_size]
        decoder_cell = torch.zeros_like(decoder_hidden)  # Initial cell state
        
        # If we're training (response_ids is provided)
        if response_ids is not None:
            max_length = response_ids.size(1)
            vocab_size = self.fc_out.out_features
            outputs = torch.zeros(batch_size, max_length, vocab_size).to(input_ids.device)
            
            # First input to the decoder is the start token
            decoder_input = response_ids[:, 0].unsqueeze(1)  # [batch_size, 1]
            
            for t in range(1, max_length):
                # Pass through the decoder
                embedded = self.decoder_embedding(decoder_input)  # [batch_size, seq_len, hidden_size]
                
                # Attention mechanism
                # Repeat decoder hidden state for each encoder position
                hidden_expanded = decoder_hidden.permute(1, 0, 2).expand(-1, encoder_hidden_states.size(1), -1)
                
                # Concatenate with encoder hidden states
                attention_input = torch.cat((hidden_expanded, encoder_hidden_states), dim=2)
                
                # Calculate attention scores
                attention_scores = self.attention(attention_input).squeeze(2)
                
                # Apply mask to exclude padding tokens
                attention_scores = attention_scores.masked_fill(attention_mask == 0, float('-inf'))
                
                # Apply softmax to get attention weights
                attention_weights = torch.nn.functional.softmax(attention_scores, dim=1).unsqueeze(1)
                
                # Calculate context vector
                context = torch.bmm(attention_weights, encoder_hidden_states)
                
                # Combine context vector with decoder input
                lstm_input = torch.cat((embedded, context), dim=2)  # [batch_size, 1, 1536]
                lstm_input = self.lstm_input_projection(lstm_input)  # [batch_size, 1, 768]

                
                # Pass through LSTM
                output, (decoder_hidden, decoder_cell) = self.decoder_lstm(
                    lstm_input, 
                    (decoder_hidden, decoder_cell)
                )
                
                # Generate output
                prediction = self.fc_out(output)  # [batch_size, 1, vocab_size]
                outputs[:, t] = prediction.squeeze(1)
                
                # Teacher forcing: decide whether to use ground truth or model output
                teacher_force = torch.rand(1).item() < teacher_forcing_ratio
                
                if teacher_force and t < max_length - 1:
                    decoder_input = response_ids[:, t].unsqueeze(1)
                else:
                    # Get the most likely next word
                    top1 = prediction.argmax(2)
                    decoder_input = top1
            
            return outputs
        
        else:  # Inference mode
            return None  # Implement inference logic in a separate function

# Initialize model
def initialize_model(tokenizer):
    vocab_size = tokenizer.vocab_size
    model = BERTPersonaChatbot(vocab_size)
    return model