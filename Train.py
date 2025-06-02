import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from Encoder import *
from Tokenization import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--resume_checkpoint', type=str, default=None, 
                   help='Path to checkpoint to resume from (e.g., checkpoint_epoch_1.pt)')
parser.add_argument('--num_epochs', type=int, default=3,
                   help='Total number of epochs to train (including resumed epochs)')
args = parser.parse_args()

# Define model save path
MODEL_SAVE_PATH = '/content/drive/MyDrive/Persona/bert_persona_chatbot'
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)  

# Training function with improved resume capability
def train_model(model, train_loader, val_loader, tokenizer, num_epochs=3, learning_rate=2e-5, resume_checkpoint=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    train_losses = []
    val_losses = []
    
    # Initialize training variables
    start_epoch = 0
    step = 0
    best_val_loss = float('inf')

    # Resume training if checkpoint provided
    if resume_checkpoint and os.path.exists(resume_checkpoint):
        print(f"Loading checkpoint: {resume_checkpoint}")
        checkpoint = torch.load(resume_checkpoint, map_location=device)
        
        # Load model and optimizer states
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Resume training progress
        start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
        step = checkpoint.get('step', 0)
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        
        # Load previous losses if available
        if 'train_losses' in checkpoint:
            train_losses = checkpoint['train_losses']
        if 'val_losses' in checkpoint:
            val_losses = checkpoint['val_losses']
        
        print(f"Resuming training from epoch {start_epoch} (previously trained {checkpoint['epoch']} epochs)")
    
    # Calculate remaining epochs
    remaining_epochs = num_epochs - start_epoch
    if remaining_epochs <= 0:
        print(f"Training already completed {num_epochs} epochs")
        return model, train_losses, val_losses

    for epoch in range(start_epoch, start_epoch + remaining_epochs):
        start_time = time.time()
        model.train()
        epoch_train_loss = 0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            response_ids = batch['response_ids'].to(device)
            response_mask = batch['response_mask'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                response_ids=response_ids,
                response_mask=response_mask,
                teacher_forcing_ratio=0.5
            )

            outputs = outputs[:, 1:, :]
            targets = response_ids[:, 1:]
            loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            step += 1

            # Save checkpoint every 500 steps
            if step % 500 == 0:
                checkpoint_path = os.path.join(MODEL_SAVE_PATH, f"checkpoint_epoch{epoch+1}_step{step}.pt")
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_losses': train_losses,
                    'val_losses': val_losses,
                    'val_loss': best_val_loss
                }, checkpoint_path)
                print(f"[Checkpoint saved at {checkpoint_path}]")

        avg_train_loss = epoch_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        epoch_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                response_ids = batch['response_ids'].to(device)
                response_mask = batch['response_mask'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    response_ids=response_ids,
                    response_mask=response_mask,
                    teacher_forcing_ratio=0.0
                )

                outputs = outputs[:, 1:, :]
                targets = response_ids[:, 1:]
                loss = criterion(outputs.reshape(-1, outputs.shape[-1]), targets.reshape(-1))
                epoch_val_loss += loss.item()

        avg_val_loss = epoch_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Update best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_path = os.path.join(MODEL_SAVE_PATH, "best_model.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved with val_loss: {best_val_loss:.4f}")

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{num_epochs} completed in {epoch_time:.2f}s")
        print(f"Train Loss: {avg_train_loss:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")

        # Save epoch checkpoint
        checkpoint_path = os.path.join(MODEL_SAVE_PATH, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_losses': train_losses,
            'val_losses': val_losses
        }, checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(MODEL_SAVE_PATH, "final_model.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved to {final_model_path}")

    # Plot losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Training Loss')
    plt.plot(range(1, len(val_losses)+1), val_losses, 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(MODEL_SAVE_PATH, 'loss_plot.png'))
    plt.show()

    return model, train_losses, val_losses

# Run training
def run_training():
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained(f"{MODEL_SAVE_PATH}/tokenizer")
    model = initialize_model(tokenizer)

    # Load your dataloaders
    # train_loader = ...
    # val_loader = ...

    model, train_losses, val_losses = train_model(
        model, 
        train_loader, 
        val_loader, 
        tokenizer,
        num_epochs=args.num_epochs,
        learning_rate=2e-5,
        resume_checkpoint=args.resume_checkpoint
    )
    
    return model

if __name__ == "__main__":
    model = run_training()