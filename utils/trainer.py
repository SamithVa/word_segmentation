"""
Training and evaluation utilities.
"""

import torch
import os
from tqdm import tqdm


def train_epoch(model, dataloader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: the neural network model
        dataloader: training data loader
        criterion: loss function
        optimizer: optimizer
        device: torch device (cuda/cpu)
        
    Returns:
        avg_loss: average loss over the epoch
        accuracy: accuracy over the epoch
    """
    model.train()
    total_loss = 0
    total_correct = 0
    total_count = 0
    
    progress_bar = tqdm(dataloader, desc="Training")
    for chars, tags in progress_bar:
        chars, tags = chars.to(device), tags.to(device)
        
        # Forward pass
        outputs = model(chars)  # [batch, seq_len, num_classes]
        
        # Calculate loss (ignore padding)
        loss = criterion(outputs.view(-1, outputs.size(-1)), tags.view(-1))
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Statistics
        total_loss += loss.item()
        mask = tags != 0  # Non-padding positions
        predictions = outputs.argmax(dim=-1)
        total_correct += ((predictions == tags) & mask).sum().item()
        total_count += mask.sum().item()
        
        progress_bar.set_postfix({'loss': loss.item(), 'acc': total_correct / total_count if total_count > 0 else 0})
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_count if total_count > 0 else 0
    return avg_loss, accuracy


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate the model.
    
    Args:
        model: the neural network model
        dataloader: evaluation data loader
        criterion: loss function
        device: torch device (cuda/cpu)
        
    Returns:
        avg_loss: average loss
        accuracy: accuracy
    """
    model.eval()
    total_loss = 0
    total_correct = 0
    total_count = 0
    
    with torch.no_grad():
        for chars, tags in tqdm(dataloader, desc="Evaluating"):
            chars, tags = chars.to(device), tags.to(device)
            
            outputs = model(chars)
            loss = criterion(outputs.view(-1, outputs.size(-1)), tags.view(-1))
            
            total_loss += loss.item()
            mask = tags != 0
            predictions = outputs.argmax(dim=-1)
            total_correct += ((predictions == tags) & mask).sum().item()
            total_count += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_count if total_count > 0 else 0
    return avg_loss, accuracy


def train_model(model, train_loader, criterion, optimizer, scheduler, device,
                num_epochs, model_path):
    """
    Complete training loop with model saving.

    Args:
        model: the neural network model
        train_loader: training data loader
        criterion: loss function
        optimizer: optimizer
        scheduler: learning rate scheduler (optional)
        device: torch device (cuda/cpu)
        num_epochs: number of training epochs
        model_path: path to save best model

    Returns:
        train_losses: list of training losses
        train_accuracies: list of training accuracies
    """
    model.to(device)

    train_losses = []
    train_accuracies = []
    best_loss = float('inf')

    for epoch in range(num_epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"{'='*50}")

        # Train for one epoch
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")

        # Update learning rate if scheduler is provided
        if scheduler is not None:
            scheduler.step()
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Learning Rate: {current_lr}")

        # Save best model
        if train_loss < best_loss:
            best_loss = train_loss

            # Ensure directory exists
            os.makedirs(os.path.dirname(model_path), exist_ok=True)

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, model_path)
            print(f"✓ Best model saved (loss: {best_loss:.4f})")

    return train_losses, train_accuracies
