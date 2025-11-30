"""Training utilities for Chinese word segmentation."""

import torch
import os
from tqdm import tqdm


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss, total_correct, total_count = 0, 0, 0
    
    for chars, tags in tqdm(dataloader, desc="Training", leave=False):
        chars, tags = chars.to(device), tags.to(device)
        
        outputs = model(chars)
        loss = criterion(outputs.view(-1, outputs.size(-1)), tags.view(-1))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        mask = tags != 0
        total_correct += ((outputs.argmax(dim=-1) == tags) & mask).sum().item()
        total_count += mask.sum().item()
    
    return total_loss / len(dataloader), total_correct / total_count if total_count > 0 else 0


def evaluate(model, dataloader, criterion, device):
    """Evaluate the model."""
    model.eval()
    total_loss, total_correct, total_count = 0, 0, 0
    
    with torch.no_grad():
        for chars, tags in tqdm(dataloader, desc="Validating", leave=False):
            chars, tags = chars.to(device), tags.to(device)
            outputs = model(chars)
            
            total_loss += criterion(outputs.view(-1, outputs.size(-1)), tags.view(-1)).item()
            mask = tags != 0
            total_correct += ((outputs.argmax(dim=-1) == tags) & mask).sum().item()
            total_count += mask.sum().item()
    
    return total_loss / len(dataloader), total_correct / total_count if total_count > 0 else 0


def train_model(model, train_loader, criterion, optimizer, scheduler, device,
                num_epochs, model_path, val_loader, use_wandb=False):
    """Training loop with validation. Saves best model based on val_loss."""
    model.to(device)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Save best model
        saved = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            torch.save(model.state_dict(), model_path)
            saved = " ✓"
        
        print(f"Epoch {epoch+1}/{num_epochs} | Train: {train_loss:.4f}/{train_acc:.4f} | Val: {val_loss:.4f}/{val_acc:.4f}{saved}")
        
        if scheduler:
            scheduler.step()
        
        if use_wandb:
            import wandb
            wandb.log({'train_loss': train_loss, 'train_acc': train_acc, 'val_loss': val_loss, 'val_acc': val_acc})
