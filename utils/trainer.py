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
                num_epochs, model_path, val_loader=None, patience=5, use_wandb=False):
    """Training loop with validation and early stopping."""
    model.to(device)
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses, train_accs = [], []

    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        log = {'train_loss': train_loss, 'train_acc': train_acc, 'epoch': epoch + 1}
        msg = f"Epoch {epoch+1}/{num_epochs} | Train: loss={train_loss:.4f}, acc={train_acc:.4f}"
        
        # Validate
        if val_loader:
            val_loss, val_acc = evaluate(model, val_loader, criterion, device)
            log.update({'val_loss': val_loss, 'val_acc': val_acc})
            msg += f" | Val: loss={val_loss:.4f}, acc={val_acc:.4f}"
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(model.state_dict(), model_path)
                msg += " ✓"
            else:
                patience_counter += 1
        else:
            if train_loss < best_val_loss:
                best_val_loss = train_loss
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                torch.save(model.state_dict(), model_path)
                msg += " ✓"
        
        print(msg)
        
        if scheduler:
            scheduler.step()
        
        if use_wandb:
            import wandb
            wandb.log(log)
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return train_losses, train_accs
