#!/usr/bin/env python3
"""
Test the improved training visualization with sample data.
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.visualization import plot_training_history

# Sample training data that mimics the problematic LSTM training
train_losses = [1.2345, 0.9876, 0.8234, 0.7123, 0.6234, 0.5432, 0.4678, 0.3890, 0.3345, 0.2890]
train_accuracies = [0.6543, 0.7234, 0.7812, 0.8234, 0.8567, 0.8790, 0.9012, 0.9189, 0.9323, 0.9456]

print("Testing improved visualization...")
print(f"Sample losses: {train_losses}")
print(f"Sample accuracies: {train_accuracies}")

# Generate the improved plot
plot_training_history(
    train_losses,
    train_accuracies,
    output_path='improved_training_history_test.png',
    model_name='Test LSTM'
)

print("\n✅ Improved visualization test completed!")
print("📊 Check 'improved_training_history_test.png' to see the enhanced plot")