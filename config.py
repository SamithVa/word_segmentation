"""Configuration for Chinese Word Segmentation Project"""


# Data paths
DATA_DIR = "icwb2-data"
TRAINING_DIR = f"{DATA_DIR}/training"
TESTING_DIR = f"{DATA_DIR}/testing"
GOLD_DIR = f"{DATA_DIR}/gold"

# Datasets
DATASETS = {
    'pku': {
        'train': f"{TRAINING_DIR}/pku_training.utf8",
        'test': f"{TESTING_DIR}/pku_test.utf8",
        'gold': f"{GOLD_DIR}/pku_test_gold.utf8"
    },
    'msr': {
        'train': f"{TRAINING_DIR}/msr_training.utf8",
        'test': f"{TESTING_DIR}/msr_test.utf8",
        'gold': f"{GOLD_DIR}/msr_test_gold.utf8"
    },
    'cityu': {
        'train': f"{TRAINING_DIR}/cityu_training.utf8",
        'test': f"{TESTING_DIR}/cityu_test.utf8",
        'gold': f"{GOLD_DIR}/cityu_test_gold.utf8"
    },
    'as': {
        'train': f"{TRAINING_DIR}/as_training.utf8",
        'test': f"{TESTING_DIR}/as_test.utf8",
        'gold': f"{GOLD_DIR}/as_testing_gold.utf8"
    }
}

# Output paths
OUTPUT_DIR = "outputs"
SAVED_MODELS_DIR = f"{OUTPUT_DIR}/saved_models"
RESULTS_DIR = f"{OUTPUT_DIR}/results"

# HMM configuration
HMM_SMOOTHING = 1e-3
HMM_USE_PREPROCESS = True

# Special tokens
NUM_TOKEN = '<NUM>'
ENG_TOKEN = '<ENG>'
PAD_TOKEN = '<PAD>'
UNK_TOKEN = '<UNK>'

# Training configuration - Shared across all models
TRAINING_CONFIG = {
    'batch_size': 512,
    'learning_rate': 0.001,
    'epochs': 10,
    'gradient_clip': 5.0,
    'device': 'cuda',  # Will be updated in each script based on availability
    'max_len': 512, 
    'scheduler_step_size': 10,
    'scheduler_gamma': 0.1
}

# Model-specific configurations - Fair comparison with consistent embedding dimensions
SHARED_EMBEDDING_DIM = 128
SHARED_HIDDEN_DIM = 256
SHARED_NUM_LAYERS = 2
SHARED_DROPOUT = 0.3

RNN_CONFIG = {
    'd_model': SHARED_EMBEDDING_DIM,  # Embedding dimension
    'hidden_dim': SHARED_HIDDEN_DIM,   # Hidden state dimension
    'num_layers': SHARED_NUM_LAYERS,
    'dropout': SHARED_DROPOUT
}

LSTM_CONFIG = {
    'embedding_dim': SHARED_EMBEDDING_DIM,  # Embedding dimension
    'hidden_dim': SHARED_HIDDEN_DIM,        # Hidden state dimension
    'num_layers': SHARED_NUM_LAYERS,
    'dropout': SHARED_DROPOUT
}

TRANSFORMER_CONFIG = {
    'd_model': SHARED_EMBEDDING_DIM,      # Embedding dimension
    'nhead': 4,                            # Number of attention heads (d_model should be divisible by nhead)
    'num_layers': SHARED_NUM_LAYERS,
    'dropout': SHARED_DROPOUT
}

# Backward compatibility
BATCH_SIZE = TRAINING_CONFIG['batch_size']
LEARNING_RATE = TRAINING_CONFIG['learning_rate']
EPOCHS = TRAINING_CONFIG['epochs']
GRADIENT_CLIP = TRAINING_CONFIG['gradient_clip']
DEVICE = TRAINING_CONFIG['device']
MAX_LEN = TRAINING_CONFIG['max_len']

# Evaluation
SAVE_PREDICTIONS = True
DETAILED_REPORT = True

# Gradio UI
UI_TITLE = "Chinese Word Segmentation Comparison"
UI_DESCRIPTION = "Compare different segmentation methods on Chinese text"