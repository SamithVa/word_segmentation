# Chinese Word Segmentation System (中文分词系统)

This project implements and compares various Chinese Word Segmentation (CWS) algorithms, ranging from traditional rule-based methods to statistical models and modern deep learning approaches. It was developed to meet the requirements of an NLP course project.

## 📋 Project Requirements & Features

The project fulfills the following requirements:

1.  **Traditional Matching Algorithms**: Implemented Forward Maximum Matching (FMM), Backward Maximum Matching (BMM), and Bidirectional Maximum Matching (BiMM).
2.  **Statistical Machine Learning**: Implemented Hidden Markov Model (HMM) with Viterbi decoding (without neural networks).
3.  **Deep Learning**: Implemented a **Transformer**-based segmentation model from scratch using PyTorch (without using pre-trained models like BERT).
4.  **Performance Comparison**: Detailed evaluation and comparison of all algorithms on the SIGHAN Bakeoff 2005 datasets (PKU, MSR, CITYU, AS).
5.  **Interactive UI**: A user-friendly web interface built with **Gradio** to demonstrate and compare segmentation results in real-time.

## 📂 Directory Structure

```
.
├── config.py                      # Configuration file with all settings
├── models/                        # Model implementations
│   ├── hmm_tokenizer.py          # HMM implementation
│   ├── classical.py               # FMM, BMM, BiMM implementations
│   ├── rnn_model.py               # RNN model
│   ├── lstm_model.py              # LSTM model
│   ├── transformer_model.py       # Transformer model
│   └── crf_model.py              # CRF model
├── scripts/                       # Training and evaluation scripts
│   ├── hmm.py                    # Train and evaluate HMM model
│   ├── test_classical.py         # Test and evaluate classical methods
│   ├── train_rnn.py              # Train RNN model
│   ├── train_lstm.py             # Train LSTM model
│   ├── train_transformer.py      # Train Transformer model
│   └── test_models.py           # Test all models
├── utils/                         # Shared utilities
│   ├── dataset.py                # Dataset helper functions
│   ├── evaluation.py             # Segmentation metrics
│   └── visualization.py          # Plotting functions
├── outputs/                       # Generated outputs
│   ├── saved_models/              # Trained model weights
│   └── results/                  # Evaluation results
├── ui_gradio.py                   # Gradio Web UI
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

### Architecture Benefits
- **Simplicity**: Clean, flat structure that's easy to navigate
- **Modularity**: Each model is self-contained in its own file
- **Reusability**: Shared utilities reduce code duplication
- **Consistency**: Standardized configuration through `config.py`
- **Easy to Extend**: Add new models by simply adding a new file to `models/`

## 🛠️ Installation

1.  **Navigate to the project directory** (or clone if applicable).

2.  **Install Dependencies**:
    Ensure you have Python 3.8+ installed. Install the required packages using pip:

    ```bash
    pip install -r requirements.txt
    ```
    *Key dependencies: `torch`, `numpy`, `gradio`, `tqdm`, `matplotlib`.*

## 🚀 Usage

### 1. Run the Interactive UI (Recommended)
The easiest way to test the system is through the Gradio web interface.

```bash
python ui_gradio.py
```
This will launch a local web server (usually at `http://127.0.0.1:7860`). You can input Chinese sentences and see segmentation results from FMM, HMM, and Transformer models side-by-side.

### 2. Train Individual Models

#### HMM Model
```bash
python scripts/hmm.py
```

#### RNN Model
```bash
python scripts/train_rnn.py
```

#### LSTM Model
```bash
python scripts/train_lstm.py
```

#### Transformer Model
```bash
python scripts/train_transformer.py
```

**Note**: Training neural models (RNN, LSTM, Transformer) requires significant computational resources. A GPU is recommended.

### 3. Test All Models
To quickly test all models:

```bash
python scripts/test_models.py
```

### 4. Using Models in Code

```python
from models import HMMTokenizer, FMM, BMM, BiMM
from config import DATASETS

# Initialize and load HMM
hmm = HMMTokenizer()
hmm.load('outputs/saved_models/hmm/model.pkl')

# Initialize classical methods with dictionary
fmm = FMM()
fmm.load_dict(DATASETS['pku']['train'])

# Segment text
text = "即将来临时"
result = hmm.tokenize(text)
print(f"HMM: {' / '.join(result)}")
```

## 📊 Performance Comparison

We evaluated the algorithms on four standard datasets from SIGHAN Bakeoff 2005. Here is a summary of the F1 scores:

| Algorithm | Avg Precision | Avg Recall | Avg F1 |
|-----------|---------------|------------|--------|
| **FMM**   | 0.8659        | 0.8750     | 0.8703 |
| **BMM**   | 0.8674        | 0.8764     | **0.8718** |
| **HMM**   | 0.7772        | 0.7816     | 0.7793 |
| **Transformer** | 0.6685  | 0.7928     | 0.7253 |

*See `result_comparison.md` for detailed breakdown by dataset.*

### Analysis
- **FMM/BMM**: Achieved the best performance (~87% F1). This indicates that for this specific dataset, a comprehensive dictionary and greedy matching strategy are very effective. BMM slightly outperformed FMM.
- **HMM**: Performed moderately (~78% F1). While it can handle some unknown words better than dictionary methods, the simple statistical assumptions limit its accuracy compared to the strong baseline of maximum matching with a good dictionary.
- **Transformer**: The current implementation achieved ~72% F1. This lower score (compared to SOTA) is likely due to training from scratch on a relatively small dataset without pre-trained embeddings (like BERT) and limited training epochs.

## 🔧 Adding New Models

To add a new segmentation model to the project:

1. Create a new file in `models/` directory (e.g., `new_model.py`)
2. Implement your model class with `train()` and `tokenize()` methods
3. Add your model to `models/__init__.py`
4. Create a training script in `scripts/` directory
5. Update configuration in `config.py` if needed

Example:
```python
# models/new_model.py
class NewModelTokenizer:
    def __init__(self, param1, param2):
        self.param1 = param1
        self.param2 = param2

    def train(self, data):
        # Training logic
        pass

    def tokenize(self, text):
        # Tokenization logic
        pass
```

## 📝 References
- SIGHAN Bakeoff 2005 Dataset
- "Attention Is All You Need" (Vaswani et al., 2017) for the Transformer architecture.
