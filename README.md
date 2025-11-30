# Chinese Word Segmentation (中文分词)

A comparison of Chinese word segmentation algorithms, from classical rule-based methods to deep learning approaches.

## Features

- **Classical Methods**: FMM, BMM, BiMM (dictionary-based)
- **Statistical**: HMM with Viterbi decoding
- **Deep Learning**: RNN, BiLSTM, Transformer
- **External**: THULAC library integration
- **Interactive UI**: Gradio web interface for real-time comparison

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train a model
python scripts/train.py --model lstm

# Evaluate
python scripts/eval.py --model lstm

# Launch UI
python ui_gradio.py
```

## Models

| Model | Type | F1 Score | Speed |
|-------|------|----------|-------|
| LSTM | Deep Learning | 0.9278 | 56K chars/s |
| Transformer | Deep Learning | 0.8800 | 27K chars/s |
| BiMM | Dictionary | 0.8739 | 193K chars/s |
| BMM | Dictionary | 0.8718 | 342K chars/s |
| FMM | Dictionary | 0.8703 | 366K chars/s |
| RNN | Deep Learning | 0.8230 | 91K chars/s |
| THULAC | External | 0.8114 | 57K chars/s |
| HMM | Statistical | 0.7848 | 101K chars/s |

## Project Structure

```
├── models/              # Model implementations
│   ├── classical.py     # FMM, BMM, BiMM
│   ├── hmm_seg.py       # HMM
│   ├── rnn_seg.py       # RNN
│   ├── lstm_seg.py      # BiLSTM
│   └── transformer_seg.py
├── scripts/
│   ├── train.py         # Training script
│   └── eval.py          # Evaluation script
├── utils/               # Utilities (dataset, evaluation, tokenizer)
├── outputs/             # Saved models and results
├── icwb2-data/          # SIGHAN Bakeoff 2005 datasets
├── ui_gradio.py         # Web interface
└── config.py            # Configuration
```

## Usage

### Training

```bash
python scripts/train.py --model <model_type>
```
Available: `hmm`, `fmm`, `bmm`, `bimm`, `rnn`, `lstm`, `transformer`

### Evaluation

```bash
python scripts/eval.py --model <model_type>
python scripts/eval.py --model all  # Evaluate all models
```

### Web UI

```bash
python ui_gradio.py
# Open http://localhost:7860
```

## Configuration

Edit `config.py` to customize:
- Training parameters (batch_size, epochs, learning_rate)
- Model architectures (embedding_dim, hidden_dim, num_layers)
- Dataset paths

## Results

See [result_comparison.md](result_comparison.md) for detailed benchmarks on PKU, MSR, CITYU, and AS datasets.

## Requirements

- Python 3.8+
- PyTorch (for neural models)
- Gradio (for UI)
- THULAC (optional)
