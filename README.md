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
├── fmm_bmm/                # Traditional Matching Algorithms
│   └── forward_backward.py # Implementation of FMM, BMM, BiMM
├── hmm/                    # Statistical Algorithms
│   └── hmm.py              # Implementation of HMM
├── transformer_based/      # Deep Learning Algorithms
│   ├── tranf.py            # Transformer model implementation (Training & Inference)
│   └── inference.py        # Inference scripts
├── icwb2-data/             # SIGHAN Bakeoff 2005 Dataset
│   ├── gold/               # Gold standard segmentations
│   ├── training/           # Training data
│   └── testing/            # Test data
├── ui_gradio.py            # Gradio Web UI Application
├── comparison_result.md    # Detailed performance comparison report
├── plot_analysis.py        # Analysis and plotting scripts
├── requirements.txt        # Python dependencies
└── README.md               # Project documentation
```

## 🛠️ Installation

1.  **Clone the repository** (if applicable) or navigate to the project directory.

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

### 2. Traditional Algorithms (FMM/BMM/BiMM)
To run and evaluate the rule-based algorithms:

```bash
cd fmm_bmm
python forward_backward.py
```
This script will:
- Load dictionaries from the training data.
- Test on example sentences.
- Evaluate F1 scores on all test datasets.

### 3. Hidden Markov Model (HMM)
To train and evaluate the HMM model:

```bash
cd hmm
python hmm.py
```
This script will:
- Train the HMM parameters (Start, Transition, Emission probabilities) using the training data.
- Save the model to `hmm_model.pkl`.
- Evaluate performance on test datasets.

### 4. Transformer Model
To train the Transformer model from scratch:

```bash
cd transformer_based
python tranf.py
```
This script will:
- Build vocabulary from training data.
- Train a Transformer encoder model for BMES sequence tagging.
- Save the best model to `transformer_seg_best.pth`.
- Generate training loss/accuracy plots.
- Evaluate on test datasets.

**Note**: Training a Transformer requires significant computational resources. A GPU is recommended.

## 📊 Performance Comparison

We evaluated the algorithms on four standard datasets from SIGHAN Bakeoff 2005. Here is a summary of the F1 scores:

| Algorithm | Avg Precision | Avg Recall | Avg F1 |
|-----------|---------------|------------|--------|
| **FMM**   | 0.8659        | 0.8750     | 0.8703 |
| **BMM**   | 0.8674        | 0.8764     | **0.8718** |
| **HMM**   | 0.7772        | 0.7816     | 0.7793 |
| **Transformer** | 0.6685  | 0.7928     | 0.7253 |

*See `comparison_result.md` for detailed breakdown by dataset.*

### Analysis
- **FMM/BMM**: Achieved the best performance (~87% F1). This indicates that for this specific dataset, a comprehensive dictionary and greedy matching strategy are very effective. BMM slightly outperformed FMM.
- **HMM**: Performed moderately (~78% F1). While it can handle some unknown words better than dictionary methods, the simple statistical assumptions limit its accuracy compared to the strong baseline of maximum matching with a good dictionary.
- **Transformer**: The current implementation achieved ~72% F1. This lower score (compared to SOTA) is likely due to training from scratch on a relatively small dataset without pre-trained embeddings (like BERT) and limited training epochs.

## 📝 References
- SIGHAN Bakeoff 2005 Dataset
- "Attention Is All You Need" (Vaswani et al., 2017) for the Transformer architecture.
