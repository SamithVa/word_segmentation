
# Chinese Word Segmentation Algorithm Comparison

## Overall Performance Comparison

| Algorithm | Avg Precision | Avg Recall | Avg F1 |
|-----------|---------------|------------|--------|
| **FMM**   | 0.8659        | 0.8750     | 0.8703 |
| **BMM**   | 0.8674        | 0.8764     | 0.8718 |
| **HMM**   | 0.7772        | 0.7816     | 0.7793 |
| **Transformer** | 0.6685  | 0.7928     | 0.7253 |

## Detailed Results by Dataset

### FMM (Forward Maximum Matching)

| Dataset | Precision | Recall | F1     |
|---------|-----------|--------|--------|
| PKU     | 0.8638    | 0.8480 | 0.8558 |
| MSR     | 0.8971    | 0.9081 | 0.9026 |
| CITYU   | 0.8449    | 0.8684 | 0.8565 |
| AS      | 0.8577    | 0.8754 | 0.8665 |
| **Avg** | **0.8659**| **0.8750** | **0.8703** |

### BMM (Backward Maximum Matching)

| Dataset | Precision | Recall | F1     |
|---------|-----------|--------|--------|
| PKU     | 0.8654    | 0.8496 | 0.8574 |
| MSR     | 0.8994    | 0.9101 | 0.9047 |
| CITYU   | 0.8474    | 0.8709 | 0.8590 |
| AS      | 0.8574    | 0.8752 | 0.8662 |
| **Avg** | **0.8674**| **0.8764** | **0.8718** |

### HMM (Hidden Markov Model)

| Dataset | Precision | Recall | F1     |
|---------|-----------|--------|--------|
| PKU     | 0.7911    | 0.7761 | 0.7835 |
| MSR     | 0.7716    | 0.7967 | 0.7840 |
| CITYU   | 0.7546    | 0.7571 | 0.7559 |
| AS      | 0.7914    | 0.7964 | 0.7939 |
| **Avg** | **0.7772**| **0.7816** | **0.7793** |

### Transformer (Deep Learning)

| Dataset | Precision | Recall | F1     |
|---------|-----------|--------|--------|
| PKU     | 0.6330    | 0.7550 | 0.6886 |
| MSR     | 0.6789    | 0.8051 | 0.7366 |
| CITYU   | 0.6258    | 0.7640 | 0.6880 |
| AS      | 0.7363    | 0.8470 | 0.7877 |
| **Avg** | **0.6685**| **0.7928** | **0.7253** |

## Key Findings

### 🏆 Best Performers
- **Best F1 Score**: BMM with 0.8718 (slightly outperforms FMM)
- **Best Precision**: BMM with 0.8674
- **Best Recall**: BMM with 0.8764

### 📊 Performance Ranking
1. **BMM**: 0.8718 ⭐⭐⭐⭐⭐
2. **FMM**: 0.8703 ⭐⭐⭐⭐⭐
3. **HMM**: 0.7793 ⭐⭐⭐
4. **Transformer**: 0.7253 ⭐⭐

### 💡 Analysis

- **Dictionary-based methods (FMM/BMM)** show superior performance across all datasets, suggesting that:
  - The vocabulary extracted from training data is comprehensive
  - The greedy matching strategy works well for Chinese word segmentation
  - BMM has a slight edge over FMM by choosing longer matches when ambiguous

- **HMM** shows moderate performance (~10% lower than FMM/BMM), indicating:
  - Statistical modeling adds complexity without proportional benefit
  - BMES tagging may lose some granular dictionary information
  - Good for handling unknown words but at the cost of overall accuracy

- **Transformer** shows lower performance than expected, possibly due to:
  - Insufficient training data or epochs
  - Hyperparameter tuning needed
  - Model may benefit from pre-trained embeddings (like BERT)
  - Training was interrupted or incomplete

### 🎯 Recommendations

1. **For Production**: Use **BMM** - best balance of accuracy and speed
2. **For Speed**: Use **FMM** - marginally lower accuracy but faster inference
3. **For Flexibility**: Use **HMM** - better handles out-of-vocabulary words
4. **For Research**: Improve **Transformer** with pre-training and more data