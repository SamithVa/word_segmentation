
# Chinese Word Segmentation Algorithm Comparison

## Overall Performance Comparison

| Algorithm | Avg Precision | Avg Recall | Avg F1 |
|-----------|---------------|------------|--------|
| **LSTM**  | 0.9324        | 0.9330     | 0.9327 |
| **BMM**   | 0.8674        | 0.8764     | 0.8718 |
| **FMM**   | 0.8659        | 0.8750     | 0.8703 |
| **RNN**   | 0.7920        | 0.8572     | 0.8233 |
| **HMM**   | 0.7772        | 0.7816     | 0.7793 |
| **Transformer** | 0.6685  | 0.7928     | 0.7253 |

## Detailed Results by Dataset

## Classical Methods

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

### BIMM (Bi-directional Maximal Matching)

| Dataset | Precision | Recall | F1     |
|---------|-----------|--------|--------|
| PKU     | 0.8670    | 0.8504 | 0.8586 |
| MSR     | 0.9003    | 0.9104 | 0.9054 |
| CITYU   | 0.8496    | 0.8726 | 0.8609 |
| AS      | 0.8623    | 0.8796 | 0.8709 |
| **Avg** | **0.8698**| **0.8782** | **0.8739** |

### HMM (Hidden Markov Model)

| Dataset | Precision | Recall | F1     |
|---------|-----------|--------|--------|
| PKU     | 0.7911    | 0.7761 | 0.7835 |
| MSR     | 0.7716    | 0.7967 | 0.7840 |
| CITYU   | 0.7546    | 0.7571 | 0.7559 |
| AS      | 0.7914    | 0.7964 | 0.7939 |
| **Avg** | **0.7772**| **0.7816** | **0.7793** |

## Neural Network Methods

All below training is done with `batch_size = 512`.

### BI-LSTM (Bidirectional LSTM)

Model parameters: 3,359,749

| Dataset | Precision | Recall | F1 |
|---------|-----------|--------|----|
| PKU     | 0.9252    | 0.9202 | 0.9227 |
| MSR     | 0.9335    | 0.9311 | 0.9323 |
| CITYU   | 0.9384    | 0.9420 | 0.9402 |
| AS      | 0.9362    | 0.9419 | 0.9390 |
| **Avg** | **0.9333**| **0.9338** | **0.9335** |

### RNN (Vanilla RNN)

Model parameters: 1,221,381

| Dataset | Precision | Recall | F1 |
|---------|-----------|--------|----|
| PKU     | 0.8037    | 0.8417 | 0.8222 |
| MSR     | 0.8081    | 0.8727 | 0.8392 |
| CITYU   | 0.7802    | 0.8532 | 0.8151 |
| AS      | 0.7997    | 0.8733 | 0.8349 |
| **Avg** | **0.7979**| **0.8602** | **0.8278** |

### Transformer

Model parameters: 2,176,389


### THULAC Library

| Dataset | Precision | Recall  | F1      |
|---------|-----------|---------|---------|
| PKU     | 0.9224    | 0.9233  | 0.9228  |
| MSR     | 0.8317    | 0.8764  | 0.8534  |
| CITYU   | 0.7234    | 0.7392  | 0.7312  |
| AS      | 0.7320    | 0.7446  | 0.7383  |
| **Average** | **0.8024**    | **0.8209**  | **0.8114**  |






## Key Findings

### 🏆 Best Performers
- **Best F1 Score**: LSTM with 0.9327
- **Best Precision**: LSTM with 0.9324
- **Best Recall**: LSTM with 0.9330

### 📊 Performance Ranking
1. **LSTM**: 0.9327 ⭐⭐⭐⭐⭐
2. **BMM**: 0.8718 ⭐⭐⭐⭐⭐
3. **FMM**: 0.8703 ⭐⭐⭐⭐⭐
4. **RNN**: 0.8233 ⭐⭐⭐⭐
5. **HMM**: 0.7793 ⭐⭐⭐
6. **Transformer**: 0.7253 ⭐⭐

### 💡 Analysis

- **Neural Network methods (LSTM/RNN)** show the highest performance, demonstrating the power of deep learning for sequence modeling in Chinese word segmentation. LSTM particularly excels with its ability to capture long-range dependencies.

- **Dictionary-based methods (FMM/BMM)** remain strong contenders but are outperformed by neural approaches, suggesting that learned representations can better capture segmentation patterns than rule-based matching.

- **RNN** provides solid performance but is surpassed by LSTM, indicating that gating mechanisms help in handling the complexities of Chinese text.

- **HMM** shows moderate performance (~15% lower than LSTM), indicating that while statistical modeling helps, it cannot match the representational capacity of deep neural networks.

- **Transformer** shows lower performance than expected, possibly due to:
  - Insufficient training data or epochs
  - Hyperparameter tuning needed
  - Model may benefit from pre-trained embeddings (like BERT)
  - Training was interrupted or incomplete

### 🎯 Recommendations

1. **For Production**: Use **LSTM** - highest accuracy and robust performance
2. **For Speed**: Use **FMM** or **BMM** - fast inference with good accuracy
3. **For Flexibility**: Use **HMM** - better handles out-of-vocabulary words
4. **For Research**: Improve **Transformer** with pre-training, larger datasets, or explore advanced architectures like BERT