
# Chinese Word Segmentation Algorithm Comparison

## Overall Performance Comparison

| Algorithm | Avg Precision | Avg Recall | Avg F1 |
|-----------|---------------|------------|--------|
| **LSTM**  | 0.9305        | 0.9326     | 0.9315 |
| **BIMM**  | 0.8686        | 0.8770     | 0.8727 |
| **BMM**   | 0.8674        | 0.8764     | 0.8718 |
| **FMM**   | 0.8659        | 0.8750     | 0.8703 |
| **RNN**   | 0.7820        | 0.8566     | 0.8176 |
| **HMM**   | 0.7804        | 0.7893     | 0.7848 |
| **Transformer** | 0.6410  | 0.7712     | 0.7000 |
| **THULAC** | 0.8024        | 0.8209     | 0.8114 |

## Detailed Results by Dataset

## Classical Methods

### FMM (Forward Maximum Matching)

| Dataset | Precision | Recall | F1 | Speed (chars/sec) | Total Characters | Processing Time (s) |
|---------|-----------|--------|----|-------------------|------------------|---------------------|
| PKU     | 0.8638    | 0.8480 | 0.8558 | 181204.90 | 172733 | 0.9532 |
| MSR     | 0.8971    | 0.9081 | 0.9026 | 273766.89 | 184358 | 0.6734 |
| CITYU   | 0.8449    | 0.8684 | 0.8565 | 241621.31 | 67744 | 0.2804 |
| AS      | 0.8577    | 0.8754 | 0.8665 | 630038.80 | 197681 | 0.3138 |
| **Average** | **0.8659** | **0.8750** | **0.8703** | **331657.98** | **622516** | **2.2208** |

### BMM (Backward Maximum Matching)

| Dataset | Precision | Recall | F1 | Speed (chars/sec) | Total Characters | Processing Time (s) |
|---------|-----------|--------|----|-------------------|------------------|---------------------|
| PKU     | 0.8654    | 0.8496 | 0.8574 | 197127.47 | 172733 | 0.8763 |
| MSR     | 0.8994    | 0.9101 | 0.9047 | 296025.14 | 184358 | 0.6228 |
| CITYU   | 0.8474    | 0.8709 | 0.8590 | 280209.55 | 67744 | 0.2418 |
| AS      | 0.8574    | 0.8752 | 0.8662 | 674081.34 | 197681 | 0.2933 |
| **Average** | **0.8674** | **0.8764** | **0.8718** | **361860.88** | **622516** | **2.0341** |

### BIMM (Bi-directional Maximal Matching)

| Dataset | Precision | Recall | F1 | Speed (chars/sec) | Total Characters | Processing Time (s) |
|---------|-----------|--------|----|-------------------|------------------|---------------------|
| PKU     | 0.8667    | 0.8501 | 0.8584 | 83003.76 | 172733 | 2.0810 |
| MSR     | 0.9006    | 0.9107 | 0.9057 | 122992.89 | 184358 | 1.4989 |
| CITYU   | 0.8483    | 0.8712 | 0.8596 | 119970.13 | 67744 | 0.5647 |
| AS      | 0.8586    | 0.8758 | 0.8671 | 302878.42 | 197681 | 0.6527 |
| **Average** | **0.8686** | **0.8770** | **0.8727** | **157211.30** | **622516** | **4.7973** |


### HMM (Hidden Markov Model)

| Dataset | Precision | Recall | F1 | Speed (chars/sec) | Total Characters | Processing Time (s) |
|---------|-----------|--------|----|-------------------|------------------|---------------------|
| PKU     | 0.7950    | 0.7887 | 0.7919 | 100290.51 | 172733 | 1.7223 |
| MSR     | 0.7729    | 0.7972 | 0.7849 | 101111.91 | 184358 | 1.8233 |
| CITYU   | 0.7614    | 0.7730 | 0.7672 | 102634.62 | 67744 | 0.6601 |
| AS      | 0.7923    | 0.7981 | 0.7952 | 97988.98 | 197681 | 2.0174 |
| **Average** | **0.7804** | **0.7893** | **0.7848** | **100506.50** | **622516** | **6.2231** |

## Neural Network Methods

All below training is done with `batch_size = 512`.

### BI-LSTM (Bidirectional LSTM)

Hidden-dim: 256
Model parameters: 3,359,749

| Dataset | Precision | Recall | F1 | Speed (chars/sec) | Total Characters | Processing Time (s) |
|---------|-----------|--------|----|-------------------|------------------|---------------------|
| PKU     | 0.9221    | 0.9187 | 0.9204 | 26665.49 | 172733 | 6.4778 |
| MSR     | 0.9306    | 0.9324 | 0.9315 | 24107.83 | 184358 | 7.6472 |
| CITYU   | 0.9377    | 0.9421 | 0.9399 | 23943.45 | 67744 | 2.8293 |
| AS      | 0.9316    | 0.9372 | 0.9344 | 15954.87 | 197681 | 12.3900 |
| **Average** | **0.9305** | **0.9326** | **0.9315** | **22667.91** | **622516** | **29.3443** |

### RNN

Model parameters: 1,221,381

| Dataset | Precision | Recall | F1 | Speed (chars/sec) | Total Characters | Processing Time (s) |
|---------|-----------|--------|----|-------------------|------------------|---------------------|
| PKU     | 0.7950    | 0.8489 | 0.8211 | 131558.54 | 172733 | 1.3130 |
| MSR     | 0.7874    | 0.8667 | 0.8252 | 97032.82 | 184358 | 1.9000 |
| CITYU   | 0.7642    | 0.8449 | 0.8025 | 95898.28 | 67744 | 0.7064 |
| AS      | 0.7813    | 0.8660 | 0.8215 | 39424.44 | 197681 | 5.0142 |
| **Average** | **0.7820** | **0.8566** | **0.8176** | **90978.52** | **622516** | **8.9335** |

### Transformer

Model parameters: 2,176,389

| Dataset | Precision | Recall | F1 | Speed (chars/sec) | Total Characters | Processing Time (s) |
|---------|-----------|--------|----|-------------------|------------------|---------------------|
| PKU     | 0.6120    | 0.7359 | 0.6683 | 48807.36 | 172733 | 3.5391 |
| MSR     | 0.6528    | 0.7862 | 0.7133 | 26252.83 | 184358 | 7.0224 |
| CITYU   | 0.6021    | 0.7438 | 0.6655 | 26105.54 | 67744 | 2.5950 |
| AS      | 0.6970    | 0.8189 | 0.7530 | 8205.22 | 197681 | 24.0921 |
| **Average** | **0.6410** | **0.7712** | **0.7000** | **27342.74** | **622516** | **37.2486** |


### THULAC Library

| Dataset | Precision | Recall | F1 | Speed (chars/sec) | Total Characters | Processing Time (s) |
|---------|-----------|--------|----|-------------------|------------------|---------------------|
| PKU     | 0.9224    | 0.9233 | 0.9228 | 51272.33 | 172733 | 3.3689 |
| MSR     | 0.8317    | 0.8764 | 0.8534 | 52358.86 | 184358 | 3.5210 |
| CITYU   | 0.7234    | 0.7392 | 0.7312 | 63979.33 | 67744 | 1.0588 |
| AS      | 0.7320    | 0.7446 | 0.7383 | 60830.39 | 197681 | 3.2497 |
| **Average** | **0.8024** | **0.8209** | **0.8114** | **57110.23** | **622516** | **11.1985** |


## Key Findings

### 🏆 Best Performers
- **Best F1 Score**: LSTM with 0.9315
- **Best Precision**: LSTM with 0.9305
- **Best Recall**: LSTM with 0.9326

### 📊 Performance Ranking
1. **LSTM**: 0.9315 ⭐⭐⭐⭐⭐
2. **BIMM**: 0.8727 ⭐⭐⭐⭐⭐
3. **BMM**: 0.8718 ⭐⭐⭐⭐⭐
4. **FMM**: 0.8703 ⭐⭐⭐⭐⭐
5. **THULAC**: 0.8114 ⭐⭐⭐⭐
6. **RNN**: 0.8176 ⭐⭐⭐⭐
7. **HMM**: 0.7848 ⭐⭐⭐
8. **Transformer**: 0.7000 ⭐⭐

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