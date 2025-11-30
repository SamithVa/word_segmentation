# Chinese Word Segmentation Algorithm Comparison

## Overall Performance Comparison

| Algorithm | Avg Precision | Avg Recall | Avg F1 | Avg Speed (chars/sec) |
|-----------|---------------|------------|--------|----------------------|
| **LSTM**  | 0.9266        | 0.9290     | 0.9278 | 56,297               |
| **Transformer** | 0.8684  | 0.8918     | 0.8800 | 27,345               |
| **BiMM**  | 0.8698        | 0.8782     | 0.8739 | 193,371              |
| **BMM**   | 0.8674        | 0.8764     | 0.8718 | 342,412              |
| **FMM**   | 0.8659        | 0.8750     | 0.8703 | 365,858              |
| **RNN**   | 0.7902        | 0.8589     | 0.8230 | 91,034               |
| **THULAC**| 0.8024        | 0.8209     | 0.8114 | 57,110               |
| **HMM**   | 0.7804        | 0.7893     | 0.7848 | 100,507              |

---

## Detailed Results by Dataset

### Classical Methods

#### FMM (Forward Maximum Matching)

| Dataset | Precision | Recall | F1 | Speed (chars/sec) | Total Characters | Processing Time (s) |
|---------|-----------|--------|----|-------------------|------------------|---------------------|
| PKU     | 0.8638    | 0.8480 | 0.8558 | 196,875 | 172,733 | 0.88 |
| MSR     | 0.8971    | 0.9081 | 0.9026 | 291,033 | 184,358 | 0.63 |
| CITYU   | 0.8449    | 0.8684 | 0.8565 | 272,587 | 67,744  | 0.25 |
| AS      | 0.8577    | 0.8754 | 0.8665 | 702,937 | 197,681 | 0.28 |
| **Average** | **0.8659** | **0.8750** | **0.8703** | **365,858** | **622,516** | **2.04** |

#### BMM (Backward Maximum Matching)

| Dataset | Precision | Recall | F1 | Speed (chars/sec) | Total Characters | Processing Time (s) |
|---------|-----------|--------|----|-------------------|------------------|---------------------|
| PKU     | 0.8654    | 0.8496 | 0.8574 | 191,416 | 172,733 | 0.90 |
| MSR     | 0.8994    | 0.9101 | 0.9047 | 293,669 | 184,358 | 0.63 |
| CITYU   | 0.8474    | 0.8709 | 0.8590 | 259,681 | 67,744  | 0.26 |
| AS      | 0.8574    | 0.8752 | 0.8662 | 624,881 | 197,681 | 0.32 |
| **Average** | **0.8674** | **0.8764** | **0.8718** | **342,412** | **622,516** | **2.11** |

#### BiMM (Bidirectional Maximum Matching)

| Dataset | Precision | Recall | F1 | Speed (chars/sec) | Total Characters | Processing Time (s) |
|---------|-----------|--------|----|-------------------|------------------|---------------------|
| PKU     | 0.8670    | 0.8504 | 0.8586 | 104,216 | 172,733 | 1.66 |
| MSR     | 0.9004    | 0.9105 | 0.9054 | 154,832 | 184,358 | 1.19 |
| CITYU   | 0.8496    | 0.8726 | 0.8609 | 148,881 | 67,744  | 0.46 |
| AS      | 0.8623    | 0.8795 | 0.8708 | 365,554 | 197,681 | 0.54 |
| **Average** | **0.8698** | **0.8782** | **0.8739** | **193,371** | **622,516** | **3.84** |

---

### Statistical Methods

#### HMM (Hidden Markov Model)

| Dataset | Precision | Recall | F1 | Speed (chars/sec) | Total Characters | Processing Time (s) |
|---------|-----------|--------|----|-------------------|------------------|---------------------|
| PKU     | 0.7950    | 0.7887 | 0.7919 | 100,291 | 172,733 | 1.72 |
| MSR     | 0.7729    | 0.7972 | 0.7849 | 101,112 | 184,358 | 1.82 |
| CITYU   | 0.7614    | 0.7730 | 0.7672 | 102,635 | 67,744  | 0.66 |
| AS      | 0.7923    | 0.7981 | 0.7952 | 97,989  | 197,681 | 2.02 |
| **Average** | **0.7804** | **0.7893** | **0.7848** | **100,507** | **622,516** | **6.22** |

---

### Neural Network Methods

#### RNN (Recurrent Neural Network)

- **Model Parameters**: 1,221,381

| Dataset | Precision | Recall | F1 | Speed (chars/sec) | Total Characters | Processing Time (s) |
|---------|-----------|--------|----|-------------------|------------------|---------------------|
| PKU     | 0.8005    | 0.8438 | 0.8216 | 131,469 | 172,733 | 1.31 |
| MSR     | 0.8033    | 0.8739 | 0.8371 | 96,354  | 184,358 | 1.91 |
| CITYU   | 0.7675    | 0.8476 | 0.8056 | 96,485  | 67,744  | 0.70 |
| AS      | 0.7895    | 0.8702 | 0.8279 | 39,827  | 197,681 | 4.96 |
| **Average** | **0.7902** | **0.8589** | **0.8230** | **91,034** | **622,516** | **8.89** |

#### BiLSTM (Bidirectional LSTM)

- **Model Parameters**: 1,650,437
- **Hidden Dimension**: 128

| Dataset | Precision | Recall | F1 | Speed (chars/sec) | Total Characters | Processing Time (s) |
|---------|-----------|--------|----|-------------------|------------------|---------------------|
| PKU     | 0.9215    | 0.9185 | 0.9200 | 78,790  | 172,733 | 2.19 |
| MSR     | 0.9290    | 0.9311 | 0.9300 | 59,996  | 184,358 | 3.07 |
| CITYU   | 0.9338    | 0.9404 | 0.9371 | 59,234  | 67,744  | 1.14 |
| AS      | 0.9221    | 0.9260 | 0.9240 | 27,170  | 197,681 | 7.28 |
| **Average** | **0.9266** | **0.9290** | **0.9278** | **56,297** | **622,516** | **13.68** |

#### Transformer

- **Model Parameters**: 1,452,421

| Dataset | Precision | Recall | F1 | Speed (chars/sec) | Total Characters | Processing Time (s) |
|---------|-----------|--------|----|-------------------|------------------|---------------------|
| PKU     | 0.8231    | 0.8492 | 0.8359 | 48,332  | 172,733 | 3.57  |
| MSR     | 0.8690    | 0.8901 | 0.8794 | 26,590  | 184,358 | 6.93  |
| CITYU   | 0.8749    | 0.8997 | 0.8871 | 26,202  | 67,744  | 2.59  |
| AS      | 0.9066    | 0.9284 | 0.9174 | 8,255   | 197,681 | 23.95 |
| **Average** | **0.8684** | **0.8918** | **0.8800** | **27,345** | **622,516** | **37.04** |

---

### External Library

#### THULAC (THU Lexical Analyzer for Chinese)

| Dataset | Precision | Recall | F1 | Speed (chars/sec) | Total Characters | Processing Time (s) |
|---------|-----------|--------|----|-------------------|------------------|---------------------|
| PKU     | 0.9224    | 0.9233 | 0.9228 | 51,272  | 172,733 | 3.37 |
| MSR     | 0.8317    | 0.8764 | 0.8534 | 52,359  | 184,358 | 3.52 |
| CITYU   | 0.7234    | 0.7392 | 0.7312 | 63,979  | 67,744  | 1.06 |
| AS      | 0.7320    | 0.7446 | 0.7383 | 60,830  | 197,681 | 3.25 |
| **Average** | **0.8024** | **0.8209** | **0.8114** | **57,110** | **622,516** | **11.20** |

---

## Key Findings

### 🏆 Best Performers

| Metric | Algorithm | Score |
|--------|-----------|-------|
| **Best F1 Score** | LSTM | 0.9278 |
| **Best Precision** | LSTM | 0.9266 |
| **Best Recall** | LSTM | 0.9290 |
| **Fastest Speed** | FMM | 365,858 chars/sec |

### 📊 Performance Ranking (by F1 Score)

| Rank | Algorithm | F1 Score | Rating |
|------|-----------|----------|--------|
| 1 | **LSTM** | 0.9278 | ⭐⭐⭐⭐⭐ |
| 2 | **Transformer** | 0.8800 | ⭐⭐⭐⭐⭐ |
| 3 | **BiMM** | 0.8739 | ⭐⭐⭐⭐ |
| 4 | **BMM** | 0.8718 | ⭐⭐⭐⭐ |
| 5 | **FMM** | 0.8703 | ⭐⭐⭐⭐ |
| 6 | **RNN** | 0.8230 | ⭐⭐⭐ |
| 7 | **THULAC** | 0.8114 | ⭐⭐⭐ |
| 8 | **HMM** | 0.7848 | ⭐⭐ |

### ⚡ Speed Ranking

| Rank | Algorithm | Speed (chars/sec) |
|------|-----------|-------------------|
| 1 | FMM | 365,858 |
| 2 | BMM | 342,412 |
| 3 | BiMM | 193,371 |
| 4 | HMM | 100,507 |
| 5 | RNN | 91,034 |
| 6 | THULAC | 57,110 |
| 7 | LSTM | 56,297 |
| 8 | Transformer | 27,345 |

---

## 💡 Analysis

### Neural Network Methods
- **LSTM** achieves the highest performance (F1: 0.9278), demonstrating the power of bidirectional context and gating mechanisms for sequence modeling.
- **Transformer** shows strong performance (F1: 0.8800), ranking second overall. Its attention mechanism captures long-range dependencies effectively.
- **RNN** provides moderate performance (F1: 0.8230), limited by its unidirectional nature and vanishing gradient issues.

### Classical Dictionary-Based Methods
- **BiMM/BMM/FMM** show consistent performance (~0.87 F1) with extremely fast inference speeds (up to 365K chars/sec).
- These methods are ~13x faster than Transformer while only ~5% lower in F1 score.
- Excellent choice when speed is critical and accuracy requirements are moderate.

### Statistical Methods
- **HMM** shows the lowest performance (F1: 0.7848), indicating limitations of first-order Markov assumptions for Chinese segmentation.

### External Library
- **THULAC** performs well on PKU dataset (F1: 0.9228) but shows significant variance across datasets.
- Performance drops notably on CITYU and AS datasets, suggesting training data domain mismatch.

---

## 🎯 Recommendations

| Use Case | Recommended Algorithm | Reason |
|----------|----------------------|--------|
| **Production (High Accuracy)** | LSTM | Highest F1 score (0.9278) |
| **Production (Balanced)** | Transformer | Good accuracy + reasonable speed |
| **Real-time Processing** | FMM/BMM | 6-13x faster than neural methods |
| **Resource Constrained** | BiMM | Best accuracy among fast methods |
| **PKU Domain** | THULAC | Excellent PKU-specific performance |
| **Research/Baseline** | HMM | Simple, interpretable model |
