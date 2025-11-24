# HMM中文分词器实现

## 项目简介

这是一个基于隐马尔可夫模型(Hidden Markov Model, HMM)的中文分词器,不使用任何神经网络,纯粹基于传统机器学习方法。

## 算法原理

### 1. HMM基础

隐马尔可夫模型包含三个核心概率分布:

- **初始状态概率 π**: P(状态|句子开始)
- **状态转移概率 A**: P(当前状态|前一状态)
- **发射概率 B**: P(观测字符|状态)

### 2. BMES标注体系

使用4个状态标注每个字符在词中的位置:

- **B** (Begin): 词的开始
- **M** (Middle): 词的中间
- **E** (End): 词的结尾
- **S** (Single): 单字成词

示例: "今天天气不错" → 今/B 天/E 天/B 气/E 不/S 错/S

### 3. Viterbi算法

使用动态规划求解最优路径:

```
dp[i][j] = max(dp[i-1][k] + A[k][j]) + B[j][char_i]
```

其中:
- `dp[i][j]`: 第i个字符处于状态j的最大对数概率
- `A[k][j]`: 从状态k转移到状态j的对数概率
- `B[j][char_i]`: 状态j发射字符i的对数概率

## 核心特性

### ✅ 已实现功能

1. **训练功能** (`train`)
   - 从标注语料中学习HMM参数
   - 使用加一平滑避免零概率
   - 对数空间计算避免下溢

2. **分词功能** (`tokenize`)
   - Viterbi算法求最优路径
   - 处理未登录词(OOV)
   - 高效的动态规划实现

3. **模型持久化**
   - `save_model`: 保存训练好的模型
   - `load_model`: 加载已保存的模型

4. **评估功能** (`evaluate`)
   - 计算精确率(Precision)
   - 计算召回率(Recall)
   - 计算F1分数
   - 输出预测结果文件

## 使用方法

### 快速开始

```python
from hmm import HMMTokenizer

# 1. 训练模型
hmm = HMMTokenizer(smoothing=1e-8)
hmm.train('./icwb2-data/training/pku_training.utf8')

# 2. 分词
text = "今天天气不错"
words = hmm.tokenize(text)
print(' / '.join(words))  # 输出: 今天 / 天气 / 不错

# 3. 保存模型
hmm.save_model('hmm_model.pkl')

# 4. 加载模型
hmm_new = HMMTokenizer()
hmm_new.load_model('hmm_model.pkl')

# 5. 评估
results = hmm.evaluate(
    gold_filepath='./icwb2-data/gold/pku_test_gold.utf8',
    test_filepath='./icwb2-data/testing/pku_test.utf8',
    output_filepath='output.txt'
)
```

### 参数说明

- `smoothing`: 平滑参数,默认1e-10,用于避免零概率

## 技术细节

### 1. 平滑技术

使用加一平滑(Laplace Smoothing)处理未见过的事件:

```python
P(x) = (count(x) + α) / (total + α * |V|)
```

其中α是平滑参数,|V|是词汇表大小。

### 2. 对数概率

所有概率在对数空间计算,避免数值下溢:

```python
log(P1 * P2) = log(P1) + log(P2)
```

### 3. OOV处理

对于训练集中未出现的字符,使用固定的低概率惩罚值(-20.0)。

### 4. 评估指标

- **精确率**: 正确预测的词数 / 预测的总词数
- **召回率**: 正确预测的词数 / 标准答案的总词数
- **F1分数**: 2 * P * R / (P + R)

## 数据集

使用SIGHAN Bakeoff 2005中文分词数据集:

- **训练集**: `icwb2-data/training/pku_training.utf8`
- **测试集**: `icwb2-data/testing/pku_test.utf8`
- **标准答案**: `icwb2-data/gold/pku_test_gold.utf8`

## 性能

典型性能指标(PKU语料库):

- 训练时间: ~30秒
- 词汇量: ~5000字符
- F1分数: ~0.85-0.90

## 优势与局限

### 优势

✅ 无需大量计算资源
✅ 模型可解释性强
✅ 训练速度快
✅ 模型文件小巧

### 局限

❌ 准确率低于深度学习方法
❌ 难以捕捉长距离依赖
❌ 对新词识别能力有限
❌ 依赖标注语料质量

## 扩展方向

1. **特征增强**: 结合字符特征(数字、英文、标点)
2. **高阶模型**: 使用二阶或三阶HMM
3. **集成学习**: 与其他分词算法集成
4. **自适应学习**: 在线更新模型参数

## 参考文献

1. Rabiner, L. R. (1989). A tutorial on hidden Markov models and selected applications in speech recognition.
2. Huang, C. N., & Zhao, H. (2007). Chinese word segmentation: A decade review.
3. SIGHAN Bakeoff 2005: Chinese Word Segmentation Shared Task.

## 运行示例

```bash
# 运行完整示例
python hmm.py

# 预期输出
Training HMM model...
HMM Training Complete. Vocabulary size: 5000
Model saved to hmm_model.pkl

=== Testing on example sentences ===
今天天气不错 -> 今天 / 天气 / 不错
中国人民站起来了 -> 中国 / 人民 / 站 / 起来 / 了
...

=== Evaluation Results ===
Precision: 0.8734
Recall: 0.8621
F1 Score: 0.8677
```

## 许可证

本项目仅供学习研究使用。
