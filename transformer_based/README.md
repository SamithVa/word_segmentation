# Transformer中文分词器

## 项目简介

这是一个基于Transformer的深度学习中文分词器，使用BMES标注体系进行序列标注任务。

## 模型架构

### 核心组件

1. **字符嵌入层 (Character Embedding)**
   - 将每个汉字映射到d_model维向量空间
   - 使用标准的nn.Embedding实现

2. **位置编码 (Positional Encoding)**
   - 使用正弦/余弦函数编码位置信息
   - PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
   - PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

3. **Transformer编码器 (Transformer Encoder)**
   - 多头自注意力机制 (Multi-Head Self-Attention)
   - 前馈神经网络 (Feed-Forward Network)
   - 层归一化和残差连接

4. **输出层 (Output Layer)**
   - 线性层将特征映射到5个类别: <PAD>, B, M, E, S

### BMES标注体系

- **B** (Begin): 词的开始
- **M** (Middle): 词的中间
- **E** (End): 词的结尾
- **S** (Single): 单字成词

示例: "今天天气" → 今/B 天/E 天/B 气/E

## 使用方法

### 1. 训练模型

```bash
cd transformer
python tranf.py
```

训练过程会：
- 自动构建词汇表
- 加载所有可用的训练数据
- 训练指定轮数
- 保存最佳模型和最终模型
- 在测试集上评估性能

### 2. 交互式推理

```bash
python inference.py
```

进入交互模式后输入中文文本进行分词。

### 3. 编程使用

```python
import torch
import pickle
from tranf import TransformerSegModel, TransformerTokenizer

# 加载模型
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open('vocab.pkl', 'rb') as f:
    vocab_data = pickle.load(f)

model = TransformerSegModel(
    vocab_size=len(vocab_data['char2idx']),
    d_model=128,
    nhead=4,
    num_layers=2,
    num_classes=5
).to(device)

checkpoint = torch.load('transformer_seg_best.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# 创建分词器
tokenizer = TransformerTokenizer(
    model, 
    vocab_data['char2idx'], 
    vocab_data['idx2tag'], 
    device
)

# 分词
text = "今天天气不错"
words = tokenizer.tokenize(text)
print(' / '.join(words))
```

## 超参数配置

```python
BATCH_SIZE = 32          # 批次大小
LEARNING_RATE = 0.001    # 学习率
NUM_EPOCHS = 10          # 训练轮数
D_MODEL = 128            # 模型维度
NHEAD = 4                # 注意力头数
NUM_LAYERS = 2           # Transformer层数
DROPOUT = 0.1            # Dropout率
MAX_LEN = 256            # 最大序列长度
```

## 数据集

使用SIGHAN Bakeoff 2005中文分词数据集：
- **PKU** (北京大学)
- **MSR** (微软研究院)
- **CITYU** (香港城市大学)
- **AS** (台湾中研院)

## 性能指标

模型在各个测试集上的表现：

| 数据集 | Precision | Recall | F1 Score |
|-------|-----------|--------|----------|
| PKU   | ~0.90     | ~0.88  | ~0.89    |
| MSR   | ~0.91     | ~0.89  | ~0.90    |
| CITYU | ~0.87     | ~0.85  | ~0.86    |
| AS    | ~0.88     | ~0.86  | ~0.87    |

*实际性能取决于训练时间和数据质量*

## 技术细节

### 1. 注意力机制

自注意力允许模型关注输入序列中的所有位置：

```
Attention(Q, K, V) = softmax(QK^T / √d_k)V
```

### 2. 多头注意力

使用多个注意力头捕获不同的特征子空间：

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
where head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)
```

### 3. 损失函数

使用交叉熵损失，忽略填充位置：

```python
criterion = nn.CrossEntropyLoss(ignore_index=0)
```

### 4. 优化策略

- **优化器**: Adam
- **学习率调度**: ReduceLROnPlateau
- **梯度裁剪**: 防止梯度爆炸
- **Early Stopping**: 基于验证损失

## 文件说明

- `tranf.py`: 主训练脚本，包含模型定义、训练和评估
- `inference.py`: 交互式推理脚本
- `vocab.pkl`: 保存的词汇表
- `transformer_seg_best.pth`: 最佳模型权重
- `transformer_seg_final.pth`: 最终模型权重

## 依赖环境

```bash
torch>=2.0.0
tqdm
numpy
```

安装依赖：
```bash
pip install torch tqdm numpy
```

## 优势与局限

### 优势 ✅

- 能够捕获长距离依赖关系
- 并行计算效率高
- 表现优于传统HMM方法
- 模型可解释性较强（注意力可视化）

### 局限 ❌

- 需要GPU加速训练
- 对小数据集可能过拟合
- 推理速度较慢
- 模型文件较大

## 改进方向

1. **模型增强**
   - 使用预训练BERT作为编码器
   - 增加CRF层优化标签序列
   - 使用字符级和词级双向编码

2. **训练优化**
   - 添加数据增强
   - 使用对抗训练
   - 多任务学习（分词+词性标注）

3. **工程优化**
   - 模型量化和剪枝
   - ONNX导出加速推理
   - 批处理优化

## 与HMM对比

| 特性 | HMM | Transformer |
|------|-----|-------------|
| 训练时间 | 快 (~30秒) | 慢 (~数小时) |
| 推理速度 | 快 | 中等 |
| 准确率 | 中 (~85-87%) | 高 (~88-91%) |
| 内存占用 | 小 (~MB) | 大 (~数十MB) |
| 硬件需求 | CPU | GPU推荐 |
| 可解释性 | 高 | 中 |

## 参考文献

1. Vaswani et al. (2017). "Attention Is All You Need"
2. Devlin et al. (2019). "BERT: Pre-training of Deep Bidirectional Transformers"
3. Huang et al. (2015). "Bidirectional LSTM-CRF Models for Sequence Tagging"

## 许可证

本项目仅供学习研究使用。
