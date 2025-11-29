"""
Generate model comparison plot using the visualization utilities.
"""

from utils.visualization import plot_model_comparison

# Data from result_comparison.md
models = ['LSTM', 'BMM', 'FMM', 'RNN', 'HMM', 'Transformer']
precision = [0.9324, 0.8674, 0.8659, 0.7979, 0.7772, 0.6685]
recall = [0.9330, 0.8764, 0.8750, 0.8602, 0.7816, 0.7928]
f1_scores = [0.9327, 0.8718, 0.8703, 0.8278, 0.7793, 0.7253]

# Generate the plot
plot_model_comparison(models, precision, recall, f1_scores, 'outputs/model_comparison.png')