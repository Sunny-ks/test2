import pandas as pd
from sklearn.metrics import hamming_loss, accuracy_score, classification_report, jaccard_score

# Example DataFrame
# df = your_dataframe

# Convert binary columns to list of true labels
def get_true_labels(row):
    return [col for col in ['S', 'H', 'V', 'HR', 'SH', 'S3', 'H2', 'V2'] if row[col] == 1]

df['ground_truth'] = df.apply(get_true_labels, axis=1)

# Convert `response` into list format
df['predicted'] = df['response'].str.split(',')

# One-hot encode ground truth and predictions for metrics
from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
y_true = mlb.fit_transform(df['ground_truth'])
y_pred = mlb.transform(df['predicted'])

# Hamming Loss
hamming = hamming_loss(y_true, y_pred)

# Subset Accuracy
subset_accuracy = accuracy_score(y_true, y_pred)

# Precision, Recall, F1 (Micro and Macro)
from sklearn.metrics import precision_recall_fscore_support

precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
    y_true, y_pred, average='micro'
)

precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
    y_true, y_pred, average='macro'
)

# Jaccard Score
jaccard = jaccard_score(y_true, y_pred, average='samples')


import matplotlib.pyplot as plt
import numpy as np

# Metrics data
metrics = {
    "Hamming Loss": 0.05,
    "Subset Accuracy": 0.82,
    "Precision (Micro)": 0.88,
    "Recall (Micro)": 0.84,
    "F1 Score (Micro)": 0.86,
    "Precision (Macro)": 0.85,
    "Recall (Macro)": 0.80,
    "F1 Score (Macro)": 0.82,
    "Jaccard Index": 0.78
}

# 1. Bar Plot for Overall Metrics
plt.figure(figsize=(10, 6))
plt.barh(list(metrics.keys()), list(metrics.values()), color='skyblue', edgecolor='black')
plt.xlabel('Metric Value')
plt.title('Overall Evaluation Metrics for Multilabel Classification')
plt.xlim(0, 1)
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 2. Micro vs Macro Comparison Bar Plot
labels = ['Precision', 'Recall', 'F1 Score']
micro_scores = [metrics["Precision (Micro)"], metrics["Recall (Micro)"], metrics["F1 Score (Micro)"]]
macro_scores = [metrics["Precision (Macro)"], metrics["Recall (Macro)"], metrics["F1 Score (Macro)"]]

x = np.arange(len(labels))  # Label positions

plt.figure(figsize=(8, 6))
plt.bar(x - 0.2, micro_scores, 0.4, label='Micro', color='lightblue', edgecolor='black')
plt.bar(x + 0.2, macro_scores, 0.4, label='Macro', color='orange', edgecolor='black')
plt.xticks(x, labels)
plt.ylabel('Score')
plt.ylim(0, 1)
plt.title('Micro vs Macro Scores for Precision, Recall, and F1 Score')
plt.legend()
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

