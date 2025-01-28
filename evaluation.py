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
