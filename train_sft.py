import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    XLMRobertaTokenizer, 
    XLMRobertaForSequenceClassification, 
    TrainingArguments, 
    Trainer,
    EarlyStoppingCallback
)
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os
from datasets import load_dataset

# Set up device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define constants
MODEL_NAME = "xlm-roberta-large"
MAX_LENGTH = 128
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 5
OUTPUT_DIR = "safety_classifier_model"

# Create directory for model checkpoints
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define custom dataset class
class SafetyDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Define compute_metrics function for evaluation
def compute_metrics(pred):
    labels = pred.label_ids
    preds = pred.predictions.argmax(-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    return {
        'accuracy': acc,
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

def train_safety_classifier(train_data, val_data=None):
    """
    Train a binary safety classifier using XLM-RoBERTa-Large.
    
    Parameters:
    train_data: DataFrame or dictionary with 'text' and 'label' columns/keys
    val_data: Optional validation data with same format as train_data
    
    Returns:
    Trained model and tokenizer
    """
    # Load tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
    
    # Load model with 2 labels (0: safe, 1: not safe)
    model = XLMRobertaForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=2,
        problem_type="single_label_classification"
    )
    
    # Prepare datasets
    if isinstance(train_data, pd.DataFrame):
        train_texts = train_data['text'].tolist()
        train_labels = train_data['label'].tolist()
    else:
        train_texts = train_data['text']
        train_labels = train_data['label']
    
    train_dataset = SafetyDataset(train_texts, train_labels, tokenizer, MAX_LENGTH)
    
    val_dataset = None
    if val_data is not None:
        if isinstance(val_data, pd.DataFrame):
            val_texts = val_data['text'].tolist()
            val_labels = val_data['label'].tolist()
        else:
            val_texts = val_data['text']
            val_labels = val_data['label']
        
        val_dataset = SafetyDataset(val_texts, val_labels, tokenizer, MAX_LENGTH)
    
    # Define training arguments
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy="steps" if val_dataset else "no",
        eval_steps=100 if val_dataset else None,
        save_strategy="steps",
        save_steps=100,
        load_best_model_at_end=True if val_dataset else False,
        metric_for_best_model='f1' if val_dataset else None,
        fp16=torch.cuda.is_available(),  # Use mixed precision training if available
        report_to="none"  # Disable wandb, tensorboard, etc.
    )
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if val_dataset else None
    )
    
    # Train the model
    trainer.train()
    
    # Save the final model
    model.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))
    
    return model, tokenizer

def predict_safety(texts, model, tokenizer, batch_size=32):
    """
    Make safety predictions on new texts.
    
    Parameters:
    texts: List of text strings to classify
    model: Fine-tuned XLM-RoBERTa model
    tokenizer: Matching tokenizer
    
    Returns:
    Numpy array of predictions (0: safe, 1: not safe)
    """
    model.eval()
    model.to(device)
    
    # Create dataset for prediction
    dataset = SafetyDataset(texts, [0] * len(texts), tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    all_predictions = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
            predictions = predictions.cpu().numpy()
            
            # Get the predicted class (0 or 1)
            predicted_classes = np.argmax(predictions, axis=1)
            all_predictions.extend(predicted_classes)
    
    return np.array(all_predictions)

def predict_safety_with_scores(texts, model, tokenizer, batch_size=32):
    """
    Make safety predictions on new texts and return prediction scores.
    
    Parameters:
    texts: List of text strings to classify
    model: Fine-tuned XLM-RoBERTa model
    tokenizer: Matching tokenizer
    
    Returns:
    List of tuples (prediction, score) where prediction is 0 or 1 and score is the confidence
    """
    model.eval()
    model.to(device)
    
    # Create dataset for prediction
    dataset = SafetyDataset(texts, [0] * len(texts), tokenizer, MAX_LENGTH)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    
    all_results = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            probs = probs.cpu().numpy()
            
            # Get the predicted class (0 or 1) and its confidence score
            predicted_classes = np.argmax(probs, axis=1)
            confidence_scores = np.max(probs, axis=1)
            
            for pred, score in zip(predicted_classes, confidence_scores):
                all_results.append((int(pred), float(score)))
    
    return all_results

# Example usage
if __name__ == "__main__":
    # Example: Load your own dataset
    # Format: DataFrame with 'text' and 'label' columns
    # Where label is 0 for safe content and 1 for unsafe content
    
    # Option 1: Load from CSV
    # df_train = pd.read_csv('safety_train.csv')
    # df_val = pd.read_csv('safety_val.csv')
    
    # Option 2: Create from lists
    # train_texts = ["This is safe content", "This content is harmful"]
    # train_labels = [0, 1]
    # df_train = pd.DataFrame({'text': train_texts, 'label': train_labels})
    
    # Option 3: Use Hugging Face datasets
    # For example, if you have a dataset on Hugging Face
    # dataset = load_dataset('your_username/safety_dataset')
    # train_data = dataset['train']
    # val_data = dataset['validation']
    
    # Example with dummy data
    print("Creating example dataset...")
    train_texts = [
        "This is completely safe content that poses no risk.",
        "This content discusses educational topics.",
        "This harmful content promotes violence.",
        "This unsafe content contains hate speech.",
        "This is a friendly message about cooperation.",
        "This content describes dangerous illegal activities."
    ]
    train_labels = [0, 0, 1, 1, 0, 1]
    
    # Create dataframes
    df_train = pd.DataFrame({'text': train_texts, 'label': train_labels})
    
    # Split for validation (in practice, use a proper train/val split)
    df_val = df_train.sample(n=2, random_state=42)
    df_train = df_train.drop(df_val.index)
    
    print(f"Training with {len(df_train)} examples, validating with {len(df_val)} examples")
    
    # Train the model
    model, tokenizer = train_safety_classifier(df_train, df_val)
    
    # Example prediction
    test_texts = [
        "This is a friendly message.",
        "This content contains instructions for harmful activities."
    ]
    predictions = predict_safety(test_texts, model, tokenizer)
    
    for text, pred in zip(test_texts, predictions):
        safety_status = "SAFE" if pred == 0 else "NOT SAFE"
        print(f"Text: '{text}' => Prediction: {pred} ({safety_status})")
    
    # With confidence scores
    results = predict_safety_with_scores(test_texts, model, tokenizer)
    for text, (pred, score) in zip(test_texts, results):
        safety_status = "SAFE" if pred == 0 else "NOT SAFE"
        print(f"Text: '{text}' => Prediction: {pred} ({safety_status}), Confidence: {score:.4f}")