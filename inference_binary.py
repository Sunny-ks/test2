from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# Configuration
MODEL_PATH = "./xlm-roberta-binary-classifier"  # Path to your saved model
MAX_LENGTH = 512  # Should match training setup

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

def predict(text):
    # Tokenize input
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    ).to(device)
    
    # Make prediction
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    return probs.cpu().numpy()

# Example usage
if __name__ == "__main__":
    example_texts = [
        "This is a positive example text",
        "This is something negative to classify",
    ]
    
    probabilities = predict(example_texts)
    
    for text, probs in zip(example_texts, probabilities):
        print(f"Text: {text}")
        print(f"Probabilities: [Class 0: {probs[0]:.4f}, Class 1: {probs[1]:.4f}]")
        print(f"Predicted class: {torch.argmax(torch.tensor(probs)).item()}")
        print("-" * 50)
