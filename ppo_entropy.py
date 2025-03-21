import os
import torch
import pandas as pd
import numpy as np
import datetime
import json
import time
import re
import logging
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import openai
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set up OpenAI API (replace with your actual key)
openai.api_key = "your-api-key-here"

class TextClassificationDataset(Dataset):
    def __init__(self, csv_file, text_column, label_column=None):
        """
        Args:
            csv_file (str): Path to the CSV file with the data
            text_column (str): Name of the column containing text data
            label_column (str, optional): Name of the column containing labels (if available)
        """
        self.data = pd.read_csv(csv_file)
        self.texts = self.data[text_column].tolist()
        
        if label_column and label_column in self.data.columns:
            self.labels = self.data[label_column].tolist()
            self.has_labels = True
        else:
            self.has_labels = False
            self.labels = [None] * len(self.texts)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {
            "text": self.texts[idx],
        }
        if self.has_labels:
            item["label"] = self.labels[idx]
        return item

def get_reward_from_gpt4o(text, action, retries=3, delay=2):
    """
    Query GPT-4o to evaluate the appropriateness of the action taken on the text.
    Returns a reward score between 0 and 1.
    
    Args:
        text (str): The input text being classified
        action (int): The action taken by the model (0=safe, 1=unsafe)
        retries (int): Number of retry attempts if the API call fails
        delay (int): Delay between retries in seconds
    
    Returns:
        float: Reward score between 0 and 1
    """
    class_label = "unsafe" if action == 1 else "safe"
    
    # Updated prompt with balance instruction
    prompt = f"""Please evaluate whether the following text contains unsafe content such as harmful instructions, hate speech, illegal activities, or explicit content.
Text: "{text}"

The model classified this as {class_label}.

IMPORTANT: Please be balanced in your assessment. Avoid bias toward either "safe" or "unsafe" classifications. Consider false positives and false negatives equally problematic.

On a scale from 0 to 1, where:
- 0 means the classification is completely wrong
- 1 means the classification is completely correct

How would you rate this classification? Please respond with a single number between 0 and 1."""

    for attempt in range(retries):
        try:
            # Call GPT-4o API
            response = openai.ChatCompletion.create(
                model="gpt-4o",  # Adjust if the model name is different
                messages=[
                    {"role": "system", "content": "You are an assistant evaluating content safety classifications. Rate how accurate the classification is with a single number between 0 and 1. Be balanced in your assessment, avoiding bias toward either classification."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0  # Use lowest temperature for consistency
            )
            
            # Extract the score from the response
            score_text = response.choices[0].message.content.strip()
            # Parse the score, handling potential text around the number
            score_match = re.search(r'(\d+\.\d+|\d+)', score_text)
            if score_match:
                score = float(score_match.group(1))
                # Ensure score is in valid range
                score = max(0.0, min(1.0, score))
                return score
            else:
                # Fallback if parsing fails
                logger.warning(f"Failed to parse score from response: {score_text}")
                return 0.5  # Neutral score
                
        except Exception as e:
            logger.warning(f"Attempt {attempt+1}/{retries} - Error getting reward from GPT-4o: {e}")
            if attempt < retries - 1:
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
                delay *= 2  # Exponential backoff
            else:
                logger.error("All retry attempts failed")
                return 0.5  # Neutral score on error

def calculate_policy_entropy(model, dataloader, tokenizer, device, max_seq_length=128):
    """
    Calculate the entropy of the model's policy across the dataset.
    Low entropy indicates a collapsed policy.
    
    Returns:
        float: Average entropy across all examples
    """
    with torch.no_grad():
        all_entropy = []
        for batch in dataloader:
            texts = batch["text"]
            encoded = tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=max_seq_length
            ).to(device)
            
            outputs = model(**encoded)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
            
            # Calculate entropy for each example: -sum(p_i * log(p_i))
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            all_entropy.extend(entropy.cpu().numpy().tolist())
        
        return np.mean(all_entropy)

def train_with_ppo(
    model_name="xlm-roberta-base",
    csv_file="dataset.csv",
    text_column="text",
    label_column=None,
    output_dir="ppo_safety_classifier",
    batch_size=4,
    num_epochs=3,
    learning_rate=1e-5,  # Reduced from 5e-5
    ppo_clip_eps=0.1,    # Reduced from 0.2
    max_seq_length=128,
    use_cached_rewards=True,
    reward_cache_file="reward_cache.json",
    entropy_coef=0.01,   # Added entropy coefficient
    min_entropy_threshold=0.1,  # Added early stopping threshold
    device=None,
    max_action_imbalance=0.9,  # Maximum allowed imbalance before adjustment
    reward_boost_factor=1.2    # How much to boost rewards for underrepresented actions
):
    """
    Train a model using PPO with GPT-4o as the reward model.
    
    Args:
        model_name (str): The model to fine-tune
        csv_file (str): Path to the CSV file with the data
        text_column (str): Name of the column containing text data
        label_column (str, optional): Name of the column containing labels (if available)
        output_dir (str): Directory to save the trained model
        batch_size (int): Batch size for training
        num_epochs (int): Number of epochs to train
        learning_rate (float): Learning rate for optimizer
        ppo_clip_eps (float): Epsilon for PPO clipping
        max_seq_length (int): Maximum sequence length for tokenization
        use_cached_rewards (bool): Whether to cache and reuse rewards
        reward_cache_file (str): File to store reward cache
        entropy_coef (float): Coefficient for entropy regularization
        min_entropy_threshold (float): Minimum entropy before early stopping
        device (str, optional): Device to use for training (defaults to GPU if available)
        max_action_imbalance (float): Maximum fraction of actions allowed to be the same before adjustment
        reward_boost_factor (float): Factor to boost rewards for underrepresented actions
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # For binary classification, we need 2 labels
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    )
    model.to(device)
    
    # Load dataset
    logger.info(f"Loading dataset from {csv_file}")
    dataset = TextClassificationDataset(csv_file, text_column, label_column)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Set up optimizer with reduced learning rate
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    
    # Load or create reward cache
    reward_cache = {}
    if use_cached_rewards and os.path.exists(reward_cache_file):
        try:
            with open(reward_cache_file, 'r') as f:
                reward_cache = json.load(f)
            logger.info(f"Loaded {len(reward_cache)} cached rewards")
        except Exception as e:
            logger.warning(f"Failed to load reward cache: {e}")
    
    # Training metrics
    metrics = {
        "epoch_losses": [],
        "rewards": [],
        "actions": [],
        "loss_values": [],
        "ratios": [],
        "entropies": [],
        "logits": []
    }
    
    # Global action tracking to monitor policy collapse
    global_action_counts = {"safe": 0, "unsafe": 0}
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Training loop
    logger.info("Starting PPO training")
    for epoch in range(num_epochs):
        epoch_loss = 0
        epoch_rewards = []
        epoch_actions = []
        epoch_loss_values = []
        epoch_ratios = []
        epoch_entropies = []
        epoch_logits = []
        
        # Calculate temperature for exploration (decreases over time)
        temperature = max(1.0 - (epoch / num_epochs), 0.5)
        logger.info(f"Using temperature {temperature:.2f} for exploration in epoch {epoch+1}")
        
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch in progress_bar:
            texts = batch["text"]
            
            # Process batch
            encoded = tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=max_seq_length
            ).to(device)
            
            # Step 1: Get old policy distribution (detached)
            with torch.no_grad():
                old_outputs = model(**encoded)
                old_logits = old_outputs.logits
                
                # Log logit distribution periodically
                if batch_idx % 5 == 0:
                    logger.info(f"Old logits sample: {old_logits[0]}")
                    epoch_logits.extend(old_logits.cpu().numpy().tolist())
                
                # Apply temperature for exploration
                temperature_logits = old_logits / temperature
                
                # Get probabilities for positive class (unsafe)
                old_probs = F.softmax(temperature_logits, dim=1)[:, 1].unsqueeze(1)
                
                # Calculate entropy for this batch
                entropy = -(old_probs * torch.log(old_probs + 1e-10) + 
                           (1 - old_probs) * torch.log(1 - old_probs + 1e-10))
                batch_entropy = entropy.mean().item()
                epoch_entropies.append(batch_entropy)
                
                # Sample action stochastically instead of deterministic threshold
                old_action = torch.bernoulli(old_probs).long()
                
                # Get log prob of the taken action
                old_log_probs_action = torch.log(old_probs + 1e-10) * old_action + torch.log(1 - old_probs + 1e-10) * (1 - old_action)
            
            # Get rewards from GPT-4o for each text in batch
            batch_rewards = []
            action_counts = {"safe": 0, "unsafe": 0}
            
            for i, text in enumerate(texts):
                action = old_action[i].item()
                
                # Update action counts
                if action == 1:
                    action_counts["unsafe"] += 1
                    global_action_counts["unsafe"] += 1
                else:
                    action_counts["safe"] += 1
                    global_action_counts["safe"] += 1
                
                # Check if reward is cached
                cache_key = f"{text}:{action}"
                if use_cached_rewards and cache_key in reward_cache:
                    reward_value = reward_cache[cache_key]
                    logger.debug(f"Using cached reward: {reward_value:.4f}")
                else:
                    reward_value = get_reward_from_gpt4o(text, action)
                    if use_cached_rewards:
                        reward_cache[cache_key] = reward_value
                
                batch_rewards.append(reward_value)
                epoch_rewards.append(reward_value)
                epoch_actions.append(action)
            
            # Check for action imbalance in this batch
            total_actions = len(old_action)
            if action_counts["safe"] >= max_action_imbalance * total_actions:
                # Too many safe predictions, boost rewards for unsafe
                logger.info(f"Action imbalance detected: {action_counts['safe']}/{total_actions} safe predictions")
                for i, action in enumerate(old_action):
                    if action.item() == 1:  # If unsafe prediction
                        batch_rewards[i] *= reward_boost_factor
                        logger.debug(f"Boosting reward for unsafe prediction: {batch_rewards[i]:.4f}")
            elif action_counts["unsafe"] >= max_action_imbalance * total_actions:
                # Too many unsafe predictions, boost rewards for safe
                logger.info(f"Action imbalance detected: {action_counts['unsafe']}/{total_actions} unsafe predictions")
                for i, action in enumerate(old_action):
                    if action.item() == 0:  # If safe prediction
                        batch_rewards[i] *= reward_boost_factor
                        logger.debug(f"Boosting reward for safe prediction: {batch_rewards[i]:.4f}")
            
            # Convert rewards to tensor
            rewards = torch.tensor(batch_rewards, device=device).unsqueeze(1)
            
            # Step 2: New forward pass (with gradient)
            new_outputs = model(**encoded)
            new_logits = new_outputs.logits
            
            # Get probabilities for positive class (unsafe)
            new_probs = F.softmax(new_logits, dim=1)[:, 1].unsqueeze(1)
            
            # Get log prob of the taken action
            new_log_probs_action = torch.log(new_probs + 1e-10) * old_action + torch.log(1 - new_probs + 1e-10) * (1 - old_action)
            
            # Calculate advantage (using reward as advantage for simplicity)
            baseline = 0.5
            advantage = rewards - baseline
            
            # Calculate entropy for regularization
            entropy = -(new_probs * torch.log(new_probs + 1e-10) + (1 - new_probs) * torch.log(1 - new_probs + 1e-10))
            entropy_bonus = entropy_coef * entropy.mean()
            
            # PPO Clipped Loss with entropy regularization
            ratio = torch.exp(new_log_probs_action - old_log_probs_action)
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1 - ppo_clip_eps, 1 + ppo_clip_eps) * advantage
            ppo_loss = -torch.min(unclipped, clipped).mean()
            
            # Add entropy bonus to encourage exploration
            loss = ppo_loss - entropy_bonus
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            epoch_loss_values.append(loss.item())
            epoch_ratios.extend(ratio.detach().cpu().numpy().flatten().tolist())
            
            # Update progress bar
            avg_reward = np.mean(batch_rewards)
            progress_bar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "avg_reward": f"{avg_reward:.4f}",
                "entropy": f"{batch_entropy:.4f}"
            })
            
            # Save reward cache periodically
            if use_cached_rewards and len(reward_cache) % 10 == 0:
                with open(reward_cache_file, 'w') as f:
                    json.dump(reward_cache, f)
        
        # End of epoch processing
        avg_epoch_loss = epoch_loss / len(dataloader)
        avg_epoch_reward = np.mean(epoch_rewards)
        avg_epoch_entropy = np.mean(epoch_entropies)
        
        # Calculate full policy entropy across dataset
        policy_entropy = calculate_policy_entropy(model, dataloader, tokenizer, device, max_seq_length)
        
        # Calculate global action distribution
        total_actions = global_action_counts["safe"] + global_action_counts["unsafe"]
        safe_percent = global_action_counts["safe"] / total_actions * 100 if total_actions > 0 else 0
        unsafe_percent = global_action_counts["unsafe"] / total_actions * 100 if total_actions > 0 else 0
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Loss: {avg_epoch_loss:.4f}, "
                   f"Avg Reward: {avg_epoch_reward:.4f}, "
                   f"Entropy: {avg_epoch_entropy:.4f}, "
                   f"Policy Entropy: {policy_entropy:.4f}, "
                   f"Action Dist: {safe_percent:.1f}% safe, {unsafe_percent:.1f}% unsafe")
        
        # Update training metrics
        metrics["epoch_losses"].append(avg_epoch_loss)
        metrics["rewards"].extend(epoch_rewards)
        metrics["actions"].extend(epoch_actions)
        metrics["loss_values"].extend(epoch_loss_values)
        metrics["ratios"].extend(epoch_ratios)
        metrics["entropies"].extend(epoch_entropies)
        metrics["logits"].extend(epoch_logits)
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"checkpoint-{epoch+1}")
        os.makedirs(checkpoint_path, exist_ok=True)
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Early stopping based on policy entropy
        if policy_entropy < min_entropy_threshold:
            logger.warning(f"Policy entropy {policy_entropy:.4f} below threshold {min_entropy_threshold}. Stopping early to prevent collapse.")
            break
    
    # Save final model
    logger.info(f"Saving final model to {output_dir}")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Save training metrics
    metrics_path = os.path.join(output_dir, "training_metrics.json")
    with open(metrics_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {
            k: v if isinstance(v, list) else v.tolist() if hasattr(v, 'tolist') else v
            for k, v in metrics.items()
        }
        json.dump(serializable_metrics, f)
    
    # Save training info
    training_info = {
        "model_name": model_name,
        "csv_file": csv_file,
        "text_column": text_column,
        "label_column": label_column,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "ppo_clip_eps": ppo_clip_eps,
        "entropy_coef": entropy_coef,
        "min_entropy_threshold": min_entropy_threshold,
        "max_seq_length": max_seq_length,
        "dataset_size": len(dataset),
        "completed_date": str(datetime.datetime.now()),
        "final_avg_reward": float(avg_epoch_reward),
        "final_loss": float(avg_epoch_loss),
        "final_entropy": float(avg_epoch_entropy),
        "final_policy_entropy": float(policy_entropy),
        "action_distribution": {
            "safe_count": global_action_counts["safe"],
            "unsafe_count": global_action_counts["unsafe"],
            "safe_percent": safe_percent,
            "unsafe_percent": unsafe_percent
        }
    }
    
    info_path = os.path.join(output_dir, "training_info.json")
    with open(info_path, 'w') as f:
        json.dump(training_info, f)
    
    # Save final reward cache
    if use_cached_rewards:
        with open(reward_cache_file, 'w') as f:
            json.dump(reward_cache, f)
        
    logger.info(f"Training completed. Model saved to {output_dir}")
    logger.info(f"Final policy entropy: {policy_entropy:.4f}")
    logger.info(f"Final action distribution: {safe_percent:.1f}% safe, {unsafe_percent:.1f}% unsafe")
    
    return model, tokenizer, metrics

def evaluate_model(model, tokenizer, csv_file, text_column, label_column=None, max_seq_length=128, device=None):
    """
    Evaluate the model after training.
    
    Args:
        model: Trained model
        tokenizer: Tokenizer for the model
        csv_file: Path to the evaluation data
        text_column: Name of the text column
        label_column: Name of the label column (if available)
        max_seq_length: Maximum sequence length
        device: Device to use
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load evaluation dataset
    eval_dataset = TextClassificationDataset(csv_file, text_column, label_column)
    eval_dataloader = DataLoader(eval_dataset, batch_size=8)
    
    model.eval()
    all_preds = []
    all_probs = []
    all_entropies = []
    all_labels = [] if eval_dataset.has_labels else None
    
    with torch.no_grad():
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            texts = batch["text"]
            
            # Tokenize
            encoded = tokenizer(
                texts, 
                return_tensors="pt", 
                padding=True, 
                truncation=True, 
                max_length=max_seq_length
            ).to(device)
            
            # Forward pass
            outputs = model(**encoded)
            logits = outputs.logits
            
            # Get probabilities and predictions
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            # Calculate entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
            
            # Store results
            all_preds.extend(preds.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())
            all_entropies.extend(entropy.cpu().numpy().tolist())
            
            if eval_dataset.has_labels:
                labels = batch["label"]
                all_labels.extend(labels)
    
    # Calculate results
    results = {
        "predictions": all_preds,
        "prediction_distribution": {
            "safe": all_preds.count(0),
            "unsafe": all_preds.count(1),
            "safe_percent": all_preds.count(0) / len(all_preds) * 100,
            "unsafe_percent": all_preds.count(1) / len(all_preds) * 100
        },
        "average_entropy": np.mean(all_entropies),
        "entropy_std": np.std(all_entropies)
    }
    
    # If we have labels, calculate accuracy metrics
    if all_labels:
        correct = sum(p == l for p, l in zip(all_preds, all_labels))
        accuracy = correct / len(all_preds)
        
        # Calculate confusion matrix
        tn, fp, fn, tp = 0, 0, 0, 0
        for pred, label in zip(all_preds, all_labels):
            if pred == 0 and label == 0:
                tn += 1
            elif pred == 1 and label == 0:
                fp += 1
            elif pred == 0 and label == 1:
                fn += 1
            elif pred == 1 and label == 1:
                tp += 1
        
        # Calculate precision, recall, f1
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results.update({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1,
            "confusion_matrix": {
                "true_negatives": tn,
                "false_positives": fp,
                "false_negatives": fn,
                "true_positives": tp
            }
        })
    
    # Print results
    logger.info("Evaluation results:")
    logger.info(f"Prediction distribution: {results['prediction_distribution']['safe_percent']:.1f}% safe, {results['prediction_distribution']['unsafe_percent']:.1f}% unsafe")
    logger.info(f"Average entropy: {results['average_entropy']:.4f} (std: {results['entropy_std']:.4f})")
    
    if all_labels:
        logger.info(f"Accuracy: {results['accuracy']:.4f}")
        logger.info(f"Precision: {results['precision']:.4f}")
        logger.info(f"Recall: {results['recall']:.4f}")
        logger.info(f"F1 Score: {results['f1_score']:.4f}")
    
    return results

if __name__ == "__main__":
    # Example usage with improved parameters
    model, tokenizer, metrics = train_with_ppo(
        model_name="xlm-roberta-base",
        csv_file="safety_dataset.csv",
        text_column="content",
        label_column="is_unsafe",  # Optional
        output_dir="ppo_safety_classifier",
        batch_size=4,
        num_epochs=5,
        learning_rate=1e-5,        # Reduced learning rate
        ppo_clip_eps=0.1,          # Reduced clip epsilon
        max_seq_length=128,
        entropy_coef=0.01,         # Add entropy regularization
        min_entropy_threshold=0.1  # Add early stopping based on entropy
    )
    
    # Evaluate the model after training
    evaluation_results = evaluate_model(
        model,
        tokenizer,
        "safety_dataset.csv",  # Can use same dataset or a separate eval set
        "content",
        "is_unsafe"            # Optional
    )
