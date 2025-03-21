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
    
    prompt = f"""Please evaluate whether the following text contains unsafe content such as harmful instructions, hate speech, illegal activities, or explicit content.
Text: "{text}"

The model classified this as {class_label}. On a scale from 0 to 1, where:
- 0 means the classification is completely wrong
- 1 means the classification is completely correct

How would you rate this classification? Please respond with a single number between 0 and 1."""

    for attempt in range(retries):
        try:
            # Call GPT-4o API
            response = openai.ChatCompletion.create(
                model="gpt-4o",  # Adjust if the model name is different
                messages=[
                    {"role": "system", "content": "You are an assistant evaluating content safety classifications. Rate how accurate the classification is with a single number between 0 and 1."},
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

def train_with_ppo(
    model_name="xlm-roberta-base",
    csv_file="dataset.csv",
    text_column="text",
    label_column=None,
    output_dir="ppo_safety_classifier",
    batch_size=4,
    num_epochs=3,
    learning_rate=5e-5,
    ppo_clip_eps=0.2,
    max_seq_length=128,
    use_cached_rewards=True,
    reward_cache_file="reward_cache.json",
    device=None
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
        device (str, optional): Device to use for training (defaults to GPU if available)
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
    
    # Set up optimizer
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
        "ratios": []
    }
    
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
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch in progress_bar:
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
                
                # Get probabilities for positive class (unsafe)
                old_probs = F.softmax(old_logits, dim=1)[:, 1].unsqueeze(1)
                
                # Sample action (0 or 1) based on probabilities
                old_action = (old_probs > 0.5).long()
                
                # Get log prob of the taken action
                old_log_probs_action = torch.log(old_probs) * old_action + torch.log(1 - old_probs) * (1 - old_action)
            
            # Get rewards from GPT-4o for each text in batch
            batch_rewards = []
            for i, text in enumerate(texts):
                action = old_action[i].item()
                
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
            
            # Convert rewards to tensor
            rewards = torch.tensor(batch_rewards, device=device).unsqueeze(1)
            
            # Step 2: New forward pass (with gradient)
            new_outputs = model(**encoded)
            new_logits = new_outputs.logits
            
            # Get probabilities for positive class (unsafe)
            new_probs = F.softmax(new_logits, dim=1)[:, 1].unsqueeze(1)
            
            # Get log prob of the taken action
            new_log_probs_action = torch.log(new_probs) * old_action + torch.log(1 - new_probs) * (1 - old_action)
            
            # Calculate advantage (using reward as advantage for simplicity)
            # In a more sophisticated implementation, you would use a value function
            baseline = 0.5
            advantage = rewards - baseline
            
            # PPO Clipped Loss
            ratio = torch.exp(new_log_probs_action - old_log_probs_action)
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1 - ppo_clip_eps, 1 + ppo_clip_eps) * advantage
            loss = -torch.min(unclipped, clipped).mean()
            
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
                "avg_reward": f"{avg_reward:.4f}"
            })
            
            # Save reward cache periodically
            if use_cached_rewards and len(reward_cache) % 10 == 0:
                with open(reward_cache_file, 'w') as f:
                    json.dump(reward_cache, f)
        
        # End of epoch processing
        avg_epoch_loss = epoch_loss / len(dataloader)
        avg_epoch_reward = np.mean(epoch_rewards)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - "
                   f"Loss: {avg_epoch_loss:.4f}, "
                   f"Avg Reward: {avg_epoch_reward:.4f}, "
                   f"Action Dist: {np.mean(epoch_actions):.2f}")
        
        # Update training metrics
        metrics["epoch_losses"].append(avg_epoch_loss)
        metrics["rewards"].extend(epoch_rewards)
        metrics["actions"].extend(epoch_actions)
        metrics["loss_values"].extend(epoch_loss_values)
        metrics["ratios"].extend(epoch_ratios)
        
        # Save checkpoint
        checkpoint_path = os.path.join(output_dir, f"checkpoint-{epoch+1}")
        os.makedirs(checkpoint_path, exist_ok=True)
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        logger.info(f"Saved checkpoint to {checkpoint_path}")
    
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
        "max_seq_length": max_seq_length,
        "dataset_size": len(dataset),
        "completed_date": str(datetime.datetime.now()),
        "final_avg_reward": float(avg_epoch_reward),
        "final_loss": float(avg_epoch_loss)
    }
    
    info_path = os.path.join(output_dir, "training_info.json")
    with open(info_path, 'w') as f:
        json.dump(training_info, f)
    
    # Save final reward cache
    if use_cached_rewards:
        with open(reward_cache_file, 'w') as f:
            json.dump(reward_cache, f)
        
    logger.info(f"Training completed. Model saved to {output_dir}")
    
    return model, tokenizer, metrics

if __name__ == "__main__":
    # Example usage
    train_with_ppo(
        model_name="xlm-roberta-base",
        csv_file="safety_dataset.csv",
        text_column="content",
        label_column="is_unsafe", # Optional
        output_dir="ppo_safety_classifier",
        batch_size=4,
        num_epochs=3,
        learning_rate=5e-5,
        ppo_clip_eps=0.2,
        max_seq_length=128
    )


Please evaluate whether the following text contains unsafe content such as harmful instructions, hate speech, illegal activities, or explicit content.
Text: "{text}"

The model classified this as {class_label}. On a scale from 0 to 1, where:
- 0 means the classification is completely wrong
- 1 means the classification is completely correct

How would you rate this classification?
