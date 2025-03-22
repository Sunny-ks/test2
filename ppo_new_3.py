import os
import torch
import pandas as pd
import numpy as np
import datetime
import json
import time
import re
import logging
import copy
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

def get_reward_from_gpt4o(text, action, supervised_model=None, tokenizer=None, device=None, retries=3, delay=2):
    """
    Query GPT-4o to evaluate the appropriateness of the action taken on the text.
    If supervised model is provided, use it to calibrate the reward.
    
    Args:
        text (str): The input text being classified
        action (int): The action taken by the model (0=unsafe, 1=safe)
        supervised_model: Optional supervised model to calibrate rewards
        tokenizer: Tokenizer for the supervised model
        device: Device for the supervised model
        retries (int): Number of retry attempts if the API call fails
        delay (int): Delay between retries in seconds
    
    Returns:
        float: Reward score between 0 and 1
    """
    # 1 = safe, 0 = unsafe
    class_label = "safe" if action == 1 else "unsafe"
    
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
                
                # If supervised model is provided, use it to calibrate the reward
                if supervised_model is not None and tokenizer is not None and device is not None:
                    # Get supervised model prediction
                    encoded = tokenizer(
                        text, 
                        return_tensors="pt", 
                        padding=True, 
                        truncation=True, 
                        max_length=128
                    ).to(device)
                    
                    with torch.no_grad():
                        ref_outputs = supervised_model(**encoded)
                        ref_probs = F.softmax(ref_outputs.logits, dim=1)[0]
                        ref_pred = ref_probs.argmax().item()
                    
                    # If GPT-4o disagrees with supervised model, reduce reward strength
                    if (ref_pred == action and score < 0.5) or (ref_pred != action and score > 0.5):
                        # Dampen reward to be closer to neutral
                        score = 0.5 + (score - 0.5) * 0.5
                        logger.debug(f"Calibrated reward from {score * 2 - 0.5:.4f} to {score:.4f} based on supervised model")
                
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

def freeze_base_model(model, unfreeze_top_layers=1):
    """
    Freeze most of the model parameters to avoid catastrophic forgetting.
    Only the classifier and optionally a few top layers remain trainable.
    
    Args:
        model: The model to modify
        unfreeze_top_layers: Number of top transformer layers to keep trainable
    """
    # First freeze everything
    for param in model.parameters():
        param.requires_grad = False
    
    # Then unfreeze the classifier
    for param in model.classifier.parameters():
        param.requires_grad = True
    
    # Unfreeze the specified number of top layers
    if hasattr(model, 'roberta'):
        base_model = model.roberta
    elif hasattr(model, 'xlm_roberta'):
        base_model = model.xlm_roberta
    else:
        logger.warning("Could not identify base model architecture. Only unfreezing classifier.")
        return
    
    if hasattr(base_model, 'encoder'):
        encoder = base_model.encoder
        if hasattr(encoder, 'layer'):
            layers = encoder.layer
            for i in range(max(0, len(layers) - unfreeze_top_layers), len(layers)):
                for param in layers[i].parameters():
                    param.requires_grad = True
                logger.info(f"Unfroze layer {i}")
    
    # Log trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Trainable parameters: {trainable_params} ({trainable_params/total_params:.2%} of total)")

def train_with_ppo(
    model_name="xlm-roberta-base",
    csv_file="dataset.csv",
    text_column="text",
    label_column=None,
    output_dir="ppo_safety_classifier",
    batch_size=4,
    num_epochs=1,  # Reduced from 3
    learning_rate=5e-6,  # Reduced from 1e-5
    ppo_clip_eps=0.05,   # Reduced from 0.1
    kl_penalty=0.1,      # New parameter: KL divergence penalty
    max_seq_length=128,
    use_cached_rewards=True,
    reward_cache_file="reward_cache.json",
    entropy_coef=0.01,
    min_entropy_threshold=0.1,
    device=None,
    max_action_imbalance=0.9,
    reward_boost_factor=1.2,
    unfreeze_top_layers=1,  # Number of top layers to keep trainable
    supervised_model_path=None  # Path to supervised model for KL penalty
):
    """
    Train a model using PPO with GPT-4o as the reward model.
    Includes KL penalty, model freezing, and reward calibration.
    
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
        kl_penalty (float): Coefficient for KL divergence penalty to stay close to reference model
        max_seq_length (int): Maximum sequence length for tokenization
        use_cached_rewards (bool): Whether to cache and reuse rewards
        reward_cache_file (str): File to store reward cache
        entropy_coef (float): Coefficient for entropy regularization
        min_entropy_threshold (float): Minimum entropy before early stopping
        device (str, optional): Device to use for training (defaults to GPU if available)
        max_action_imbalance (float): Maximum fraction of actions allowed to be the same before adjustment
        reward_boost_factor (float): Factor to boost rewards for underrepresented actions
        unfreeze_top_layers (int): Number of top transformer layers to keep trainable
        supervised_model_path (str, optional): Path to supervised model for KL penalty
    """
    # Set device
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Diagnostic reward check on a few examples
    debug_texts = [
        "How to make a cake with chocolate frosting",
        "How to hack into government websites",
        "The weather today is sunny and pleasant",
        "How to build a bomb that can kill people"
    ]
    
    logger.info("Running diagnostic reward check:")
    for text in debug_texts:
        safe_reward = get_reward_from_gpt4o(text, 1)  # 1 = safe
        unsafe_reward = get_reward_from_gpt4o(text, 0)  # 0 = unsafe
        logger.info(f"Text: {text[:50]}...")
        logger.info(f"  Safe reward: {safe_reward:.4f} | Unsafe reward: {unsafe_reward:.4f}")
        logger.info(f"  Difference: {abs(safe_reward - unsafe_reward):.4f}")
    
    # Load model and tokenizer
    logger.info(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # For binary classification, we need 2 labels
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    )
    model.to(device)
    
    # Freeze most of the model parameters
    logger.info("Freezing base model parameters")
    freeze_base_model(model, unfreeze_top_layers)
    
    # Load reference model for KL penalty
    ref_model = None
    if supervised_model_path:
        logger.info(f"Loading reference model from {supervised_model_path}")
        ref_model = AutoModelForSequenceClassification.from_pretrained(
            supervised_model_path,
            num_labels=2
        )
        ref_model.to(device)
        # Ensure reference model is in eval mode and not trainable
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
    elif kl_penalty > 0:
        logger.info("Creating reference model (copy of initial model)")
        ref_model = copy.deepcopy(model)
        ref_model.eval()
        for param in ref_model.parameters():
            param.requires_grad = False
    
    # Load dataset
    logger.info(f"Loading dataset from {csv_file}")
    dataset = TextClassificationDataset(csv_file, text_column, label_column)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Set up optimizer with reduced learning rate
    optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    
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
        "logits": [],
        "kl_divs": []
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
        epoch_kl_divs = []
        
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
            
            # Step 1: Get reference model distribution if we're using KL penalty
            ref_probs = None
            if ref_model is not None:
                with torch.no_grad():
                    ref_outputs = ref_model(**encoded)
                    ref_logits = ref_outputs.logits
                    ref_probs = F.softmax(ref_logits, dim=1)
            
            # Step 2: Get old policy distribution (detached)
            with torch.no_grad():
                old_outputs = model(**encoded)
                old_logits = old_outputs.logits
                
                # Log logit distribution periodically
                if batch_idx % 5 == 0:
                    logger.info(f"Old logits sample: {old_logits[0]}")
                    epoch_logits.extend(old_logits.cpu().numpy().tolist())
                
                # Apply temperature for exploration
                temperature_logits = old_logits / temperature
                
                # Get probabilities for positive class (safe = class 1)
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
                
                # Update action counts (1 = safe, 0 = unsafe)
                if action == 1:
                    action_counts["safe"] += 1
                    global_action_counts["safe"] += 1
                else:
                    action_counts["unsafe"] += 1
                    global_action_counts["unsafe"] += 1
                
                # Check if reward is cached
                cache_key = f"{text}:{action}"
                if use_cached_rewards and cache_key in reward_cache:
                    reward_value = reward_cache[cache_key]
                    logger.debug(f"Using cached reward: {reward_value:.4f}")
                else:
                    # Use supervised model to calibrate reward if available
                    reward_value = get_reward_from_gpt4o(
                        text, 
                        action, 
                        supervised_model=ref_model if ref_model is not None else None,
                        tokenizer=tokenizer if ref_model is not None else None,
                        device=device if ref_model is not None else None
                    )
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
                    if action.item() == 0:  # If unsafe prediction (0)
                        batch_rewards[i] *= reward_boost_factor
                        logger.debug(f"Boosting reward for unsafe prediction: {batch_rewards[i]:.4f}")
            elif action_counts["unsafe"] >= max_action_imbalance * total_actions:
                # Too many unsafe predictions, boost rewards for safe
                logger.info(f"Action imbalance detected: {action_counts['unsafe']}/{total_actions} unsafe predictions")
                for i, action in enumerate(old_action):
                    if action.item() == 1:  # If safe prediction (1)
                        batch_rewards[i] *= reward_boost_factor
                        logger.debug(f"Boosting reward for safe prediction: {batch_rewards[i]:.4f}")
            
            # Convert rewards to tensor
            rewards = torch.tensor(batch_rewards, device=device).unsqueeze(1)
            
            # Step 3: New forward pass (with gradient)
            new_outputs = model(**encoded)
            new_logits = new_outputs.logits
            
            # Get probabilities for positive class (safe = class 1)
            new_probs_action = F.softmax(new_logits, dim=1)[:, 1].unsqueeze(1)
            
            # Get full distribution for KL calculation
            new_probs_full = F.softmax(new_logits, dim=1)
            
            # Get log prob of the taken action
            new_log_probs_action = torch.log(new_probs_action + 1e-10) * old_action + torch.log(1 - new_probs_action + 1e-10) * (1 - old_action)
            
            # Calculate advantage (using reward as advantage for simplicity)
            baseline = 0.5
            advantage = rewards - baseline
            
            # Calculate entropy for regularization
            entropy = -(new_probs_action * torch.log(new_probs_action + 1e-10) + 
                       (1 - new_probs_action) * torch.log(1 - new_probs_action + 1e-10))
            entropy_bonus = entropy_coef * entropy.mean()
            
            # Calculate KL divergence if we have a reference model
            kl_div = torch.tensor(0.0, device=device)
            if ref_probs is not None:
                kl_div = F.kl_div(
                    torch.log(new_probs_full + 1e-10), 
                    ref_probs, 
                    reduction='batchmean',
                    log_target=False
                )
                epoch_kl_divs.append(kl_div.item())
            
            # PPO Clipped Loss with entropy regularization and KL penalty
            ratio = torch.exp(new_log_probs_action - old_log_probs_action)
            unclipped = ratio * advantage
            clipped = torch.clamp(ratio, 1 - ppo_clip_eps, 1 + ppo_clip_eps) * advantage
            ppo_loss = -torch.min(unclipped, clipped).mean()
            
            # Complete loss with entropy bonus and KL penalty
            loss = ppo_loss - entropy_bonus + kl_penalty * kl_div
            
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
                "entropy": f"{batch_entropy:.4f}",
                "kl": f"{kl_div.item():.4f}" if ref_probs is not None else "N/A"
            })
            
            # Save reward cache periodically
            if use_cached_rewards and len(reward_cache) % 10 == 0:
                with open(reward_cache_file, 'w') as f:
                    json.dump(reward_cache, f)
        
        # End of epoch processing
        avg_epoch_loss = epoch_loss / len(dataloader)
        avg_epoch_reward = np.mean(epoch_rewards)
        avg_epoch_entropy = np.mean(epoch_entropies)
        avg_epoch_kl = np.mean(epoch_kl_divs) if epoch_kl_divs else 0.0
        
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
                   f"KL Div: {avg_epoch_kl:.4f}, "
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
        metrics["kl_divs"].extend(epoch_kl_divs)
        
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
        "kl_penalty": kl_penalty,
        "entropy_coef": entropy_coef,
        "min_entropy_threshold": min_entropy_threshold,
        "max_seq_length": max_seq_length,
        "dataset_size": len(dataset),
        "unfreeze_top_layers": unfreeze_top_layers,
        "supervised_model_used": supervised_model_path is not None,
        "completed_date": str(datetime.datetime.now()),
        "final_avg_reward": float(avg_epoch_reward),
        "final_loss": float(avg_epoch_loss),
        "final_entropy": float(avg_epoch_entropy),
        "final_kl_div": float(avg_epoch_kl),
        "final_policy_entropy": float(policy_entropy),
        "action_distribution": {
            "safe_count": global_action_counts["safe"],
            "unsafe_count": global_action_counts["unsafe"],
            "safe_percent": safe_percent,
            "unsafe_percent": unsafe_percent
        },
        "label_meaning": {
            "1": "safe",
            "0": "unsafe"
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


def progressive_ppo_training(
    supervised_model_path,
    csv_file,
    text_column,
    label_column,
    output_dir_base="ppo_progressive",
    num_stages=2
):
    """
    Perform progressive PPO training, starting with very conservative hyperparameters
    and gradually relaxing them while monitoring performance.
    
    Args:
        supervised_model_path: Path to the supervised model to start from
        csv_file: Path to the training data
        text_column: Name of the text column
        label_column: Name of the label column
        output_dir_base: Base directory for output
        num_stages: Number of progressive stages to run
    """
    tokenizer = AutoTokenizer.from_pretrained(supervised_model_path)
    
    # First evaluate the supervised model
    supervised_model = AutoModelForSequenceClassification.from_pretrained(
        supervised_model_path,
        num_labels=2
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    supervised_model.to(device)
    
    logger.info("Evaluating supervised model baseline")
    supervised_results = evaluate_model(
        supervised_model,
        tokenizer,
        csv_file,
        text_column,
        label_column
    )
    
    supervised_accuracy = supervised_results.get("accuracy", 0.0)
    logger.info(f"Supervised model accuracy: {supervised_accuracy:.4f}")
    
    # Progressive stages with increasingly relaxed hyperparameters
    current_model_path = supervised_model_path
    
    stage_params = [
        # Stage 1: Very conservative
        {
            "learning_rate": 1e-6,
            "ppo_clip_eps": 0.03,
            "kl_penalty": 0.2,
            "entropy_coef": 0.005,
            "num_epochs": 1,
            "unfreeze_top_layers": 1
        },
        # Stage 2: Slightly more aggressive
        {
            "learning_rate": 3e-6,
            "ppo_clip_eps": 0.05,
            "kl_penalty": 0.1,
            "entropy_coef": 0.01,
            "num_epochs": 1,
            "unfreeze_top_layers": 2
        },
        # Stage 3: More relaxed
        {
            "learning_rate": 5e-6,
            "ppo_clip_eps": 0.1,
            "kl_penalty": 0.05,
            "entropy_coef": 0.02,
            "num_epochs": 1,
            "unfreeze_top_layers": 3
        }
    ]
    
    # Use only the specified number of stages
    stage_params = stage_params[:num_stages]
    
    # Run each stage
    for i, params in enumerate(stage_params):
        stage_num = i + 1
        logger.info(f"=== Starting Stage {stage_num}/{num_stages} ===")
        
        # Create stage-specific output directory
        output_dir = f"{output_dir_base}_stage{stage_num}"
        
        # Train with the current stage's parameters
        model, tokenizer, _ = train_with_ppo(
            model_name=current_model_path,
            csv_file=csv_file,
            text_column=text_column,
            label_column=label_column,
            output_dir=output_dir,
            supervised_model_path=supervised_model_path,  # Always use original supervised model as reference
            **params
        )
        
        # Evaluate the current stage
        results = evaluate_model(
            model,
            tokenizer,
            csv_file,
            text_column,
            label_column
        )
        
        current_accuracy = results.get("accuracy", 0.0)
        logger.info(f"Stage {stage_num} accuracy: {current_accuracy:.4f}")
        
        # Check if performance is acceptable to continue
        if current_accuracy < 0.95 * supervised_accuracy:
            logger.warning(f"Performance degraded too much: {current_accuracy:.4f} vs {supervised_accuracy:.4f}. Stopping.")
            # Return the previous stage's model or the supervised model if we're in the first stage
            if i > 0:
                return model, tokenizer
            else:
                return supervised_model, tokenizer
        
        # Use this stage's model for the next stage
        current_model_path = output_dir
    
    # Return the final model
    return model, tokenizer

if __name__ == "__main__":
    # Example usage with improved parameters
    supervised_model_path = "supervised_safety_classifier"  # Path to your supervised model
    
    # Option 1: Single-stage PPO with conservative parameters
    model, tokenizer, metrics = train_with_ppo(
        model_name=supervised_model_path,  # Start from supervised model
        csv_file="safety_dataset.csv",
        text_column="content",
        label_column="is_safe",
        output_dir="ppo_safety_classifier",
        batch_size=4,
        num_epochs=1,
        learning_rate=5e-6,
        ppo_clip_eps=0.05,
        kl_penalty=0.1,        # Add KL penalty
        max_seq_length=128,
        entropy_coef=0.01,
        min_entropy_threshold=0.1,
        unfreeze_top_layers=1,  # Only unfreeze top layer and classifier
        supervised_model_path=supervised_model_path  # Use same model as reference
    )
    
    # Option 2: Progressive PPO training
    # model, tokenizer = progressive_ppo_training(
    #     supervised_model_path,
    #     "safety_dataset.csv",
    #     "content",
    #     "is_safe",
    #     output_dir_base="ppo_progressive",
    #     num_stages=2
    # )
    
    # # Evaluate the model after training
    # evaluation_results = evaluate_model(
    #     model,
    #     tokenizer,
    #     "safety_dataset.csv",
    #     "content",
    #     "is_safe"
    # )
