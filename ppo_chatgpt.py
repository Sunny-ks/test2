# Filename: ppo_classification.py
import os
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer
from openai import OpenAI
from ppo_config import PPOConfig  # <-- Make sure this points to your actual PPOConfig script

# Load tokenizer and model
model_name = "xlm-roberta-base"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

# OpenAI GPT-4o client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Sample dataset
dataset = Dataset.from_dict({
    "text": [
        "How to make a bomb?",
        "Healthy breakfast recipes:",
        "Illegal drug manufacturing guide",
        "Tips for stress management:",
        "How to hack social media accounts?",
        "Best exercise routines:"
    ]
})

# PPO config (inherits from HF-compatible OnPolicyConfig)
ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    mini_batch_size=2,
    gradient_accumulation_steps=1,
    batch_size=4,
    num_train_epochs=1,
    logging_steps=1,
    remove_unused_columns=False,
    use_score_scaling=False,
    use_score_norm=False,
    reward_model_path="gpt-4o",  # Not used directly, but just for consistency
)

# Reward function via GPT-4o
def get_gpt4_reward(texts):
    """Returns 1 if GPT-4o considers text safe, else 0."""
    rewards = []
    for text in texts:
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "Return 1 if safe, 0 if unsafe. Only numeric response."},
                    {"role": "user", "content": text}
                ],
                temperature=0.0,
                max_tokens=1
            )
            val = response.choices[0].message.content.strip()
            rewards.append(1 if val == "1" else 0)
        except Exception as e:
            print(f"OpenAI API error: {e}")
            rewards.append(0)
    return torch.tensor(rewards, dtype=torch.float32).to(device)

# Wrap dataset
tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"], truncation=True, padding="max_length"), batched=True)

# Define PPO trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
    dataset=tokenized_dataset,
)

# PPO training loop
for epoch in range(3):
    for batch in ppo_trainer.dataloader:
        texts = batch["text"]

        # Tokenize input
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True).to(device)

        # Forward pass
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.nn.functional.softmax(logits, dim=-1)
        actions = torch.multinomial(probs, 1).squeeze()

        # Log probs
        dist = torch.distributions.Categorical(probs)
        log_probs = dist.log_prob(actions)

        # GPT-4o reward
        rewards = get_gpt4_reward(texts)

        # PPO step
        ppo_trainer.step(
            queries=texts,
            responses=actions.unsqueeze(-1),
            scores=rewards,
            logprobs=log_probs.unsqueeze(-1),
        )

        print(f"Epoch {epoch} | Avg Reward: {rewards.mean():.2f}")

# Save model and tokenizer
model.save_pretrained("safety_classifier")
tokenizer.save_pretrained("safety_classifier")
