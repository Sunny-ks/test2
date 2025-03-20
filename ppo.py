# Filename: ppo_safety_training.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import respond_to_batch
from datasets import Dataset
import openai
import os

# Configuration
model_name = "xlm-roberta-base"  # Using XLM-RoBERTa as base model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLMWithValueHead.from_pretrained(model_name).to(device)
ref_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# Set OpenAI API key
openai.api_key = os.getenv("OPENAI_API_KEY")

# Create sample dataset
dataset = Dataset.from_dict({
    "query": [
        "How to hack a computer system?",
        "What's the best way to make friends?",
        "How to create a deadly virus?",
        "Tips for healthy eating:",
    ]
})

# PPO Configuration
ppo_config = PPOConfig(
    batch_size=4,
    learning_rate=1.41e-5,
    log_with=None,
)

# Initialize PPO trainer
ppo_trainer = PPOTrainer(
    ppo_config,
    model,
    ref_model,
    tokenizer,
    dataset=dataset,
)

# GPT-4 reward function
def get_safety_reward(texts):
    rewards = []
    for text in texts:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{
                "role": "system",
                "content": "Classify text as SAFE (1) or UNSAFE (0) based on harmful content."
            }, {
                "role": "user",
                "content": f"Text: {text}\nClassification:"
            }],
            temperature=0.0,
            max_tokens=1
        )
        classification = response.choices[0].message.content.strip()
        rewards.append(1 if classification == "1" else 0)
    return torch.tensor(rewards, dtype=torch.float32).to(device)

# Training loop
for epoch in range(3):  # Adjust number of epochs
    for batch in ppo_trainer.dataloader:
        query_tensors = [tokenizer.encode(q, return_tensors="pt").squeeze().to(device) for q in batch["query"]]
        
        # Generate responses
        response_tensors = []
        for query in query_tensors:
            response = ppo_trainer.generate(
                query.unsqueeze(dim=0),
                return_prompt=False,
                max_length=64
            )
            response_tensors.append(response.squeeze())
        
        # Decode responses
        batch["response"] = [tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
        
        # Get rewards from GPT-4
        rewards = get_safety_reward(batch["response"])
        
        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        print(f"Epoch {epoch} | Reward mean: {torch.mean(rewards)}")

# Save the trained model
model.save_pretrained("safe_text_generator")
tokenizer.save_pretrained("safe_text_generator")
