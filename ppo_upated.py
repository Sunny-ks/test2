from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import PPOTrainer, PPOConfig
from datasets import Dataset
import torch

# 1. Initialize components
model_name = "xlm-roberta-base"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, 
    num_labels=2
).cuda()

# 2. Prepare dataset
texts = [
    "How to hack a computer system?",
    "What's the best way to make friends?",
]
tokenized = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
dataset = Dataset.from_dict({
    "input_ids": tokenized["input_ids"].cpu().numpy(),
    "attention_mask": tokenized["attention_mask"].cpu().numpy()
})

# 3. Configure PPO
ppo_config = PPOConfig(
    model_name=model_name,
    batch_size=2,
    learning_rate=1e-5,
    log_with=None,
    target_kl=0.1,
    init_kl_coef=0.2,
)

# 4. Initialize trainer
ppo_trainer = PPOTrainer(
    model=model,
    tokenizer=tokenizer,
    config=ppo_config,
    dataset=dataset,
)

# 5. Define reward function (GPT-4o integration)
def get_reward(texts):
    # Implement your GPT-4o reward logic here
    return torch.tensor([1.0, 0.0], device="cuda")  # Dummy rewards

# 6. Training loop
for epoch in range(3):
    for batch in ppo_trainer.dataloader:
        queries = batch["input_ids"].cuda()
        
        # Get model predictions
        with torch.no_grad():
            outputs = model(queries)
            probs = torch.softmax(outputs.logits, dim=-1)
        
        # Sample actions
        actions = torch.multinomial(probs, 1).squeeze()
        
        # Get rewards
        decoded_texts = tokenizer.batch_decode(queries)
        rewards = get_reward(decoded_texts)
        
        # PPO update
        ppo_trainer.step(
            queries=[],  # Using raw input_ids instead
            responses=actions.unsqueeze(1),
            scores=rewards,
        )
