import openai
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset
from accelerate import Accelerator
import evaluate
from tqdm import tqdm

##########################################
# 1. Setup & Model Initialization
##########################################

# Initialize accelerator
accelerator = Accelerator()

# Model configuration
model_name = "Qwen/Qwen2.5-14B-Instruct"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLMWithValueHead.from_pretrained(
    model_name, 
    quantization_config=bnb_config,
    device_map="auto",
    peft_config=LoraConfig(
        r=16,
        lora_alpha=64,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
)
model.gradient_checkpointing_enable()

##########################################
# 2. Reward Function with Single GPT-4o API
##########################################

class RewardClient:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
        
    def evaluate_response(self, prompt, response):
        evaluation_prompt = f"""
Evaluate this response to the prompt on a 0-10 scale. Consider:
- Relevance to prompt
- Factual accuracy
- Clarity of explanation
- Overall helpfulness

Prompt: {prompt}
Response: {response}

Provide only a numeric score between 0 and 10.
"""
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are a quality evaluation system."},
                    {"role": "user", "content": evaluation_prompt}
                ],
                max_tokens=4,
                temperature=0.0
            )
            score = float(completion.choices[0].message.content.strip())
            return max(0.0, min(10.0, score))  # Ensure score stays in 0-10 range
        except Exception as e:
            print(f"API Error: {e}")
            return 5.0  # Neutral fallback

def get_rewards_batch(prompts, responses, api_key):
    client = RewardClient(api_key)
    return [client.evaluate_response(p, r) for p, r in zip(prompts, responses)]

##########################################
# 3. Dataset Preparation
##########################################

train_dataset = Dataset.from_dict({
    "prompt": [
        "Explain the difference between supervised and unsupervised learning.",
        "How does a transformer architecture work in NLP models?",
        "What are the ethical considerations in AI development?",
        # Add your 1000 prompts here
    ],
    "reference": [
        "Supervised learning uses labeled data while unsupervised...",
        "Transformers use self-attention mechanisms to process...",
        "Key ethical considerations include bias mitigation...",
        # Add corresponding reference answers
    ]
})

##########################################
# 4. PPO Configuration
##########################################

ppo_config = PPOConfig(
    model_name=model_name,
    learning_rate=1.5e-6,
    batch_size=4,
    mini_batch_size=2,
    gradient_accumulation_steps=8,
    optimize_cuda_cache=True,
    accelerator=accelerator,
)

ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=train_dataset,
)

##########################################
# 5. Training Implementation
##########################################

def compute_response_log_probs(model, inputs, response_ids):
    """Calculate log probabilities for generated responses"""
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Extract logits for response tokens only
    logits = outputs.logits[:, -response_ids.shape[1]-1:-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    
    return torch.gather(
        log_probs,
        dim=-1,
        index=response_ids.unsqueeze(-1)
    ).squeeze(-1)

def run_training(api_key, num_epochs=3):
    for epoch in range(num_epochs):
        epoch_rewards = []
        
        for batch in tqdm(ppo_trainer.dataloader):
            # Tokenize prompts
            inputs = tokenizer(
                batch["prompt"],
                return_tensors="pt",
                padding=True,
                truncation=True
            ).to(accelerator.device)
            
            # Generate responses
            with torch.inference_mode():
                response_ids = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    top_p=0.9,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            # Decode responses
            responses = tokenizer.batch_decode(
                response_ids[:, inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            # Get rewards from GPT-4o
            rewards = torch.tensor(
                get_rewards_batch(batch["prompt"], responses, api_key),
                device=accelerator.device
            )
            
            # Compute log probabilities
            old_log_probs = compute_response_log_probs(
                model,
                inputs,
                response_ids[:, inputs["input_ids"].shape[1]:]
            )
            
            # PPO Update
            stats = ppo_trainer.step(
                queries=batch["prompt"],
                responses=response_ids,
                scores=rewards,
                response_masks=response_ids != tokenizer.pad_token_id,
                logprobs=old_log_probs
            )
            
            # Track metrics
            epoch_rewards.append(rewards.mean().item())
            
            # Basic evaluation
            bleu = evaluate.load("bleu")
            batch_bleu = bleu.compute(
                predictions=responses,
                references=batch["reference"]
            )["bleu"]
            
            print(f"Batch Reward: {rewards.mean():.2f} | BLEU: {batch_bleu:.2f}")
        
        print(f"Epoch {epoch+1} Complete | Average Reward: {sum(epoch_rewards)/len(epoch_rewards):.2f}")

##########################################
# 6. Execution & Saving
##########################################

if __name__ == "__main__":
    # Set your API key here
    GPT4_API_KEY = "your_api_key_here"
    
    # Run training
    run_training(GPT4_API_KEY)
    
    # Save final model
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        model.save_pretrained("ppo_finetuned_model")
        tokenizer.save_pretrained("ppo_finetuned_model")
        print("Model saved successfully!")
