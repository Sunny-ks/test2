import openai
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from datasets import Dataset
from accelerate import Accelerator
import deepspeed
import evaluate
from tqdm import tqdm

##########################################
# 1. Setup & Model Initialization
##########################################

# 🔹 Enable Multi-GPU Training with DeepSpeed ZeRO-2
accelerator = Accelerator()

# 🔹 Load Qwen2.5-14B Model with QLoRA
model_name = "Qwen/Qwen2.5-14B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

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
# 2. Enhanced Reward Function with API Clients
##########################################

class RewardClient:
    def __init__(self, api_key):
        self.client = openai.OpenAI(api_key=api_key)
        
    def evaluate_response(self, prompt, response):
        evaluation_prompt = f"""..."""  # Your prompt template
        
        try:
            completion = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[...],
                max_tokens=4
            )
            return float(completion.choices[0].message.content.strip())
        except Exception as e:
            print(f"API Error: {e}")
            return 0.0

def get_rewards_batch(prompts, responses):
    client1 = RewardClient("your_api_key_1")
    client2 = RewardClient("your_api_key_2")
    
    rewards = []
    for i, (p, r) in enumerate(zip(prompts, responses)):
        if i % 2 == 0:
            rewards.append(client1.evaluate_response(p, r))
        else:
            rewards.append(client2.evaluate_response(p, r))
    return rewards

##########################################
# 3. Dataset with Reference Answers
##########################################

dataset = Dataset.from_dict({
    "prompt": [
        "Explain the significance of reinforcement learning in AI.",
        # ... add all 1000 prompts
    ],
    "reference": [
        "Reinforcement learning is crucial because...",  # Actual reference answers
        # ... corresponding references
    ]
})

##########################################
# 4. PPO Configuration
##########################################

config = PPOConfig(
    model_name=model_name,
    learning_rate=3e-6,
    batch_size=4,
    mini_batch_size=4,
    gradient_accumulation_steps=16,
    optimize_cuda_cache=True,
    accelerator=accelerator,
)

ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=None,
    tokenizer=tokenizer,
    dataset=dataset,
)

##########################################
# 5. Training Loop Fixes
##########################################

def compute_log_probs(model, inputs, response_ids):
    """Compute log probabilities for generated responses only"""
    with torch.no_grad():
        outputs = model(**inputs)
        
    # Slice logits to exclude prompt
    logits = outputs.logits[:, -response_ids.shape[1]-1:-1, :]
    log_probs = F.log_softmax(logits, dim=-1)
    
    # Gather log probabilities for actual generated tokens
    return torch.gather(
        log_probs, 
        dim=-1, 
        index=response_ids.unsqueeze(-1)
    ).squeeze(-1)

num_epochs = 2
for epoch in range(num_epochs):
    for batch in tqdm(ppo_trainer.dataloader):
        # Generate responses
        inputs = tokenizer(batch["prompt"], return_tensors="pt", padding=True, truncation=True).to("cuda")
        
        with torch.inference_mode():
            response_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
            
        # Extract response text
        batch["response"] = tokenizer.batch_decode(
            response_ids[:, inputs["input_ids"].shape[1]:], 
            skip_special_tokens=True
        )
        
        # Compute rewards
        rewards = torch.tensor(get_rewards_batch(
            batch["prompt"], 
            batch["response"]
        )).to("cuda")
        
        # Compute log probs for PPO
        old_log_probs = compute_log_probs(model, inputs, response_ids[:, inputs["input_ids"].shape[1]:])
        
        # PPO Update
        stats = ppo_trainer.step(
            queries=batch["prompt"],
            responses=response_ids,
            scores=rewards,
            response_masks=response_ids != tokenizer.pad_token_id,
            logprobs=old_log_probs
        )
        
        # Evaluation with actual references
        bleu_scores = [
            evaluate.load("bleu").compute(
                predictions=[resp],
                references=[ref]
            )["bleu"] for resp, ref in zip(batch["response"], batch["reference"])
        ]
        
        print(f"Batch BLEU: {sum(bleu_scores)/len(bleu_scores):.2f}")

##########################################
# 6. Safe Model Saving
##########################################

accelerator.wait_for_everyone()
if accelerator.is_main_process:
    model.save_pretrained("fine_tuned_model")
    tokenizer.save_pretrained("fine_tuned_model")
