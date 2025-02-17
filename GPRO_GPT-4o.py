import os
import torch
from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

# Define OpenAI API Key
import openai
openai.api_key = os.getenv("OPENAI_API_KEY")

# Function to get reward score from GPT-4o
def gpt4o_reward_func(completions, prompts, **kwargs):
    rewards = []
    for prompt, completion in zip(prompts, completions):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an AI evaluator that scores response quality from 1 (poor) to 10 (excellent)."},
                    {"role": "user", "content": f"Evaluate the quality of the following AI-generated response.\n\n"
                                                     f"**Prompt:** {prompt}\n"
                                                     f"**Completion:** {completion}\n\n"
                                                     f"Rate this completion on a scale from 1 to 10 based on accuracy, clarity, and relevance. Just return a number."}
                ],
                temperature=0.0
            )
            score = float(response["choices"][0]["message"]["content"].strip())
        except Exception as e:
            print(f"Error calling GPT-4o: {e}")
            score = 5.0  # Default score if call fails
        
        rewards.append(score)
    return rewards

# Load Mistral Model with QLoRA (4-bit quantization)
model_name = "mistralai/Mistral-7B-v0.1"

# QLoRA 4-bit quantization with bitsandbytes
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",  # Normal Float 4 for better precision
    bnb_4bit_compute_dtype=torch.float16,  # Compute in float16
)

# Load Model with QLoRA
torch_dtype = torch.float16  # Ensure dtype is consistent
model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, torch_dtype=torch_dtype)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load Dataset
dataset = load_dataset("trl-lib/tldr", split="train")

# Define GRPO Configuration with Extended Context Length and vLLM Optimization
training_args = GRPOConfig(
    output_dir="Mistral-GRPO-GPT4o",
    logging_steps=50,  # Log every 50 steps (Adjust if dataset is smaller)
    save_steps=500,  # Save model every 500 steps (Prevents losing progress)
    evaluation_strategy="steps",  # Evaluate periodically instead of just at end
    eval_steps=100,  # Evaluate every 100 steps
    use_vllm=True,  # Enable vLLM for fast inference
    num_generations=8,  # Generate 8 completions per prompt
    max_prompt_length=12288,  # Increased context size
    vllm_max_model_len=12288,  # Ensure vLLM can handle large context sizes
    num_train_epochs=3,  # Set number of training epochs
    learning_rate=2e-6,  # Lower learning rate for stability
    optim="adamw_torch",  # AdamW optimizer
    weight_decay=0.01,  # Prevents overfitting
    lr_scheduler_type="cosine",  # Cosine decay for smoother learning rate
    warmup_ratio=0.1,  # 10% warm-up
    per_device_train_batch_size=8,  # Train batch size (Adjust based on GPU)
    per_device_eval_batch_size=4,  # Eval batch size (Smaller to fit memory)
    gradient_accumulation_steps=4,  # Accumulates gradients to simulate larger batch size
    logging_dir="./logs",  # Set directory for logs
    report_to="wandb",  # Enables Weights & Biases logging (optional)
)

# Initialize GRPO Trainer
trainer = GRPOTrainer(
    model=model,
    reward_funcs=gpt4o_reward_func,  # Using GPT-4o as reward model
    args=training_args,
    train_dataset=dataset,
    tokenizer=tokenizer
)

# Start Training
trainer.train()
