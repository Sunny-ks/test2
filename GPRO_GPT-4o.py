
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")  # Set OpenAI API Key

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
                temperature=0.0  # Ensures consistent evaluations
            )
            score = float(response["choices"][0]["message"]["content"].strip())
        except Exception as e:
            print(f"Error calling GPT-4o: {e}")
            score = 5.0  # Assign default score if GPT-4o call fails

        rewards.append(score)
    return rewards


from datasets import load_dataset
from trl import GRPOTrainer, GRPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load Mistral Model
model_name = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Load Dataset
dataset = load_dataset("trl-lib/tldr", split="train")

# Define GRPO Configuration
training_args = GRPOConfig(
    output_dir="Mistral-GRPO-GPT4o", 
    logging_steps=10,
    use_vllm=True,  # Enables faster generation using vLLM
    num_generations=8  # Generates 8 completions per prompt
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
