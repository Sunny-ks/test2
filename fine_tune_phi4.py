### pip install transformers peft trl datasets bitsandbytes accelerate deepspeed pandas torch
### torchrun --nproc_per_node=4 fine_tune_phi4.py

import os
import sys
import logging
import pandas as pd
import torch
import transformers
import torch.distributed as dist
from peft import LoraConfig, PeftModel
from trl import SFTTrainer
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig
)

"""
Fine-tuning Phi-4 using:
  - **torchrun for Multi-GPU Training**
  - **QLoRA (4-bit) with 12,288 sequence length**
  - **Distributed Training Across 4 A100 GPUs**
"""

# Initialize distributed training
dist.init_process_group(backend="nccl")

# Get local rank for multi-GPU setup
local_rank = int(os.environ.get("LOCAL_RANK", 0))
device = torch.device(f"cuda:{local_rank}")

logger = logging.getLogger(__name__)

###################
# Hyperparameters
###################
training_config = {
    "bf16": True,
    "do_eval": True,
    "learning_rate": 2e-5,
    "log_level": "info",
    "logging_steps": 20,
    "logging_strategy": "steps",
    "lr_scheduler_type": "cosine",
    "num_train_epochs": 1,
    "max_steps": -1,
    "output_dir": "./checkpoint_dir",
    "overwrite_output_dir": True,
    "per_device_eval_batch_size": 1,
    "per_device_train_batch_size": 1,
    "remove_unused_columns": True,
    "save_steps": 100,
    "save_total_limit": 1,
    "seed": 42,
    "gradient_checkpointing": True,
    "gradient_checkpointing_kwargs": {"use_reentrant": False},
    "gradient_accumulation_steps": 8,
    "warmup_ratio": 0.1,
    "fp16": False,  # Ensure mixed precision is disabled
    "bf16_full_eval": True,
}

#############################
# QLoRA (4-bit Quantization)
#############################
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True
)

peft_config = {
    "r": 64,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "bias": "none",
    "task_type": "CAUSAL_LM",
    "target_modules": "all-linear",
    "modules_to_save": None,
}

train_conf = TrainingArguments(
    **training_config,
    ddp_find_unused_parameters=False,  # Optimize multi-GPU training
    dataloader_num_workers=4,  # Adjust based on system specs
    torch_compile=False,  # Ensure stability
)

###############
# Setup Logging
###############
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log_level = train_conf.get_process_log_level()
logger.setLevel(log_level)
transformers.utils.logging.set_verbosity(log_level)

logger.warning(
    f"Process rank: {dist.get_rank()}, device: {device}, n_gpu: {torch.cuda.device_count()}"
    + f" distributed training: {dist.is_initialized()}, 16-bits training: {train_conf.bf16}"
)
logger.info(f"Training/evaluation parameters {train_conf}")
logger.info(f"PEFT parameters {peft_conf}")

################
# Model Loading
################
checkpoint_path = "microsoft/Phi-4"  # Phi-4 Model
model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    device_map={"": local_rank},  # Assign each instance to a specific GPU
    quantization_config=bnb_config
)

model = AutoModelForCausalLM.from_pretrained(checkpoint_path, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)

# Setting sequence length to 12,288
tokenizer.model_max_length = 12288
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

##################
# Load CSV Dataset
##################
train_csv_file = "train_data.csv"  # Change to your actual train CSV file path
test_csv_file = "test_data.csv"    # Change to your actual test CSV file path

# Load train & test data from CSV
train_df = pd.read_csv(train_csv_file)
test_df = pd.read_csv(test_csv_file)

# Ensure CSV files contain the expected columns
required_columns = {"system", "user", "assistant"}
assert required_columns.issubset(train_df.columns), "Train CSV must contain 'system', 'user', and 'assistant' columns"
assert required_columns.issubset(test_df.columns), "Test CSV must contain 'system', 'user', and 'assistant' columns"

def format_messages(example):
    """
    Formats dataset messages using Phi-4's <|im_start|> format.
    """
    chat_template = (
        f"<|im_start|>system<|im_sep|>{example['system']}<|im_end|>\n"
        f"<|im_start|>user<|im_sep|>{example['user']}<|im_end|>\n"
        f"<|im_start|>assistant<|im_sep|>{example['assistant']}<|im_end|>"
    )
    return {"text": chat_template}

# Apply formatting
train_data = train_df.apply(format_messages, axis=1)
test_data = test_df.apply(format_messages, axis=1)

# Convert to Hugging Face Dataset
train_dataset = Dataset.from_pandas(train_data)
test_dataset = Dataset.from_pandas(test_data)

###########
# Training
###########
trainer = SFTTrainer(
    model=model,
    args=train_conf,
    peft_config=peft_conf,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    max_seq_length=12288,  # Updated to 12,288
    dataset_text_field="text",
    tokenizer=tokenizer,
    packing=True
)

train_result = trainer.train()
metrics = train_result.metrics
trainer.log_metrics("train", metrics)
trainer.save_metrics("train", metrics)
trainer.save_state()

#############
# Evaluation
#############
tokenizer.padding_side = "left"
metrics = trainer.evaluate()
metrics["eval_samples"] = len(test_dataset)
trainer.log_metrics("eval", metrics)
trainer.save_metrics("eval", metrics)

############
# Save model
############
trainer.save_model(train_conf.output_dir)

# Shutdown process group
dist.destroy_process_group()
