import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

# Initialize Distributed Training
def setup():
    dist.init_process_group("nccl")  # Using NVIDIA NCCL for multi-GPU
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup():
    dist.destroy_process_group()

def main():
    setup()
    
    MODEL_NAME = "xlm-roberta-large"
    
    # Load CSV dataset and split
    dataset = load_dataset("csv", data_files={"train": "train.csv"})["train"]
    split = dataset.train_test_split(test_size=0.1, seed=42)
    train_dataset = split["train"]
    test_dataset = split["test"]

    print(f"Training samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenization function
    def preprocess_function(examples):
        return tokenizer(
            examples["text"], 
            padding="max_length", 
            truncation=True, 
            max_length=512
        )

    # Tokenize datasets
    train_dataset = train_dataset.map(preprocess_function, batched=True)
    test_dataset = test_dataset.map(preprocess_function, batched=True)

    # Set format for PyTorch
    train_dataset = train_dataset.rename_column("binary_classifier", "labels")
    test_dataset = test_dataset.rename_column("binary_classifier", "labels")
    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = model.to(torch.cuda.current_device())

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./xlm-roberta-binary-classifier",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        num_train_epochs=5,
        logging_dir="./logs",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=True,
        optim="adamw_torch",
        report_to="none",
        ddp_find_unused_parameters=False,  # Critical for DDP efficiency
        dataloader_pin_memory=True,       # Optimize data transfer
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Train and save
    trainer.train()
    trainer.save_model("./xlm-roberta-binary-classifier")
    tokenizer.save_pretrained("./xlm-roberta-binary-classifier")

    cleanup()

if __name__ == "__main__":
    main()
