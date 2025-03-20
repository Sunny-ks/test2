import os
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, random_split, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from datasets import load_dataset
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
    
    # Load CSV dataset
    dataset = load_dataset("csv", data_files={"full": "train.csv"})["full"]

    # Split dataset into 90% train, 10% test
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    print(f"Training samples: {train_size}, Test samples: {test_size}")

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

    # Tokenize train and test datasets
    train_dataset = train_dataset.map(preprocess_function)
    test_dataset = test_dataset.map(preprocess_function)

    # Convert label column to PyTorch format
    train_dataset = train_dataset.rename_column("binary_classifier", "labels")
    test_dataset = test_dataset.rename_column("binary_classifier", "labels")

    train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

    # Load model and move it to the appropriate GPU
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    model = model.to(torch.cuda.current_device())
    model = DDP(model, device_ids=[torch.cuda.current_device()])

    # Training Arguments
    training_args = TrainingArguments(
        output_dir="./xlm-roberta-binary-classifier",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=16,  # Adjust for memory usage
        per_device_eval_batch_size=16,
        gradient_accumulation_steps=2,  # Increase for large batch sizes
        learning_rate=2e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        num_train_epochs=5,
        logging_dir="./logs",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        fp16=True,  # Mixed precision training
        optim="adamw_torch",
        report_to="none",  # Change to "wandb" if using Weights & Biases
        ddp_find_unused_parameters=False,  # Optimize DDP
    )

    # Use DistributedSampler for multi-GPU efficiency
    train_sampler = DistributedSampler(train_dataset)
    test_sampler = DistributedSampler(test_dataset, shuffle=False)

    train_dataloader = DataLoader(
        train_dataset, batch_size=training_args.per_device_train_batch_size, sampler=train_sampler
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=training_args.per_device_eval_batch_size, sampler=test_sampler
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    metrics = trainer.evaluate()
    print(metrics)

    # Save the trained model
    trainer.save_model("./xlm-roberta-binary-classifier")
    tokenizer.save_pretrained("./xlm-roberta-binary-classifier")

    cleanup()

if __name__ == "__main__":
    main()
