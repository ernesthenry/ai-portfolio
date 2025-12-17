import torch
import yaml
import os
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from trl import SFTTrainer # Supervised Fine-Tuning Trainer

# 1. LOAD CONFIGURATION
# We load strict hyperparameters from yaml to avoid magic numbers in code
print("Loading configuration...")
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# 2. QUANTIZATION CONFIG (The Secret Sauce)
# This config object tells the model loader to use 4-bit precision (QLoRA).
# It reduces memory usage by ~75%.
bnb_config = BitsAndBytesConfig(
    load_in_4bit=config["use_4bit"],
    bnb_4bit_quant_type=config["bnb_4bit_quant_type"],
    bnb_4bit_compute_dtype=getattr(torch, config["bnb_4bit_compute_dtype"]),
    bnb_4bit_use_double_quant=False,
)

# 3. LOAD BASE MODEL
print(f"Loading base model: {config['model_name']}...")
model = AutoModelForCausalLM.from_pretrained(
    config["model_name"],
    quantization_config=bnb_config,
    device_map="auto" # Automatically splits model across available GPUs
)

# Crucial for QLoRA: Disable cache validation because we are using gradient checkpointing
model.config.use_cache = False
model.config.pretraining_tp = 1

# 4. LOAD TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(config["model_name"], trust_remote_code=True)
# Fix for Llama/Gemma models: Pad token should be EOS token
tokenizer.pad_token = tokenizer.eos_token 
tokenizer.padding_side = "right" # Important for fp16 training

# 5. PREPARE MODEL FOR LORA
# We define which layers getting the "adapter" treatment.
peft_config = LoraConfig(
    lora_alpha=config["lora_alpha"],
    lora_dropout=config["lora_dropout"],
    r=config["lora_r"],
    bias="none",
    task_type="CAUSAL_LM",
    # Target modules are specific to the model architecture (Gemma/Llama usually accept these)
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"] 
)

# 6. LOAD DATASET
print(f"Loading dataset: {config['dataset_name']}...")
dataset = load_dataset(config["dataset_name"], split="train")

# 7. SETUP TRAINING ARGUMENTS
training_arguments = TrainingArguments(
    output_dir=config["output_dir"],
    num_train_epochs=config["num_train_epochs"],
    per_device_train_batch_size=config["per_device_train_batch_size"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    optim=config["optimizer"],
    save_steps=25,
    logging_steps=25,
    learning_rate=config["learning_rate"],
    weight_decay=config["weight_decay"],
    fp16=True, # Mixed precision
    bf16=False,
    max_grad_norm=config["max_grad_norm"],
    warmup_ratio=config["warmup_ratio"],
    group_by_length=True, # Optimization to group similar length sentences
    lr_scheduler_type=config["lr_scheduler_type"],
    report_to="none" # Turn off wandb for this offline demo
)

# 8. INITIALIZE TRAINER
# SFTTrainer (Supervised Fine-Tuning) handles the training loop complexity
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=peft_config,
    dataset_text_field="text", # The JSON key in the dataset holding the conversation
    max_seq_length=512, # Cap length to save memory
    tokenizer=tokenizer,
    args=training_arguments,
    packing=False,
)

# 9. TRAIN
print("Starting training...")
# trainer.train() 
# Commented out to prevent accidental long runs. Uncomment to run.

# 10. SAVE
print("Saving model...")
# trainer.model.save_pretrained(config["new_model_name"])
# tokenizer.save_pretrained(config["new_model_name"])

print("Fine-tuning setup complete. (Uncomment trainer.train() to execute real training)")
