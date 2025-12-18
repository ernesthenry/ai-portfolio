from unsloth import FastLanguageModel, PatchDPOTrainer
from unsloth import is_bfloat16_supported
from trl import DPOTrainer, DPOConfig
from datasets import load_dataset

# CONCEPT: DPO (Direct Preference Optimization)
# It is the modern, stable implementation of RLHF (Reinforcement Learning from Human Feedback).
# Instead of a Reward Model + PPO (complex), we directly feed "Chosen" vs "Rejected" pairs.

def run_dpo_training():
    print("--- Starting DPO Alignment (Simulated) ---")

    # 1. LOAD PRETRAINED MODEL (SFT Phase completed)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = "unsloth/zephyr-sft",
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
    )

    # 2. LOAD PREFERENCE DATASET
    # Format: {prompt, chosen, rejected}
    # "Chosen": The polite/safe answer.
    # "Rejected": The raw/toxic/incorrect answer.
    dataset = load_dataset("intel/orca_dpo_pairs", split="train[:1%]")
    
    # 3. DEFINE DPO TRAINER
    # This aligns the model probability distribution to prefer "Chosen" over "Rejected"
    dpo_trainer = DPOTrainer(
        model = model,
        args = DPOConfig(
            per_device_train_batch_size = 2,
            gradient_accumulation_steps = 4,
            warmup_ratio = 0.1,
            num_train_epochs = 1,
            learning_rate = 5e-6,
            fp16 = not is_bfloat16_supported(),
            bf16 = is_bfloat16_supported(),
            logging_steps = 1,
            output_dir = "dpo_model",
        ),
        beta = 0.1, # The strength of the KL divergence penalty (keeping it close to reference)
        train_dataset = dataset,
        tokenizer = tokenizer,
        max_length = 1024,
        max_prompt_length = 512,
    )

    # 4. TRAINING (Mock for portfolio, requires GPU)
    print("Initializing Trainer... (GPU Required for full run)")
    # dpo_trainer.train()
    print("✅ Training Configuration Ready. This would convert SFT model -> Chat Model aligned with human values.")

if __name__ == "__main__":
    run_dpo_training()
