import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import yaml
import os

# Configuration
config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

base_model_name = config["model_name"]
adapter_path = config["new_model_name"] # Where we saved the adapters

def run_inference():
    print(f"Loading base model: {base_model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_name,
        load_in_4bit=True,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    print("Loading LoRA adapters...")
    # This is the magic step: Merging the tiny qualified weights with the frozen base model
    # If the adapter path doesn't exist (because we didn't train yet), this might fail.
    # For demo, we just use the base model.
    try:
        model = PeftModel.from_pretrained(base_model, adapter_path)
    except:
        print("Warning: Adapter not found. Generating with Base Model only.")
        model = base_model

    tokenizer = AutoTokenizer.from_pretrained(base_model_name)

    # Inference Loop
    def generate_response(prompt):
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7
            )

        print(f"\nPROMPT: {prompt}")
        print(f"RESPONSE: {tokenizer.decode(outputs[0], skip_special_tokens=True)}")

    prompt = "Human: Explain quantum physics to a 5 year old. Assistant:"
    generate_response(prompt)

if __name__ == "__main__":
    run_inference()
