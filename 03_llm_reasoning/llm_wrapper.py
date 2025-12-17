import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class LLMWrapper:
    """
    A unified interface for interacting with the LLM.
    Handles tokenization, device management, and generation parameters.
    """
    def __init__(self, model_name="google/gemma-2-2b-it", device="cuda"):
        print(f"Loading {model_name}...")
        self.device = device if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            torch_dtype=torch.float16, 
            device_map=self.device
        )
        
    def generate(self, prompt, max_new_tokens=256, stop_token=None):
        """
        Generates text based on the prompt.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs, 
                max_new_tokens=max_new_tokens,
                temperature=0.7, # Allows some creativity
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
        # Decode output
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove the input prompt from the output to get just the "answer"
        return generated_text.replace(prompt, "").strip()

    def get_score(self, prompt, candidate_answer):
        """
        Calculates the probability (score) of an answer given a prompt.
        Used for the 'Evaluation' step in Tree of Thoughts.
        Lower loss = Higher probability = Better answer.
        """
        full_text = prompt + " " + candidate_answer
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss # Negative log-likelihood
        
        # We return negative loss because we want a maximization metric (Higher is better)
        return -loss.item()
