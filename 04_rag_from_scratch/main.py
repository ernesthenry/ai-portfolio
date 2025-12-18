import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from vector_db import VectorDB

class RAGApplication:
    def __init__(self, db_path):
        # 1. Initialize Vector DB (The Long-Term Memory)
        self.db = VectorDB(db_path)

        # 2. Initialize LLM (The Reasoning Brain)
        # Using a tiny model (Gemma-2b or TinyLlama) so it runs on most machines.
        model_name = "google/gemma-2b-it" 
        print(f"Loading Generative Model: {model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.llm = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )

    def generate_answer(self, query):
        # --- STEP 1: RETRIEVAL ---
        print(f"\nScanning Database for: '{query}'...")
        results = self.db.search(query, top_k=3)
        
        # Combine the retrieved chunks into a single context block
        context_block = "\n".join([f"- {r['text']}" for r in results])
        print(f"Retrieved Context:\n{context_block}\n")
        
        # --- STEP 2: AUGMENTATION ---
        # Construct the RAG Prompt
        prompt = f"""
        You are a helpful AI assistant. Use the following Context to answer the User's Question.
        If the answer is not in the context, say "I don't know".
        
        Context:
        {context_block}
        
        User Question: {query}
        
        Answer:
        """
        
        # --- STEP 3: GENERATION ---
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.llm.device)
        
        with torch.no_grad():
            outputs = self.llm.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.7,
                do_sample=True
            )
            
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up output to just show the answer (removing the prompt)
        final_answer = response.split("Answer:")[-1].strip()
        return final_answer

if __name__ == "__main__":
    # Initialize
    app = RAGApplication("knowledge_base.txt")
    
    # Run a Query that requires the specific data file
    user_query = "What is the top speed of the Lunaris-X?"
    
    answer = app.generate_answer(user_query)
    
    print("-" * 30)
    print(f"FINAL ANSWER:\n{answer}")
    print("-" * 30)
