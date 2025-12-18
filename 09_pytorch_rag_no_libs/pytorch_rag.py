import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

class PyTorchRAG:
    def __init__(self, device="cpu"):
        self.device = device

        # 1. THE ENCODER (MiniLM)
        # We use a pre-trained Transformer to turn text into tensors (embeddings)
        print("Loading Encoder...")
        self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
        self.encoder = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(self.device)

        # The Knowledge Base (Just a list of strings and a Tensor matrix)
        self.documents = []
        self.doc_embeddings = None

    def _mean_pooling(self, model_output, attention_mask):
        """
        Standard BERT pooling strategy: Average all token vectors to get one sentence vector.
        """
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def embed_text(self, text_list):
        """
        Converts a list of strings to a PyTorch Tensor [Batch_Size, 384]
        """
        encoded_input = self.tokenizer(text_list, padding=True, truncation=True, return_tensors='pt').to(self.device)

        with torch.no_grad():
            model_output = self.encoder(**encoded_input)

        # Pool the outputs into a single vector
        embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings (L2 Norm) so Dot Product equals Cosine Similarity
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings

    def index_documents(self, docs):
        """
        Step 1: Ingest Data
        """
        print(f"Indexing {len(docs)} documents...")
        self.documents = docs
        self.doc_embeddings = self.embed_text(docs) # Shape: [N_Docs, 384]
        print(f"Index shape: {self.doc_embeddings.shape}")

    def retrieve(self, query, k=2):
        """
        Step 2: Vector Search (Matrix Multiplication)
        """
        # A. Embed Query
        query_emb = self.embed_text([query]) # Shape: [1, 384]

        # B. Similarity Search (Dot Product)
        # [1, 384] x [384, N_Docs] = [1, N_Docs]
        scores = torch.matmul(query_emb, self.doc_embeddings.T)

        # C. Top-K Retrieval
        # PyTorch has a built-in topk function
        top_scores, top_indices = torch.topk(scores, k=min(k, len(self.documents)))

        results = []
        for score, idx in zip(top_scores[0], top_indices[0]):
            results.append({
                "score": score.item(),
                "content": self.documents[idx.item()]
            })
        return results

    def generate(self, query):
        """
        Step 3: End-to-End Generation (Mocked for simplicity or connect to HuggingFace)
        """
        retrieved = self.retrieve(query)
        context = "\n".join([f"- {r['content']}" for r in retrieved])

        # In a real PyTorch LLM project, you would pass this to a decoder model.
        # Here we demonstrate the Prompt Construction.
        prompt = f"""
        CONTEXT:
        {context}
        
        QUESTION: {query}
        
        ANSWER (Generated based on Context):
        """
        return prompt, retrieved

# --- EXECUTION ---
if __name__ == "__main__":
    rag = PyTorchRAG()
    
    # 1. Knowledge Base
    kb = [
        "PyTorch is an open-source machine learning library developed by Facebook's AI Research lab.",
        "TensorFlow is a free and open-source software library for machine learning and artificial intelligence.",
        "RAG stands for Retrieval-Augmented Generation.",
        "A Tensor is a multi-dimensional array used in deep learning."
    ]
    
    # 2. Index
    rag.index_documents(kb)
    
    # 3. Query
    query = "What does PyTorch do?"
    prompt, sources = rag.generate(query)
    
    print("\n--- RETRIEVAL RESULTS ---")
    for s in sources:
        print(f"[{s['score']:.4f}] {s['content']}")
        
    print("\n--- FINAL PROMPT TO LLM ---")
    print(prompt)
