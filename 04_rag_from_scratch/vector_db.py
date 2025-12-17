import numpy as np
from sentence_transformers import SentenceTransformer

class VectorDB:
    """
    A custom Vector Database built from scratch.
    Demonstrates Chunking, Embedding, and Matrix Multiplication (Search).
    """
    def __init__(self, doc_file_path):
        # 1. Initialize the Embedding Model
        # 'all-MiniLM-L6-v2' is efficient and creates 384-dimensional vectors
        print("Loading Embedding Model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.chunk_size = 100 # Characters
        self.overlap = 20     # Characters
        
        # Load and process data
        self.chunks = self._load_and_chunk(doc_file_path)
        self.db_matrix = self._build_index()

    def _load_and_chunk(self, path):
        """
        Reads text and slices it into sliding windows.
        """
        with open(path, 'r') as f:
            text = f.read().replace('\n', ' ')
        
        chunks = []
        # Simple sliding window chunking
        for i in range(0, len(text), self.chunk_size - self.overlap):
            chunk = text[i:i + self.chunk_size]
            if len(chunk) > 50: # Discard tiny chunks (noise)
                chunks.append(chunk)
        
        print(f"Created {len(chunks)} text chunks.")
        return chunks

    def _build_index(self):
        """
        Converts all text chunks into a NumPy matrix.
        Shape: [Num_Chunks, Embedding_Dim]
        """
        print("Vectorizing chunks...")
        embeddings = self.model.encode(self.chunks)
        
        # L2 Normalization: v / ||v||
        # This ensures that the dot product is equivalent to Cosine Similarity.
        # It puts all vectors on the "surface of a hypersphere".
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized_embeddings = embeddings / norms
        
        return normalized_embeddings

    def search(self, query, top_k=3):
        """
        The Search Algorithm:
        1. Embed Query
        2. Dot Product against DB
        3. Sort and Return
        """
        # 1. Encode query
        query_vec = self.model.encode([query])
        query_vec = query_vec / np.linalg.norm(query_vec)
        
        # 2. Similarity (Dot Product)
        # [1, 384] @ [384, N] = [1, N]
        scores = np.dot(query_vec, self.db_matrix.T)[0]
        
        # 3. Get Top K
        # argsort is ascending, so take last K and reverse
        top_indices = np.argsort(scores)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            results.append({
                "score": float(scores[idx]),
                "text": self.chunks[idx]
            })
            
        return results
