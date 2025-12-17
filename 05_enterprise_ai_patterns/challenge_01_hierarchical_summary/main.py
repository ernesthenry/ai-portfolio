import os
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI

# 1. SETUP
# We use a cheap model (gpt-3.5 or gpt-4o-mini) for the map step to save money
llm = ChatOpenAI(temperature=0, model_name="gpt-4o-mini")

def hierarchical_summarization(file_path):
    print(f"--- Processing {file_path} ---")
    
    # 2. LOAD & CHUNK
    # We pretend to load a long narrative file
    # For demo purposes, create a dummy file if it doesn't exist
    if not os.path.exists(file_path):
        with open(file_path, "w") as f:
            f.write("Chapter 1: The beginning was dark... " * 500)
            f.write("Chapter 10: The ending was bright... " * 500)

    loader = TextLoader(file_path)
    docs = loader.load()

    # Split into large chunks (e.g., 2000 tokens) to fit in context
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=100)
    split_docs = text_splitter.split_documents(docs)
    print(f"Split into {len(split_docs)} chunks.")

    # 3. MAP REDUCE CHAIN
    # 'map_reduce' strategy:
    # A. Runs a summary prompt on every chunk independently (Map)
    # B. Combines those summaries into one final prompt (Reduce)
    chain = load_summarize_chain(
        llm, 
        chain_type="map_reduce",
        verbose=True # Set to True to see the thought process in console
    )

    # 4. EXECUTE
    final_summary = chain.run(split_docs)
    
    print("\n--- FINAL HIERARCHICAL SUMMARY ---")
    print(final_summary)

if __name__ == "__main__":
    hierarchical_summarization("long_narrative.txt")
