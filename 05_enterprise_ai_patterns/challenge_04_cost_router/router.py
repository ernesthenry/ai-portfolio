from semantic_router import Route
from semantic_router.encoders import OpenAIEncoder
from semantic_router.layer import RouteLayer

# 1. DEFINE ROUTES
# We define "intents" by giving examples.
chitchat = Route(
    name="chitchat",
    utterances=[
        "Hello", "Hi", "How are you?", "Good morning", "Thanks"
    ],
)

politics = Route(
    name="politics",
    utterances=[
        "Who is the president?", "What do you think about the election?", "government policy"
    ],
)

summarize = Route(
    name="summarize",
    utterances=[
        "Summarize this text", "Give me a TLDR", "What is the main point?"
    ],
)

# 2. INITIALIZE ROUTER LAYER
# We use a fast, cheap encoder to classify the query
encoder = OpenAIEncoder()
rl = RouteLayer(encoder=encoder, routes=[chitchat, politics, summarize])

def run_router_logic(query):
    # 3. DECIDE
    route = rl(query)
    
    print(f"Query: '{query}' -> Route: {route.name}")
    
    # 4. EXECUTE (Mockup)
    if route.name == "chitchat":
        print("ACTION: Return static response (Cost: $0.00)")
    elif route.name == "summarize":
        print("ACTION: Send to Llama-3-70B (Cost: $0.001)")
    elif route.name == "politics":
        print("ACTION: Block request (Safety Layer)")
    else:
        print("ACTION: Send to GPT-4 (Cost: $0.03) - Default Fallback")

if __name__ == "__main__":
    queries = [
        "Hello there!", 
        "Summarize the meeting notes for me.", 
        "Explain quantum physics in detail." # Should hit default
    ]
    
    for q in queries:
        run_router_logic(q)
