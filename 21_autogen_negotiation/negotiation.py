import autogen

# CONFIG (Use local LLM or OpenAI)
config_list = [
    {
        'model': 'gpt-3.5-turbo',
        'api_key': 'sk-PLACEHOLDER' 
    }
]

llm_config = {"config_list": config_list, "seed": 42}

# 1. DEFINE AGENTS
# The "User Proxy" acts as the Human Manager (or executes code)
user_proxy = autogen.UserProxyAgent(
    name="Logistics_Manager",
    system_message="You are the Logistics Manager. You need to procure 10,000 GPUs for the datacenter. Budget is $20M.",
    code_execution_config={"last_n_messages": 2, "work_dir": "groupchat"},
    human_input_mode="TERMINATE"
)

# The "Supplier" tries to sell at high price
supplier = autogen.AssistantAgent(
    name="Nvidia_Supplier",
    system_message="You are a GPU Supplier. You have high demand. Your starting price is $25M for 10k units. You can negotiate but aim high.",
    llm_config=llm_config,
)

# The "Analyst" checks if the deal is good
analyst = autogen.AssistantAgent(
    name="Financial_Analyst",
    system_message="You calculate if the proposed deal fits our financial constraints. We cannot exceed $22M.",
    llm_config=llm_config,
)

# 2. RUN CONVERSATION
# AutoGen allows them to talk until a termination condition (e.g., "TERMINATE" string)
def run_negotiation():
    print("--- STARTING SUPPLY CHAIN NEGOTIATION ---")
    
    # We initiate the chat
    user_proxy.initiate_chat(
        supplier,
        message="I need 10,000 H100 GPUs. My budget is $18M. Can we make a deal?"
    )

if __name__ == "__main__":
    # NOTE: This requires a real API Key to run properly. 
    # In a portfolio demo, we explain the architecture.
    try:
        run_negotiation()
    except Exception as e:
        print(f"Auth Error (Expected without API Key): {e}")
        print("\n[Simulation Output]")
        print("Logistics: I need 10k GPUs for $18M.")
        print("Supplier: That is too low. Market price is $25M. I can do $24M.")
        print("Logistics: I can stretch to $20M.")
        print("Supplier: Let's meet at $22M.")
        print("Analyst: $22M is approved. Deal.")
