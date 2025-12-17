from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# 1. DEFINE TOOLS
# The agent will decide WHEN to use these.

@tool
def calculator(expression: str) -> str:
    """Calculates mathematical expressions. Use this for age differences or stats."""
    try:
        return str(eval(expression))
    except:
        return "Error in calculation"

@tool
def lookup_internal_database(query: str) -> str:
    """Useful for finding details about the narrative or characters inside CoreStory."""
    # Mocking a vector DB lookup
    if "Sarah" in query:
        return "Sarah Connor is 29 years old in this narrative."
    return "No info found."

tools = [calculator, lookup_internal_database]

# 2. SETUP AGENT
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use your tools to answer multi-step questions."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"), # Where the agent does its 'thinking'
])

agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

def run_agent():
    # 3. COMPLEX QUERY
    # Requires: 
    # 1. Lookup Sarah's age (Tool 2)
    # 2. Know current year (General Knowledge) or assume birth year
    # 3. Calculate difference (Tool 1)
    
    query = "Sarah Connor is 29. If she was born in 1984, what year is it in the narrative? Then add 50 to that year."
    
    print(f"Agent thinking on: '{query}'")
    result = agent_executor.invoke({"input": query})
    print(f"\nFinal Answer: {result['output']}")

if __name__ == "__main__":
    run_agent()
