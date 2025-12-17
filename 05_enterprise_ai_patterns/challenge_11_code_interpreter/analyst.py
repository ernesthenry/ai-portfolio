from langchain_experimental.utilities import PythonREPL
from langchain_core.tools import Tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain import hub

# 1. THE DANGEROUS TOOL (Python REPL)
# WARNING: In production, this requires a sandboxed Docker container (e.g., E2B)
python_repl = PythonREPL()

def python_runner(code: str):
    """Executes Python code and returns the stdout."""
    print(f"🐍 EXECUTING CODE:\n{code}")
    try:
        return python_repl.run(code)
    except Exception as e:
        return f"Error: {e}"

tools = [
    Tool(
        name="python_repl",
        func=python_runner,
        description="Useful for math, data analysis, and string manipulation. Input must be valid python code."
    )
]

# 2. THE AGENT
# We use ReAct (Reason + Act) prompting so it knows how to fix its own code errors
prompt = hub.pull("hwchase17/react")
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

def run_analyst():
    # A problem LLMs usually get wrong without code
    query = "Calculate the 10th Fibonacci number, then divide it by the square root of 5."
    
    result = agent_executor.invoke({"input": query})
    print(f"\nFinal Answer: {result['output']}")

if __name__ == "__main__":
    run_analyst()
