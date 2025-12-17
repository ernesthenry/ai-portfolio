import operator
from typing import Annotated, List, TypedDict
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

# 1. DEFINE STATE
# This is the "Shared Memory" that passes between agents
class AgentState(TypedDict):
    topic: str
    research_notes: str
    analysis: str
    final_draft: str
    # 'operator.add' means: when a node returns 'messages', append them to the list
    messages: Annotated[List[str], operator.add] 

# 2. DEFINE AGENTS (NODES)
llm = ChatOpenAI(model="gpt-3.5-turbo")

def researcher_agent(state: AgentState):
    print("--- RESEARCHER WORKING ---")
    topic = state['topic']
    # In a real app, this would use a Search Tool
    response = llm.invoke(f"List 3 key factual bullet points about: {topic}")
    return {"research_notes": response.content}

def analyst_agent(state: AgentState):
    print("--- ANALYST WORKING ---")
    notes = state['research_notes']
    response = llm.invoke(f"Analyze these notes for sentiment and trends: {notes}")
    return {"analysis": response.content}

def writer_agent(state: AgentState):
    print("--- WRITER WORKING ---")
    analysis = state['analysis']
    response = llm.invoke(f"Write a short LinkedIn post based on this analysis: {analysis}")
    return {"final_draft": response.content}

# 3. BUILD THE GRAPH (The Workflow)
workflow = StateGraph(AgentState)

# Add nodes
workflow.add_node("Researcher", researcher_agent)
workflow.add_node("Analyst", analyst_agent)
workflow.add_node("Writer", writer_agent)

# Add edges (The flow logic)
workflow.set_entry_point("Researcher")
workflow.add_edge("Researcher", "Analyst")
workflow.add_edge("Analyst", "Writer")
workflow.add_edge("Writer", END)

# Compile
app = workflow.compile()

# 4. RUN
if __name__ == "__main__":
    inputs = {"topic": "The impact of AI on Junior Developers in 2024"}
    result = app.invoke(inputs)
    
    print("\n--- FINAL OUTPUT ---")
    print(result['final_draft'])
