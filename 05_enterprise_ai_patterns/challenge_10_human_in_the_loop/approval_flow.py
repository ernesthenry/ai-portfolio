from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver # In prod, use Postgres/Redis
from typing import TypedDict

# 1. STATE
class State(TypedDict):
    draft: str
    feedback: str
    is_approved: bool

# 2. NODES
def drafter(state: State):
    print("🤖 Bot: Drafting content...")
    return {"draft": "CoreStory Q3 revenue is up 200%. We are firing 10% of staff."}

def human_review_node(state: State):
    # This node doesn't generate. It just reflects the state for the human.
    pass 

def publisher(state: State):
    if state.get("is_approved"):
        print(f"🚀 PUBLISHED: {state['draft']}")
    else:
        print("🛑 REJECTED by Human.")
    return {}

# 3. GRAPH WITH INTERRUPT
workflow = StateGraph(State)
workflow.add_node("drafter", drafter)
workflow.add_node("human_review", human_review_node)
workflow.add_node("publisher", publisher)

workflow.set_entry_point("drafter")
workflow.add_edge("drafter", "human_review")
workflow.add_edge("human_review", "publisher")

# MEMORY IS KEY FOR HITL
checkpointer = MemorySaver()
app = workflow.compile(checkpointer=checkpointer, interrupt_before=["publisher"])

# 4. SIMULATE THE FLOW
if __name__ == "__main__":
    thread_config = {"configurable": {"thread_id": "thread-1"}}
    
    # Step 1: Run until the interrupt
    print("--- RUN 1 (Bot drafts) ---")
    app.invoke({"feedback": ""}, config=thread_config)
    
    # ... The system halts here. Imagine 2 hours pass ...
    
    # Step 2: Human Inspects State
    current_state = app.get_state(thread_config)
    print(f"\n👨‍💻 Human sees draft: '{current_state.values['draft']}'")
    
    # Step 3: Human Updates State (Approval)
    print("--- HUMAN APPROVES ---")
    app.update_state(thread_config, {"is_approved": True})
    
    # Step 4: Resume
    print("--- RUN 2 (Resuming) ---")
    for event in app.stream(None, config=thread_config):
        pass
