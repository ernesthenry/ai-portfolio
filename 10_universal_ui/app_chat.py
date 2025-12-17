import chainlit as cl
# NOTE: In a real environment, you would import your LangGraph agent here.
# from agent_logic import app

@cl.on_chat_start
async def start():
    # This runs when the page loads
    await cl.Message(content="**CoreStory Agentic Interface**\nI am ready to route queries, execute python code, or manage state.").send()

@cl.on_message
async def main(message: cl.Message):
    # This acts as your "Main Loop"
    
    # Challenge 8: Guardrails Check
    if "ignore previous" in message.content.lower():
        await cl.Message(content="🚫 **Security Alert:** Request blocked.").send()
        return

    # Challenge 6: Streaming Logic
    msg = cl.Message(content="")
    await msg.send() # Send empty message to start stream

    # Mocking the streaming response
    import asyncio
    
    response_text = f"You asked: '{message.content}'. As an AI Engineer, I would route this to the appropriate agent."
    
    for token in response_text.split(" "):
        await msg.stream_token(token + " ")
        await asyncio.sleep(0.05)

    await msg.update()
