from crewai import Agent, Task, Crew, Process
import os

# Simulated "Call" Input
incoming_call_transcript = """
Customer: Hi, I'm calling because my internet has been down for 3 days. I work from home and this is unacceptable.
I've already restarted the router five times. I want a refund for this month and a technician sent out immediately.
My account number is 8842-112.
"""

# 1. DEFINE AGENTS
triage_agent = Agent(
    role='Call Triage Specialist',
    goal='Categorize the call and extract key details (Sentiment, Intent, Account ID).',
    backstory='You are the first line of defense. You listen to the voice transcript and structure the data.',
    verbose=True,
    allow_delegation=False
)

support_agent = Agent(
    role='L2 Technical Support',
    goal='Determine the technical resolution and compensation eligibility.',
    backstory='You are an expert technician. You know that "3 days down" qualifies for a 50% refund.',
    verbose=True,
    allow_delegation=False
)

qa_agent = Agent(
    role='Quality Assurance Manager',
    goal='Ensure the final response is empathetic and professional.',
    backstory='You review all agent responses to ensure 5-star customer satisfaction.',
    verbose=True,
    allow_delegation=False
)

# 2. DEFINE TASKS
task1 = Task(
    description=f'Analyze this call transcript: "{incoming_call_transcript}". Extract the Issue, Sentiment (0-10 anger score), and Customer Demands.',
    agent=triage_agent
)

task2 = Task(
    description='Based on the Triage analysis, determine the correct action play. If anger > 8, approve the refund. Schedule the technician.',
    agent=support_agent
)

task3 = Task(
    description='Draft the final "Voice Script" that the Text-to-Speech engine will read back to the customer. It must be polite but concise.',
    agent=qa_agent
)

# 3. RUN CREW
crew = Crew(
    agents=[triage_agent, support_agent, qa_agent],
    tasks=[task1, task2, task3],
    process=Process.sequential,
    verbose=2
)

if __name__ == "__main__":
    print("📞 Incoming Call Detected...")
    result = crew.kickoff()
    print("\n\n########################")
    print("## FINAL RESPONSE SCRIPT ##")
    print("########################\n")
    print(result)
