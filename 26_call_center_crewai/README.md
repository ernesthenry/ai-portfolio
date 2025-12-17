# CrewAI Call Center: Autonomous Support Team

**Concept:** Simulating a full "Call Center Floor" using AI Agents.

**The Business Problem:**
A single LLM calling "Chat Completion" lacks process. It might promise a refund it isn't authorized to give.
We need a **Chain of Command**: Triage -> Tech Support -> Supervisor Approval.

**The Solution:**
**CrewAI** (Agent Orchestration).

1.  **Triage Agent:** "Listens" (processing STT output) and tags the ticket.
2.  **Support Agent:** Checks the "Policy Knowledge Base" (context).
3.  **QA Agent:** Polishes the script before it is sent to the TTS engine.

**Architecture:**
`User Audio` -> `Whisper STT` -> `CrewAI Kickoff` -> `Agents Collaborate` -> `Final Script` -> `ElevenLabs TTS`.
