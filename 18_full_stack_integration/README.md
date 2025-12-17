# Full Stack AI Integration

**Goal:** Prove you can connect your AI models to a real-world frontend.

**The Architecture:**

1.  **Frontend:** HTML/JS (or React/React Native).
2.  **Protocol:** REST API or WebSockets (for streaming).
3.  **Backend:** FastAPI (Python).

**How it works:**
The `index.html` file demonstrates the client-side logic:

- Taking user input.
- Sending it to the backend endpoint.
- Handling the usage of `fetch` or `EventSource` (for SSE) to display text token-by-token.
