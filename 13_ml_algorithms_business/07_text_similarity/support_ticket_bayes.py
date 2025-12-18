from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline
import pandas as pd

# ---------------------------------------------------------
# BUSINESS PROBLEM: SUPPORT TICKET CLASSIFICATION
# ---------------------------------------------------------
# Scenario: Helpdesk is overwhelmed. Need to route tickets automatically.
# Classes: [Billing, Technical, General]
# ---------------------------------------------------------

def run_ticket_routing():
    # Training Data (Mock)
    data = [
        ("My credit card was charged twice", "Billing"),
        ("I cannot login to my account", "Technical"),
        ("Where is the office located?", "General"),
        ("Invoice is incorrect", "Billing"),
        ("Server is down, error 500", "Technical"),
        ("What are your business hours?", "General"),
        ("Payment failed", "Billing"),
        ("App crashes on startup", "Technical"),
        ("Do you have a phone number?", "General")
    ]
    df = pd.DataFrame(data, columns=['text', 'category'])
    
    # ALGORITHM: NAIVE BAYES
    # Why? It works exceptionally well on Text Counts (Bag of Words).
    # It assumes independence between words (Naive) but is very fast and effective.
    model = make_pipeline(CountVectorizer(), MultinomialNB())
    model.fit(df['text'], df['category'])
    
    # New Tickets
    new_tickets = [
        "I need a refund please",
        "The website is very slow",
        "Can I visit your HQ?"
    ]
    
    preds = model.predict(new_tickets)
    
    print("--- Naive Bayes Ticket Router ---")
    for ticket, category in zip(new_tickets, preds):
        print(f"Ticket: '{ticket}' -> Routed to: [{category}]")

if __name__ == "__main__":
    run_ticket_routing()
