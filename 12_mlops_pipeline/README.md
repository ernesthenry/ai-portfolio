# MLOps Pipeline: Registry & Monitoring

**Business Goal:** Ensure models in production are reliable, versioned, and monitored for degradation.

## Components

### 1. Model Registry (`registry.py`)

Simulates **MLflow**.

- Solves: "Which version of the model is currently running?"
- Actions: Serializes the model object (`.pkl`) and saves metadata/metrics (`.json`) so we can audit performance later.

### 2. Drift Monitoring (`monitor.py`)

Simulates **EvidentlyAI** or **Fiddler**.

- Solves: "Why is the model failing today when it worked yesterday?"
- Logic: Uses **Kolmogorov-Smirnov (KS) Test** to compare the statistical distribution of Training Data vs Live Inference Data.
- Alert: If `p-value < 0.05`, the data has fundamentally changed (Drift), and we need to retrain.
