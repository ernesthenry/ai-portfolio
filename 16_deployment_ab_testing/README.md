# Deployment & Validation

**Goal:** The final mile. A model is useless if it isn't deployed and validated.

## 1. Dockerization (`Dockerfile`)

The industry standard for packaging. This file deploys our **FastAPI Streaming Service**.

- Uses `python:slim` for small image size.
- Optimizes caching layers.

## 2. A/B Testing (`ab_test_simulator.py`)

Before replacing a model, we test it.

- Implements **Z-Test for Proportions** from scratch.
- Calculates **P-Value** to ensure the improvement isn't just luck.
- This is the "Data Science" way to make business decisions.
