# LLM Benchmarking & Optimization



## The Goal

In production, "It works" isn't enough. We need to know:

1.  **Latency:** P95 and P99 metrics (Tail Latency).
2.  **Throughput:** Tokens per second (TPS).
3.  **Cost:** Token usage vs GPU hours.

## The Script (`benchmark.py`)

A comprehensive async load tester that:

- Simulates high-concurrency traffic.
- Calculates P50/P95/P99 latency distribution.
- Measures System Throughput (essential for capacity planning).
