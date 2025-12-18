import time
import timeit
import asyncio
import numpy as np

# SIMULATED LLM ENDPOINT
# In real life, this would call OpenAI/vLLM/TGI
async def mock_llm_generation(tokens_to_gen: int):
    # Simulate: 30ms Time To First Token (TTFT), then 10ms per token
    await asyncio.sleep(0.03)
    for _ in range(tokens_to_gen):
        await asyncio.sleep(0.01)
    return "token " * tokens_to_gen

async def benchmark_request(request_id: str, tokens: int):
    start = time.perf_counter()
    await mock_llm_generation(tokens)
    end = time.perf_counter()
    latency = end - start
    return latency, tokens

async def run_load_test(concurrency=10, total_requests=50):
    print(f"--- Starting Load Test (Concurrency: {concurrency}) ---")
    
    tasks = []
    for i in range(total_requests):
        # Randomize output length to simulate real traffic
        output_len = np.random.randint(50, 500)
        tasks.append(benchmark_request(f"req_{i}", output_len))
    
    # Run in batches of 'concurrency'
    results = []
    for i in range(0, total_requests, concurrency):
        batch = tasks[i : i + concurrency]
        batch_results = await asyncio.gather(*batch)
        results.extend(batch_results)
        print(f"Processed batch {i // concurrency + 1}...")

    # METRICS CALCULATION
    latencies = [r[0] for r in results]
    total_tokens = sum([r[1] for r in results])
    total_time = sum(latencies)
    
    p50 = np.percentile(latencies, 50)
    p95 = np.percentile(latencies, 95)
    p99 = np.percentile(latencies, 99)
    
    print("\n--- BENCHMARK REPORT ---")
    print(f"Total Requests: {total_requests}")
    print(f"Total Tokens Gen: {total_tokens}")
    print(f"Throughput: {total_tokens / total_time:.2f} tok/sec (Effective)")
    print(f"Latency P50: {p50:.4f}s")
    print(f"Latency P95: {p95:.4f}s (SLO Violation Threshold)")
    print(f"Latency P99: {p99:.4f}s")

if __name__ == "__main__":
    asyncio.run(run_load_test())
