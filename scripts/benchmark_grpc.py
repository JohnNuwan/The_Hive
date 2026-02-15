import time
import asyncio
import json
import statistics
import logging
from uuid import uuid4

# Imports shared
from shared import SwarmGRPCClient
from shared.redis_client import get_redis_client, init_redis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("benchmark")

async def benchmark_redis(iterations: int = 100):
    logger.info(f"🏁 Starting Redis Benchmark ({iterations} iterations)...")
    redis = await init_redis()
    latencies = []
    
    for i in range(iterations):
        start = time.perf_counter()
        await redis.send_to_agent(
            source="benchmark",
            target="nervous",
            action="BENCHMARK",
            payload={"iter": i, "id": str(uuid4())}
        )
        end = time.perf_counter()
        latencies.append((end - start) * 1000) # ms
        
    return latencies

async def benchmark_grpc(iterations: int = 100):
    logger.info(f"🏁 Starting gRPC Benchmark ({iterations} iterations)...")
    client = SwarmGRPCClient()
    if not client.connect():
        logger.error("❌ Failed to connect to gRPC server")
        return []
    
    latencies = []
    for i in range(iterations):
        start = time.perf_counter()
        success = client.send_signal(
            source="benchmark",
            target="nervous",
            action="BENCHMARK",
            payload={"iter": i, "id": str(uuid4())},
            priority=1
        )
        end = time.perf_counter()
        if success:
            latencies.append((end - start) * 1000) # ms
        else:
            logger.warning(f"⚠️ gRPC call {i} failed")
            
    client.close()
    return latencies

def print_stats(name: str, latencies: list):
    if not latencies:
        logger.error(f"No results for {name}")
        return
        
    avg = statistics.mean(latencies)
    med = statistics.median(latencies)
    p99 = statistics.quantiles(latencies, n=100)[98]
    min_val = min(latencies)
    max_val = max(latencies)
    
    print(f"\n📊 --- {name} Results ---")
    print(f"  Count:  {len(latencies)}")
    print(f"  Avg:    {avg:.3f} ms")
    print(f"  Median: {med:.3f} ms")
    print(f"  P99:    {p99:.3f} ms")
    print(f"  Min/Max: {min_val:.3f}/{max_val:.3f} ms")

async def main():
    iterations = 200
    
    redis_results = await benchmark_redis(iterations)
    grpc_results = await benchmark_grpc(iterations)
    
    print_stats("Redis Pub/Sub", redis_results)
    print_stats("gRPC Unary", grpc_results)
    
    if grpc_results and redis_results:
        improvement = (statistics.mean(redis_results) / statistics.mean(grpc_results))
        print(f"\n🚀 Speedup Factor: {improvement:.1f}x faster with gRPC\n")

if __name__ == "__main__":
    asyncio.run(main())
