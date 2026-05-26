"""Session 2 — latency benchmark for /chat/sync vs /chat/stream.

Measures from a real client over loopback so the numbers include FastAPI,
uvicorn, SSE framing, and async scheduling — everything except real-network
latency. Reports P50/P95 for total latency on both endpoints, and TTFT for
the streaming endpoint.

Usage:
    # 1. Start the API
    uv run uvicorn app.main:app --port 8000

    # 2. Run the bench in another shell
    uv run python scripts/bench_latency.py --repeats 3 --rows 0,1,2,3,4

Outputs raw timings to scripts/bench_results.csv so you can re-aggregate
without re-running.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import httpx

REPO_ROOT = Path(__file__).parent.parent
TESTSET = REPO_ROOT / "eval" / "datasets" / "synthetic_testset_raw.csv"
PARAPHRASES = REPO_ROOT / "eval" / "datasets" / "bench_paraphrases.json"
RESULTS = Path(__file__).parent / "bench_results.csv"


async def flush_redis(host: str, port: int) -> None:
    """FLUSHALL between phases so each endpoint sees its own cold-path misses.
    Without this, whichever phase runs second gets 100% cache hits — the cache
    populated by the first phase. That makes the second phase's tail look fake.

    FLUSHALL also drops the L2 vector index (it lives in Redis), so we recreate
    it here. Otherwise the second phase would silently L2-miss on everything.
    """
    sys.path.insert(0, str(REPO_ROOT))
    from services.semantic_cache import get_cache  # noqa: E402
    import redis.asyncio as redis
    client = redis.Redis(host=host, port=port)
    try:
        await client.flushall()
    finally:
        await client.aclose()
    await get_cache().ensure_index()


def load_queries(rows: list[int]) -> list[str]:
    with open(TESTSET) as f:
        all_rows = list(csv.DictReader(f))
    return [all_rows[i]["user_input"] for i in rows]


def load_paraphrase_groups(rows: list[int]) -> list[list[str]]:
    """Each group = [original, paraphrase_1, ..., paraphrase_9] for one base query.
    First call per group is L1+L2 cold-miss (full pipeline + populate L2).
    Subsequent calls are unique text (L1 miss) but semantically close (L2 hit, ideally).
    """
    with open(PARAPHRASES) as f:
        data = json.load(f)
    return [data[str(i)] for i in rows]


async def bench_sync(client: httpx.AsyncClient, query: str) -> dict:
    """Single /chat/sync request. client_total_ms is the source of truth."""
    t0 = time.perf_counter()
    r = await client.post("/chat/sync", json={"query": query}, timeout=60.0)
    client_total_ms = (time.perf_counter() - t0) * 1000
    r.raise_for_status()
    body = r.json()
    return {
        "endpoint": "sync",
        "query": query,
        "client_ttft_ms": client_total_ms,  # no streaming → TTFT == total
        "client_total_ms": client_total_ms,
        "server_ttft_ms": body["total_ms"],
        "server_total_ms": body["total_ms"],
        "retrieval_ms": body["retrieval_ms"],
        "tokens": None,
    }


async def bench_stream(client: httpx.AsyncClient, query: str) -> dict:
    """Single /chat/stream request. Parses SSE, captures client TTFT on the
    first non-empty token frame (matches server convention)."""
    t0 = time.perf_counter()
    client_ttft_ms: float | None = None
    server_ttft_ms: float | None = None
    server_total_ms: float | None = None
    retrieval_ms: float | None = None
    tokens = 0

    async with client.stream(
        "POST", "/chat/stream", json={"query": query}, timeout=60.0
    ) as r:
        r.raise_for_status()
        current_event: str | None = None
        async for line in r.aiter_lines():
            if not line:
                current_event = None
                continue
            if line.startswith("event: "):
                current_event = line[7:].strip()
            elif line.startswith("data: "):
                payload = json.loads(line[6:])
                if current_event == "meta":
                    retrieval_ms = payload.get("retrieval_ms")
                elif current_event == "ttft":
                    if client_ttft_ms is None:
                        client_ttft_ms = (time.perf_counter() - t0) * 1000
                    server_ttft_ms = payload.get("ttft_ms")
                elif current_event == "token":
                    if client_ttft_ms is None:
                        # Defensive: if for some reason ttft event was missed,
                        # fall back to first token frame.
                        client_ttft_ms = (time.perf_counter() - t0) * 1000
                    tokens += 1
                elif current_event == "done":
                    server_total_ms = payload.get("total_ms")
                    server_ttft_ms = payload.get("ttft_ms", server_ttft_ms)

    client_total_ms = (time.perf_counter() - t0) * 1000
    return {
        "endpoint": "stream",
        "query": query,
        "client_ttft_ms": client_ttft_ms,
        "client_total_ms": client_total_ms,
        "server_ttft_ms": server_ttft_ms,
        "server_total_ms": server_total_ms,
        "retrieval_ms": retrieval_ms,
        "tokens": tokens,
    }


def percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


def report(samples: list[dict]) -> None:
    def stat(rows: list[dict], key: str) -> tuple[float, float, float]:
        vals = [r[key] for r in rows if r.get(key) is not None]
        if not vals:
            return float("nan"), float("nan"), float("nan")
        return statistics.median(vals), percentile(vals, 95), statistics.mean(vals)

    sync = [s for s in samples if s["endpoint"] == "sync"]
    stream = [s for s in samples if s["endpoint"] == "stream"]

    print()
    print("=" * 78)
    print(f"{'':<28}{'P50':>14}{'P95':>14}{'mean':>14}{'N':>8}")
    print("-" * 78)

    if sync:
        p50, p95, mean = stat(sync, "client_total_ms")
        print(f"{'sync   total (client)':<28}{p50:>12.0f}ms{p95:>12.0f}ms{mean:>12.0f}ms{len(sync):>8}")

    if stream:
        p50, p95, mean = stat(stream, "client_ttft_ms")
        print(f"{'stream TTFT (client)':<28}{p50:>12.0f}ms{p95:>12.0f}ms{mean:>12.0f}ms{len(stream):>8}")
        p50, p95, mean = stat(stream, "client_total_ms")
        print(f"{'stream total (client)':<28}{p50:>12.0f}ms{p95:>12.0f}ms{mean:>12.0f}ms{len(stream):>8}")
        p50, p95, mean = stat(stream, "server_ttft_ms")
        print(f"{'stream TTFT (server)':<28}{p50:>12.0f}ms{p95:>12.0f}ms{mean:>12.0f}ms{len(stream):>8}")

    if sync and stream:
        sync_p50 = statistics.median([s["client_total_ms"] for s in sync])
        stream_ttft_p50 = statistics.median(
            [s["client_ttft_ms"] for s in stream if s["client_ttft_ms"] is not None]
        )
        ratio = sync_p50 / stream_ttft_p50 if stream_ttft_p50 else float("nan")
        print("-" * 78)
        print(f"  perceived-latency win (P50): sync_total / stream_ttft = {ratio:.1f}x faster")
    print("=" * 78)


def write_csv(samples: list[dict]) -> None:
    fields = [
        "endpoint", "query", "client_ttft_ms", "client_total_ms",
        "server_ttft_ms", "server_total_ms", "retrieval_ms", "tokens",
    ]
    with open(RESULTS, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in samples:
            w.writerow({k: s.get(k) for k in fields})
    print(f"\nraw timings → {RESULTS}")


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--rows", default="0,1,2,3,4",
                    help="comma-separated row indices from synthetic_testset_raw.csv")
    ap.add_argument("--repeats", type=int, default=3,
                    help="repeats per query per endpoint (after warmup)")
    ap.add_argument("--warmup", type=int, default=2,
                    help="warmup requests per endpoint (discarded)")
    ap.add_argument("--flush-between-phases", action="store_true", default=True,
                    help="FLUSHALL Redis between sync and stream phases so "
                         "each endpoint sees its own cold-path misses")
    ap.add_argument("--no-flush-between-phases", dest="flush_between_phases",
                    action="store_false")
    ap.add_argument("--redis-host", default="localhost")
    ap.add_argument("--redis-port", type=int, default=6379)
    ap.add_argument("--paraphrase-mode", action="store_true",
                    help="Send unique paraphrases per base query instead of "
                         "repeating identical text. Forces L1 misses so L2 "
                         "is the only thing that can short-circuit. Uses "
                         "eval/datasets/bench_paraphrases.json.")
    args = ap.parse_args()

    row_indices = [int(x) for x in args.rows.split(",")]

    # Build the query plan: list of lists, one per base query.
    # Repeat mode: [[q0, q0, ...], [q1, q1, ...], ...]   ← all identical inside a group
    # Paraphrase mode: [[orig0, para0_1, ...], [orig1, para1_1, ...], ...]   ← unique inside a group
    if args.paraphrase_mode:
        groups = load_paraphrase_groups(row_indices)
        groups = [g[: args.repeats] for g in groups]
        warmup_q = groups[0][0]
    else:
        queries = load_queries(row_indices)
        groups = [[q] * args.repeats for q in queries]
        warmup_q = queries[0]

    async with httpx.AsyncClient(base_url=args.base_url) as client:
        # Warmup — discarded. Cold imports, connection pool, OpenAI route.
        for _ in range(args.warmup):
            await bench_sync(client, warmup_q)
            await bench_stream(client, warmup_q)

        # Phase-separated: sync first, then flush, then stream. Interleaving
        # the two endpoints lets the first one warm the cache for the second,
        # which makes the second phase's P95 a lie.
        if args.flush_between_phases:
            await flush_redis(args.redis_host, args.redis_port)

        samples: list[dict] = []
        print("sync: ", end="", flush=True)
        for group in groups:
            for q in group:
                samples.append(await bench_sync(client, q))
                print(".", end="", flush=True)
        print()

        if args.flush_between_phases:
            await flush_redis(args.redis_host, args.redis_port)

        print("stream: ", end="", flush=True)
        for group in groups:
            for q in group:
                samples.append(await bench_stream(client, q))
                print(".", end="", flush=True)
        print()

    write_csv(samples)
    report(samples)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        sys.exit(130)
