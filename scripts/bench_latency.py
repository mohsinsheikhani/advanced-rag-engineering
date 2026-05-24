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
RESULTS = Path(__file__).parent / "bench_results.csv"


def load_queries(rows: list[int]) -> list[str]:
    with open(TESTSET) as f:
        all_rows = list(csv.DictReader(f))
    return [all_rows[i]["user_input"] for i in rows]


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
    args = ap.parse_args()

    row_indices = [int(x) for x in args.rows.split(",")]
    queries = load_queries(row_indices)

    async with httpx.AsyncClient(base_url=args.base_url) as client:
        # Warmup — discarded. Cold imports, connection pool, OpenAI route.
        warmup_q = queries[0]
        for _ in range(args.warmup):
            await bench_sync(client, warmup_q)
            await bench_stream(client, warmup_q)

        samples: list[dict] = []
        for q in queries:
            for _ in range(args.repeats):
                samples.append(await bench_sync(client, q))
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
