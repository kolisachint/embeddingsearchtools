#!/usr/bin/env python3
"""Time bulk indexing through the daemon, the way a real caller drives it.

The `index` subcommand is not the path that matters: consumers run `serve` and
push `bulk` batches, so that is what this measures. Reports throughput and the
per-batch spread, because a mean alone hides a long tail.

Usage:
  scripts/bench-index.py --corpus chunks.jsonl [--model pack/bge-small]
                         [--limit 2000] [--batch 48] [--hybrid]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, help="JSONL of {id,text}")
    ap.add_argument("--binary", default="./target/release/embsearch")
    ap.add_argument("--model", help="model pack dir; omit for the bundled model")
    ap.add_argument("--limit", type=int, default=0, help="0 = whole corpus")
    ap.add_argument("--batch", type=int, default=48, help="chunks per bulk request")
    ap.add_argument("--hybrid", action="store_true")
    ap.add_argument("--store", help="store dir; default is a temp dir, removed after")
    ap.add_argument("--json", help="write a machine-readable record here")
    args = ap.parse_args()

    records = []
    with open(args.corpus, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
            if args.limit and len(records) >= args.limit:
                break
    if not records:
        print("empty corpus", file=sys.stderr)
        return 1

    store = args.store or tempfile.mkdtemp(prefix="embsearch-bench-")
    owned = args.store is None

    cmd = [args.binary, "serve", "--path", store]
    if args.hybrid:
        cmd.append("--hybrid")
    if args.model:
        cmd += ["--model", args.model]

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    try:
        # The daemon prints its readiness banner to stderr before serving; model
        # load is not indexing and must not be counted as it.
        ready_line = proc.stderr.readline().strip()
        load_done = time.monotonic()

        batch_times = []
        indexed = 0
        t0 = time.monotonic()
        for offset in range(0, len(records), args.batch):
            batch = records[offset : offset + args.batch]
            req = json.dumps({"op": "bulk", "items": batch})
            b0 = time.monotonic()
            proc.stdin.write(req + "\n")
            proc.stdin.flush()
            resp_line = proc.stdout.readline()
            b1 = time.monotonic()
            if not resp_line:
                err = proc.stderr.read()
                print(f"daemon closed early: {err}", file=sys.stderr)
                return 1
            resp = json.loads(resp_line)
            if not resp.get("ok", False):
                print(f"bulk failed: {resp}", file=sys.stderr)
                return 1
            batch_times.append(b1 - b0)
            indexed += len(batch)
            if offset and (offset // args.batch) % 25 == 0:
                rate = indexed / (b1 - t0)
                print(
                    f"  {indexed}/{len(records)}  {rate:.0f} chunks/s",
                    file=sys.stderr,
                )
        elapsed = time.monotonic() - t0
    finally:
        try:
            proc.stdin.close()
        except Exception:
            pass
        proc.wait(timeout=30)
        if owned:
            shutil.rmtree(store, ignore_errors=True)

    batch_times.sort()

    def pct(p: float) -> float:
        return batch_times[min(len(batch_times) - 1, int(p / 100 * len(batch_times)))]

    rate = indexed / elapsed
    result = {
        "model": args.model or "bundled",
        "modelBanner": ready_line,
        "chunks": indexed,
        "batchSize": args.batch,
        "hybrid": args.hybrid,
        "indexSeconds": round(elapsed, 2),
        "chunksPerSecond": round(rate, 1),
        "batchMs": {
            "mean": round(1000 * sum(batch_times) / len(batch_times), 1),
            "p50": round(1000 * pct(50), 1),
            "p90": round(1000 * pct(90), 1),
            "max": round(1000 * batch_times[-1], 1),
        },
        "ortThreadsEnv": os.environ.get("EMBSEARCH_ORT_THREADS", "(unset)"),
        # What the full corpus would cost at this rate, which is the number the
        # eval budget is actually set against.
        "projected22kMinutes": round(22012 / rate / 60, 1),
    }
    print(json.dumps(result, indent=2))
    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(result, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
