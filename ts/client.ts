/**
 * TypeScript client for the `embsearch` stdio daemon.
 *
 * Spawns `embsearch serve` once and talks newline-delimited JSON over its
 * stdin/stdout. Because the process stays alive, the model and index are loaded
 * a single time and every query after startup is hot — this is the low-latency
 * path, as opposed to spawning the binary per request.
 *
 * Zero dependencies; only Node's built-in `child_process`.
 *
 * ```ts
 * const client = new EmbSearchClient({ binaryPath: "./embsearch", storePath: "./store" });
 * await client.ready();
 * await client.add("doc1", "hello world");
 * const hits = await client.query("greeting", 5);
 * await client.close();
 * ```
 */

import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";

export interface SearchResult {
  id: string;
  score: number;
}

export interface DaemonInfo {
  modelId: string;
  dim: number;
  count: number;
}

export interface BulkResult {
  inserted: number;
  updated: number;
}

export interface EmbSearchOptions {
  /** Path to the `embsearch` binary. */
  binaryPath: string;
  /** Store directory passed as `--path`. */
  storePath: string;
  /** Metric for a freshly created store. Default: "cosine". */
  metric?: "cosine" | "dot" | "euclidean";
  /** Optional model dir (only meaningful for an onnx-built binary). */
  modelPath?: string;
}

interface Pending {
  resolve: (value: any) => void;
  reject: (err: Error) => void;
}

export class EmbSearchClient {
  private proc: ChildProcessWithoutNullStreams;
  private queue: Pending[] = [];
  private buffer = "";
  private closed = false;
  private readyPromise: Promise<void>;

  constructor(opts: EmbSearchOptions) {
    const args = ["serve", "--path", opts.storePath];
    if (opts.metric) args.push("--metric", opts.metric);
    if (opts.modelPath) args.push("--model", opts.modelPath);

    this.proc = spawn(opts.binaryPath, args, {
      stdio: ["pipe", "pipe", "pipe"],
    });

    this.proc.stdout.setEncoding("utf8");
    this.proc.stdout.on("data", (chunk: string) => this.onStdout(chunk));

    // Readiness = the daemon answering a ping. This probes the actual
    // request loop instead of grepping stderr for a banner string.
    this.readyPromise = this.send({ op: "ping" }).then(() => undefined);

    this.proc.on("exit", (code) => {
      this.closed = true;
      const err = new Error(`embsearch daemon exited (code ${code})`);
      for (const p of this.queue.splice(0)) p.reject(err);
    });
  }

  /** Resolves once the daemon has loaded the model + index. */
  ready(): Promise<void> {
    return this.readyPromise;
  }

  private onStdout(chunk: string): void {
    this.buffer += chunk;
    let nl: number;
    // Each response is one line; dispatch FIFO against the pending queue.
    while ((nl = this.buffer.indexOf("\n")) !== -1) {
      const line = this.buffer.slice(0, nl).trim();
      this.buffer = this.buffer.slice(nl + 1);
      if (!line) continue;
      const pending = this.queue.shift();
      if (!pending) continue;
      let msg: any;
      try {
        msg = JSON.parse(line);
      } catch (e) {
        pending.reject(new Error(`bad response: ${line}`));
        continue;
      }
      if (msg.ok) pending.resolve(msg);
      else pending.reject(new Error(msg.error ?? "unknown error"));
    }
  }

  private send<T = any>(req: Record<string, unknown>): Promise<T> {
    if (this.closed) return Promise.reject(new Error("client is closed"));
    return new Promise<T>((resolve, reject) => {
      this.queue.push({ resolve, reject });
      this.proc.stdin.write(JSON.stringify(req) + "\n");
    });
  }

  /** Search for the top-`k` matches for `text`. */
  async query(text: string, k = 10): Promise<SearchResult[]> {
    const res = await this.send({ op: "query", text, k });
    return res.results ?? [];
  }

  /**
   * Batched insert-or-replace. One embedding inference for the whole batch —
   * the fast path for bulk indexing. Keep batches modest (e.g. 32–64) so a
   * concurrent query is not stuck behind a huge inference.
   */
  async bulk(items: Array<{ id: string; text: string }>): Promise<BulkResult> {
    const res = await this.send({ op: "bulk", items });
    return {
      inserted: res.inserted_count ?? 0,
      updated: res.updated_count ?? 0,
    };
  }

  /** Model id, dimensionality, and live vector count of the daemon. */
  async info(): Promise<DaemonInfo> {
    const res = await this.send({ op: "info" });
    return {
      modelId: res.model_id ?? "",
      dim: res.dim ?? 0,
      count: res.count ?? 0,
    };
  }

  /** Reclaim tombstoned rows left behind by `remove`. */
  async compact(): Promise<void> {
    await this.send({ op: "compact" });
  }

  /** Search using a precomputed query vector. */
  async queryVector(vector: number[], k = 10): Promise<SearchResult[]> {
    const res = await this.send({ op: "query", vector, k });
    return res.results ?? [];
  }

  /** Insert a new record. Rejects if `id` already exists. */
  async add(id: string, text: string): Promise<void> {
    await this.send({ op: "add", id, text });
  }

  /** Replace the text of an existing record. */
  async update(id: string, text: string): Promise<void> {
    await this.send({ op: "update", id, text });
  }

  /** Insert or replace. Resolves to `true` if newly inserted. */
  async upsert(id: string, text: string): Promise<boolean> {
    const res = await this.send({ op: "upsert", id, text });
    return res.inserted === true;
  }

  /** Remove a record. Resolves to `true` if it existed. */
  async remove(id: string): Promise<boolean> {
    const res = await this.send({ op: "remove", id });
    return res.removed === true;
  }

  /** Persist the index to the store directory. */
  async save(): Promise<void> {
    await this.send({ op: "save" });
  }

  /** Number of live vectors. */
  async count(): Promise<number> {
    const res = await this.send({ op: "count" });
    return res.count ?? 0;
  }

  /** Shut the daemon down, closing stdin so it exits cleanly. */
  async close(): Promise<void> {
    if (this.closed) return;
    this.proc.stdin.end();
    await new Promise<void>((resolve) => this.proc.on("exit", () => resolve()));
  }
}
