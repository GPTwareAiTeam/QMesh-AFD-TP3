#!/usr/bin/env python3
"""DSpark 1017 eight-context exact-token workload harness.

Permanent harness for the AFD-TP3 experiment tree.  It owns only its fixtures
and artifacts; it never starts, stops, or restarts the engine.

What it measures
----------------
Eight context types, two distinct prompts per type, and both token pairs:
  * 1024 prompt / 128 completion
  * 128 prompt / 1024 completion
Each cell is timed twice after warmup.  Prompt text is cut to the *exact*
served token count with /tokenize (add_special_tokens=false) and /detokenize;
completion is forced to the exact budget with max_tokens == min_tokens and
ignore_eos=true.  Forced continuations are valid for timing only, never for
quality; they are marked `forced_continuation: true` and
`invalid_for_quality: true`.

A separate natural-termination probe runs with ignore_eos=false and
min_tokens=0.  Quality uses the served official chat template
(`/v1/chat/completions`, `chat_template_kwargs={"thinking": false}`) so the
task is an instruction turn rather than a raw continuation; timed rows stay
on `/v1/completions` and are never scored for quality.  Quality prompts keep
the complete fixture body (labelled `quality_short`/`quality_long`); exact
128/1024-token cuts apply only to timed rows, because a truncated task can
remove facts the validator asks for.  It records finish_reason, JSON/tool
validity, task checks, and source-echo rejection.
A small fixed-seed sampled subset is reported separately and never mixed into
the greedy aggregates.

Streaming accounting
--------------------
The harness consumes OpenAI SSE and takes prompt/completion token counts from
the final `usage` object, not from the number of SSE chunks (a chunk may carry
more than one token).  TTFT is first content-chunk time.  Decode rate is
(usage.completion_tokens - 1) / (last content chunk - first content chunk),
labelled as first-to-last actual decode count.  Prefill throughput is
prompt_tokens / TTFT and is explicitly labelled a proxy, not a server GPU
prefill duration.

Metrics
-------
For every timed/quality request the harness snapshots /metrics immediately
before and after, stores parsed snapshots under metrics/, refuses to start if
the engine is already busy, and records counter deltas.  Acceptance/spec
counters are whatever the running build exposes; absent names are simply
absent.

Usage
-----
  # tokenizer-only dry run; no completions, safe before a window
  python3 eval/dspark_workload_eval.py --check

  # write the benchmark matrix plan without any completion request
  python3 eval/dspark_workload_eval.py --plan-only \
      --out-dir eval-out/dspark_eval_1017/round1

  # inside a granted exclusive window
  python3 eval/dspark_workload_eval.py --execute --label k0 \
      --out-dir eval-out/dspark_eval_1017/round1

  # matched comparison of two completed runs (medians per cell)
  python3 eval/dspark_workload_eval.py --compare DIR_A DIR_B \
      --out-dir DIR_CMP

  # matched chat quality only, no timed replay
  python3 eval/dspark_workload_eval.py --execute --label k3_chatq \
      --out-dir DIR_Q --quality-only --quality-mode chat --quality-sampled
"""

from __future__ import annotations

import argparse
import ast
import difflib
import http.client
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from fixtures_contexts import CONTEXT_TYPES, FIXTURES  # noqa: E402

SCHEMA_ROW = "dspark-workload-eval-row-v1"
SCHEMA_SUMMARY = "dspark-workload-eval-summary-v1"
SCHEMA_PLAN = "dspark-workload-eval-plan-v1"
SCHEMA_COMPARE = "dspark-workload-eval-compare-v1"
MODEL = "deepseek-ai/DeepSeek-V4.1-Flash"
DEFAULT_OUT = os.environ.get("DSPARK_EVAL_OUT", "eval-out/dspark_eval_1017/round1")
PAIR_SPECS = [
    # pair_id, prompt tokens, completion tokens, body key
    ("1024p_128c", 1024, 128, "long_body"),
    ("128p_1024c", 128, 1024, "short_body"),
]
PAIR_BY_ID = {p[0]: p for p in PAIR_SPECS}
METRIC_KEEP = re.compile(
    r"^(vllm:|afd|dspark|spec|accept|.*(?:spec|accept|draft).*$)")
SELECTED_DELTA_PREFIXES = (
    "vllm:prompt_tokens",
    "vllm:generation_tokens",
    "vllm:request_success",
    "vllm:num_requests",
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_prompt_tokens",
    "vllm:request_generation_tokens",
    "vllm:spec",
    "vllm:accept",
    "vllm:iteration_tokens",
    "vllm:request_decode_time",
    "vllm:kv_cache_usage",
    "vllm:prefix_cache",
)
METRIC_LINE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)(?P<labels>\{[^}]*\})?\s+"
    r"(?P<value>[^\s]+)$")


# ---------------------------------------------------------------------------
# small utilities
# ---------------------------------------------------------------------------

def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value)


def append_jsonl(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        f.flush()


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)


def median_or_none(values):
    vals = [v for v in values if v is not None]
    return statistics.median(vals) if vals else None


def spread(values):
    vals = [v for v in values if v is not None]
    if not vals:
        return {"n": 0, "median": None, "min": None, "max": None,
                "stdev": None}
    return {
        "n": len(vals),
        "median": statistics.median(vals),
        "min": min(vals),
        "max": max(vals),
        "stdev": statistics.stdev(vals) if len(vals) > 1 else 0.0,
    }


def fmt(value, nd=3):
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.{nd}f}"
    return str(value)


# ---------------------------------------------------------------------------
# served tokenizer client
# ---------------------------------------------------------------------------

class Tokenizer:
    """Thin client for the served tokenizer.  No local checkpoint assumptions."""

    def __init__(self, base_url: str, model: str, timeout: float = 120.0):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._count_cache = {}
        self._ids_cache = {}
        self._detok_cache = {}
        self.calls = 0

    def _post(self, path: str, payload: dict, timeout: float | None = None):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            self.base_url + path, data=data,
            headers={"Content-Type": "application/json"})
        self.calls += 1
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            return json.loads(r.read())

    def count(self, text: str) -> int:
        key = sha256_text(text)
        if key not in self._count_cache:
            rep = self._post("/tokenize", {
                "model": self.model, "prompt": text,
                "add_special_tokens": False,
            })
            self._count_cache[key] = int(rep["count"])
        return self._count_cache[key]

    def ids(self, text: str) -> list[int]:
        key = sha256_text(text)
        if key not in self._ids_cache:
            rep = self._post("/tokenize", {
                "model": self.model, "prompt": text,
                "add_special_tokens": False,
            })
            self._ids_cache[key] = list(rep["tokens"])
        return self._ids_cache[key]

    def detokenize(self, ids: list[int]) -> str:
        key = hashlib.sha256(
            json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        if key not in self._detok_cache:
            rep = self._post("/detokenize", {
                "model": self.model, "tokens": list(ids),
            })
            self._detok_cache[key] = rep["prompt"]
        return self._detok_cache[key]


# ---------------------------------------------------------------------------
# exact-token prompt construction
# ---------------------------------------------------------------------------

BOUNDARY_PUNCT = ".!?:;)]}\n"
# Natural separators only.  They are inserted between the truncated source
# document and the question so an exact token target remains reachable when
# tokenization merges across the boundary.  They never dominate a prompt.
CONNECTORS = ("", " ", "\n", ".\n", ".", "\n\n", "  ", "\t")


def boundary_score(body_text: str) -> int:
    """Prefer cuts that do not end mid-word.  Never used to pad."""
    if not body_text:
        return 0
    stripped = body_text.rstrip()
    if stripped and stripped[-1] in BOUNDARY_PUNCT:
        return 2
    if body_text[-1:].isspace():
        return 1
    return 0


def build_exact_prompt(tok: Tokenizer, head: str, body: str, tail: str,
                       target: int) -> dict:
    """Return a prompt whose served token count is exactly `target`.

    The body is truncated at a token boundary and optionally followed by a
    whitespace/punctuation separator; it is never padded with filler.  A
    small window is searched for a cut that also lands on a sentence-ish
    boundary, but exactness is the hard requirement.
    """
    h_count = tok.count(head)
    t_count = tok.count(tail)
    need = target - h_count - t_count
    body_ids = tok.ids(body)
    if need < 16:
        raise ValueError(
            f"head+tail ({h_count}+{t_count}) leave only {need} body tokens "
            f"for target {target}")
    if len(body_ids) < need:
        raise ValueError(
            f"body has {len(body_ids)} tokens, need at least {need} for "
            f"target {target}")

    def assemble(n, connector):
        body_text = tok.detokenize(body_ids[:n])
        return head + body_text + connector + tail, body_text, n, connector

    lo = max(16, need - 8)
    hi = min(len(body_ids), need + 8)
    for connector in CONNECTORS:
        candidates = []
        for n in range(lo, hi + 1):
            text, body_text, used, con = assemble(n, connector)
            candidates.append({
                "n": used, "count": tok.count(text), "text": text,
                "body_text": body_text, "connector": con,
                "score": boundary_score(body_text),
                "distance": abs(used - need),
            })
        exact = [c for c in candidates if c["count"] == target]
        if exact:
            best = sorted(
                exact, key=lambda c: (-c["score"], c["distance"]))[0]
            return _prompt_result(best, head, tail, target, h_count, t_count,
                                  len(body_ids), 1)

    # No exact combination in the window: walk the cut position, testing the
    # separator set at each step.  This is a last-resort exactness path.
    seen = {}
    cut = need
    last = None
    for iteration in range(60):
        closest = None
        for connector in CONNECTORS:
            text, body_text, used, con = assemble(cut, connector)
            item = {"n": used, "count": tok.count(text), "text": text,
                    "body_text": body_text, "connector": con,
                    "score": boundary_score(body_text),
                    "distance": abs(used - need)}
            if item["count"] == target:
                return _prompt_result(
                    item, head, tail, target, h_count, t_count,
                    len(body_ids), iteration + 1)
            if closest is None or abs(item["count"] - target) < abs(
                    closest["count"] - target):
                closest = item
        last = closest
        delta = closest["count"] - target
        step = -delta if delta else (1 if delta > 0 else -1)
        new_cut = min(len(body_ids), max(16, cut + step))
        if new_cut == cut:
            new_cut = min(len(body_ids), max(16, cut + (1 if step > 0
                                                       else -1)))
        if new_cut == cut:
            break
        if new_cut in seen:
            break
        seen[new_cut] = True
        cut = new_cut
    raise RuntimeError(
        f"could not hit exact token target {target} for prompt; "
        f"last count {last['count'] if last else None}, body cut {cut}")


def _prompt_result(best, head, tail, target, h_count, t_count,
                   body_available, iterations):
    text = best["text"]
    body_text = best["body_text"]
    connector = best.get("connector", "")
    return {
        "text": text,
        "target": target,
        "actual": best["count"],
        "head_tokens": h_count,
        "tail_tokens": t_count,
        "connector": connector,
        "connector_chars": len(connector),
        "body_tokens_used": best["n"],
        "body_tokens_available": body_available,
        "boundary_score": best["score"],
        "body_tail_preview": body_text[-80:],
        "iterations": iterations,
        "sha256": sha256_text(text),
        "chars": len(text),
    }


def build_full_prompt(tok: Tokenizer, head: str, body: str, tail: str) -> dict:
    """Quality prompt keeps the whole fixture body.

    Timed prompts are exact-token cuts by design.  Quality is a separate
    instruction probe and must not inherit a truncation that removes facts
    the validator asks for (for example the second site or the event time).
    """
    text = head + body + tail
    n_head = tok.count(head)
    n_tail = tok.count(tail)
    n_body = tok.count(body)
    return {
        "text": text,
        "target": None,
        "actual": tok.count(text),
        "head_tokens": n_head,
        "tail_tokens": n_tail,
        "connector": "",
        "connector_chars": 0,
        "body_tokens_used": n_body,
        "body_tokens_available": n_body,
        "boundary_score": 0,
        "body_tail_preview": body[-80:],
        "full_body": True,
        "sha256": sha256_text(text),
        "chars": len(text),
    }


# ---------------------------------------------------------------------------
# /metrics parsing and snapshots
# ---------------------------------------------------------------------------

def parse_metrics(text: str) -> dict:
    out = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        m = METRIC_LINE.match(line)
        if not m:
            continue
        name = m.group("name")
        labels = m.group("labels") or ""
        key = name + labels
        try:
            out[key] = float(m.group("value"))
        except ValueError:
            continue
    return out


def fetch_metrics_text(base_url: str, timeout: float = 10.0) -> str:
    req = urllib.request.Request(base_url.rstrip("/") + "/metrics")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read().decode("utf-8", errors="replace")


def fetch_metrics(base_url: str, timeout: float = 10.0) -> dict:
    return parse_metrics(fetch_metrics_text(base_url, timeout=timeout))


def keep_metric(key: str) -> bool:
    return bool(METRIC_KEEP.search(key))


def snapshot_metrics(base_url: str) -> dict:
    raw = fetch_metrics_text(base_url)
    series = {k: v for k, v in parse_metrics(raw).items()
              if keep_metric(k)}
    return {"timestamp": now_iso(), "series": series,
            "raw_sha256": hashlib.sha256(raw.encode()).hexdigest(),
            "raw_chars": len(raw), "raw": raw}


def discover_engine_container(port: int = 8001) -> str | None:
    """Best-effort container lookup by published engine port (CPU only)."""
    try:
        proc = subprocess.run(
            ["docker", "ps", "--filter", f"publish={port}",
             "--format", "{{.Names}}"],
            capture_output=True, text=True, timeout=15)
        names = [line.strip() for line in (proc.stdout or "").splitlines()
                 if line.strip()]
        return names[0] if names else None
    except Exception:  # noqa: BLE001
        return None


def capture_engine_log(out_dir: Path, since_iso: str,
                       container: str | None = None,
                       port: int = 8001) -> dict:
    """Best-effort docker log capture; never a serving action."""
    log_path = out_dir / "metrics" / "engine.log"
    err_path = out_dir / "metrics" / "engine_log_error.txt"
    resolved = container or discover_engine_container(port)
    if not resolved:
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_path.write_text(
            f"no container found publishing port {port}\n", encoding="utf-8")
        return {"path": None, "container": None,
                "error": f"no container found publishing port {port}"}
    try:
        proc = subprocess.run(
            ["docker", "logs", "--since", since_iso, resolved],
            capture_output=True, text=True, timeout=120)
        text = proc.stdout or ""
        if proc.stderr:
            text += "\n--- stderr ---\n" + proc.stderr
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.write_text(text, encoding="utf-8")
        return {"path": str(log_path), "chars": len(text),
                "returncode": proc.returncode, "container": resolved}
    except Exception as exc:  # noqa: BLE001
        err_path.parent.mkdir(parents=True, exist_ok=True)
        err_path.write_text(f"{type(exc).__name__}: {exc}\n",
                            encoding="utf-8")
        return {"path": None, "container": resolved,
                "error": f"{type(exc).__name__}: {exc}"}


def metric_value(series: dict, name: str, default=None):
    for key, value in series.items():
        if key == name or key.startswith(name + "{"):
            return value
    return default


def selected_deltas(before: dict, after: dict) -> dict:
    deltas = {}
    for key, value in after.items():
        if no_selected(key):
            continue
        old = before.get(key)
        if old is None:
            continue
        if value != old:
            deltas[key] = value - old
    return deltas


def no_selected(key: str) -> bool:
    return not any(prefix in key for prefix in SELECTED_DELTA_PREFIXES)


def counter_total(series: dict, name: str) -> float:
    total = 0.0
    found = False
    for key, value in series.items():
        if key == name or key.startswith(name + "{"):
            total += value
            found = True
    return total if found else 0.0


def wait_engine_idle(base_url: str, timeout: float = 5.0) -> dict:
    deadline = time.monotonic() + timeout
    last = {}
    while True:
        last = fetch_metrics(base_url)
        running = metric_value(last, "vllm:num_requests_running", 0.0)
        waiting = metric_value(last, "vllm:num_requests_waiting", 0.0)
        if running == 0 and waiting == 0:
            return last
        if time.monotonic() >= deadline:
            return last
        time.sleep(0.2)


# ---------------------------------------------------------------------------
# streaming request
# ---------------------------------------------------------------------------

def stream_completion(base_url: str, payload: dict,
                      timeout: float = 600.0,
                      endpoint: str = "completions") -> dict:
    parsed = urllib.parse.urlparse(base_url)
    host = parsed.hostname
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    path = (parsed.path.rstrip("/") if parsed.path else "") + \
        f"/v1/{endpoint}"
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    body = json.dumps(payload).encode("utf-8")
    started = time.perf_counter()
    started_iso = now_iso()
    result = {
        "http_status": None,
        "ttft_s": None,
        "e2e_s": None,
        "decode_window_s": None,
        "decode_tok_s": None,
        "completion_tok_s_e2e": None,
        "prefill_tok_s_proxy": None,
        "prompt_tokens": None,
        "completion_tokens": None,
        "total_tokens": None,
        "finish_reason": None,
        "sse_events": 0,
        "text_chunks": 0,
        "tokens_per_text_chunk": None,
        "chunk_gap_ms": {"median": None, "p90": None, "max": None},
        "output_text": "",
        "output_sha256": None,
        "output_chars": 0,
        "reasoning_text": "",
        "reasoning_chars": 0,
        "reasoning_chunks": 0,
        "reasoning_tokens": None,
        "usage_source": "usage",
        "error": None,
        "started_iso": started_iso,
        "ended_iso": None,
    }
    try:
        conn.request(
            "POST", path, body=body,
            headers={
                "Content-Type": "application/json",
                "Accept": "text/event-stream",
                "Connection": "close",
            })
        resp = conn.getresponse()
        result["http_status"] = resp.status
        if resp.status != 200:
            err_body = resp.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                f"HTTP {resp.status}: {err_body[:2000]}")
        first = None
        last = None
        prev = None
        gaps = []
        parts = []
        reasoning_parts = []
        usage = None
        finish = None
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            result["sse_events"] += 1
            if obj.get("error"):
                raise RuntimeError(json.dumps(obj["error"])[:2000])
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            for choice in choices:
                delta = choice.get("text")
                if delta is None and isinstance(choice.get("delta"), dict):
                    inner = choice["delta"]
                    delta = inner.get("content")
                    reasoning = (inner.get("reasoning_content")
                                 or inner.get("reasoning"))
                    if reasoning:
                        reasoning_parts.append(reasoning)
                        result["reasoning_chunks"] += 1
                if choice.get("finish_reason"):
                    finish = choice["finish_reason"]
                if delta is None or delta == "":
                    continue
                moment = time.perf_counter()
                if first is None:
                    first = moment
                else:
                    gaps.append((moment - prev) * 1000.0)
                prev = moment
                last = moment
                result["text_chunks"] += 1
                parts.append(delta)
        ended = time.perf_counter()
        text = "".join(parts)
        reasoning_text = "".join(reasoning_parts)
        result["output_text"] = text
        result["output_sha256"] = sha256_text(text)
        result["output_chars"] = len(text)
        result["reasoning_text"] = reasoning_text
        result["reasoning_chars"] = len(reasoning_text)
        result["finish_reason"] = finish
        if usage:
            result["prompt_tokens"] = usage.get("prompt_tokens")
            result["completion_tokens"] = usage.get("completion_tokens")
            result["total_tokens"] = usage.get("total_tokens")
            details = usage.get("completion_tokens_details") or {}
            result["reasoning_tokens"] = details.get("reasoning_tokens")
        else:
            result["usage_source"] = "missing_usage"
        if result["completion_tokens"] is None:
            # Never treat chunk count as token count.  Tokenize the text and
            # mark the fallback explicitly; it is not used for forced-budget
            # validity.
            result["usage_source"] = "tokenize_fallback"
        result["ttft_s"] = (first - started) if first is not None else None
        result["e2e_s"] = ended - started
        if first is not None and last is not None and last > first:
            result["decode_window_s"] = last - first
        n_out = result["completion_tokens"]
        if n_out and n_out > 1 and result["decode_window_s"]:
            result["decode_tok_s"] = (n_out - 1) / result["decode_window_s"]
        if n_out and result["e2e_s"]:
            result["completion_tok_s_e2e"] = n_out / result["e2e_s"]
        if result["prompt_tokens"] and result["ttft_s"]:
            result["prefill_tok_s_proxy"] = (
                result["prompt_tokens"] / result["ttft_s"])
        if result["text_chunks"] and n_out:
            result["tokens_per_text_chunk"] = n_out / result["text_chunks"]
        if gaps:
            ordered = sorted(gaps)
            result["chunk_gap_ms"] = {
                "median": statistics.median(ordered),
                "p90": ordered[min(len(ordered) - 1,
                                   int(0.9 * (len(ordered) - 1)))],
                "max": ordered[-1],
            }
    except Exception as exc:  # noqa: BLE001 — recorded as a failed row
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["ended_iso"] = now_iso()
        result["e2e_s"] = time.perf_counter() - started
    finally:
        try:
            conn.close()
        except Exception:
            pass
    return result


# ---------------------------------------------------------------------------
# request payloads
# ---------------------------------------------------------------------------

def timed_payload(prompt: str, completion_tokens: int, seed: int) -> dict:
    return {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": completion_tokens,
        "min_tokens": completion_tokens,
        "ignore_eos": True,
        "temperature": 0.0,
        "top_p": 1.0,
        "seed": seed,
        "n": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
        "add_special_tokens": False,
    }


def quality_payload(prompt: str, max_tokens: int, seed: int,
                    temperature: float, top_p: float) -> dict:
    return {
        "model": MODEL,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "min_tokens": 0,
        "ignore_eos": False,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "n": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
        "add_special_tokens": False,
    }


CHAT_QUALITY_SYSTEM = (
    "You are a precise assistant. Follow the user's final task exactly. "
    "Do not continue or repeat the source document unless the task asks for "
    "it. For JSON or tool-call tasks, output only the requested JSON value."
)


def chat_quality_payload(prompt: str, max_tokens: int, seed: int,
                         temperature: float, top_p: float) -> dict:
    """Quality goes through the served model's official chat template.

    The timed path stays on raw /v1/completions. Quality needs an instruction
    turn, not a forced raw continuation, so it must use the chat endpoint.
    """
    return {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": CHAT_QUALITY_SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max(1024, max_tokens),
        "min_tokens": 0,
        "ignore_eos": False,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "n": 1,
        "stream": True,
        "stream_options": {"include_usage": True},
        # Use the model's official V4.1 chat template with reasoning disabled
        # so the answer budget is not consumed by a hidden reasoning block.
        "chat_template_kwargs": {"thinking": False},
    }


def longest_source_overlap(output: str, source: str) -> int:
    if not output or not source:
        return 0
    matcher = difflib.SequenceMatcher(None, source, output, autojunk=False)
    return matcher.find_longest_match(
        0, len(source), 0, len(output)).size


def source_echo_check(output: str, source: str,
                      validator_name: str) -> dict:
    # Code fixes must reproduce the supplied module; every other task must
    # answer rather than copy the source document/instruction.
    overlap = longest_source_overlap(output, source)
    if validator_name == "python_ast":
        return {"source_echo_chars": overlap, "no_source_echo": True}
    return {
        "source_echo_chars": overlap,
        "no_source_echo": overlap < 160,
    }


# ---------------------------------------------------------------------------
# quality validators
# ---------------------------------------------------------------------------

def extract_json_value(text: str):
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.S | re.I)
    haystack = fenced.group(1) if fenced else text
    decoder = json.JSONDecoder()
    candidates = []
    for i, ch in enumerate(haystack):
        if ch not in "[{":
            continue
        try:
            value, end = decoder.raw_decode(haystack[i:])
        except json.JSONDecodeError:
            continue
        candidates.append((end, value))
    if not candidates:
        raise ValueError("no parseable JSON value")
    return max(candidates, key=lambda item: item[0])[1]


def extract_python(text: str) -> str:
    blocks = re.findall(r"```(?:python)?\s*(.*?)```", text, re.S | re.I)
    if blocks:
        return "\n\n".join(blocks)
    stripped = text.strip()
    if stripped.startswith(("import ", "from ", "def ", "class ", "#!")):
        return stripped
    m = re.search(r"(?m)^(?:import |from |def |class )", text)
    return text[m.start():] if m else text


def validate_python(text: str, expect: dict) -> dict:
    code = extract_python(text)
    checks = {"code_present": bool(code.strip())}
    try:
        ast.parse(code)
        checks["python_parse"] = True
    except SyntaxError as exc:
        checks["python_parse"] = False
        checks["syntax_error"] = f"{exc.msg} at line {exc.lineno}"
    required = expect.get("required", [])
    forbidden = expect.get("forbidden", [])
    checks["required_present"] = all(token in code for token in required)
    checks["missing_required"] = [
        token for token in required if token not in code]
    checks["forbidden_present"] = [
        token for token in forbidden if token in code]
    passed = (
        checks.get("python_parse", False)
        and checks["required_present"]
        and not checks["forbidden_present"]
    )
    return {"pass": passed, "checks": checks}


def validate_json_order(text: str, expect: dict) -> dict:
    checks = {}
    try:
        obj = extract_json_value(text)
        checks["json_parse"] = True
    except ValueError as exc:
        return {"pass": False,
                "checks": {"json_parse": False, "error": str(exc)}}
    checks["is_object"] = isinstance(obj, dict)
    if not isinstance(obj, dict):
        return {"pass": False, "checks": checks}
    order_id = obj.get("order_id")
    checks["order_id_format"] = bool(
        isinstance(order_id, str) and re.fullmatch(r"ord_[0-9]{8}", order_id))
    customer = obj.get("customer")
    checks["customer_object"] = (
        isinstance(customer, dict)
        and isinstance(customer.get("id"), str)
        and isinstance(customer.get("email"), str)
        and customer["email"].count("@") == 1)
    lines = obj.get("lines")
    checks["lines_nonempty"] = isinstance(lines, list) and 1 <= len(lines) <= 20
    checks["lines_expected"] = (
        isinstance(lines, list) and len(lines) == expect.get("lines"))
    line_ok = True
    if isinstance(lines, list):
        for line in lines:
            if not isinstance(line, dict):
                line_ok = False
                break
            sku = line.get("sku")
            qty = line.get("quantity")
            price = line.get("unit_price_cents")
            disc = line.get("discount_cents", 0)
            if not (isinstance(sku, str)
                    and re.fullmatch(r"sku_[A-Z0-9]{6}", sku)):
                line_ok = False
            qty_ok = (isinstance(qty, int) and not isinstance(qty, bool)
                      and 1 <= qty <= 99)
            price_ok = (isinstance(price, int) and not isinstance(price, bool)
                        and price >= 1)
            if not qty_ok:
                line_ok = False
            if not price_ok:
                line_ok = False
            if not (isinstance(disc, int) and not isinstance(disc, bool)
                    and qty_ok and price_ok and 0 <= disc <= qty * price):
                line_ok = False
    checks["line_fields"] = line_ok
    shipping = obj.get("shipping")
    checks["shipping_object"] = (
        isinstance(shipping, dict)
        and shipping.get("method") in {"standard", "express", "overnight"}
        and isinstance(shipping.get("country"), str)
        and bool(re.fullmatch(r"[A-Z]{2}", shipping["country"])))
    if isinstance(shipping, dict):
        checks["shipping_method_expected"] = (
            shipping.get("method") == expect.get("method"))
        if shipping.get("method") == "overnight":
            checks["overnight_postal"] = isinstance(
                shipping.get("postal_code"), str)
        if shipping.get("method") == "express":
            checks["express_locale_or_postal"] = bool(
                shipping.get("postal_code")
                or (isinstance(customer, dict) and customer.get("locale")))
    checks["no_total_cents"] = "total_cents" not in obj
    notes = obj.get("notes")
    checks["notes_length"] = notes is None or (
        isinstance(notes, str) and len(notes) <= 280)
    tags = obj.get("tags")
    checks["tags_ok"] = tags is None or (
        isinstance(tags, list)
        and len(tags) <= 8
        and len({t.lower() for t in tags}) == len(tags)
        and all(isinstance(t, str)
                and re.fullmatch(r"[a-z][a-z0-9-]{0,23}", t) for t in tags))
    passed = all([
        checks["is_object"], checks["order_id_format"],
        checks["customer_object"], checks["lines_nonempty"],
        checks["lines_expected"], checks["line_fields"],
        checks["shipping_object"], checks["shipping_method_expected"],
        checks["no_total_cents"], checks["notes_length"], checks["tags_ok"],
    ])
    return {"pass": passed, "checks": checks}


def validate_json_errors(text: str, expect: dict) -> dict:
    checks = {}
    try:
        obj = extract_json_value(text)
        checks["json_parse"] = True
    except ValueError as exc:
        return {"pass": False,
                "checks": {"json_parse": False, "error": str(exc)}}
    checks["is_array"] = isinstance(obj, list)
    checks["array_len"] = isinstance(obj, list) and len(obj) == expect.get(
        "array_len")
    codes = []
    envelope_ok = True
    if isinstance(obj, list):
        for env in obj:
            if not isinstance(env, dict):
                envelope_ok = False
                continue
            rid = env.get("request_id")
            if not (isinstance(rid, str)
                    and re.fullmatch(r"req_[a-f0-9]{12}", rid)):
                envelope_ok = False
            if not isinstance(env.get("retryable"), bool):
                envelope_ok = False
            err = env.get("error")
            if not isinstance(err, dict):
                envelope_ok = False
                continue
            code = err.get("code")
            codes.append(code)
            if not isinstance(err.get("message"), str) or not (
                1 <= len(err["message"]) <= 200):
                envelope_ok = False
            details = err.get("details")
            if details is not None:
                if not isinstance(details, list) or not all(
                        isinstance(d, dict)
                        and isinstance(d.get("field"), str)
                        and isinstance(d.get("reason"), str)
                        for d in details):
                    envelope_ok = False
            if code == "invalid_order":
                if env.get("retryable") is not False:
                    envelope_ok = False
                if not isinstance(details, list) or not details:
                    envelope_ok = False
            if code == "rate_limited":
                if env.get("retryable") is not True:
                    envelope_ok = False
                retry = env.get("retry_after_seconds")
                if not (isinstance(retry, int) and 1 <= retry <= 300):
                    envelope_ok = False
    checks["envelopes_ok"] = envelope_ok
    checks["codes"] = codes
    checks["codes_expected"] = set(codes) == set(expect.get("codes", []))
    passed = all([
        checks["is_array"], checks["array_len"], checks["envelopes_ok"],
        checks["codes_expected"],
    ])
    return {"pass": passed, "checks": checks}


def validate_tool_call(text: str, expect: dict) -> dict:
    checks = {}
    try:
        obj = extract_json_value(text)
        checks["json_parse"] = True
    except ValueError as exc:
        return {"pass": False,
                "checks": {"json_parse": False, "error": str(exc)}}
    checks["is_object"] = isinstance(obj, dict)
    if not isinstance(obj, dict):
        return {"pass": False, "checks": checks}
    checks["exact_keys"] = set(obj) == {"name", "arguments"}
    checks["no_function_wrapper"] = "function" not in obj
    checks["tool_name"] = obj.get("name") == expect.get("tool_name")
    args = obj.get("arguments")
    checks["arguments_object"] = isinstance(args, dict)
    required = expect.get("required_args", [])
    checks["required_args"] = (
        isinstance(args, dict)
        and all(name in args for name in required))
    allowed = expect.get("allowed_args")
    checks["arg_keys"] = (
        isinstance(args, dict)
        and (allowed is None or set(args) <= set(allowed)))
    expected_values = expect.get("arg_values", {})
    checks["arg_values"] = (
        isinstance(args, dict)
        and all(args.get(key) == value
                for key, value in expected_values.items()))
    string_args = expect.get("string_args", required)
    array_args = expect.get("array_args", [])
    checks["arg_types"] = isinstance(args, dict) and all(
        isinstance(args.get(name), str) for name in string_args
        if name in args) and all(
        isinstance(args.get(name), list) for name in array_args
        if name in args)
    passed = all([
        checks["is_object"], checks["exact_keys"],
        checks["no_function_wrapper"], checks["tool_name"],
        checks["arguments_object"], checks["required_args"],
        checks["arg_keys"], checks["arg_values"], checks["arg_types"],
    ])
    return {"pass": passed, "checks": checks}


def validate_numeric(text: str, expect: dict) -> dict:
    numbers = []
    for raw in re.findall(r"-?\d[\d,]*(?:\.\d+)?", text):
        try:
            numbers.append(float(raw.replace(",", "")))
        except ValueError:
            continue
    lo = expect.get("min")
    hi = expect.get("max")
    hits = [n for n in numbers if lo <= n <= hi]
    return {
        "pass": bool(hits),
        "checks": {"numbers_found": len(numbers), "in_range": len(hits),
                   "range": [lo, hi]},
    }


def validate_contains(text: str, expect: dict) -> dict:
    lowered = text.lower()
    needed = expect.get("contains", [])
    missing = [item for item in needed if item.lower() not in lowered]
    any_groups = expect.get("contains_any", [])
    missing_any = [
        group for group in any_groups
        if not any(item.lower() in lowered for item in group)
    ]
    return {
        "pass": not missing and not missing_any,
        "checks": {"required": needed, "missing": missing,
                   "any_groups": any_groups, "missing_any": missing_any},
    }


def validate_prose(text: str, expect: dict) -> dict:
    minimum = expect.get("min_completion_tokens", 10)
    degenerate = False
    if len(text) >= 64:
        for width in (4, 8, 16):
            chunk = text[-width:]
            if chunk.strip() and text.endswith(chunk * min(8, len(text) // width)):
                degenerate = True
                break
    return {
        "pass": bool(text.strip()) and not degenerate,
        "checks": {"nonempty": bool(text.strip()),
                   "degenerate_tail": degenerate,
                   "min_completion_tokens": minimum},
    }


VALIDATORS = {
    "prose": validate_prose,
    "numeric_range": validate_numeric,
    "python_ast": validate_python,
    "json_order": validate_json_order,
    "json_errors": validate_json_errors,
    "tool_call": validate_tool_call,
    "contains": validate_contains,
}


# ---------------------------------------------------------------------------
# matrix construction
# ---------------------------------------------------------------------------

def fixture_digest() -> str:
    return sha256_file(HERE / "fixtures_contexts.py")


def build_prompt_manifest(tok: Tokenizer) -> tuple[dict, list]:
    """Build every timed and quality prompt.  Returns (manifest, audit)."""
    manifest = {}
    audit = []
    for fixture in FIXTURES:
        ctype = fixture["type"]
        row = {
            "context_type": ctype,
            "long_body_tokens": tok.count(fixture["long_body"]),
            "short_body_tokens": tok.count(fixture["short_body"]),
            "variants": [],
        }
        for variant in fixture["variants"]:
            prompt_id = variant["prompt_id"]
            head = variant["head"]
            tail = variant["tail"]
            head_count = tok.count(head)
            tail_count = tok.count(tail)
            variant_row = {
                "prompt_id": prompt_id,
                "head_tokens": head_count,
                "tail_tokens": tail_count,
                "needs": {},
            }
            for pair_id, prompt_tokens, _completion, body_key in PAIR_SPECS:
                need = prompt_tokens - head_count - tail_count
                available = tok.count(fixture[body_key])
                variant_row["needs"][pair_id] = {
                    "body_tokens_needed": need,
                    "body_tokens_available": available,
                    "ok": available >= need and need >= 16,
                }
                built = build_exact_prompt(
                    tok, head, fixture[body_key], tail, prompt_tokens)
                if built["actual"] != prompt_tokens:
                    raise RuntimeError(
                        f"{prompt_id}/{pair_id}: built {built['actual']} "
                        f"tokens, target {prompt_tokens}")
                key = (ctype, prompt_id, pair_id, "timed")
                manifest[key] = built
            q = variant["quality"]
            q_pair = ("quality_short" if q["pair"] == "short"
                      else "quality_long")
            body_key = f"{q['pair']}_body"
            built = build_full_prompt(tok, head, fixture[body_key], tail)
            manifest[(ctype, prompt_id, q_pair, "quality")] = built
            variant_row["quality"] = {
                "pair": q["pair"],
                "actual_prompt_tokens": built["actual"],
                "full_body": True,
            }
            row["variants"].append(variant_row)
        audit.append(row)
    return manifest, audit


def timed_cells() -> list:
    cells = []
    for fixture in FIXTURES:
        for variant in fixture["variants"]:
            for pair_id, prompt_tokens, completion_tokens, _body in PAIR_SPECS:
                cells.append({
                    "context_type": fixture["type"],
                    "prompt_id": variant["prompt_id"],
                    "pair": pair_id,
                    "prompt_tokens_target": prompt_tokens,
                    "completion_tokens_target": completion_tokens,
                })
    return cells


def quality_cells() -> list:
    cells = []
    for fixture in FIXTURES:
        for variant in fixture["variants"]:
            q = variant["quality"]
            pair_id = ("quality_short" if q["pair"] == "short"
                       else "quality_long")
            cells.append({
                "context_type": fixture["type"],
                "prompt_id": variant["prompt_id"],
                "pair": pair_id,
                "prompt_tokens_target": None,
                "prompt_full_body": True,
                "max_tokens": q["max_tokens"],
                "validator": q["validator"],
                "expect": q["expect"],
                "sampling": "greedy",
                "temperature": 0.0,
                "top_p": 1.0,
            })
    return cells


def sampled_cells(all_quality: list) -> list:
    seen = set()
    out = []
    for cell in all_quality:
        if cell["context_type"] in seen:
            continue
        seen.add(cell["context_type"])
        sampled = dict(cell)
        sampled["sampling"] = "sampled"
        sampled["temperature"] = 0.7
        sampled["top_p"] = 0.95
        out.append(sampled)
    return out


# ---------------------------------------------------------------------------
# plan / execution
# ---------------------------------------------------------------------------

def make_plan(label: str, out_dir: Path, repeats: int, seed: int,
              warmup: int, manifest: dict, audit: list) -> dict:
    cells = timed_cells()
    order = []
    for repeat in range(1, repeats + 1):
        shuffled = list(cells)
        random.Random(seed + repeat * 1009).shuffle(shuffled)
        for index, cell in enumerate(shuffled):
            order.append({
                **cell,
                "repeat": repeat,
                "pass_index": repeat,
                "order_index": len(order),
            })
    q_cells = quality_cells()
    s_cells = sampled_cells(q_cells)
    prompt_rows = []
    for key, built in sorted(manifest.items()):
        ctype, prompt_id, pair_id, kind = key
        prompt_rows.append({
            "kind": kind,
            "context_type": ctype,
            "prompt_id": prompt_id,
            "pair": pair_id,
            **{k: v for k, v in built.items() if k != "text"},
        })
    return {
        "schema": SCHEMA_PLAN,
        "created_at": now_iso(),
        "label": label,
        "out_dir": str(out_dir),
        "base_url": None,
        "model": MODEL,
        "seed": seed,
        "repeats": repeats,
        "warmup_requests": warmup,
        "pairs": [
            {"pair": p[0], "prompt_tokens": p[1],
             "completion_tokens": p[2]} for p in PAIR_SPECS
        ],
        "context_types": CONTEXT_TYPES,
        "fixtures_sha256": fixture_digest(),
        "timed_cells": cells,
        "timed_order": order,
        "quality_cells": q_cells,
        "sampled_cells": s_cells,
        "forced_continuation": {
            "max_tokens_equals_min_tokens": True,
            "ignore_eos": True,
            "valid_for_quality": False,
        },
        "tokenizer": {
            "endpoint": "/tokenize",
            "detokenize_endpoint": "/detokenize",
            "add_special_tokens": False,
            "alignment_verified_against_usage_prompt_tokens": True,
        },
        "timed_request_count": len(order),
        "quality_request_count": len(q_cells) + len(s_cells),
        "prompts": prompt_rows,
        "fixture_audit": audit,
    }


def write_prompts_jsonl(path: Path, manifest: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for key in sorted(manifest):
            ctype, prompt_id, pair_id, kind = key
            built = manifest[key]
            row = {
                "kind": kind,
                "context_type": ctype,
                "prompt_id": prompt_id,
                "pair": pair_id,
                "target": built["target"],
                "actual": built["actual"],
                "head_tokens": built["head_tokens"],
                "tail_tokens": built["tail_tokens"],
                "connector": built.get("connector", ""),
                "connector_chars": built.get("connector_chars", 0),
                "body_tokens_used": built["body_tokens_used"],
                "body_tokens_available": built["body_tokens_available"],
                "boundary_score": built["boundary_score"],
                "body_tail_preview": built["body_tail_preview"],
                "sha256": built["sha256"],
                "chars": built["chars"],
                "text": built["text"],
            }
            f.write(json.dumps(row, sort_keys=True,
                               separators=(",", ":")) + "\n")


def run_request(base_url: str, payload: dict, row: dict, out_dir: Path,
                timeout: float, metrics: bool, request_tag: str,
                endpoint: str = "completions") -> dict:
    before = None
    before_file = None
    before_prom = None
    after_file = None
    after_prom = None
    metrics_error = None
    if metrics:
        try:
            before = snapshot_metrics(base_url)
            before_file = f"metrics/{safe_name(request_tag)}.before.json"
            before_prom = f"metrics/{safe_name(request_tag)}.before.prom"
            (out_dir / before_prom).parent.mkdir(
                parents=True, exist_ok=True)
            (out_dir / before_prom).write_text(before["raw"],
                                               encoding="utf-8")
            write_json(out_dir / before_file, {
                "timestamp": before["timestamp"],
                "raw_sha256": before["raw_sha256"],
                "raw_chars": before["raw_chars"],
                "series": before["series"],
            })
        except Exception as exc:  # noqa: BLE001
            metrics_error = f"before: {type(exc).__name__}: {exc}"
    result = stream_completion(base_url, payload, timeout=timeout,
                               endpoint=endpoint)
    row.update({k: v for k, v in result.items() if k != "started_iso"})
    row["started_iso"] = result["started_iso"]
    row["ended_iso"] = result["ended_iso"]
    if metrics:
        try:
            if before is not None:
                wait_engine_idle(base_url, timeout=5.0)
            after = snapshot_metrics(base_url)
            after_file = f"metrics/{safe_name(request_tag)}.after.json"
            after_prom = f"metrics/{safe_name(request_tag)}.after.prom"
            (out_dir / after_prom).parent.mkdir(
                parents=True, exist_ok=True)
            (out_dir / after_prom).write_text(after["raw"],
                                              encoding="utf-8")
            write_json(out_dir / after_file, {
                "timestamp": after["timestamp"],
                "raw_sha256": after["raw_sha256"],
                "raw_chars": after["raw_chars"],
                "series": after["series"],
            })
            if before is not None:
                row["counter_deltas"] = selected_deltas(
                    before["series"], after["series"])
                row["num_requests_running_before"] = metric_value(
                    before["series"], "vllm:num_requests_running", 0.0)
                row["num_requests_running_after"] = metric_value(
                    after["series"], "vllm:num_requests_running", 0.0)
                row["num_requests_waiting_before"] = metric_value(
                    before["series"], "vllm:num_requests_waiting", 0.0)
                row["num_requests_waiting_after"] = metric_value(
                    after["series"], "vllm:num_requests_waiting", 0.0)
                row["metrics_series_before"] = len(before["series"])
                row["metrics_series_after"] = len(after["series"])
        except Exception as exc:  # noqa: BLE001
            metrics_error = (metrics_error + "; " if metrics_error else "") + \
                f"after: {type(exc).__name__}: {exc}"
    row["metrics_before_file"] = before_file
    row["metrics_after_file"] = after_file
    row["metrics_before_prom"] = before_prom
    row["metrics_after_prom"] = after_prom
    row["metrics_ok"] = metrics and metrics_error is None
    if metrics_error:
        row["metrics_error"] = metrics_error
    return row


def health_check(base_url: str, model: str) -> dict:
    with urllib.request.urlopen(base_url + "/health", timeout=10) as r:
        if r.status != 200:
            raise RuntimeError(f"/health returned {r.status}")
    with urllib.request.urlopen(base_url + "/v1/models", timeout=10) as r:
        models = json.loads(r.read())
    ids = [item.get("id") for item in models.get("data", [])]
    if model not in ids:
        raise RuntimeError(f"model {model} not served; ids={ids}")
    series = fetch_metrics(base_url)
    running = metric_value(series, "vllm:num_requests_running", 0.0)
    waiting = metric_value(series, "vllm:num_requests_waiting", 0.0)
    if running or waiting:
        raise RuntimeError(
            f"engine is busy before run: running={running} waiting={waiting}; "
            "refusing to collect an isolated baseline")
    return {"model_ids": ids, "running": running, "waiting": waiting,
            "metrics_series": len(series)}


def execute(args, out_dir: Path, manifest: dict, plan: dict) -> None:
    base = args.base_url.rstrip("/")
    health = health_check(base, args.model)
    if args.resume_timed:
        meta_path = out_dir / "run_meta.json"
        if not meta_path.exists():
            raise RuntimeError(
                f"--resume-timed requires existing {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        run_id = meta["run_id"]
        meta["resumed_at"] = now_iso()
        meta["health_resume"] = health
        write_json(out_dir / "health_resume.json", {
            "timestamp": now_iso(), **health})
    else:
        run_id = (
            f"{args.label}-"
            f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}")
        meta = {
            "schema": "dspark-workload-eval-meta-v1",
            "run_id": run_id,
            "label": args.label,
            "created_at": now_iso(),
            "base_url": base,
            "model": args.model,
            "seed": args.seed,
            "sampled_seed": args.sampled_seed,
            "repeats": args.repeats,
            "warmup_requests": args.warmup,
            "out_dir": str(out_dir),
            "fixtures_sha256": fixture_digest(),
            "health_before": health,
            "engine_container": args.engine_container,
            "command": " ".join(sys.argv),
        }
        write_json(out_dir / "health_before.json", {
            "timestamp": now_iso(), **health})
        write_json(out_dir / "run_meta.json", meta)

    timed_path = out_dir / "timed.jsonl"
    quality_path = out_dir / "quality.jsonl"
    sampled_path = out_dir / "quality_sampled.jsonl"
    warmup_path = out_dir / "warmup.jsonl"
    if args.quality_only:
        for path in (timed_path, quality_path, sampled_path, warmup_path):
            if path.exists():
                path.unlink()
    elif args.resume_timed:
        if not timed_path.exists():
            raise RuntimeError(
                f"--resume-timed requires existing {timed_path}")
        for path in (quality_path, sampled_path):
            if path.exists():
                path.unlink()
    else:
        for path in (timed_path, quality_path, sampled_path, warmup_path):
            if path.exists():
                path.unlink()


    # Warmup: one short-prompt and one long-prompt forced decode, discarded.
    if args.warmup and not args.resume_timed and not args.quality_only:
        first = FIXTURES[0]
        warm_cells = [
            (first["type"], first["variants"][0]["prompt_id"],
             "128p_1024c", 16),
            (first["type"], first["variants"][0]["prompt_id"],
             "1024p_128c", 16),
        ]
        for index in range(min(args.warmup, len(warm_cells))):
            ctype, prompt_id, pair_id, completion = warm_cells[index]
            built = manifest[(ctype, prompt_id, pair_id, "timed")]
            tag = f"{run_id}-warmup-{index + 1}"
            row = {
                "schema": SCHEMA_ROW, "run_id": run_id,
                "label": args.label, "phase": "warmup",
                "request_id": tag, "context_type": ctype,
                "prompt_id": prompt_id, "pair": pair_id,
                "prompt_tokens_target": PAIR_BY_ID[pair_id][1],
                "completion_tokens_target": completion,
                "seed": args.seed, "temperature": 0.0,
                "ignore_eos": True, "min_tokens": completion,
                "max_tokens": completion, "stream": True,
                "forced_continuation": True, "invalid_for_quality": True,
            }
            row = run_request(
                base, timed_payload(built["text"], completion, args.seed),
                row, out_dir, args.timeout, False, tag)
            append_jsonl(warmup_path, row)
            print(f"[warmup] {index + 1}/{args.warmup} "
                  f"{pair_id} ttft={fmt(row.get('ttft_s'))} "
                  f"decode={fmt(row.get('decode_tok_s'), 2)} tok/s",
                  flush=True)

    prompt_exact_abort = not args.allow_prompt_drift
    order = [] if args.quality_only else plan["timed_order"]
    if args.quality_only:
        timed_rows = []
        print("[quality-only] skipping all timed requests", flush=True)
    elif args.resume_timed:
        timed_rows = read_jsonl(timed_path)
        if len(timed_rows) != len(order):
            raise RuntimeError(
                f"--resume-timed found {len(timed_rows)} rows, "
                f"plan needs {len(order)}")
        expected = {
            (c["context_type"], c["prompt_id"], c["pair"], c["repeat"])
            for c in order}
        observed = {
            (r.get("context_type"), r.get("prompt_id"), r.get("pair"),
             r.get("repeat")) for r in timed_rows}
        if observed != expected:
            raise RuntimeError("--resume-timed rows do not match the plan")
        for row in timed_rows:
            if (row.get("error") or not row.get("prompt_exact")
                    or not row.get("forced_budget_exact")):
                raise RuntimeError(
                    "--resume-timed refuses invalid row "
                    f"{row.get('request_id')}: error={row.get('error')} "
                    f"prompt_exact={row.get('prompt_exact')} "
                    f"budget_exact={row.get('forced_budget_exact')}")
        print(f"[resume] reusing {len(timed_rows)} valid timed rows; "
              "quality only", flush=True)
    else:
        timed_rows = []
        for position, cell in enumerate(order, start=1):
            key = (cell["context_type"], cell["prompt_id"], cell["pair"],
                   "timed")
            built = manifest[key]
            prompt_tokens = cell["prompt_tokens_target"]
            completion_tokens = cell["completion_tokens_target"]
            tag = (f"{run_id}-timed-{cell['context_type']}-"
                   f"{cell['prompt_id']}-{cell['pair']}-r{cell['repeat']}")
            row = {
                "schema": SCHEMA_ROW,
                "run_id": run_id,
                "label": args.label,
                "phase": "timed",
                "request_id": tag,
                "context_type": cell["context_type"],
                "prompt_id": cell["prompt_id"],
                "pair": cell["pair"],
                "repeat": cell["repeat"],
                "order_index": cell["order_index"],
                "prompt_tokens_target": prompt_tokens,
                "completion_tokens_target": completion_tokens,
                "seed": args.seed,
                "temperature": 0.0,
                "top_p": 1.0,
                "ignore_eos": True,
                "min_tokens": completion_tokens,
                "max_tokens": completion_tokens,
                "stream": True,
                "forced_continuation": True,
                "invalid_for_quality": True,
                "prompt_sha256": built["sha256"],
            }
            row = run_request(
                base, timed_payload(built["text"], completion_tokens,
                                    args.seed),
                row, out_dir, args.timeout, args.metrics, tag)
            row["prompt_exact"] = (
                row.get("prompt_tokens") == prompt_tokens)
            row["forced_budget_exact"] = (
                row.get("completion_tokens") == completion_tokens
                and row.get("finish_reason") == "length")
            append_jsonl(timed_path, row)
            timed_rows.append(row)
            print(
                f"[timed] {position}/{len(order)} {cell['context_type']} "
                f"{cell['prompt_id']} {cell['pair']} r{cell['repeat']} "
                f"ttft={fmt(row.get('ttft_s'), 3)}s "
                f"decode={fmt(row.get('decode_tok_s'), 2)} tok/s "
                f"e2e={fmt(row.get('e2e_s'), 2)}s "
                f"out={row.get('completion_tokens')} "
                f"{'ERR ' + row['error'] if row.get('error') else ''}",
                flush=True)
            if prompt_exact_abort and not row["prompt_exact"]:
                raise RuntimeError(
                    f"prompt token drift at {tag}: "
                    f"usage={row.get('prompt_tokens')} "
                    f"target={prompt_tokens}; stopping to protect exactness")
            if row.get("error"):
                raise RuntimeError(
                    f"timed request failed at {tag}: {row['error']}")

    quality_rows = []
    if not args.skip_quality:
        quality_rows = run_quality(args, out_dir, manifest, plan, run_id,
                                   quality_path, sampled=False)
        if args.quality_sampled:
            run_quality(args, out_dir, manifest, plan, run_id, sampled_path,
                        sampled=True)
    engine_port = urllib.parse.urlparse(base).port or 8001
    log_info = capture_engine_log(
        out_dir, meta["created_at"],
        container=args.engine_container, port=engine_port)
    write_json(out_dir / "engine_log.json", {
        "timestamp": now_iso(), "since": meta["created_at"], **log_info})
    summary = summarize(out_dir, args.label, run_id, plan, timed_rows,
                        quality_rows,
                        read_jsonl(sampled_path) if sampled_path.exists() else [])
    summary["engine_log"] = log_info
    write_json(out_dir / "summary.json", summary)
    (out_dir / "tables.txt").write_text(render_tables(summary),
                                        encoding="utf-8")
    print_summary(summary)
    print(f"\nartifacts: {out_dir}", flush=True)


def run_quality(args, out_dir, manifest, plan, run_id, path, sampled):
    cells = plan["sampled_cells"] if sampled else plan["quality_cells"]
    rows = []
    phase = "quality_sampled" if sampled else "quality"
    mode = getattr(args, "quality_mode", "chat")
    for index, cell in enumerate(cells, start=1):
        built = manifest[(cell["context_type"], cell["prompt_id"],
                          cell["pair"], "quality")]
        tag = (f"{run_id}-{phase}-{cell['context_type']}-"
               f"{cell['prompt_id']}")
        seed = args.sampled_seed if sampled else args.seed
        row = {
            "schema": SCHEMA_ROW,
            "run_id": run_id,
            "label": args.label,
            "phase": phase,
            "request_id": tag,
            "context_type": cell["context_type"],
            "prompt_id": cell["prompt_id"],
            "pair": cell["pair"],
            "prompt_tokens_target": built["actual"],
            "prompt_full_body": True,
            "completion_tokens_target": None,
            "max_tokens": (max(1024, cell["max_tokens"])
                           if mode == "chat" else cell["max_tokens"]),
            "seed": seed,
            "temperature": cell["temperature"],
            "top_p": cell["top_p"],
            "ignore_eos": False,
            "min_tokens": 0,
            "stream": True,
            "forced_continuation": False,
            "invalid_for_quality": False,
            "sampling": cell["sampling"],
            "validator": cell["validator"],
            "prompt_sha256": built["sha256"],
            "quality_mode": mode,
            "request_endpoint": (
                "/v1/chat/completions" if mode == "chat"
                else "/v1/completions"),
            "chat_thinking": False if mode == "chat" else None,
        }
        if mode == "chat":
            payload = chat_quality_payload(
                built["text"], cell["max_tokens"], seed,
                cell["temperature"], cell["top_p"])
            endpoint = "chat/completions"
        else:
            payload = quality_payload(
                built["text"], cell["max_tokens"], seed,
                cell["temperature"], cell["top_p"])
            endpoint = "completions"
        row = run_request(
            base_url=args.base_url.rstrip("/"),
            payload=payload, row=row, out_dir=out_dir,
            timeout=args.timeout, metrics=args.metrics,
            request_tag=tag, endpoint=endpoint)
        if mode == "chat":
            row["prompt_exact"] = None
            row["prompt_exact_note"] = (
                "not applicable: official chat template adds tokens")
        else:
            row["prompt_exact"] = (
                row.get("prompt_tokens") == cell["prompt_tokens_target"])
        row["natural_termination"] = row.get("finish_reason") == "stop"
        validator = VALIDATORS.get(cell["validator"])
        if row.get("error") or validator is None:
            validation = {
                "pass": False,
                "checks": {"error": row.get("error") or
                           f"unknown validator {cell['validator']}"},
            }
        else:
            validation = validator(row.get("output_text", ""), cell["expect"])
        echo = source_echo_check(
            row.get("output_text", ""), built["text"], cell["validator"])
        validation["checks"].update(echo)
        row["validator_contract_pass"] = validation["pass"]
        row["validator_pass"] = bool(
            validation["pass"] and echo["no_source_echo"])
        row["source_echo_chars"] = echo["source_echo_chars"]
        row["no_source_echo"] = echo["no_source_echo"]
        row["validator_checks"] = validation["checks"]
        row["quality_pass"] = bool(
            row["validator_pass"] and row["natural_termination"]
            and not row.get("error"))
        append_jsonl(path, row)
        rows.append(row)
        print(
            f"[{phase}] {index}/{len(cells)} {cell['context_type']} "
            f"{cell['prompt_id']} valid={row['validator_pass']} "
            f"contract={row['validator_contract_pass']} "
            f"echo={row['source_echo_chars']} "
            f"natural={row['natural_termination']} "
            f"tokens={row.get('completion_tokens')} "
            f"{'ERR ' + row['error'] if row.get('error') else ''}",
            flush=True)
    return rows


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------

def read_jsonl(path: Path) -> list:
    rows = []
    if not path.exists():
        return rows
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def cell_stats(rows: list) -> dict:
    valid = [r for r in rows if not r.get("error")]
    return {
        "n": len(rows),
        "n_valid": len(valid),
        "n_prompt_exact": sum(1 for r in valid if r.get("prompt_exact")),
        "n_forced_budget_exact": sum(
            1 for r in valid if r.get("forced_budget_exact")),
        "ttft_s": spread([r.get("ttft_s") for r in valid]),
        "decode_tok_s": spread([r.get("decode_tok_s") for r in valid]),
        "completion_tok_s_e2e": spread(
            [r.get("completion_tok_s_e2e") for r in valid]),
        "e2e_s": spread([r.get("e2e_s") for r in valid]),
        "prompt_tokens": spread([r.get("prompt_tokens") for r in valid]),
        "completion_tokens": spread(
            [r.get("completion_tokens") for r in valid]),
        "prefill_tok_s_proxy": spread(
            [r.get("prefill_tok_s_proxy") for r in valid]),
        "finish_reasons": sorted({r.get("finish_reason") for r in valid
                                  if r.get("finish_reason")}),
        "distinct_outputs": len({
            r.get("output_sha256") for r in valid
            if r.get("output_sha256")}),
    }


def aggregate_counters(rows: list) -> dict:
    totals = {}
    for row in rows:
        for key, delta in (row.get("counter_deltas") or {}).items():
            totals[key] = totals.get(key, 0.0) + delta
    return totals


def sum_metric_prefix(totals: dict, prefix: str) -> float:
    return sum(value for key, value in totals.items()
               if key == prefix or key.startswith(prefix + "{"))


def spec_summary(counter_totals: dict) -> dict:
    drafts = sum_metric_prefix(
        counter_totals, "vllm:spec_decode_num_drafts_total")
    draft_tokens = sum_metric_prefix(
        counter_totals, "vllm:spec_decode_num_draft_tokens_total")
    accepted = sum_metric_prefix(
        counter_totals, "vllm:spec_decode_num_accepted_tokens_total")
    per_position = {}
    for key, value in counter_totals.items():
        if not key.startswith(
                "vllm:spec_decode_num_accepted_tokens_per_pos_total"):
            continue
        m = re.search(r'position="(\d+)"', key)
        if m:
            pos = int(m.group(1))
            per_position[pos] = per_position.get(pos, 0.0) + value
    present = bool(
        drafts or draft_tokens or accepted or per_position)
    return {
        "present": present,
        "drafts_total": drafts,
        "draft_tokens_total": draft_tokens,
        "accepted_tokens_total": accepted,
        "acceptance_ratio": (
            accepted / draft_tokens if draft_tokens else None),
        "mean_accepted_length": (
            1.0 + accepted / drafts if drafts else None),
        "accepted_per_position": {
            str(pos): value for pos, value in sorted(per_position.items())},
        "per_position_acceptance_ratio": {
            str(pos): (value / drafts if drafts else None)
            for pos, value in sorted(per_position.items())},
        "note": ("spec counters absent when SPEC_TOKENS=0; a present=false "
                 "cell means this build exposed no spec families"),
    }


def prefix_cache_summary(counter_totals: dict) -> dict:
    hits = sum_metric_prefix(counter_totals, "vllm:prefix_cache_hits_total")
    queries = sum_metric_prefix(
        counter_totals, "vllm:prefix_cache_queries_total")
    return {"hits": hits, "queries": queries,
            "hit_ratio": (hits / queries if queries else None)}


def summarize(out_dir: Path, label: str, run_id: str, plan: dict,
              timed_rows: list, quality_rows: list,
              sampled_rows: list) -> dict:
    valid_timed = [r for r in timed_rows if not r.get("error")]
    by_cell = {}
    for row in valid_timed:
        key = (row["context_type"], row["prompt_id"], row["pair"])
        by_cell.setdefault(key, []).append(row)
    cells = []
    for key in sorted(by_cell):
        stats = cell_stats(by_cell[key])
        cells.append({
            "context_type": key[0], "prompt_id": key[1], "pair": key[2],
            **stats,
        })
    by_type = {}
    for row in valid_timed:
        by_type.setdefault((row["context_type"], row["pair"]), []).append(row)
    per_type = []
    for key in sorted(by_type):
        stats = cell_stats(by_type[key])
        per_type.append({
            "context_type": key[0], "pair": key[1],
            "cells": len({(r["prompt_id"]) for r in by_type[key]}),
            "repeats_per_cell": max(
                len([r for r in by_type[key]
                     if r["prompt_id"] == p]) for p in {
                         r["prompt_id"] for r in by_type[key]}),
            **stats,
        })
    per_pair = {}
    for pair_id in [p[0] for p in PAIR_SPECS]:
        rows = [r for r in valid_timed if r["pair"] == pair_id]
        if rows:
            per_pair[pair_id] = cell_stats(rows)
    q_by = {}
    for row in quality_rows:
        q_by.setdefault((row["context_type"], row["validator"]), []).append(row)
    quality = []
    for key in sorted(q_by):
        rows = q_by[key]
        quality.append({
            "context_type": key[0],
            "validator": key[1],
            "sampling": "greedy",
            "n": len(rows),
            "quality_pass": sum(1 for r in rows if r.get("quality_pass")),
            "validator_pass": sum(
                1 for r in rows if r.get("validator_pass")),
            "natural_termination": sum(
                1 for r in rows if r.get("natural_termination")),
            "prompt_exact": sum(1 for r in rows if r.get("prompt_exact")),
            "validator_contract_pass": sum(
                1 for r in rows if r.get("validator_contract_pass")),
            "no_source_echo": sum(1 for r in rows if r.get("no_source_echo")),
            "source_echo_chars_max": max(
                [r.get("source_echo_chars") or 0 for r in rows] or [0]),
            "completion_tokens": spread(
                [r.get("completion_tokens") for r in rows]),
            "reasoning_chars_max": max(
                [r.get("reasoning_chars") or 0 for r in rows] or [0]),
            "reasoning_tokens_max": max(
                [r.get("reasoning_tokens") or 0 for r in rows] or [0]),
            "finish_reasons": sorted({
                r.get("finish_reason") for r in rows
                if r.get("finish_reason")}),
        })
    sampled_by = {}
    for row in sampled_rows:
        sampled_by.setdefault((row["context_type"], row["validator"]),
                              []).append(row)
    sampled = []
    for key in sorted(sampled_by):
        rows = sampled_by[key]
        sampled.append({
            "context_type": key[0], "validator": key[1],
            "sampling": "sampled", "n": len(rows),
            "quality_pass": sum(1 for r in rows if r.get("quality_pass")),
            "validator_pass": sum(
                1 for r in rows if r.get("validator_pass")),
            "natural_termination": sum(
                1 for r in rows if r.get("natural_termination")),
            "validator_contract_pass": sum(
                1 for r in rows if r.get("validator_contract_pass")),
            "no_source_echo": sum(1 for r in rows if r.get("no_source_echo")),
            "source_echo_chars_max": max(
                [r.get("source_echo_chars") or 0 for r in rows] or [0]),
            "completion_tokens": spread(
                [r.get("completion_tokens") for r in rows]),
            "reasoning_chars_max": max(
                [r.get("reasoning_chars") or 0 for r in rows] or [0]),
            "reasoning_tokens_max": max(
                [r.get("reasoning_tokens") or 0 for r in rows] or [0]),
        })
    return {
        "schema": SCHEMA_SUMMARY,
        "generated_at": now_iso(),
        "run_id": run_id,
        "label": label,
        "out_dir": str(out_dir),
        "fixtures_sha256": plan["fixtures_sha256"],
        "seed": plan["seed"],
        "repeats": plan["repeats"],
        "pairs": plan["pairs"],
        "context_types": CONTEXT_TYPES,
        "timed_request_count": len(timed_rows),
        "timed_valid_count": len(valid_timed),
        "timed_all_prompt_exact": (
            all(r.get("prompt_exact") for r in valid_timed)
            if valid_timed else None),
        "timed_all_forced_budget_exact": (
            all(r.get("forced_budget_exact") for r in valid_timed)
            if valid_timed else None),
        "timed_forced_invalid_for_quality": True,
        "prefill_metric_note": (
            "prefill_tok_s_proxy is prompt_tokens/TTFT, a client-side proxy, "
            "not a server GPU prefill duration"),
        "decode_metric_note": (
            "decode_tok_s uses usage.completion_tokens (authoritative) over "
            "first-to-last content-chunk time; SSE chunks are never counted "
            "as tokens"),
        "cells": cells,
        "per_type": per_type,
        "per_pair": per_pair,
        "counter_totals": aggregate_counters(valid_timed),
        "spec_decode": spec_summary(aggregate_counters(valid_timed)),
        "prefix_cache": prefix_cache_summary(
            aggregate_counters(valid_timed)),
        "quality": quality,
        "quality_sampled": sampled,
        "quality_mode": (
            quality_rows[0].get("quality_mode") if quality_rows else None),
        "quality_note": (
            "quality probes use the official /v1/chat/completions template "
            "with ignore_eos=false, min_tokens=0, max_tokens>=1024, natural "
            "stop, and source-echo rejection; timed rows remain raw forced "
            "completions and are never mixed into quality"),
    }


def render_tables(summary: dict) -> str:
    lines = []
    lines.append(f"label: {summary['label']}  run: {summary['run_id']}  "
                 f"quality_mode: {summary.get('quality_mode')}")
    lines.append(f"timed rows: {summary['timed_valid_count']}/"
                 f"{summary['timed_request_count']} valid; "
                 f"prompt exact all={summary['timed_all_prompt_exact']}; "
                 f"forced budget exact all="
                 f"{summary['timed_all_forced_budget_exact']}")
    lines.append("")
    lines.append("Per context type and token pair (medians over prompts x "
                 "repeats; pairs never mixed)")
    header = ("context_type", "pair", "n", "ttft_s", "decode_tok_s",
              "e2e_s", "prompt_tok", "completion_tok", "distinct_out")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in summary["per_type"]:
        lines.append("| " + " | ".join([
            row["context_type"], row["pair"], str(row["n"]),
            fmt(row["ttft_s"]["median"], 3),
            fmt(row["decode_tok_s"]["median"], 2),
            fmt(row["e2e_s"]["median"], 2),
            fmt(row["prompt_tokens"]["median"], 0),
            fmt(row["completion_tokens"]["median"], 0),
            f"{row['distinct_outputs']}/{row['n_valid']}",
        ]) + " |")
    lines.append("")
    lines.append("Per cell (two distinct prompts, both pairs, repeats)")
    header = ("context_type", "prompt", "pair", "n", "ttft_s",
              "decode_tok_s", "e2e_s", "out_tok")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---"] * len(header)) + "|")
    for row in summary["cells"]:
        lines.append("| " + " | ".join([
            row["context_type"], row["prompt_id"], row["pair"],
            str(row["n_valid"]),
            fmt(row["ttft_s"]["median"], 3),
            fmt(row["decode_tok_s"]["median"], 2),
            fmt(row["e2e_s"]["median"], 2),
            fmt(row["completion_tokens"]["median"], 0),
        ]) + " |")
    lines.append("")
    if summary["quality"]:
        lines.append("Natural-termination quality probes (greedy; separate "
                     "from forced timed rows)")
        header = ("context_type", "validator", "n", "quality_pass",
                  "validator_pass", "natural_stop", "source_echo_chars",
                  "prompt_exact")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in summary["quality"]:
            lines.append("| " + " | ".join([
                row["context_type"], row["validator"], str(row["n"]),
                str(row["quality_pass"]), str(row["validator_pass"]),
                str(row["natural_termination"]),
                str(row.get("source_echo_chars_max", "n/a")),
                ("N/A" if summary.get("quality_mode") == "chat"
                 else str(row["prompt_exact"])),
            ]) + " |")
        lines.append("")
    if summary["quality_sampled"]:
        lines.append("Sampled subset (temperature 0.7, fixed seed; reported "
                     "separately)")
        header = ("context_type", "n", "quality_pass", "validator_pass",
                  "natural_stop")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in summary["quality_sampled"]:
            lines.append("| " + " | ".join([
                row["context_type"], str(row["n"]),
                str(row["quality_pass"]), str(row["validator_pass"]),
                str(row["natural_termination"]),
            ]) + " |")
        lines.append("")
    lines.append(f"prefill note: {summary['prefill_metric_note']}")
    lines.append(f"decode note: {summary['decode_metric_note']}")
    return "\n".join(lines) + "\n"


def print_summary(summary: dict) -> None:
    print("\n" + render_tables(summary), flush=True)


# ---------------------------------------------------------------------------
# matched comparison
# ---------------------------------------------------------------------------

def load_timed(directory: Path) -> list:
    return read_jsonl(directory / "timed.jsonl")


def matched_medians(rows: list) -> dict:
    by_cell = {}
    for row in rows:
        if row.get("error") or not row.get("forced_budget_exact"):
            continue
        key = (row["context_type"], row["prompt_id"], row["pair"])
        by_cell.setdefault(key, []).append(row)
    out = {}
    for key, cells in by_cell.items():
        out[key] = {
            "n": len(cells),
            "ttft_s": median_or_none([r.get("ttft_s") for r in cells]),
            "decode_tok_s": median_or_none(
                [r.get("decode_tok_s") for r in cells]),
            "e2e_s": median_or_none([r.get("e2e_s") for r in cells]),
            "prompt_tokens": median_or_none(
                [r.get("prompt_tokens") for r in cells]),
            "completion_tokens": median_or_none(
                [r.get("completion_tokens") for r in cells]),
            "label": cells[0].get("label"),
        }
    return out


def compare_runs(dir_a: Path, dir_b: Path, out_dir: Path) -> dict:
    a = matched_medians(load_timed(dir_a))
    b = matched_medians(load_timed(dir_b))
    matched = sorted(set(a) & set(b))
    cells = []
    for key in matched:
        left, right = a[key], b[key]
        deltas = {}
        for metric in ("ttft_s", "decode_tok_s", "e2e_s"):
            if left[metric] and right[metric]:
                deltas[metric + "_pct"] = (
                    (right[metric] - left[metric]) / left[metric] * 100.0)
        cells.append({
            "context_type": key[0], "prompt_id": key[1], "pair": key[2],
            "a": left, "b": right, "delta_pct_b_minus_a": deltas,
            "n_matched": min(left["n"], right["n"]),
        })
    by_type_pair = {}
    for cell in cells:
        by_type_pair.setdefault(
            (cell["context_type"], cell["pair"]), []).append(cell)
    aggregate = []
    for key in sorted(by_type_pair):
        group = by_type_pair[key]
        aggregate.append({
            "context_type": key[0], "pair": key[1],
            "matched_cells": len(group),
            "decode_tok_s_pct": median_or_none(
                [c["delta_pct_b_minus_a"].get("decode_tok_s_pct")
                 for c in group]),
            "ttft_s_pct": median_or_none(
                [c["delta_pct_b_minus_a"].get("ttft_s_pct")
                 for c in group]),
            "e2e_s_pct": median_or_none(
                [c["delta_pct_b_minus_a"].get("e2e_s_pct")
                 for c in group]),
        })
    result = {
        "schema": SCHEMA_COMPARE,
        "generated_at": now_iso(),
        "dir_a": str(dir_a),
        "dir_b": str(dir_b),
        "label_a": next(iter(a.values()))["label"] if a else None,
        "label_b": next(iter(b.values()))["label"] if b else None,
        "cells_a": len(a), "cells_b": len(b), "matched_cells": len(matched),
        "missing_in_a": sorted(f"{k[0]}/{k[1]}/{k[2]}" for k in set(b) - set(a)),
        "missing_in_b": sorted(f"{k[0]}/{k[1]}/{k[2]}" for k in set(a) - set(b)),
        "cells": cells,
        "per_type_pair": aggregate,
        "note": "delta_pct is (B - A) / A; matched by "
                "(context_type, prompt_id, pair); pairs never mixed"
    }
    write_json(out_dir / "compare.json", result)
    lines = [f"A={result['label_a']}  B={result['label_b']}  "
             f"matched={len(matched)} cells"]
    lines.append("| context_type | pair | cells | decode_tok_s % | "
                 "ttft_s % | e2e_s % |")
    lines.append("|---|---|---|---|---|---|")
    for row in aggregate:
        lines.append("| " + " | ".join([
            row["context_type"], row["pair"], str(row["matched_cells"]),
            fmt(row["decode_tok_s_pct"], 2),
            fmt(row["ttft_s_pct"], 2),
            fmt(row["e2e_s_pct"], 2),
        ]) + " |")
    (out_dir / "compare.txt").write_text("\n".join(lines) + "\n",
                                         encoding="utf-8")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="DSpark 1017 eight-context exact-token harness")
    p.add_argument("--base-url", default="http://127.0.0.1:8001")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--label", default="k0")
    p.add_argument("--out-dir", default=DEFAULT_OUT,
                   help="artifact directory (default: $DSPARK_EVAL_OUT or "
                        "eval-out/dspark_eval_1017/round1)")
    p.add_argument("--seed", type=int, default=1017)
    p.add_argument("--sampled-seed", type=int, default=424242)
    p.add_argument("--repeats", type=int, default=2)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--timeout", type=float, default=600.0)
    p.add_argument("--engine-container", default=None,
                   help="docker container for post-run log capture; "
                        "default: discover by published --base-url port")
    p.add_argument("--metrics", action="store_true", default=True)
    p.add_argument("--no-metrics", dest="metrics", action="store_false")
    p.add_argument("--quality-sampled", action="store_true", default=True)
    p.add_argument("--no-quality-sampled", dest="quality_sampled",
                   action="store_false")
    p.add_argument("--skip-quality", action="store_true")
    p.add_argument("--quality-mode", choices=("chat", "raw"),
                   default="chat",
                   help="quality transport: official chat template (default) "
                        "or legacy raw completion")
    p.add_argument("--quality-only", action="store_true",
                   help="run quality probes only; no warmup or timed requests")
    p.add_argument("--allow-prompt-drift", action="store_true")
    p.add_argument("--execute", action="store_true",
                   help="send serving requests; requires a granted window")
    p.add_argument("--plan-only", action="store_true")
    p.add_argument("--check", action="store_true",
                   help="tokenizer-only fixture/prompt audit; no completions")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--resume-timed", action="store_true",
                   help="reuse a complete timed.jsonl in --out-dir and run "
                        "only quality phases plus summary")
    p.add_argument("--compare", nargs=2, metavar=("DIR_A", "DIR_B"))
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir).resolve()

    if args.compare:
        compare_dir = out_dir
        compare_dir.mkdir(parents=True, exist_ok=True)
        result = compare_runs(Path(args.compare[0]).resolve(),
                              Path(args.compare[1]).resolve(),
                              compare_dir)
        print(f"matched {result['matched_cells']} cells; "
              f"wrote {compare_dir / 'compare.json'}")
        print((compare_dir / "compare.txt").read_text())
        return 0

    tok = Tokenizer(args.base_url, args.model, timeout=args.timeout)
    manifest, audit = build_prompt_manifest(tok)
    plan = make_plan(args.label, out_dir, args.repeats, args.seed,
                     args.warmup, manifest, audit)
    plan["base_url"] = args.base_url

    if args.check or args.plan_only or not args.execute:
        print("fixture/prompt audit (tokenizer only, no completions):")
        for row in audit:
            print(f"  {row['context_type']}: long={row['long_body_tokens']} "
                  f"short={row['short_body_tokens']} tokens")
            for variant in row["variants"]:
                for pair_id, need in variant["needs"].items():
                    status = "OK" if need["ok"] else "FAIL"
                    print(f"    {variant['prompt_id']} {pair_id}: "
                          f"need {need['body_tokens_needed']} "
                          f"have {need['body_tokens_available']} {status}")
        out_dir.mkdir(parents=True, exist_ok=True)
        write_prompts_jsonl(out_dir / "prompts.jsonl", manifest)
        write_json(out_dir / "matrix_plan.json", plan)
        print(f"\nprompts: {len(manifest) - len(plan['quality_cells'])} "
              f"timed exact-token + {len(plan['quality_cells'])} "
              f"quality full-body prompts built")
        print(f"timed requests: {len(plan['timed_order'])} "
              f"({len(plan['timed_cells'])} cells x {args.repeats} repeats)")
        print(f"quality requests: {plan['quality_request_count']} "
              f"(greedy + sampled subset)")
        print(f"plan: {out_dir / 'matrix_plan.json'}")
        if not args.execute:
            print("\nNo serving requests sent. Pass --execute inside a "
                  "granted exclusive window.")
        return 0

    if out_dir.exists() and not args.overwrite:
        if any((out_dir / name).exists() for name in (
                "timed.jsonl", "summary.json", "matrix_plan.json")):
            print(f"refusing to overwrite existing artifacts in {out_dir}; "
                  "use a unique --out-dir or --overwrite", file=sys.stderr)
            return 2
    out_dir.mkdir(parents=True, exist_ok=True)
    write_prompts_jsonl(out_dir / "prompts.jsonl", manifest)
    write_json(out_dir / "matrix_plan.json", plan)
    execute(args, out_dir, manifest, plan)
    return 0


if __name__ == "__main__":
    sys.exit(main())
