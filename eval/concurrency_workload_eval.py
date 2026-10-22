#!/usr/bin/env python3
"""Bounded, synchronized concurrency benchmark for the served V4.1 engine.

Campaign
--------
Four fixture types (code, prose, json, toolagent), two workload shapes
(prefill: 2048 prompt / 1 forced output; decode: 512 prompt / 2048 forced
output), four concurrency levels (1, 2, 4, 8), two measured repeats, and one
excluded warmup request per execution configuration.  That is 64 measured
groups and 240 measured requests.

Every group is a gate-synchronized release: all C requests are connected and
waiting on an asyncio gate; the gate release timestamp starts the group clock.
Groups run one at a time, after the engine reports idle.

Exactness
---------
Prompts are built with the served /tokenize + /detokenize endpoints (via the
existing ``dspark_workload_eval.Tokenizer``/``build_exact_prompt``), and the
same prompt is reused for all requests in a (type, shape).  Prefix caching is
required to be off.  Completion is forced with ``max_tokens == min_tokens``
and ``ignore_eos``.  The final SSE usage object is the completion-token
authority; ``return_token_ids`` supplies the actual chunk token IDs.  Prompt
tokens are verified from usage and from the first chunk's prompt_token_ids.
Forced continuations are timing evidence only, never quality evidence.

Engineering accounting
----------------------
* TTFT = first chunk carrying token IDs - request send.
* first_chunk_tokens = len(token_ids) in that first content chunk.
* decode_tokens_after_first = usage.completion_tokens - first_chunk_tokens.
* decode duration = last token chunk - first token chunk.  A chunk may carry
  several tokens; SSE chunk count is never used as a token count.
* engine prefill is request-scoped only: the engine's finished-request event
  (prefill_time_s, queue_time_s separate) joined by response id.  Median and
  peak engine prefill tokens/s are emitted only from those events.
* if request events are absent, the Prometheus per-request prefill histograms
  are used only for the group aggregate (sum computed tokens / sum prefill
  wall); median/peak stay null and the source is labelled
  ``engine_histogram_group_delta``.  TTFT is never relabelled as prefill and
  GPU time is never relabelled as engine prefill.

Usage
-----
  python3 eval/concurrency_workload_eval.py --list          # offline plan
  python3 eval/concurrency_workload_eval.py --self-smoke    # fake HTTP server
  python3 eval/concurrency_workload_eval.py \
      --base-url http://127.0.0.1:8001 \
      --out eval-out/concurrency_1m_native \
      --engine-event-file /tmp/afd_dump/cwb-engine-events.jsonl \
      --gpu-csv /tmp/afd_dump/gpu0.csv

The live path never starts, stops, or restarts an engine.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import random
import re
import statistics
import sys
import tempfile
import threading
import time
import urllib.parse
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from fixtures_contexts import FIXTURES  # noqa: E402
from dspark_workload_eval import (  # noqa: E402
    MODEL,
    Tokenizer,
    append_jsonl,
    build_exact_prompt,
    counter_total,
    fetch_metrics_text,
    fixture_digest,
    health_check,
    metric_value,
    now_iso,
    parse_metrics,
    safe_name,
    sha256_text,
    timed_payload,
    write_json,
)

BENCHMARK = "concurrency_workload_eval"
PLAN_SCHEMA = "cwb-plan-v1"
ROW_SCHEMA = "cwb-row-v1"
GROUP_SCHEMA = "cwb-group-v1"
SUMMARY_SCHEMA = "cwb-summary-v1"
ENGINE_EVENT_SCHEMA = "dspark-engine-request-event-v1"

ENGINE_PREFILL_DEFINITION = (
    "prefill_time_s = first_token_ts - scheduled_ts: a scheduled-to-first-"
    "token engine wall interval that includes the first generated token and "
    "scheduler/chunk interleaving. It is not exclusive GPU kernel time; "
    "queue_time_s and decode_time_s are separate engine intervals.")
ENGINE_EVENT_REQUIRED_FIELDS = (
    "request_id",
    "num_prompt_tokens",
    "num_generation_tokens",
    "num_cached_tokens",
    "prefill_time_s",
    "queue_time_s",
    "decode_time_s",
    "finished_ts",
)

# The fixture file already ships these documents and exact instruction tails.
# Reusing them avoids a second, divergent corpus or tokenizer convention.
TYPE_FIXTURES = {
    "code": ("code_generation", "codegen-a"),
    "prose": ("prose", "prose-a"),
    "json": ("json_schema_output", "json-a"),
    "toolagent": ("tool_agent_exchange", "tool-a"),
}
SHAPE_SPECS = {
    "prefill": {"prompt_tokens": 2048, "completion_tokens": 1},
    "decode": {"prompt_tokens": 512, "completion_tokens": 2048},
}
TYPE_ALIASES = {
    "code": "code",
    "prose": "prose",
    "json": "json",
    "toolagent": "toolagent",
    "tool-agent": "toolagent",
    "tool_agent": "toolagent",
    "toolagentexchange": "toolagent",
}
DEFAULT_CONCURRENCY = [1, 2, 4, 8]
DEFAULT_REPEATS = 2
DEFAULT_WARMUP = 1
DEFAULT_ENGINE_EVENT_FILE = "/tmp/afd_dump/cwb-engine-events.jsonl"

# Source-inspected K5 cache geometry.  The engine-reported num_gpu_blocks is
# authoritative for the boot actually measured; these constants only make the
# byte capacity and native-context admission floor explicit.  Override them
# with --cache-capacity-json when a different image/cache layout is booted.
K5_CACHE_EVIDENCE = {
    "kv_bytes_per_block": 117216,
    "blocks_per_request_at_8k": 2121,
    "blocks_8k_derivation": (
        "8 * 257 + 65 = 2121 blocks per 8192-token request "
        "(65 = 64 compressed + 1 compressor)"),
    "blocks_per_request_at_native_1m": 10289,
    "native_1m_derivation": (
        "8 * 262 + 8192 + 1 = 10289 blocks per 1048576-token request"),
    "benchmark_blocks_per_request": {
        "2048_prompt_1_output": 2113,
        "512_prompt_2048_output": 2117,
    },
    "benchmark_c8_blocks": {
        "2048_prompt_1_output": 16904,
        "512_prompt_2048_output": 16936,
    },
    "max_inflight_requests": 8192,
    "source": (
        "source-inspected K5 compressed KV layout; engine-reported "
        "num_gpu_blocks/block_size remain the measured-boot authority"),
}

METRIC_PREFIXES = (
    "vllm:cache_config_info",
    "vllm:kv_cache_usage_perc",
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_preemptions_total",
    "vllm:prefix_cache_hits_total",
    "vllm:prompt_tokens_cached_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:request_prefill_time_seconds",
    "vllm:request_prefill_kv_computed_tokens",
    "vllm:request_decode_time_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_success_total",
    "vllm:request_admission_rejections_total",
)

LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)\s*=\s*"((?:[^"\\]|\\.)*)"')


# ---------------------------------------------------------------------------
# small numeric helpers
# ---------------------------------------------------------------------------

def percentile(values, p: float):
    vals = sorted(float(v) for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * p
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return vals[lo]
    return vals[lo] * (hi - pos) + vals[hi] * (pos - lo)


def stats_summary(values, prefix: str, p95: bool = True) -> dict:
    vals = [float(v) for v in values if v is not None]
    out = {f"{prefix}_n": len(vals)}
    if not vals:
        out.update({
            f"{prefix}_median": None,
            f"{prefix}_peak": None,
            f"{prefix}_min": None,
        })
        if p95:
            out[f"{prefix}_p95"] = None
        return out
    out.update({
        f"{prefix}_median": statistics.median(vals),
        f"{prefix}_peak": max(vals),
        f"{prefix}_min": min(vals),
    })
    if p95:
        out[f"{prefix}_p95"] = percentile(vals, 0.95)
    return out


def rate_or_none(num, den):
    try:
        num = float(num)
        den = float(den)
    except (TypeError, ValueError):
        return None
    if den <= 0:
        return None
    return num / den


def parse_labels(key: str) -> dict:
    start = key.find("{")
    if start < 0:
        return {}
    return {
        m.group(1): m.group(2).replace('\\"', '"').replace("\\\\", "\\")
        for m in LABEL_RE.finditer(key[start:])
    }


def metric_series(series: dict, name: str) -> dict:
    return {k: v for k, v in series.items()
            if k == name or k.startswith(name + "{")}


def selected_metrics(series: dict) -> dict:
    return {k: v for k, v in series.items()
            if any(k.startswith(p) for p in METRIC_PREFIXES)}


def cache_config_from_series(series: dict) -> dict:
    for key in series:
        if key.startswith("vllm:cache_config_info"):
            return parse_labels(key)
    return {}


def snapshot_row(series: dict, t_mono: float) -> dict:
    return {
        "t_mono": t_mono,
        "kv_cache_usage_perc": metric_value(series, "vllm:kv_cache_usage_perc"),
        "num_requests_running": metric_value(
            series, "vllm:num_requests_running"),
        "num_requests_waiting": metric_value(
            series, "vllm:num_requests_waiting"),
        "num_preemptions_total": counter_total(
            series, "vllm:num_preemptions_total"),
        "prompt_tokens_total": counter_total(
            series, "vllm:prompt_tokens_total"),
        "generation_tokens_total": counter_total(
            series, "vllm:generation_tokens_total"),
        "prompt_tokens_cached_total": counter_total(
            series, "vllm:prompt_tokens_cached_total"),
        "prefix_cache_hits_total": counter_total(
            series, "vllm:prefix_cache_hits_total"),
    }


def histogram_deltas(before: dict, after: dict) -> dict:
    out = {}
    for name in (
        "vllm:request_prefill_time_seconds",
        "vllm:request_prefill_kv_computed_tokens",
        "vllm:request_decode_time_seconds",
        "vllm:request_queue_time_seconds",
    ):
        safe = name.split(":", 1)[1]
        out[f"{safe}_sum_delta"] = (
            counter_total(after, name + "_sum")
            - counter_total(before, name + "_sum"))
        out[f"{safe}_count_delta"] = (
            counter_total(after, name + "_count")
            - counter_total(before, name + "_count"))
    for name in (
        "vllm:prompt_tokens_total",
        "vllm:generation_tokens_total",
        "vllm:prompt_tokens_cached_total",
        "vllm:prefix_cache_hits_total",
        "vllm:num_preemptions_total",
    ):
        out[f"{name.split(':', 1)[1]}_delta"] = (
            counter_total(after, name) - counter_total(before, name))
    return out


# ---------------------------------------------------------------------------
# fixtures and exact prompts
# ---------------------------------------------------------------------------

def fixture_for_type(ctype: str):
    fixture_type, variant_id = TYPE_FIXTURES[ctype]
    fixture = next(f for f in FIXTURES if f["type"] == fixture_type)
    variant = next(
        v for v in fixture["variants"] if v["prompt_id"] == variant_id)
    return fixture, variant


def expanded_body(tok: Tokenizer, base_body: str, need: int) -> tuple[str, int]:
    """Repeat the fixture's own long body until the cut has enough tokens.

    The fixture documents are shorter than 2048 tokens, especially for prose.
    Repetition is deterministic runtime expansion of the existing fixture, not
    a new authored corpus; the exact builder still cuts at a token boundary.
    """
    body = base_body
    repeats = 1
    while tok.count(body) < need + 16 and repeats < 12:
        body = body + "\n\n" + base_body
        repeats += 1
    if tok.count(body) < need + 16:
        raise RuntimeError(
            f"fixture body still has {tok.count(body)} tokens after "
            f"{repeats} repeats; need {need + 16}")
    return body, repeats


def build_prompt_set(tok: Tokenizer, types: list[str]) -> dict:
    prompts = {}
    for ctype in types:
        fixture, variant = fixture_for_type(ctype)
        for shape, spec in SHAPE_SPECS.items():
            target = spec["prompt_tokens"]
            head = variant["head"]
            tail = variant["tail"]
            need = target - tok.count(head) - tok.count(tail)
            if need < 16:
                raise RuntimeError(
                    f"{ctype}/{shape}: head+tail leave only {need} body tokens")
            body, repeats = expanded_body(tok, fixture["long_body"], need)
            built = build_exact_prompt(tok, head, body, tail, target)
            if built["actual"] != target:
                raise RuntimeError(
                    f"{ctype}/{shape}: built {built['actual']} tokens, "
                    f"target {target}")
            ids = tok.ids(built["text"])
            if len(ids) != target:
                raise RuntimeError(
                    f"{ctype}/{shape}: /tokenize ids {len(ids)} != {target}")
            key = (ctype, shape)
            prompts[key] = {
                "context_type": ctype,
                "fixture_type": fixture["type"],
                "variant": variant["prompt_id"],
                "shape": shape,
                "target_prompt_tokens": target,
                "completion_tokens": spec["completion_tokens"],
                "prompt_text": built["text"],
                "prompt_sha256": built["sha256"],
                "prompt_chars": built["chars"],
                "prompt_token_ids": ids,
                "body_key": "long_body",
                "body_repeats": repeats,
                "head_tokens": built["head_tokens"],
                "tail_tokens": built["tail_tokens"],
                "body_tokens_used": built["body_tokens_used"],
                "connector": built["connector"],
                "boundary_score": built["boundary_score"],
                "fitting_iterations": built["iterations"],
                "add_special_tokens": False,
                "tokenizer_source": "/tokenize",
            }
    return prompts


# ---------------------------------------------------------------------------
# campaign plan
# ---------------------------------------------------------------------------

def normalize_types(values) -> list[str]:
    out = []
    for value in values:
        key = value.strip().lower()
        if key not in TYPE_ALIASES:
            raise SystemExit(
                f"unknown type {value!r}; choose from {sorted(TYPE_FIXTURES)}")
        ctype = TYPE_ALIASES[key]
        if ctype not in out:
            out.append(ctype)
    return out


def build_plan(types, shapes, concurrencies, repeats, warmup) -> dict:
    groups = []
    order = 0
    for repeat in range(1, repeats + 1):
        for ctype in types:
            for shape in shapes:
                for concurrency in concurrencies:
                    groups.append({
                        "group_id": (f"{ctype}-{shape}-c{concurrency}-"
                                     f"r{repeat}"),
                        "order_index": order,
                        "context_type": ctype,
                        "shape": shape,
                        "concurrency": concurrency,
                        "repeat": repeat,
                        "phase": "measured",
                        "requests": concurrency,
                        "prompt_tokens": SHAPE_SPECS[shape]["prompt_tokens"],
                        "completion_tokens": SHAPE_SPECS[shape][
                            "completion_tokens"],
                    })
                    order += 1
    warmups = []
    for ctype in types:
        for shape in shapes:
            for concurrency in concurrencies:
                for repeat in range(1, warmup + 1):
                    warmups.append({
                        "group_id": (f"warmup-{ctype}-{shape}-"
                                     f"c{concurrency}-w{repeat}"),
                        "context_type": ctype,
                        "shape": shape,
                        "concurrency": concurrency,
                        "repeat": repeat,
                        "phase": "warmup",
                        "requests": 1,
                        "prompt_tokens": SHAPE_SPECS[shape]["prompt_tokens"],
                        "completion_tokens": SHAPE_SPECS[shape][
                            "completion_tokens"],
                    })
    measured_requests = sum(g["requests"] for g in groups)
    return {
        "schema": PLAN_SCHEMA,
        "benchmark": BENCHMARK,
        "types": list(types),
        "shapes": shapes,
        "concurrency": list(concurrencies),
        "repeats": repeats,
        "warmup_per_configuration": warmup,
        "shape_specs": SHAPE_SPECS,
        "fixture_map": {
            ctype: {"fixture_type": TYPE_FIXTURES[ctype][0],
                    "variant": TYPE_FIXTURES[ctype][1]}
            for ctype in types
        },
        "groups": groups,
        "warmups": warmups,
        "totals": {
            "groups": len(groups),
            "measured_requests": measured_requests,
            "warmup_groups": len(warmups),
            "warmup_requests": len(warmups),
            "executions": len(types) * len(shapes) * len(concurrencies),
            "requests_per_execution_two_repeats": (
                sum(concurrencies) * repeats),
        },
        "forced": {
            "max_tokens_equals_min_tokens": True,
            "ignore_eos": True,
            "temperature": 0.0,
            "prefix_cache": "required_off",
            "invalid_for_quality": True,
        },
        "expected_240": {
            "formula": "4 types * 2 shapes * sum(C=1,2,4,8) * 2 repeats",
            "value": 4 * 2 * sum(DEFAULT_CONCURRENCY) * DEFAULT_REPEATS,
            "matches": measured_requests == 240,
        },
    }


def print_plan(plan: dict) -> None:
    print(f"{BENCHMARK}: {plan['totals']['groups']} measured groups, "
          f"{plan['totals']['measured_requests']} measured requests")
    print(f"types={plan['types']}")
    print(f"concurrency={plan['concurrency']} repeats={plan['repeats']} "
          f"warmup/config={plan['warmup_per_configuration']}")
    for shape, spec in SHAPE_SPECS.items():
        print(f"  {shape}: prompt={spec['prompt_tokens']} "
              f"completion={spec['completion_tokens']}")
    print(f"warmup requests={plan['totals']['warmup_requests']} (excluded)")
    print(f"240 formula matches: {plan['expected_240']['matches']}")
    if not plan["expected_240"]["matches"]:
        print("note: current --types/--concurrency/--repeats differ from the "
              "full 240-request campaign")


# ---------------------------------------------------------------------------
# engine event file (request-scoped timings)
# ---------------------------------------------------------------------------

class EngineEvents:
    """Incremental reader for the engine finished-request JSONL."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path) if path else None
        self.offset = 0
        self.buf = ""
        self.by_id = {}
        self.alias_candidates = {}
        self.bad_lines = 0
        self.lines = 0
        self.errors = []
        if self.path and self.path.exists():
            self.offset = self.path.stat().st_size

    def refresh(self) -> int:
        if not self.path or not self.path.exists():
            return 0
        try:
            with self.path.open("r", encoding="utf-8",
                                errors="replace") as handle:
                handle.seek(self.offset)
                data = handle.read()
                self.offset = handle.tell()
        except OSError as exc:
            self.errors.append(f"{type(exc).__name__}: {exc}")
            return 0
        if not data:
            return 0
        self.buf += data
        added = 0
        while "\n" in self.buf:
            line, self.buf = self.buf.split("\n", 1)
            line = line.strip()
            if not line:
                continue
            self.lines += 1
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                self.bad_lines += 1
                continue
            fields = dict(event)
            for nested in ("fields", "stats", "metrics"):
                if isinstance(event.get(nested), dict):
                    fields.update(event[nested])
            request_id = fields.get("request_id") or fields.get("id")
            if not request_id:
                self.bad_lines += 1
                continue
            rid = str(request_id)
            self.by_id[rid] = fields
            # vLLM appends the completion/choice index (n=1 here means "-0")
            # to finished-request IDs.  Alias only a purely numeric suffix and
            # fail closed if more than one event claims the same base.
            base, sep, suffix = rid.rpartition("-")
            if sep and base and suffix.isdigit():
                self.alias_candidates.setdefault(base, {})[rid] = fields
            added += 1
        return added

    def get(self, *request_ids):
        # Exact IDs always win over a suffix-derived alias.
        for request_id in request_ids:
            if request_id and str(request_id) in self.by_id:
                return self.by_id[str(request_id)]
        for request_id in request_ids:
            candidates = (
                self.alias_candidates.get(str(request_id))
                if request_id else None)
            if candidates and len(candidates) == 1:
                return next(iter(candidates.values()))
        return None

    async def wait_for(self, request_ids, timeout: float) -> bool:
        deadline = time.monotonic() + timeout
        while True:
            self.refresh()
            if all(self.get(rid, "cmpl-" + rid) for rid in request_ids):
                return True
            if time.monotonic() >= deadline:
                return False
            await asyncio.sleep(0.05)


def event_float(event: dict, *names):
    for name in names:
        if name in event and event[name] is not None:
            try:
                return float(event[name])
            except (TypeError, ValueError):
                continue
    return None


def event_int(event: dict, *names):
    value = event_float(event, *names)
    return int(value) if value is not None else None


def normalize_engine_event(event: dict | None) -> dict | None:
    if not event:
        return None
    return {
        "request_id": event.get("request_id") or event.get("id"),
        "finish_reason": event.get("finish_reason"),
        "num_prompt_tokens": event_int(
            event, "num_prompt_tokens", "prompt_tokens"),
        "num_generation_tokens": event_int(
            event, "num_generation_tokens", "generation_tokens"),
        "num_cached_tokens": event_int(
            event, "num_cached_tokens", "cached_tokens"),
        "num_preemptions": event_int(event, "num_preemptions", "preemptions"),
        "prefill_time_s": event_float(event, "prefill_time_s", "prefill_time"),
        "decode_time_s": event_float(event, "decode_time_s", "decode_time"),
        "queue_time_s": event_float(event, "queue_time_s", "queue_time"),
        "e2e_latency_s": event_float(
            event, "e2e_latency_s", "e2e_latency"),
        "queued_ts": event_float(event, "queued_ts"),
        "scheduled_ts": event_float(event, "scheduled_ts"),
        "first_token_ts": event_float(event, "first_token_ts"),
        "last_token_ts": event_float(event, "last_token_ts"),
        "finished_ts": event_float(event, "finished_ts"),
        "finished_iso": event.get("finished_iso"),
        "schema": event.get("schema"),
        "timestamp_source": event.get("timestamp_source"),
        "metric_definition": event.get("metric_definition"),
    }


# ---------------------------------------------------------------------------
# HTTP / SSE request path
# ---------------------------------------------------------------------------

def import_httpx():
    try:
        import httpx  # noqa: F401
    except ImportError as exc:  # pragma: no cover - environment guard
        raise SystemExit(
            "httpx is required for the async concurrency harness "
            "(pip install httpx); aiohttp/requests are not used") from exc
    return httpx


def http_limits(httpx, max_concurrency: int):
    return httpx.Limits(
        max_connections=max(16, max_concurrency * 3),
        max_keepalive_connections=max(8, max_concurrency * 2),
        keepalive_expiry=30.0,
    )


async def fetch_metrics_async(client, base_url: str) -> dict:
    response = await client.get(base_url.rstrip("/") + "/metrics",
                                timeout=10.0)
    response.raise_for_status()
    return parse_metrics(response.text)


async def fetch_models_async(client, base_url: str) -> dict:
    response = await client.get(base_url.rstrip("/") + "/v1/models",
                                timeout=10.0)
    response.raise_for_status()
    return response.json()


async def wait_idle_async(client, base_url: str, timeout: float) -> dict:
    deadline = time.monotonic() + timeout
    last = {}
    while True:
        try:
            series = await fetch_metrics_async(client, base_url)
            running = metric_value(series, "vllm:num_requests_running", 0.0)
            waiting = metric_value(series, "vllm:num_requests_waiting", 0.0)
            last = {"running": running, "waiting": waiting}
            if not running and not waiting:
                return last
        except Exception as exc:  # noqa: BLE001 - retry until deadline
            last = {"error": f"{type(exc).__name__}: {exc}"}
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"engine did not become idle within {timeout}s: {last}")
        await asyncio.sleep(0.1)


async def stream_request(client, base_url: str, payload: dict,
                         meta: dict, expected_prompt: int,
                         expected_completion: int) -> dict:
    """Consume one completion SSE stream and return one measured row."""
    request_id = payload["request_id"]
    row = {
        "schema": ROW_SCHEMA,
        "benchmark": BENCHMARK,
        "run_id": meta["run_id"],
        "group_id": meta["group_id"],
        "phase": meta["phase"],
        "context_type": meta["context_type"],
        "shape": meta["shape"],
        "concurrency": meta["concurrency"],
        "repeat": meta["repeat"],
        "request_index": meta["request_index"],
        "request_id": request_id,
        "model": payload.get("model"),
        "endpoint": "/v1/completions",
        "prompt_tokens_target": expected_prompt,
        "completion_tokens_target": expected_completion,
        "completion_tokens_usage": None,
        "prompt_tokens_usage": None,
        "observed_token_ids": 0,
        "first_chunk_tokens": None,
        "first_chunk_token_ids": None,
        "decode_tokens_after_first": None,
        "ttft_s": None,
        "e2e_s": None,
        "decode_duration_s": None,
        "decode_tok_s": None,
        "completion_tok_s_e2e": None,
        "chunk_count": 0,
        "token_chunk_count": 0,
        "usage_source": None,
        "response_id": None,
        "prompt_ids_returned": None,
        "prompt_ids_match": None,
        "token_ids_complete": True,
        "finish_reason": None,
        "length_limited": None,
        "exact_lengths": False,
        "http_status": None,
        "error": None,
        "output_chars": 0,
        "output_sha256": None,
        "engine_event_present": False,
        "engine_event_schema": None,
        "engine_event_missing_fields": [],
        "engine_timestamp_source": None,
        "engine_metric_definition": None,
        "engine_prefill_definition": ENGINE_PREFILL_DEFINITION,
        "engine_prefill_tokens_per_s": None,
        "engine_prefill_computed_tokens_per_s": None,
        "engine_decode_tokens_per_s": None,
        "engine": None,
        "gpu0_min_free_mib": None,
        "gpu0_max_util_pct": None,
        "valid": False,
        "validation_errors": [],
    }
    chunks = []
    output_parts = []
    usage = None
    first_token_time = None
    last_token_time = None
    first_chunk_ids = None
    first_prompt_ids = None
    stream_done = False

    t0 = time.monotonic()
    row["send_ts_rel_ms"] = round(
        (t0 - meta["group_start_mono"]) * 1000.0, 3)
    row["gate_wait_ms"] = row["send_ts_rel_ms"]
    url = base_url.rstrip("/") + "/v1/completions"
    try:
        async with client.stream(
                "POST", url, json=payload,
                headers={"Accept": "text/event-stream"}) as response:
            row["http_status"] = response.status_code
            if response.status_code != 200:
                body = (await response.aread()).decode(
                    "utf-8", errors="replace")[:1200]
                raise RuntimeError(
                    f"HTTP {response.status_code}: {body}")
            async for raw_line in response.aiter_lines():
                if not raw_line:
                    continue
                line = raw_line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    stream_done = True
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    row["token_ids_complete"] = False
                    row["validation_errors"].append("bad_sse_json")
                    continue
                arrived = time.monotonic()
                if event.get("id"):
                    row["response_id"] = event["id"]
                if event.get("error"):
                    raise RuntimeError(f"SSE error: {event['error']}")
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    finish = choice.get("finish_reason")
                    if finish:
                        row["finish_reason"] = finish
                    token_ids = choice.get("token_ids")
                    if choice.get("prompt_token_ids") is not None:
                        first_prompt_ids = choice["prompt_token_ids"]
                    text = choice.get("text") or ""
                    if text:
                        output_parts.append(text)
                    if token_ids is None:
                        if text:
                            row["token_ids_complete"] = False
                            row["validation_errors"].append(
                                "token_ids_missing")
                        continue
                    token_ids = list(token_ids)
                    n_tokens = len(token_ids)
                    row["chunk_count"] += 1
                    if n_tokens:
                        row["token_chunk_count"] += 1
                        row["observed_token_ids"] += n_tokens
                        if first_token_time is None:
                            first_token_time = arrived
                            row["ttft_s"] = arrived - t0
                            first_chunk_ids = token_ids
                            row["first_chunk_tokens"] = n_tokens
                        last_token_time = arrived
                    chunks.append({
                        "t_ms": round((arrived - t0) * 1000.0, 3),
                        "tokens": n_tokens,
                        "token_ids": token_ids,
                        "finish_reason": finish,
                    })
            ended = time.monotonic()
    except Exception as exc:  # noqa: BLE001 - every failure is a measured row
        ended = time.monotonic()
        row["error"] = f"{type(exc).__name__}: {exc}"
        row["validation_errors"].append("request_error")

    row["e2e_s"] = ended - t0
    row["end_rel_ms"] = round((ended - meta["group_start_mono"]) * 1000.0, 3)
    row["prompt_ids_returned"] = (
        len(first_prompt_ids) if first_prompt_ids is not None else None)
    row["first_chunk_token_ids"] = first_chunk_ids
    row["output_chars"] = sum(len(part) for part in output_parts)
    if output_parts:
        row["output_sha256"] = hashlib.sha256(
            "".join(output_parts).encode("utf-8")).hexdigest()
    row["stream_done"] = stream_done
    row["timeline"] = chunks

    if usage is not None:
        row["usage_source"] = "final_sse_usage"
        prompt_tokens = usage.get("prompt_tokens")
        completion_tokens = usage.get("completion_tokens")
        row["prompt_tokens_usage"] = prompt_tokens
        row["completion_tokens_usage"] = completion_tokens
        if first_prompt_ids is not None and (
                meta.get("prompt_token_ids") is not None):
            row["prompt_ids_match"] = (
                list(first_prompt_ids) == list(meta["prompt_token_ids"]))
        elif first_prompt_ids is not None:
            row["prompt_ids_match"] = (
                len(first_prompt_ids) == expected_prompt)
        if completion_tokens is not None:
            if first_chunk_ids is not None:
                row["decode_tokens_after_first"] = (
                    int(completion_tokens) - len(first_chunk_ids))
            if first_token_time is not None and last_token_time is not None:
                duration = last_token_time - first_token_time
                row["first_token_rel_ms"] = round(
                    (first_token_time - meta["group_start_mono"]) * 1000.0, 3)
                row["last_token_rel_ms"] = round(
                    (last_token_time - meta["group_start_mono"]) * 1000.0, 3)
                row["decode_duration_s"] = duration
                if row["decode_tokens_after_first"]:
                    row["decode_tok_s"] = rate_or_none(
                        row["decode_tokens_after_first"], duration)
            row["completion_tok_s_e2e"] = rate_or_none(
                completion_tokens, row["e2e_s"])
    else:
        row["validation_errors"].append("usage_missing")

    # --- validation ------------------------------------------------------
    errors = row["validation_errors"]
    if row["http_status"] not in (None, 200):
        errors.append("http_status")
    if row["error"]:
        pass
    if not row["stream_done"]:
        errors.append("stream_not_done")
    if usage is None:
        pass
    else:
        if row["prompt_tokens_usage"] != expected_prompt:
            errors.append("prompt_tokens_usage_mismatch")
        if row["completion_tokens_usage"] != expected_completion:
            errors.append("completion_tokens_usage_mismatch")
        if row["observed_token_ids"] != row["completion_tokens_usage"]:
            errors.append("observed_token_ids_mismatch")
        if row["prompt_ids_returned"] != expected_prompt:
            errors.append("prompt_ids_returned_mismatch")
        if row["prompt_ids_match"] is False:
            errors.append("prompt_ids_do_not_match")
    if not row["token_ids_complete"]:
        errors.append("token_ids_incomplete")
    if row["first_chunk_tokens"] is None:
        errors.append("no_content_tokens")
    if (row["decode_tokens_after_first"] is not None
            and row["decode_tokens_after_first"] < 0):
        errors.append("first_chunk_exceeds_usage")
    if row["finish_reason"] != "length":
        errors.append("finish_reason_not_length")
    row["length_limited"] = row["finish_reason"] == "length"
    row["exact_lengths"] = (
        row["prompt_tokens_usage"] == expected_prompt
        and row["completion_tokens_usage"] == expected_completion)
    row["valid"] = not errors
    if row["valid"]:
        row["validation_errors"] = []
    return row


# ---------------------------------------------------------------------------
# metrics sampling while a group runs
# ---------------------------------------------------------------------------

async def metrics_monitor(client, base_url: str, interval: float,
                          stop_event: asyncio.Event, samples: list) -> None:
    while not stop_event.is_set():
        try:
            series = await fetch_metrics_async(client, base_url)
            samples.append(snapshot_row(series, time.monotonic()))
        except Exception as exc:  # noqa: BLE001 - monitoring is best effort
            samples.append({
                "t_mono": time.monotonic(),
                "error": f"{type(exc).__name__}: {exc}",
            })
        try:
            await asyncio.wait_for(stop_event.wait(), timeout=interval)
        except asyncio.TimeoutError:
            continue


# ---------------------------------------------------------------------------
# group execution
# ---------------------------------------------------------------------------

def apply_engine_event(row: dict, event: dict | None, expected_prompt: int,
                       expected_completion: int,
                       require_engine_events: bool) -> None:
    """Attach one finished-request event and its validation status.

    The event is the authority for engine prefill/decode/queue timings.
    Client-side validation errors are preserved; engine-side errors are
    replaced so an offline replay can repair rows without inventing client
    evidence.
    """
    row["validation_errors"] = [
        error for error in row.get("validation_errors", [])
        if not str(error).startswith("engine_")]
    engine = normalize_engine_event(event)
    row["engine_event_present"] = engine is not None
    row["engine"] = engine
    row["engine_event_schema"] = event.get("schema") if event else None
    row["engine_timestamp_source"] = (
        event.get("timestamp_source") if event else None)
    row["engine_metric_definition"] = (
        event.get("metric_definition") if event else None)
    row["engine_event_missing_fields"] = [
        field for field in ENGINE_EVENT_REQUIRED_FIELDS
        if event is None or event.get(field) is None]
    if engine:
        row["engine_request_id"] = engine["request_id"]
        row["engine_prefill_time_s"] = engine["prefill_time_s"]
        row["engine_decode_time_s"] = engine["decode_time_s"]
        row["engine_queue_time_s"] = engine["queue_time_s"]
        row["engine_e2e_latency_s"] = engine["e2e_latency_s"]
        row["engine_num_prompt_tokens"] = engine["num_prompt_tokens"]
        row["engine_num_generation_tokens"] = engine[
            "num_generation_tokens"]
        row["engine_num_cached_tokens"] = engine["num_cached_tokens"]
        row["engine_num_preemptions"] = engine["num_preemptions"]
        if engine["prefill_time_s"] and engine["prefill_time_s"] > 0:
            if engine["num_prompt_tokens"] is not None:
                row["engine_prefill_tokens_per_s"] = (
                    engine["num_prompt_tokens"]
                    / engine["prefill_time_s"])
            cached = max(engine["num_cached_tokens"] or 0, 0)
            if engine["num_prompt_tokens"] is not None:
                row["engine_prefill_computed_tokens_per_s"] = (
                    max(engine["num_prompt_tokens"] - cached, 0)
                    / engine["prefill_time_s"])
        if (engine["decode_time_s"]
                and engine["decode_time_s"] > 0
                and engine["num_generation_tokens"]):
            row["engine_decode_tokens_per_s"] = (
                max(engine["num_generation_tokens"] - 1, 0)
                / engine["decode_time_s"])
        if engine["num_prompt_tokens"] != expected_prompt:
            row["validation_errors"].append(
                "engine_prompt_tokens_mismatch")
        if engine["num_generation_tokens"] != expected_completion:
            row["validation_errors"].append(
                "engine_generation_tokens_mismatch")
        if (engine["num_cached_tokens"] or 0) > 0:
            row["validation_errors"].append("engine_prefix_cache_hit")
        if engine["finish_reason"] != "length":
            row["validation_errors"].append(
                "engine_finish_reason_not_length")
        if require_engine_events:
            if row["engine_event_missing_fields"]:
                row["validation_errors"].append(
                    "engine_event_missing_fields")
            if event.get("schema") != ENGINE_EVENT_SCHEMA:
                row["validation_errors"].append(
                    "engine_event_schema_mismatch")
            definition = event.get("metric_definition") or ""
            if "scheduled-to-first-token" not in definition:
                row["validation_errors"].append(
                    "engine_prefill_definition_missing")
            if event.get("timestamp_source") != "req_stats":
                row["validation_errors"].append(
                    "engine_timestamp_source_not_req_stats")
            stamps = [
                engine["scheduled_ts"], engine["first_token_ts"],
                engine["last_token_ts"], engine["finished_ts"]]
            if all(value is not None for value in stamps):
                if not (stamps[0] <= stamps[1] <= stamps[2]
                        <= stamps[3] + 1e-6):
                    row["validation_errors"].append(
                        "engine_event_timestamp_order")
            else:
                row["validation_errors"].append(
                    "engine_event_timestamps_missing")
    elif require_engine_events:
        row["validation_errors"].append("engine_event_missing")
    row["valid"] = not row["validation_errors"]


async def run_group(client, base_url: str, group: dict, prompt: dict,
                    events: EngineEvents, args, run_id: str) -> tuple[dict, list]:
    ready = [asyncio.Event() for _ in range(group["concurrency"])]
    gate = asyncio.Event()
    rows = [None] * group["concurrency"]
    task_ids = []

    def payload_for(index: int) -> dict:
        request_id = (
            f"cwb-{run_id}-{safe_name(group['group_id'])}-i{index}-"
            f"{uuid.uuid4().hex[:8]}")
        payload = timed_payload(prompt["prompt_text"],
                                group["completion_tokens"], args.seed)
        payload["model"] = args.model
        payload["request_id"] = request_id
        payload["return_token_ids"] = True
        return payload

    async def one(index: int) -> None:
        payload = payload_for(index)
        task_ids.append(payload["request_id"])
        ready[index].set()
        await gate.wait()
        # Build meta only after the group gate has assigned its start clock;
        # reading gate_start earlier would capture 0.0.
        meta = {
            "run_id": run_id,
            "group_id": group["group_id"],
            "phase": "measured",
            "context_type": group["context_type"],
            "shape": group["shape"],
            "concurrency": group["concurrency"],
            "repeat": group["repeat"],
            "request_index": index,
            "group_start_mono": gate_start[0],
            "prompt_token_ids": prompt["prompt_token_ids"],
        }
        try:
            rows[index] = await stream_request(
                client, base_url, payload, meta,
                group["prompt_tokens"], group["completion_tokens"])
        except Exception as exc:  # noqa: BLE001 - never lose a group
            rows[index] = {
                "schema": ROW_SCHEMA, "benchmark": BENCHMARK,
                "run_id": run_id, "group_id": group["group_id"],
                "phase": "measured",
                "context_type": group["context_type"],
                "shape": group["shape"],
                "concurrency": group["concurrency"],
                "repeat": group["repeat"], "request_index": index,
                "request_id": payload.get("request_id"),
                "error": f"{type(exc).__name__}: {exc}",
                "valid": False,
                "validation_errors": ["request_error"],
            }

    gate_start = [0.0]
    tasks = [asyncio.create_task(one(i))
             for i in range(group["concurrency"])]
    await asyncio.gather(*(event.wait() for event in ready))
    group_start = time.monotonic()
    gate_start[0] = group_start
    release_iso = now_iso()
    gate.set()
    await asyncio.gather(*tasks, return_exceptions=True)
    group_end = time.monotonic()
    rows = [row for row in rows if row is not None]

    request_ids = [row.get("request_id") for row in rows if row.get("request_id")]
    events_ok = await events.wait_for(
        request_ids, args.engine_event_timeout) if request_ids else False
    events.refresh()
    for row in rows:
        response_id = row.get("response_id")
        request_id = row.get("request_id")
        event = events.get(response_id, "cmpl-" + request_id, request_id)
        apply_engine_event(
            row, event, group["prompt_tokens"], group["completion_tokens"],
            args.require_engine_events)

    group_row = {
        "schema": GROUP_SCHEMA,
        "benchmark": BENCHMARK,
        "run_id": run_id,
        "group_id": group["group_id"],
        "group_start_mono": group_start,
        "order_index": group["order_index"],
        "context_type": group["context_type"],
        "shape": group["shape"],
        "concurrency": group["concurrency"],
        "repeat": group["repeat"],
        "phase": "measured",
        "prompt_key": f"{group['context_type']}/{group['shape']}",
        "prompt_tokens": group["prompt_tokens"],
        "completion_tokens": group["completion_tokens"],
        "requests": len(rows),
        "valid_requests": sum(1 for row in rows if row.get("valid")),
        "invalid_requests": sum(1 for row in rows if not row.get("valid")),
        "release_iso": release_iso,
        "release_unix_s": time.time() - (time.monotonic() - group_start),
        "group_wall_s": group_end - group_start,
        "request_start_spread_ms": (
            max((row.get("send_ts_rel_ms") or 0.0) for row in rows)
            - min((row.get("send_ts_rel_ms") or 0.0) for row in rows)),
        "max_observed_inflight": 0,
        "active_concurrency_source": (
            "vllm:num_requests_running maximum; queued requests are recorded "
            "separately and never counted as active"),
        "active_concurrency_excludes_queued": True,
        "engine_prefill_group_wall_span_s": None,
        "engine_prefill_group_prompt_tokens": None,
        "engine_prefill_group_from_events_tokens_per_s": None,
        "engine_prefill_group_tokens_per_s": None,
        "engine_prefill_histogram_group_tokens_per_s": None,
        "engine_prefill_group_source": "none",
        "ttft_p50_s": percentile(
            [row.get("ttft_s") for row in rows], 0.5),
        "ttft_p95_s": percentile(
            [row.get("ttft_s") for row in rows], 0.95),
        "e2e_p95_s": percentile(
            [row.get("e2e_s") for row in rows], 0.95),
        "engine_events_expected": len(rows),
        "engine_events_present": sum(
            1 for row in rows if row.get("engine_event_present")),
        "engine_events_all_present": all(
            row.get("engine_event_present") for row in rows),
        "engine_events_ok": events_ok,
        "engine_event_file": str(events.path) if events.path else None,
        "engine_event_timestamp_sources": sorted({
            row.get("engine_timestamp_source") for row in rows
            if row.get("engine_event_present")}),
        "engine_prefill_definition": ENGINE_PREFILL_DEFINITION,
    }
    group_row.update(group_request_stats(rows))
    group_row["group_decode_tokens_per_s"] = group_decode_rate(rows)
    group_row["group_completion_tokens_per_s"] = rate_or_none(
        sum((row.get("completion_tokens_usage") or 0) for row in rows),
        group_row["group_wall_s"])
    group_row["group_engine_prefill_tokens_per_s"] = None
    group_row["engine_prefill_source"] = (
        "engine_events" if all(row.get("engine_event_present") for row in rows)
        else "none")

    before_series, after_series, samples = await sample_group_metrics(
        client, base_url, args, group_start, group_end)
    if before_series is not None:
        group_metrics = histogram_deltas(before_series, after_series)
        group_row["metrics"] = group_metrics
        group_row["kv_cache_usage_perc_max"] = max(
            (s.get("kv_cache_usage_perc") or 0.0)
            for s in samples if "error" not in s) if samples else None
        group_row["num_requests_running_max"] = max(
            (s.get("num_requests_running") or 0.0)
            for s in samples if "error" not in s) if samples else None
        group_row["num_requests_waiting_max"] = max(
            (s.get("num_requests_waiting") or 0.0)
            for s in samples if "error" not in s) if samples else None
        group_row["max_observed_inflight"] = int(
            group_row["num_requests_running_max"] or 0)
        group_row["preemptions_delta"] = group_metrics.get(
            "num_preemptions_total_delta")
        group_row["prefix_cache_hits_delta"] = group_metrics.get(
            "prefix_cache_hits_total_delta")
        group_row["prompt_tokens_cached_delta"] = group_metrics.get(
            "prompt_tokens_cached_total_delta")

        # Request-scoped events are the only source of median/peak.  When
        # absent, group histogram deltas still give an honest aggregate.
        computed_delta = group_metrics.get(
            "request_prefill_kv_computed_tokens_sum_delta")
        prefill_time_delta = group_metrics.get(
            "request_prefill_time_seconds_sum_delta")
        group_row["engine_prefill_histogram_group_tokens_per_s"] = (
            rate_or_none(computed_delta, prefill_time_delta))
        group_row["engine_prefill_hist_count_delta"] = group_metrics.get(
            "request_prefill_time_seconds_count_delta")
        group_row["engine_prefill_hist_time_sum_delta"] = prefill_time_delta
        if not group_row["engine_events_all_present"]:
            group_row["engine_prefill_source"] = "engine_histogram_group_delta"
        if (group_row["prefix_cache_hits_delta"] or 0) > 0 or (
                group_row["prompt_tokens_cached_delta"] or 0) > 0:
            for row in rows:
                row["validation_errors"].append("group_prefix_cache_delta")
                row["valid"] = False
            group_row["valid_requests"] = sum(
                1 for row in rows if row.get("valid"))
            group_row["invalid_requests"] = len(rows) - group_row[
                "valid_requests"]
    else:
        group_row["metrics"] = None
        group_row["engine_prefill_histogram_group_tokens_per_s"] = None

    # Per-request event values feed median/peak; group aggregate is separate.
    prefills = [
        row.get("engine_prefill_tokens_per_s") for row in rows
        if row.get("engine_prefill_tokens_per_s") is not None]
    decodes = [
        row.get("engine_decode_tokens_per_s") for row in rows
        if row.get("engine_decode_tokens_per_s") is not None]
    group_row["engine_prefill_tokens_per_s_median"] = (
        statistics.median(prefills) if prefills else None)
    group_row["engine_prefill_tokens_per_s_peak"] = (
        max(prefills) if prefills else None)
    group_row["engine_decode_tokens_per_s_median"] = (
        statistics.median(decodes) if decodes else None)
    group_row["engine_decode_tokens_per_s_peak"] = (
        max(decodes) if decodes else None)
    event_prompt_tokens = sum(
        row.get("engine_num_prompt_tokens") or 0
        for row in rows if row.get("engine_event_present"))
    event_ends = [
        (row.get("engine") or {}).get("first_token_ts") for row in rows
        if row.get("engine_event_present")]
    event_ends = [value for value in event_ends if value is not None]
    event_wall_span = (
        max(event_ends) - group_start if event_ends else None)
    event_group_rate = rate_or_none(event_prompt_tokens, event_wall_span)
    group_row["engine_prefill_group_wall_span_s"] = event_wall_span
    group_row["engine_prefill_group_prompt_tokens"] = event_prompt_tokens
    group_row["engine_prefill_group_from_events_tokens_per_s"] = (
        event_group_rate)
    group_row["engine_prefill_group_from_events_definition"] = (
        "sum of prompt tokens / (max first_token_ts - gate release) over the "
        "synchronized group: queue plus compute group wall, not a sum of "
        "per-request prefill times")
    group_row["engine_prefill_group_tokens_per_s"] = (
        event_group_rate
        if event_group_rate is not None
        else group_row.get("engine_prefill_histogram_group_tokens_per_s"))
    group_row["engine_prefill_group_source"] = (
        "engine_events" if event_group_rate is not None
        else ("engine_histogram_group_delta"
              if group_row.get("engine_prefill_histogram_group_tokens_per_s")
              else "none"))
    group_row["group_engine_prefill_tokens_per_s"] = (
        group_row["engine_prefill_group_tokens_per_s"])
    if not prefills and group_row["engine_prefill_group_tokens_per_s"]:
        group_row["engine_prefill_source"] = "engine_histogram_group_delta"
    group_row["validation_errors"] = sorted({
        error for row in rows for error in row.get("validation_errors", [])})
    group_row["valid"] = (
        all(row.get("valid") for row in rows)
        and (not args.require_engine_events
             or group_row["engine_events_all_present"])
    )
    return group_row, rows


def group_decode_rate(rows) -> float | None:
    pairs = [(row.get("first_token_rel_ms"), row.get("last_token_rel_ms"),
              row.get("decode_tokens_after_first") or 0)
             for row in rows
             if row.get("first_token_rel_ms") is not None
             and row.get("last_token_rel_ms") is not None]
    if not pairs:
        return None
    tokens = sum(pair[2] for pair in pairs)
    start = min(pair[0] for pair in pairs)
    end = max(pair[1] for pair in pairs)
    return rate_or_none(tokens, (end - start) / 1000.0)


def group_request_stats(rows) -> dict:
    prefill_rates = [
        row.get("engine_prefill_tokens_per_s") for row in rows
        if row.get("engine_prefill_tokens_per_s") is not None]
    decode_rates = [
        row.get("decode_tok_s") for row in rows
        if row.get("decode_tok_s") is not None]
    decode_counts = [
        row.get("decode_tokens_after_first") for row in rows
        if row.get("decode_tokens_after_first") is not None]
    out = {}
    out.update(stats_summary(prefill_rates, "engine_prefill_tokens_per_s"))
    out.update(stats_summary(decode_rates, "decode_tok_s"))
    out.update(stats_summary(decode_counts, "decode_tokens_after_first"))
    out.update(stats_summary(
        [row.get("ttft_s") for row in rows], "ttft_s"))
    out.update(stats_summary(
        [row.get("e2e_s") for row in rows], "e2e_s"))
    return out


async def sample_group_metrics(client, base_url: str, args,
                               group_start: float, group_end: float):
    if not args.metrics:
        return None, None, []
    try:
        before = await fetch_metrics_async(client, base_url)
    except Exception as exc:  # noqa: BLE001 - metrics are optional
        return None, None, [{"error": f"{type(exc).__name__}: {exc}"}]
    samples = []
    stop_event = asyncio.Event()
    monitor = asyncio.create_task(
        metrics_monitor(client, base_url, args.metrics_interval,
                        stop_event, samples))
    await asyncio.sleep(min(args.metrics_interval, 0.05))
    await asyncio.sleep(max(group_end - time.monotonic(), 0))
    stop_event.set()
    try:
        await asyncio.wait_for(monitor, timeout=5.0)
    except asyncio.TimeoutError:
        monitor.cancel()
    try:
        after = await fetch_metrics_async(client, base_url)
    except Exception:  # noqa: BLE001
        after = before
    return before, after, samples


# ---------------------------------------------------------------------------
# warmups
# ---------------------------------------------------------------------------

async def run_warmups(client, base_url: str, plan: dict, prompts: dict,
                      events: EngineEvents, args, run_id: str,
                      out_dir: Path) -> list:
    rows = []
    for warmup in plan["warmups"]:
        await wait_idle_async(client, base_url, args.idle_timeout)
        prompt = prompts[(warmup["context_type"], warmup["shape"])]
        request_id = (
            f"cwb-{run_id}-{safe_name(warmup['group_id'])}-"
            f"{uuid.uuid4().hex[:8]}")
        payload = timed_payload(prompt["prompt_text"],
                                warmup["completion_tokens"], args.seed)
        payload["model"] = args.model
        payload["request_id"] = request_id
        payload["return_token_ids"] = True
        meta = {
            "run_id": run_id,
            "group_id": warmup["group_id"],
            "phase": "warmup",
            "context_type": warmup["context_type"],
            "shape": warmup["shape"],
            "concurrency": warmup["concurrency"],
            "repeat": warmup["repeat"],
            "request_index": 0,
            "group_start_mono": time.monotonic(),
            "prompt_token_ids": prompt["prompt_token_ids"],
        }
        row = await stream_request(
            client, base_url, payload, meta, warmup["prompt_tokens"],
            warmup["completion_tokens"])
        await events.wait_for([request_id], min(args.engine_event_timeout, 5))
        engine = normalize_engine_event(
            events.get("cmpl-" + request_id, request_id))
        row["engine_event_present"] = engine is not None
        row["engine"] = engine
        if engine:
            row["engine_num_cached_tokens"] = engine["num_cached_tokens"]
            if engine["num_prompt_tokens"] != warmup["prompt_tokens"]:
                row["validation_errors"].append(
                    "warmup_engine_prompt_tokens_mismatch")
            if (engine["num_generation_tokens"]
                    != warmup["completion_tokens"]):
                row["validation_errors"].append(
                    "warmup_engine_generation_tokens_mismatch")
        row["valid"] = not row["validation_errors"]
        append_jsonl(out_dir / "warmup_rows.jsonl", row)
        rows.append(row)
        print(f"warmup {warmup['group_id']} ttft={row.get('ttft_s')} "
              f"e2e={row.get('e2e_s')} valid={row.get('valid')}")
    return rows


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def overlap_peak(intervals) -> int | None:
    """Maximum number of half-open intervals active at one timestamp."""
    events = []
    for start, end in intervals:
        if start is None or end is None:
            continue
        try:
            start = float(start)
            end = float(end)
        except (TypeError, ValueError):
            continue
        if end < start:
            continue
        events.append((start, 1))
        events.append((end, -1))
    if not events:
        return None
    # At equal timestamps an interval end is processed before a new start.
    events.sort(key=lambda item: (item[0], item[1]))
    current = 0
    peak = 0
    for _, delta in events:
        current += delta
        peak = max(peak, current)
    return peak


def group_actual_overlap(group: dict, rows: list) -> dict:
    active = []
    queued = []
    preemptions = 0
    event_rows = [row for row in rows if row.get("engine")]
    for row in event_rows:
        engine = row["engine"]
        scheduled = engine.get("scheduled_ts")
        end = engine.get("last_token_ts")
        if end is None:
            end = engine.get("finished_ts")
        if scheduled is not None and end is not None:
            active.append((scheduled, end))
        queued_at = engine.get("queued_ts")
        if queued_at is None and scheduled is not None:
            queue_time = engine.get("queue_time_s")
            if queue_time is not None:
                queued_at = scheduled - queue_time
        if queued_at is not None and scheduled is not None:
            queued.append((queued_at, scheduled))
        preemptions += int(engine.get("num_preemptions") or 0)
    expected = int(group.get("requests") or group.get("concurrency") or 0)
    active_peak = overlap_peak(active)
    queued_peak = overlap_peak(queued)
    concurrency = int(group.get("concurrency") or 0)
    fully = bool(
        active_peak is not None
        and concurrency
        and active_peak >= concurrency
        and len(event_rows) == expected)
    out = {
        "offered_concurrency": concurrency,
        "events_expected": expected,
        "events_seen": len(event_rows),
        "active_interval": "[scheduled_ts,last_token_ts) (finished_ts fallback)",
        "queued_interval": "[queued_ts,scheduled_ts)",
        "active_max": active_peak,
        "queued_max": queued_peak,
        "fully_overlapped": fully,
        "queueing_observed": bool(queued_peak),
        "preemptions_total": preemptions,
        "resident_guaranteed": bool(fully and preemptions == 0),
        "timestamp_source": "engine_finished_request_events",
    }
    if preemptions:
        out["warning"] = (
            "engine reported preemptions; active overlap does not prove "
            "simultaneous residency")
    return out


def group_prefill_event_rate(group: dict, rows: list):
    event_rows = [row for row in rows if row.get("engine")]
    prompt_tokens = sum(
        row["engine"].get("num_prompt_tokens") or 0 for row in event_rows)
    ends = [
        row["engine"].get("first_token_ts") for row in event_rows
        if row["engine"].get("first_token_ts") is not None]
    if not ends or not prompt_tokens:
        return None, None, None
    reference = group.get("group_start_mono")
    reference_source = "group_start_mono"
    if reference is None:
        queued = []
        for row in event_rows:
            engine = row["engine"]
            value = engine.get("queued_ts")
            if value is None and engine.get("scheduled_ts") is not None:
                queue_time = engine.get("queue_time_s")
                if queue_time is not None:
                    value = engine["scheduled_ts"] - queue_time
            if value is not None:
                queued.append(value)
        reference = min(queued) if queued else None
        reference_source = "min_queued_ts"
    wall = (
        max(ends) - float(reference)
        if reference is not None else None)
    return rate_or_none(prompt_tokens, wall), reference_source, wall


def refresh_group_rollups(rows: list, groups: list,
                          require_engine_events: bool) -> None:
    rows_by_group = {}
    for row in rows:
        rows_by_group.setdefault(row.get("group_id"), []).append(row)
    for group in groups:
        group_rows = rows_by_group.get(group.get("group_id"), [])
        group["requests"] = len(group_rows)
        group["valid_requests"] = sum(
            1 for row in group_rows if row.get("valid"))
        group["invalid_requests"] = (
            len(group_rows) - group["valid_requests"])
        group["engine_events_expected"] = len(group_rows)
        group["engine_events_present"] = sum(
            1 for row in group_rows if row.get("engine_event_present"))
        group["engine_events_all_present"] = all(
            row.get("engine_event_present") for row in group_rows)
        group["engine_events_ok"] = group["engine_events_all_present"]
        group["engine_event_timestamp_sources"] = sorted({
            row.get("engine_timestamp_source") for row in group_rows
            if row.get("engine_event_present")})
        errors = set(group.get("validation_errors") or [])
        for row in group_rows:
            errors.update(row.get("validation_errors") or [])
        group["validation_errors"] = sorted(errors)
        group["valid"] = bool(group_rows) and all(
            row.get("valid") for row in group_rows) and (
                not require_engine_events
                or group["engine_events_all_present"])
        if group.get("group_wall_s") is None:
            ends = [
                row.get("end_rel_ms") for row in group_rows
                if row.get("end_rel_ms") is not None]
            starts = [
                row.get("send_ts_rel_ms") for row in group_rows
                if row.get("send_ts_rel_ms") is not None]
            if ends and starts:
                group["group_wall_s"] = (max(ends) - min(starts)) / 1000.0
        group.update(group_request_stats(group_rows))
        group["group_decode_tokens_per_s"] = (
            group_decode_rate(group_rows)
            if group.get("shape") == "decode" else None)
        group["group_completion_tokens_per_s"] = rate_or_none(
            sum((row.get("completion_tokens_usage") or 0)
                for row in group_rows),
            group.get("group_wall_s"))
        group["engine_prefill_source"] = (
            "engine_events" if group["engine_events_all_present"] else "none")


def annotate_group_activity(rows: list, groups: list) -> None:
    rows_by_group = {}
    for row in rows:
        rows_by_group.setdefault(row.get("group_id"), []).append(row)
    for group in groups:
        group_rows = rows_by_group.get(group.get("group_id"), [])
        group["actual_overlap"] = group_actual_overlap(group, group_rows)
        if group.get("engine_prefill_group_tokens_per_s") is None:
            rate, source, wall = group_prefill_event_rate(group, group_rows)
            group["engine_prefill_group_tokens_per_s"] = rate
            group["engine_prefill_group_from_events_tokens_per_s"] = rate
            group["engine_prefill_group_wall_span_s"] = wall
            group["engine_prefill_group_reference"] = source
            if rate is not None:
                group["engine_prefill_source"] = "engine_events"
                group["engine_prefill_group_definition"] = (
                    "sum of prompt tokens / (max first_token_ts - "
                    f"{source}); reference is not gate release when the "
                    "run predates group_start_mono")
        else:
            group["engine_prefill_group_reference"] = "gate_release_runtime"
        group["group_engine_prefill_tokens_per_s"] = (
            group.get("engine_prefill_group_tokens_per_s"))
        if group.get("shape") == "prefill":
            group["group_decode_tokens_per_s"] = None


def aggregate_rows(rows, group_rows, key_fields, group_key_fields) -> dict:
    buckets = {}
    for row in rows:
        key = tuple(row.get(field) for field in key_fields)
        buckets.setdefault(key, {"rows": [], "groups": []})["rows"].append(row)
    for group in group_rows:
        key = tuple(group.get(field) for field in group_key_fields)
        if key in buckets:
            buckets[key]["groups"].append(group)
    out = {}
    for key, bucket in buckets.items():
        rows_here = bucket["rows"]
        groups_here = bucket["groups"]
        valid = [row for row in rows_here if row.get("valid")]
        valid_groups = [group for group in groups_here if group.get("valid")]
        decode_groups = [
            group for group in valid_groups if group.get("shape") == "decode"]
        prefill_groups = [
            group for group in valid_groups if group.get("shape") == "prefill"]
        decode_rows = [
            row for row in valid if row.get("shape") == "decode"]
        running = [
            group.get("num_requests_running_max") for group in groups_here
            if group.get("num_requests_running_max") is not None]
        waiting = [
            group.get("num_requests_waiting_max") for group in groups_here
            if group.get("num_requests_waiting_max") is not None]
        group_decode = [
            group.get("group_decode_tokens_per_s")
            for group in decode_groups
            if group.get("group_decode_tokens_per_s") is not None]
        group_completion = [
            group.get("group_completion_tokens_per_s")
            for group in decode_groups
            if group.get("group_completion_tokens_per_s") is not None]
        group_prefill = [
            group.get("engine_prefill_group_tokens_per_s")
            for group in prefill_groups
            if group.get("engine_prefill_group_tokens_per_s") is not None]
        overlaps = [
            group.get("actual_overlap") for group in valid_groups
            if group.get("actual_overlap")]
        active_values = [
            overlap.get("active_max") for overlap in overlaps
            if overlap.get("active_max") is not None]
        queued_values = [
            overlap.get("queued_max") for overlap in overlaps
            if overlap.get("queued_max") is not None]
        entry = {
            "key": dict(zip(key_fields, key)),
            "requests": len(rows_here),
            "valid_requests": len(valid),
            "valid_rate": (len(valid) / len(rows_here)) if rows_here else None,
            "groups": len(groups_here),
            "valid_groups": len(valid_groups),
            "group_wall_s_median": (
                statistics.median([
                    group["group_wall_s"] for group in valid_groups
                    if group.get("group_wall_s") is not None])
                if any(group.get("group_wall_s") is not None
                       for group in valid_groups) else None),
            "group_decode_tokens_per_s_median": (
                statistics.median(group_decode) if group_decode else None),
            "group_decode_tokens_per_s_peak": (
                max(group_decode) if group_decode else None),
            "group_completion_tokens_per_s_median": (
                statistics.median(group_completion)
                if group_completion else None),
            "group_prefill_tokens_per_s_median": (
                statistics.median(group_prefill) if group_prefill else None),
            "group_prefill_tokens_per_s_peak": (
                max(group_prefill) if group_prefill else None),
            "group_prefill_groups": len(prefill_groups),
            "group_decode_groups": len(decode_groups),
        }
        entry.update(stats_summary(
            [row.get("engine_prefill_tokens_per_s") for row in valid],
            "engine_prefill_tokens_per_s"))
        entry.update(stats_summary(
            [row.get("engine_prefill_computed_tokens_per_s")
             for row in valid],
            "engine_prefill_computed_tokens_per_s"))
        entry.update(stats_summary(
            [row.get("engine_decode_tokens_per_s") for row in decode_rows],
            "engine_decode_tokens_per_s"))
        entry.update(stats_summary(
            [row.get("decode_tok_s") for row in decode_rows],
            "decode_tok_s"))
        entry.update(stats_summary(
            [row.get("decode_tokens_after_first") for row in decode_rows],
            "decode_tokens_after_first"))
        entry.update(stats_summary(
            [row.get("ttft_s") for row in valid], "ttft_s"))
        entry.update(stats_summary(
            [row.get("e2e_s") for row in valid], "e2e_s"))
        entry["engine_prefill_source"] = sorted({
            group.get("engine_prefill_source") for group in groups_here
            if group.get("engine_prefill_source")})
        entry["actual_overlap"] = {
            "groups": len(valid_groups),
            "groups_with_events": len(overlaps),
            "groups_fully_overlapped": sum(
                1 for overlap in overlaps
                if overlap.get("fully_overlapped")),
            "groups_queueing_observed": sum(
                1 for overlap in overlaps
                if overlap.get("queueing_observed")),
            "preemptions_total": sum(
                int(overlap.get("preemptions_total") or 0)
                for overlap in overlaps),
            "active_interval": (
                "[scheduled_ts,last_token_ts) (finished_ts fallback)"),
            "queued_interval": "[queued_ts,scheduled_ts)",
            "source": "engine_finished_request_events",
        }
        entry["actual_overlap"].update(
            stats_summary(active_values, "active_max", p95=False))
        entry["actual_overlap"].update(
            stats_summary(queued_values, "queued_max", p95=False))
        entry["activity"] = {
            "sampling": "post_group_only",
            "proves_active_overlap": False,
            "groups_total": len(groups_here),
            "groups_metrics_present": sum(
                1 for group in groups_here
                if group.get("metrics") is not None),
            "observed_running_max": max(running) if running else None,
            "observed_waiting_max": max(waiting) if waiting else None,
            "observed_running_max_values": running,
            "observed_waiting_max_values": waiting,
            "groups_with_waiting": sum(
                1 for value in waiting if value and value > 0),
            "caveat": (
                "vllm:num_requests_running/waiting are sampled after the "
                "group gate has ended; they are post-group samples, not "
                "proof of active overlap inside the group."),
        }
        out["|".join(str(part) for part in key)] = entry
    return out


def cache_capacity_summary(cache_config: dict, max_model_len,
                           tested_lengths, tested_output_lengths,
                           expected_native_context, capacity_json_path,
                           tested_concurrency=None,
                           allow_non_native_context=False):
    def as_int(value):
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None

    def as_float(value):
        try:
            return float(str(value))
        except (TypeError, ValueError):
            return None

    evidence = dict(K5_CACHE_EVIDENCE)
    out_evidence = None
    evidence_error = None
    if capacity_json_path:
        path = Path(capacity_json_path)
        if path.exists():
            try:
                loaded = json.loads(path.read_text())
                out_evidence = loaded
                if isinstance(loaded, dict):
                    # Runtime-provided recipe evidence is authoritative; only
                    # fall back to the source-image stride/source labels.
                    evidence = dict(loaded)
                    evidence.setdefault(
                        "kv_bytes_per_block",
                        K5_CACHE_EVIDENCE["kv_bytes_per_block"])
                    evidence.setdefault(
                        "source", K5_CACHE_EVIDENCE["source"])
            except (OSError, json.JSONDecodeError, TypeError) as exc:
                out_evidence = None
                evidence_error = f"{type(exc).__name__}: {exc}"
        else:
            out_evidence = None
            evidence_error = f"missing: {path}"
    else:
        out_evidence = None
        evidence_error = None

    def cdiv(value, divisor):
        return (int(value) + int(divisor) - 1) // int(divisor)

    blocks = as_int(cache_config.get("num_gpu_blocks"))
    block_size = as_int(cache_config.get("block_size"))
    bs = block_size or 128
    engine_tokens = as_int(cache_config.get("kv_cache_size_tokens"))
    logical_product = (
        blocks * block_size if blocks is not None and block_size else None)
    stride = as_int(evidence.get("kv_bytes_per_block"))
    cache_bytes = blocks * stride if blocks is not None and stride else None
    max_batched = as_int(evidence.get("max_num_batched_tokens"))
    max_in_flight = as_int(evidence.get("max_in_flight_requests"))
    if max_in_flight is None and max_batched:
        max_in_flight = 2 * max_batched
    swa_group = as_int(evidence.get("swa_blocks_per_group"))
    if swa_group is None and max_in_flight:
        swa_group = cdiv(127 + 6 + max_in_flight, 32) + 1
    blocks_8k = as_int(evidence.get("blocks_per_request_at_8k"))
    if blocks_8k is None and swa_group:
        blocks_8k = 8 * swa_group + cdiv(8192, bs) + 1
    native_len = as_int(expected_native_context) or 1048576
    blocks_native = as_int(
        evidence.get("blocks_per_request_at_native_1m"))
    if blocks_native is None and swa_group:
        blocks_native = 8 * swa_group + cdiv(native_len, bs) + 1
    conc_8k = (blocks / blocks_8k if blocks is not None and blocks_8k
               else None)
    conc_native = (
        blocks / blocks_native
        if blocks is not None and blocks_native else None)
    engine_concurrency_1m = (
        engine_tokens / max_model_len
        if engine_tokens is not None and max_model_len else None)
    engine_concurrency_8k = (
        engine_tokens / 8192 if engine_tokens is not None else None)
    benchmark_blocks = dict(
        evidence.get("benchmark_blocks_per_request") or {})
    if not benchmark_blocks and swa_group:
        for prompt, completion in zip(tested_lengths, tested_output_lengths):
            total = prompt if completion <= 1 else prompt + completion
            benchmark_blocks[
                f"{prompt}_prompt_{completion}_output"] = (
                    8 * swa_group + cdiv(total, bs) + 1)
    benchmark_c8_blocks = dict(evidence.get("benchmark_c8_blocks") or {})
    if not benchmark_c8_blocks and benchmark_blocks:
        benchmark_c8_blocks = {
            key: value * 8 for key, value in benchmark_blocks.items()}
    benchmark_peak_c8 = (
        max(benchmark_c8_blocks.values()) if benchmark_c8_blocks else None)
    out = {
        "cache_config": cache_config,
        "formula": {
            "capacity_authority": (
                "per-length block groups (SWA layers + compressed/compressor "
                "blocks); num_gpu_blocks * block_size is a logical product, "
                "not KV capacity"),
            "num_gpu_blocks": blocks,
            "block_size": block_size,
            "logical_block_product": logical_product,
            "logical_block_product_is_capacity": False,
            "engine_reported_kv_cache_size_tokens": engine_tokens,
            "engine_reported_max_concurrency_at_8k": engine_concurrency_8k,
            "engine_reported_max_concurrency_at_max_model_len": (
                engine_concurrency_1m),
            "engine_reported_max_num_seqs": cache_config.get("max_num_seqs"),
            "engine_reported_max_num_batched_tokens": cache_config.get(
                "max_num_batched_tokens"),
            "max_num_batched_tokens": max_batched,
            "max_in_flight_requests": max_in_flight,
            "swa_blocks_per_group": swa_group,
            "swa_blocks_formula": (
                "cdiv(127 + 6 + max_in_flight_requests, 32) + 1"),
            "native_1m_blocks_formula": (
                "8 * swa_blocks_per_group + "
                "cdiv(expected_native_context, block_size) + 1"),
            "kv_bytes_per_block_stride": stride,
            "cache_bytes_from_stride": cache_bytes,
            "cache_gib_from_stride": (
                cache_bytes / (1024 ** 3) if cache_bytes is not None else None),
            "blocks_per_request_at_8k": blocks_8k,
            "eight_k_max_concurrency_from_blocks": conc_8k,
            "blocks_per_request_at_native_1m": blocks_native,
            "native_1m_max_concurrency_from_blocks": conc_native,
            "benchmark_blocks_per_request": benchmark_blocks,
            "benchmark_c8_blocks": benchmark_c8_blocks,
            "benchmark_peak_c8_blocks": benchmark_peak_c8,
            "benchmark_peak_fits_actual_pool": (
                blocks >= benchmark_peak_c8
                if blocks is not None and benchmark_peak_c8 is not None
                else None),
            "kv_cache_memory_bytes_label": cache_config.get(
                "kv_cache_memory_bytes"),
            "note": (
                "Engine kv_cache_size_tokens is authoritative for admission. "
                "Per-length block groups are source-inspected evidence; the "
                "117216-byte stride is not a uniform token capacity. Override "
                "via --cache-capacity-json for a different image."),
        },
        "tested_lengths": sorted(set(tested_lengths)),
        "distinct_tested_prompt_lengths": len(set(tested_lengths)),
        "tested_output_lengths": sorted(set(tested_output_lengths)),
        "tested_total_token_lengths": sorted(set(
            prompt + output
            for prompt, output in zip(tested_lengths, tested_output_lengths))),
        "tested_max_prompt_tokens": max(tested_lengths) if tested_lengths
        else None,
        "tested_max_total_tokens": max(
            prompt + output
            for prompt, output in zip(tested_lengths, tested_output_lengths))
        if tested_lengths and tested_output_lengths else None,
        "tested_concurrency": tested_concurrency,
        "tested_peak_concurrent_tokens": (
            max(prompt + output
                for prompt, output in zip(
                    tested_lengths, tested_output_lengths))
            * tested_concurrency
            if tested_lengths and tested_output_lengths
            and tested_concurrency else None),
        "native_context": {
            "max_model_len": max_model_len,
            "expected_native_context": expected_native_context,
            "matches_expected": (
                max_model_len == expected_native_context
                if max_model_len is not None else None),
            "allow_non_native_context": bool(allow_non_native_context),
            "admission_cap_only": True,
            "proven_1m": False,
            "max_prompt_tested": max(tested_lengths) if tested_lengths
            else None,
            "note": (
                "Native context is an admission cap. This campaign tests "
                "prompts of at most 2048 tokens; it does not prove a 1M "
                "prompt or long-context behavior."),
        },
        "prefix_cache_enabled": cache_config.get("enable_prefix_caching"),
        "gpu_memory_utilization": as_float(
            cache_config.get("gpu_memory_utilization")),
        "cache_dtype": cache_config.get("cache_dtype"),
    }
    out["cache_geometry_evidence"] = evidence
    out["evidence"] = out_evidence
    if evidence_error:
        out["evidence_error"] = evidence_error
    return out


# ---------------------------------------------------------------------------
# GPU CSV ingestion
# ---------------------------------------------------------------------------

def parse_timestamp(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        pass
    else:
        if number > 1e17:
            return number / 1e9
        if number > 1e14:
            return number / 1e6
        if number > 1e11:
            return number / 1e3
        return number
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def pick(record: dict, *names):
    for name in names:
        if name in record and record[name] not in (None, ""):
            return record[name]
    return None


def ingest_gpu_csv(path: str | None, label_filter: str | None,
                   groups: list, phases: list) -> dict:
    if not path:
        return {"status": "not_provided"}
    csv_path = Path(path)
    if not csv_path.exists():
        return {"status": "missing", "path": str(csv_path)}
    rows = []
    try:
        with csv_path.open(newline="", encoding="utf-8") as handle:
            data = list(csv.reader(handle))
        if not data:
            return {"status": "no_rows", "path": str(csv_path)}
        header = [str(cell).strip().lower() for cell in data[0]]
        has_header = bool(
            {"timestamp", "time", "ts", "iso_time", "datetime"}
            & set(header))
        if has_header:
            records = [dict(zip(header, values)) for values in data[1:]]
        else:
            # nvidia-smi sampler rows:
            # iso_timestamp,label,used_mib,free_mib,util_pct,sm_clock,power_w
            records = []
            for values in data:
                if len(values) < 4:
                    continue
                records.append({
                    "timestamp": values[0],
                    "label": values[1],
                    "mem_used_mib": values[2],
                    "mem_free_mib": values[3],
                    "util_pct": values[4] if len(values) > 4 else None,
                    "raw": values,
                })
        for record in records:
            normalized = {
                str(key).strip().lower(): value
                for key, value in record.items()
            }
            ts = parse_timestamp(pick(
                normalized, "timestamp", "time", "ts", "iso_time",
                "datetime"))
            if ts is None:
                continue
            gpu = pick(normalized, "gpu", "gpu_index", "index",
                       "device", "gpu_id")
            try:
                gpu_int = int(float(gpu)) if gpu is not None else 0
            except (TypeError, ValueError):
                gpu_int = 0
            if gpu_int != 0:
                continue
            label = pick(normalized, "phase", "label", "stage", "tag")
            if label_filter and label != label_filter:
                continue
            free = pick(normalized, "mem_free_mib", "memory_free_mib",
                        "free_mib", "free", "memory_free")
            used = pick(normalized, "mem_used_mib", "memory_used_mib",
                        "used_mib", "used", "memory_used")
            util = pick(normalized, "util_pct", "gpu_util",
                        "utilization", "gpu_utilization")
            try:
                free_f = float(free) if free is not None else None
            except (TypeError, ValueError):
                free_f = None
            try:
                used_f = float(used) if used is not None else None
            except (TypeError, ValueError):
                used_f = None
            try:
                util_f = float(util) if util is not None else None
            except (TypeError, ValueError):
                util_f = None
            rows.append({
                "timestamp": ts,
                "gpu": gpu_int,
                "label": label,
                "mem_free_mib": free_f,
                "mem_used_mib": used_f,
                "util_pct": util_f,
                "raw": normalized,
            })
    except (OSError, csv.Error) as exc:
        return {"status": "error", "path": str(csv_path),
                "error": f"{type(exc).__name__}: {exc}"}
    if not rows:
        return {"status": "no_rows", "path": str(csv_path)}

    def window(start_s, end_s):
        sample = [row for row in rows
                  if row["timestamp"] is not None
                  and start_s is not None and end_s is not None
                  and start_s <= row["timestamp"] <= end_s]
        frees = [row["mem_free_mib"] for row in sample
                 if row["mem_free_mib"] is not None]
        used = [row["mem_used_mib"] for row in sample
                if row["mem_used_mib"] is not None]
        util = [row["util_pct"] for row in sample
                if row["util_pct"] is not None]
        return {
            "samples": len(sample),
            "min_free_mib": min(frees) if frees else None,
            "max_used_mib": max(used) if used else None,
            "max_util_pct": max(util) if util else None,
            "first_ts": sample[0]["timestamp"] if sample else None,
            "last_ts": sample[-1]["timestamp"] if sample else None,
        }

    by_group = {}
    for group in groups:
        by_group[group["group_id"]] = window(
            group.get("release_unix_s"),
            (group.get("release_unix_s") or 0)
            + (group.get("group_wall_s") or 0))
    by_phase = {}
    for phase in phases:
        if phase.get("start_unix_s") is None or phase.get("end_unix_s") is None:
            continue
        name = phase["name"]
        window_row = window(phase["start_unix_s"], phase["end_unix_s"])
        if name not in by_phase:
            by_phase[name] = window_row
        else:
            existing = by_phase[name]
            existing["samples"] += window_row["samples"]
            for field in ("min_free_mib",):
                values = [existing.get(field), window_row.get(field)]
                values = [value for value in values if value is not None]
                existing[field] = min(values) if values else None
            for field in ("max_used_mib", "max_util_pct"):
                values = [existing.get(field), window_row.get(field)]
                values = [value for value in values if value is not None]
                existing[field] = max(values) if values else None
    by_label = {}
    for row in rows:
        label = row.get("label")
        if not label:
            continue
        entry = by_label.setdefault(label, {"samples": 0,
                                            "free": [], "used": [],
                                            "util": []})
        entry["samples"] += 1
        if row["mem_free_mib"] is not None:
            entry["free"].append(row["mem_free_mib"])
        if row["mem_used_mib"] is not None:
            entry["used"].append(row["mem_used_mib"])
        if row["util_pct"] is not None:
            entry["util"].append(row["util_pct"])
    label_summary = {
        label: {
            "samples": entry["samples"],
            "min_free_mib": min(entry["free"]) if entry["free"] else None,
            "max_used_mib": max(entry["used"]) if entry["used"] else None,
            "max_util_pct": max(entry["util"]) if entry["util"] else None,
        }
        for label, entry in by_label.items()
    }
    all_frees = [row["mem_free_mib"] for row in rows
                 if row["mem_free_mib"] is not None]
    measured_start = min(
        (group["release_unix_s"] for group in groups
         if group.get("release_unix_s") is not None), default=None)
    measured_end = max(
        ((group.get("release_unix_s") or 0)
         + (group.get("group_wall_s") or 0) for group in groups),
        default=None)
    measured = window(measured_start, measured_end) if groups else {
        "samples": 0, "min_free_mib": None}
    return {
        "status": "ok",
        "path": str(csv_path),
        "rows_used_gpu0": len(rows),
        "min_free_mib": min(all_frees) if all_frees else None,
        "measured": measured,
        "by_group": by_group,
        "by_phase": by_phase,
        "by_label": label_summary,
    }


# ---------------------------------------------------------------------------
# summary rendering
# ---------------------------------------------------------------------------

def markdown_summary(summary: dict) -> str:
    capacity = summary["cache_capacity"]
    overlap = summary.get("actual_overlap") or {}
    c8 = overlap.get("c8_decode") or {}
    lines = [
        f"# Concurrency workload summary ({summary['run_id']})",
        "",
        f"- groups: {summary['totals']['measured_groups']} measured "
        f"({summary['totals']['valid_groups']} valid), "
        f"{summary['totals']['warmup_requests']} warmup requests excluded",
        f"- measured requests: {summary['totals']['measured_requests']} "
        f"({summary['totals']['valid_requests']} valid)",
        f"- engine prefill source: {summary['engine_prefill']['source']} "
        f"(primary shape: {summary['engine_prefill'].get('primary_shape')})",
        f"- tested prompt lengths: {capacity['tested_lengths']}",
        f"- native max_model_len: "
        f"{capacity['native_context']['max_model_len']} "
        f"(admission cap only; proven_1m="
        f"{capacity['native_context']['proven_1m']})",
        "- prefix cache: off; prompt lengths are forced from exact tokenizer "
        "counts. These are performance measurements, not a quality sweep.",
        (lambda specs: (
            "- shapes: prefill = "
            f"{specs['prefill']['prompt_tokens']} prompt / "
            f"{specs['prefill']['completion_tokens']} output; decode = "
            f"{specs['decode']['prompt_tokens']} prompt / "
            f"{specs['decode']['completion_tokens']} output "
            f"({specs['decode']['completion_tokens'] - 1} measured decode "
            "tokens after the first chunk)."))(
                summary.get("plan", {}).get("shape_specs", SHAPE_SPECS)),
        "",
        "## By type, shape, and concurrency",
        "",
        "| type | shape | C | requests | valid | TTFT p95 ms | "
        "e2e p95 ms | engine prefill tok/s med | peak | "
        "decode tok/s med | peak | decode tokens med | "
        "group decode tok/s med |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    def fmt(value, scale=1.0, digits=2):
        if value is None:
            return "-"
        return f"{value * scale:.{digits}f}"

    for entry in sorted(
            summary["by_type_shape_concurrency"].values(),
            key=lambda item: (
                item["key"].get("context_type", ""),
                item["key"].get("shape", ""),
                item["key"].get("concurrency", 0))):
        key = entry["key"]
        lines.append(
            f"| {key.get('context_type')} | {key.get('shape')} | "
            f"{key.get('concurrency')} | {entry['requests']} | "
            f"{entry['valid_requests']} | "
            f"{fmt(entry.get('ttft_s_p95'), 1000)} | "
            f"{fmt(entry.get('e2e_s_p95'), 1000)} | "
            f"{fmt(entry.get('engine_prefill_tokens_per_s_median'))} | "
            f"{fmt(entry.get('engine_prefill_tokens_per_s_peak'))} | "
            f"{fmt(entry.get('decode_tok_s_median'))} | "
            f"{fmt(entry.get('decode_tok_s_peak'))} | "
            f"{fmt(entry.get('decode_tokens_after_first_median'))} | "
            f"{fmt(entry.get('group_decode_tokens_per_s_median'))} |")
    lines += [
        "",
        "The engine-prefill columns are primary only for `shape=prefill` "
        "rows. Decode-shape prefill is secondary and is never pooled into "
        "that primary result.",
        "",
        "## Shape x concurrency",
        "",
        "| shape | C | groups valid/total | engine prefill tok/s med | "
        "peak | group prefill tok/s med | group decode tok/s med | "
        "group completion tok/s med |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for entry in sorted(
            summary["by_shape_concurrency"].values(),
            key=lambda item: (
                item["key"].get("shape", ""),
                item["key"].get("concurrency", 0))):
        key = entry["key"]
        lines.append(
            f"| {key.get('shape')} | {key.get('concurrency')} | "
            f"{entry['valid_groups']}/{entry['groups']} | "
            f"{fmt(entry.get('engine_prefill_tokens_per_s_median'))} | "
            f"{fmt(entry.get('engine_prefill_tokens_per_s_peak'))} | "
            f"{fmt(entry.get('group_prefill_tokens_per_s_median'))} | "
            f"{fmt(entry.get('group_decode_tokens_per_s_median'))} | "
            f"{fmt(entry.get('group_completion_tokens_per_s_median'))} |")
    lines += [
        "",
        "## Actual engine interval overlap",
        "",
        "Active intervals are `[scheduled_ts,last_token_ts)` from "
        "finished-request events; queued intervals are "
        "`[queued_ts,scheduled_ts)`. These are actual request intervals, not "
        "post-group gauges.",
        "",
        "| shape | C | groups | with events | active max med | peak | "
        "queued max med | fully overlapped |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, entry in sorted(
            (overlap.get("by_shape_concurrency") or {}).items(),
            key=lambda item: (item[0].split("|")[0],
                              int(item[0].split("|")[1])
                              if item[0].split("|")[1].isdigit() else 0)):
        shape, concurrency = (key.split("|", 1) + [""])[:2]
        lines.append(
            f"| {shape} | {concurrency} | {entry.get('groups')} | "
            f"{entry.get('groups_with_events')} | "
            f"{fmt(entry.get('active_max_median'), digits=1)} | "
            f"{fmt(entry.get('active_max_peak'), digits=1)} | "
            f"{fmt(entry.get('queued_max_median'), digits=1)} | "
            f"{entry.get('groups_fully_overlapped')} |")
    lines += [
        "",
        f"- C8 decode actual-overlap check: {c8.get('groups_max_active_8')}"
        f"/{c8.get('groups_total')} groups reached active_max=8 "
        f"(with events: {c8.get('groups_with_events')}, "
        f"all_groups_actual_8={c8.get('all_groups_actual_8')}).",
        f"- Missing C8 decode groups at active_max=8: "
        f"{c8.get('missing_groups') or 'none'}.",
        "- Prefill overlap is scheduler interleaving of scheduled-to-last-"
        "token intervals; it is not C simultaneous prefill GPU kernels.",
        "- `num_requests_running`/`waiting` are sampled after the group gate "
        "ends. They are post-group observations and never prove active "
        "overlap.",
        "- Decode tokens are actual completion tokens after the first chunk, "
        "not SSE chunk counts. Engine prefill is `first_token_ts - "
        "scheduled_ts`, not a TTFT proxy.",
        "- Native 1048576 context is an admission cap only; no 1M-token "
        "prompt or long-context sweep was run in this bounded campaign.",
    ]
    gpu = summary.get("gpu") or {}
    lines += ["", "## GPU0 memory headroom", ""]
    if gpu.get("status") == "ok":
        measured = gpu.get("measured") or {}
        lines += [
            f"- source: `{gpu.get('path')}` "
            f"(GPU0 rows: {gpu.get('rows_used_gpu0')})",
            f"- overall sampled minimum free: "
            f"{fmt(gpu.get('min_free_mib'), digits=0)} MiB",
            f"- measured request window minimum free: "
            f"{fmt(measured.get('min_free_mib'), digits=0)} MiB; "
            f"maximum used: {fmt(measured.get('max_used_mib'), digits=0)} "
            "MiB",
            "",
            "| phase | samples | min free MiB | max used MiB | "
            "max util % |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, entry in (gpu.get("by_phase") or {}).items():
            lines.append(
                f"| {name} | {entry.get('samples')} | "
                f"{fmt(entry.get('min_free_mib'), digits=0)} | "
                f"{fmt(entry.get('max_used_mib'), digits=0)} | "
                f"{fmt(entry.get('max_util_pct'), digits=1)} |")
        lines.append("")
        lines.append(
            "- GPU memory is sampled separately from engine metrics; the "
            "minimum free value is an observed floor, not a reserve or a "
            "claim about future headroom.")
    else:
        lines.append(
            f"- GPU sampler evidence: {gpu.get('status', 'not_provided')}.")
    lines += [
        "",
        "Per-group rows and all raw metrics are in `summary.json`; request "
        "rows are in `rows.jsonl`; exact chunk token IDs are in "
        "`streams.jsonl`.",
        "",
    ]
    return "\n".join(lines)


def summarize(run_meta: dict, plan: dict, rows: list, warmup_rows: list,
              groups: list, prompts: dict, cache_config: dict,
              max_model_len, events: EngineEvents, gpu: dict,
              phases: list, args) -> dict:
    measured_rows = [row for row in rows if row.get("phase") == "measured"]
    refresh_group_rollups(
        measured_rows, groups,
        bool(getattr(args, "require_engine_events", False)))
    annotate_group_activity(measured_rows, groups)
    valid_rows = [row for row in measured_rows if row.get("valid")]
    group_ok = [group for group in groups if group.get("valid")]
    tested_lengths = [prompt["target_prompt_tokens"]
                      for prompt in prompts.values()]
    tested_outputs = [prompt["completion_tokens"]
                      for prompt in prompts.values()]
    by_type = aggregate_rows(
        measured_rows, groups, ["context_type"],
        ["context_type"])
    by_concurrency = aggregate_rows(
        measured_rows, groups, ["concurrency"],
        ["concurrency"])
    by_type_concurrency = aggregate_rows(
        measured_rows, groups, ["context_type", "concurrency"],
        ["context_type", "concurrency"])
    by_type_shape_concurrency = aggregate_rows(
        measured_rows, groups,
        ["context_type", "shape", "concurrency"],
        ["context_type", "shape", "concurrency"])
    by_shape_concurrency = aggregate_rows(
        measured_rows, groups, ["shape", "concurrency"],
        ["shape", "concurrency"])
    by_shape = aggregate_rows(
        measured_rows, groups, ["shape"], ["shape"])
    sources = sorted({
        group.get("engine_prefill_source") for group in groups
        if group.get("engine_prefill_source")})
    request_scoped = all(
        group.get("engine_events_all_present") for group in groups)
    prefill_rows = [
        row for row in valid_rows if row.get("shape") == "prefill"]
    decode_rows = [
        row for row in valid_rows if row.get("shape") == "decode"]
    prefill_rates = [
        row.get("engine_prefill_tokens_per_s") for row in prefill_rows
        if row.get("engine_prefill_tokens_per_s") is not None]
    decode_shape_prefill_rates = [
        row.get("engine_prefill_tokens_per_s") for row in decode_rows
        if row.get("engine_prefill_tokens_per_s") is not None]
    c8_decode_groups = [
        group for group in groups
        if group.get("shape") == "decode"
        and group.get("concurrency") == 8]
    c8_decode_with_events = [
        group for group in c8_decode_groups
        if (group.get("actual_overlap") or {}).get("active_max") is not None]
    c8_decode_active_8 = [
        group for group in c8_decode_with_events
        if (group.get("actual_overlap") or {}).get("active_max") >= 8]
    c8_missing = sorted(
        set(group.get("group_id") for group in c8_decode_groups)
        - set(group.get("group_id") for group in c8_decode_active_8))
    invalids = {}
    for row in measured_rows:
        for error in row.get("validation_errors", []):
            invalids[error] = invalids.get(error, 0) + 1
    for group in groups:
        for error in group.get("validation_errors", []):
            invalids[error] = invalids.get(error, 0) + 1
    summary = {
        "schema": SUMMARY_SCHEMA,
        "benchmark": BENCHMARK,
        "run_id": run_meta["run_id"],
        "label": run_meta["label"],
        "created_at": now_iso(),
        "base_url": run_meta["base_url"],
        "model": run_meta["model"],
        "endpoint": "/v1/completions",
        "wall_s": run_meta["wall_s"],
        "fixture_digest": fixture_digest(),
        "plan": plan,
        "totals": {
            "measured_groups": len(groups),
            "valid_groups": len(group_ok),
            "measured_requests": len(measured_rows),
            "valid_requests": len(valid_rows),
            "invalid_requests": len(measured_rows) - len(valid_rows),
            "warmup_requests": len(warmup_rows),
            "warmup_invalid": sum(
                1 for row in warmup_rows if not row.get("valid")),
        },
        "engine_prefill": {
            "source": ",".join(sources) if sources else "none",
            "request_scoped": request_scoped,
            "primary_shape": "prefill",
            "primary_scope": (
                "per-request engine prefill from valid rows whose shape is "
                "prefill; decode-shape values are secondary below"),
            "metric_definition": ENGINE_PREFILL_DEFINITION,
            "timestamp_source_counts": {
                source: sum(
                    1 for row in measured_rows
                    if row.get("engine_timestamp_source") == source)
                for source in sorted({
                    row.get("engine_timestamp_source")
                    for row in measured_rows
                    if row.get("engine_event_present")})
            },
            "exact_req_stats_timestamps": all(
                row.get("engine_timestamp_source") == "req_stats"
                for row in measured_rows if row.get("engine_event_present")),
            "median_tokens_per_s": (
                statistics.median(prefill_rates) if prefill_rates else None),
            "peak_tokens_per_s": (
                max(prefill_rates) if prefill_rates else None),
            "per_request_events": sum(
                1 for row in measured_rows if row.get("engine_event_present")),
            "events_expected": len(measured_rows),
            "event_file": str(events.path) if events.path else None,
            "event_lines_seen": events.lines,
            "bad_event_lines": events.bad_lines,
            "note": (
                "Per-request median/peak require the engine finished-request "
                "event (prefill_time_s, queue_time_s separate). The "
                "Prometheus histogram delta is aggregate-only and is never "
                "converted into per-request median/peak, TTFT proxy, or GPU "
                "time."),
            "decode_shape_secondary": {
                "shape": "decode",
                "note": (
                    f"decode groups force "
                    f"{plan['shape_specs']['decode']['prompt_tokens']}"
                    f" prompt + "
                    f"{plan['shape_specs']['decode']['completion_tokens']}"
                    " output; their engine prefill is reported only as "
                    "secondary and is never pooled into the primary prefill "
                    "result."),
                "median_tokens_per_s": (
                    statistics.median(decode_shape_prefill_rates)
                    if decode_shape_prefill_rates else None),
                "peak_tokens_per_s": (
                    max(decode_shape_prefill_rates)
                    if decode_shape_prefill_rates else None),
                "n": len(decode_shape_prefill_rates),
            },
        },
        "cache_capacity": cache_capacity_summary(
            cache_config, max_model_len, tested_lengths, tested_outputs,
            args.expected_native_context, args.cache_capacity_json,
            max(plan["concurrency"]) if plan.get("concurrency") else None,
            getattr(args, "allow_non_native_context", False)),
        "forced": plan["forced"],
        "prefix_cache": {
            "cache_config_enabled": cache_config.get(
                "enable_prefix_caching"),
            "required": "off",
            "metric_deltas_nonzero_groups": [
                group["group_id"] for group in groups
                if (group.get("prefix_cache_hits_delta") or 0) > 0
                or (group.get("prompt_tokens_cached_delta") or 0) > 0],
            "verified_by_events": request_scoped,
        },
        "metrics": {
            "enabled": bool(args.metrics),
            "interval_s": args.metrics_interval,
            "groups_with_snapshot": sum(
                1 for group in groups if group.get("metrics") is not None),
        },
        "actual_overlap": {
            "source": "engine_finished_request_events",
            "active_interval": (
                "[scheduled_ts,last_token_ts) (finished_ts fallback)"),
            "queued_interval": "[queued_ts,scheduled_ts)",
            "prefill_note": (
                "Prefill active_max is the overlap of scheduled-to-last-token "
                "intervals, i.e. scheduler interleaving of prefill work. It "
                "does not mean C simultaneous prefill GPU kernels."),
            "post_group_gauges": {
                "sampling": "post_group_only",
                "proves_active_overlap": False,
                "note": (
                    "num_requests_running/waiting are captured after the group "
                    "gate; they are reported as observed post-group samples "
                    "only."),
            },
            "by_shape_concurrency": {
                key: entry["actual_overlap"]
                for key, entry in by_shape_concurrency.items()},
            "c8_decode": {
                "groups_total": len(c8_decode_groups),
                "groups_with_events": len(c8_decode_with_events),
                "groups_max_active_8": len(c8_decode_active_8),
                "missing_groups": c8_missing,
                "all_groups_actual_8": (
                    len(c8_decode_groups) == 8
                    and len(c8_decode_active_8) == 8),
            },
        },
        "gpu": gpu,
        "phases": phases,
        "groups": groups,
        "by_type": by_type,
        "by_shape": by_shape,
        "by_concurrency": by_concurrency,
        "by_type_concurrency": by_type_concurrency,
        "by_type_shape_concurrency": by_type_shape_concurrency,
        "by_shape_concurrency": by_shape_concurrency,
        "invalid_reasons": dict(sorted(
            invalids.items(), key=lambda item: (-item[1], item[0]))),
        "validation": {
            "strict": bool(args.strict),
            "require_engine_events": bool(args.require_engine_events),
            "all_measured_valid": len(valid_rows) == len(measured_rows),
            "all_groups_valid": len(group_ok) == len(groups),
            "exit_nonzero": bool(
                args.strict and (
                    len(valid_rows) != len(measured_rows)
                    or (args.require_engine_events and not request_scoped))),
        },
    }
    return summary


# ---------------------------------------------------------------------------
# live path
# ---------------------------------------------------------------------------

async def run_live_async(args) -> int:
    httpx = import_httpx()
    base = args.base_url.rstrip("/")
    out_dir = Path(args.out)
    if out_dir.exists() and any(out_dir.iterdir()) and not args.overwrite:
        raise SystemExit(
            f"--out {out_dir} is not empty; pass --overwrite or a new path")
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()) + "-" + (
        uuid.uuid4().hex[:6])
    run_start = time.monotonic()
    run_start_unix = time.time()

    plan = build_plan(args.types, args.shapes, args.concurrency,
                      args.repeats, args.warmup)
    write_json(out_dir / "plan.json", plan)

    health = await asyncio.to_thread(health_check, base, args.model)
    limits = http_limits(httpx, max(args.concurrency))
    timeout = httpx.Timeout(args.timeout, connect=15.0, write=30.0,
                            pool=args.timeout)
    async with httpx.AsyncClient(
            limits=limits, timeout=timeout, http2=False) as client:
        models = await fetch_models_async(client, base)
        model_entry = next(
            (item for item in models.get("data", [])
             if item.get("id") == args.model), {})
        max_model_len = model_entry.get("max_model_len")
        series = await fetch_metrics_async(client, base)
        cache_config = cache_config_from_series(series)
        if (max_model_len is not None
                and int(max_model_len) != int(args.expected_native_context)):
            if not getattr(args, "allow_non_native_context", False):
                raise SystemExit(
                    f"engine max_model_len={max_model_len} != expected native "
                    f"{args.expected_native_context}; pass "
                    "--allow-non-native-context only for an explicitly "
                    "approved reduced admission cap")
            print(f"WARNING non-native admission cap {max_model_len} allowed; "
                  f"native {args.expected_native_context} is NOT proven")
        cache_blocks = cache_config.get("num_gpu_blocks")
        benchmark_blocks = K5_CACHE_EVIDENCE.get(
            "benchmark_c8_blocks") or {}
        required_blocks = (
            max(benchmark_blocks.values()) if benchmark_blocks else None)
        try:
            cache_blocks_int = int(str(cache_blocks))
        except (TypeError, ValueError):
            cache_blocks_int = None
        if cache_blocks_int is None and not getattr(
                args, "self_smoke", False):
            raise SystemExit(
                "engine did not report num_gpu_blocks; cannot verify KV "
                "admission, refusing to benchmark")
        if (required_blocks is not None
                and cache_blocks_int is not None
                and cache_blocks_int < required_blocks
                and not getattr(args, "self_smoke", False)):
            raise SystemExit(
                f"KV cache has {cache_blocks_int} blocks but C8 needs "
                f"{required_blocks}; refusing to benchmark")
        if str(cache_config.get("enable_prefix_caching", "")).lower() in (
                "true", "1", "yes"):
            raise SystemExit(
                "engine reports enable_prefix_caching=true; this campaign "
                "requires prefix caching OFF")
        for shape, spec in SHAPE_SPECS.items():
            total = spec["prompt_tokens"] + spec["completion_tokens"]
            if max_model_len is not None and total > int(max_model_len):
                raise SystemExit(
                    f"{shape} needs {total} tokens > max_model_len "
                    f"{max_model_len}")
        print(f"engine model={args.model} max_model_len={max_model_len} "
              f"cache_prefix={cache_config.get('enable_prefix_caching')}")

        tokenizer = Tokenizer(base, args.model, timeout=args.tokenizer_timeout)
        print("building exact prompts with served /tokenize ...")
        prompts = build_prompt_set(tokenizer, args.types)
        write_json(out_dir / "prompts.json", {
            "schema": "cwb-prompts-v1",
            "created_at": now_iso(),
            "tokenizer_calls": tokenizer.calls,
            "prompts": [
                {key: value for key, value in prompt.items()
                 if key != "prompt_token_ids"}
                for prompt in prompts.values()
            ],
            "prompt_token_ids_sha256": {
                f"{prompt['context_type']}/{prompt['shape']}": sha256_text(
                    json.dumps(prompt["prompt_token_ids"],
                               separators=(",", ":")))
                for prompt in prompts.values()
            },
        })
        # Exact token IDs are kept separately so the prompt JSON stays small.
        with (out_dir / "prompt_token_ids.jsonl").open("w") as handle:
            for prompt in prompts.values():
                handle.write(json.dumps({
                    "context_type": prompt["context_type"],
                    "shape": prompt["shape"],
                    "target_prompt_tokens": prompt["target_prompt_tokens"],
                    "token_ids": prompt["prompt_token_ids"],
                }, separators=(",", ":")) + "\n")

        events = EngineEvents(args.engine_event_file)
        run_meta = {
            "schema": "cwb-run-v1",
            "run_id": run_id,
            "label": args.label,
            "base_url": base,
            "model": args.model,
            "started_at": now_iso(),
            "started_unix_s": run_start_unix,
            "health": health,
            "max_model_len": max_model_len,
            "cache_config": cache_config,
            "engine_event_file": str(events.path) if events.path else None,
            "engine_event_start_offset": events.offset,
            "args": vars(args),
        }
        write_json(out_dir / "run_meta.json", run_meta)
        write_json(out_dir / "startup_metrics.json", {
            "created_at": now_iso(),
            "selected": selected_metrics(series),
        })
        phases = [{
            "name": "startup",
            "start_unix_s": run_start_unix,
            "start_mono": run_start,
            "end_unix_s": time.time(),
            "end_mono": time.monotonic(),
        }]

        warmup_rows = []
        if args.warmup:
            phase_start = time.monotonic()
            warmup_rows = await run_warmups(
                client, base, plan, prompts, events, args, run_id, out_dir)
            phases.append({
                "name": "warmup",
                "start_unix_s": run_start_unix + (phase_start - run_start),
                "start_mono": phase_start,
                "end_unix_s": time.time(),
                "end_mono": time.monotonic(),
            })

        rows = []
        groups = []
        streams_path = out_dir / "streams.jsonl"
        rows_path = out_dir / "rows.jsonl"
        groups_path = out_dir / "groups.jsonl"
        for index, group in enumerate(plan["groups"]):
            await wait_idle_async(client, base, args.idle_timeout)
            prompt = prompts[(group["context_type"], group["shape"])]
            print(f"[{index + 1}/{len(plan['groups'])}] {group['group_id']} "
                  f"prompt={group['prompt_tokens']} "
                  f"completion={group['completion_tokens']}", flush=True)
            phase_start = time.monotonic()
            phase_start_unix = time.time()
            group_row, group_rows = await run_group(
                client, base, group, prompt, events, args, run_id)
            phases.append({
                "name": group["shape"],
                "group_id": group["group_id"],
                "context_type": group["context_type"],
                "concurrency": group["concurrency"],
                "repeat": group["repeat"],
                "start_unix_s": phase_start_unix,
                "start_mono": phase_start,
                "end_unix_s": time.time(),
                "end_mono": time.monotonic(),
            })
            for row in group_rows:
                timeline = row.pop("timeline", [])
                if args.timeline:
                    append_jsonl(streams_path, {
                        "schema": "cwb-stream-v1",
                        "run_id": run_id,
                        "group_id": group["group_id"],
                        "request_id": row.get("request_id"),
                        "response_id": row.get("response_id"),
                        "prompt_tokens": row.get("prompt_tokens_usage"),
                        "completion_tokens": row.get(
                            "completion_tokens_usage"),
                        "first_chunk_tokens": row.get("first_chunk_tokens"),
                        "decode_tokens_after_first": row.get(
                            "decode_tokens_after_first"),
                        "chunks": timeline,
                    })
                append_jsonl(rows_path, row)
            append_jsonl(groups_path, group_row)
            rows.extend(group_rows)
            groups.append(group_row)
            await wait_idle_async(client, base, args.idle_timeout)
            print(f"    valid={group_row['valid_requests']}/"
                  f"{group_row['requests']} "
                  f"wall={group_row['group_wall_s']:.2f}s "
                  f"prefill_med="
                  f"{group_row['engine_prefill_tokens_per_s_median']} "
                  f"prefill_agg="
                  f"{group_row['engine_prefill_group_tokens_per_s']} "
                  f"decode_med={group_row['decode_tok_s_median']}",
                  flush=True)

        final_series = await fetch_metrics_async(client, base)
        gpu = ingest_gpu_csv(args.gpu_csv, args.gpu_label, groups, phases)
        write_json(out_dir / "phases.json", {
            "schema": "cwb-phases-v1", "phases": phases})
        write_json(out_dir / "gpu_headroom.json", gpu)
        if args.metrics_raw:
            write_json(out_dir / "final_metrics.json", {
                "created_at": now_iso(),
                "selected": selected_metrics(final_series),
            })
        for group in groups:
            gpu_group = gpu.get("by_group", {}).get(group["group_id"])
            if gpu_group:
                group["gpu0_min_free_mib"] = gpu_group.get("min_free_mib")
                group["gpu0_max_util_pct"] = gpu_group.get("max_util_pct")
        run_meta["finished_at"] = now_iso()
        run_meta["wall_s"] = time.monotonic() - run_start
        write_json(out_dir / "run_meta.json", run_meta)
        summary = summarize(
            run_meta, plan, rows, warmup_rows, groups, prompts, cache_config,
            max_model_len, events, gpu, phases, args)
        write_json(out_dir / "summary.json", summary)
        (out_dir / "summary.md").write_text(markdown_summary(summary))
        print(f"wrote {out_dir / 'summary.json'}")
        print(f"valid groups={summary['totals']['valid_groups']}/"
              f"{summary['totals']['measured_groups']} "
              f"valid requests={summary['totals']['valid_requests']}/"
              f"{summary['totals']['measured_requests']} "
              f"engine_prefill_source={summary['engine_prefill']['source']}")
        if summary["validation"]["exit_nonzero"]:
            print("STRICT FAILURE: invalid measured rows/groups")
            for reason, count in summary["invalid_reasons"].items():
                print(f"  {reason}: {count}")
            return 2
        return 0


# ---------------------------------------------------------------------------
# self smoke: fake HTTP engine, no GPU/engine/SSH
# ---------------------------------------------------------------------------

class SmokeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.active = 0
        self.max_active = 0
        self.starts = []
        self.finishes = []
        self.event_path = None
        self.errors = []


def make_smoke_handler(state: SmokeState, delay: float):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *args):
            pass

        def _json(self, value, status=200):
            body = json.dumps(value).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _sse(self, value):
            body = ("data: " + json.dumps(
                value, separators=(",", ":")) + "\n\n").encode("utf-8")
            self.wfile.write(body)
            self.wfile.flush()

        def do_GET(self):  # noqa: N802 - http.server API
            path = urllib.parse.urlparse(self.path).path
            if path == "/health":
                self._json({"status": "ok"})
            elif path == "/v1/models":
                self._json({"object": "list", "data": [{
                    "id": MODEL, "object": "model", "owned_by": "smoke",
                    "max_model_len": 1048576}]})
            elif path == "/metrics":
                body = "\n".join([
                    "# smoke metrics",
                    'vllm:num_requests_running{engine="0",model_name="'
                    + MODEL + '"} 0.0',
                    'vllm:num_requests_waiting{engine="0",model_name="'
                    + MODEL + '"} 0.0',
                    'vllm:kv_cache_usage_perc{engine="0",model_name="'
                    + MODEL + '"} 0.0',
                    'vllm:cache_config_info{block_size="128",'
                    'num_gpu_blocks="5237",kv_cache_size_tokens="20227",'
                    'enable_prefix_caching="False",'
                    'cache_dtype="nvfp4_ds_mla",'
                    'gpu_memory_utilization="0.97",'
                    'kv_cache_memory_bytes="None"} 1.0',
                    'vllm:num_preemptions_total{engine="0"} 0.0',
                    'vllm:prefix_cache_hits_total{engine="0"} 0.0',
                    'vllm:prompt_tokens_cached_total{engine="0"} 0.0',
                    'vllm:prompt_tokens_total{engine="0"} 0.0',
                    'vllm:generation_tokens_total{engine="0"} 0.0',
                    'vllm:request_prefill_time_seconds_sum{engine="0"} 0.0',
                    'vllm:request_prefill_time_seconds_count{engine="0"} 0.0',
                    'vllm:request_prefill_kv_computed_tokens_sum'
                    '{engine="0"} 0.0',
                    'vllm:request_prefill_kv_computed_tokens_count'
                    '{engine="0"} 0.0',
                    'vllm:request_decode_time_seconds_sum{engine="0"} 0.0',
                    'vllm:request_decode_time_seconds_count{engine="0"} 0.0',
                    'vllm:request_queue_time_seconds_sum{engine="0"} 0.0',
                    'vllm:request_queue_time_seconds_count{engine="0"} 0.0',
                    "",
                ]).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            else:
                self._json({"error": "not found"}, 404)

        def do_POST(self):  # noqa: N802 - http.server API
            path = urllib.parse.urlparse(self.path).path
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError):
                self._json({"error": "bad json"}, 400)
                return
            if path == "/tokenize":
                text = payload.get("prompt", "")
                self._json({"count": len(text),
                            "tokens": [ord(char) for char in text],
                            "model": MODEL})
            elif path == "/detokenize":
                ids = payload.get("tokens", [])
                self._json({"prompt": "".join(chr(int(i)) for i in ids),
                            "model": MODEL})
            elif path == "/v1/completions":
                self._completion(payload)
            else:
                self._json({"error": "not found"}, 404)

        def _completion(self, payload):
            prompt = payload.get("prompt", "")
            max_tokens = int(payload.get("max_tokens", 1))
            request_id = payload.get("request_id") or uuid.uuid4().hex
            response_id = ("cmpl-" + request_id
                           if not request_id.startswith("cmpl-")
                           else request_id)
            scheduled = time.monotonic()
            with state.lock:
                state.active += 1
                state.max_active = max(state.max_active, state.active)
                state.starts.append(scheduled)
            try:
                time.sleep(delay)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                prompt_ids = [ord(char) for char in prompt]
                total = 0
                first = None
                last = None
                rng = random.Random(hash(request_id) & 0xFFFFFFFF)
                first_batch = True
                while total < max_tokens:
                    count = min(max_tokens - total, rng.randint(3, 5))
                    token_ids = [ord("x")] * count
                    now = time.monotonic()
                    if first is None:
                        first = now
                    last = now
                    choice = {
                        "index": 0,
                        "text": "x" * count,
                        "token_ids": token_ids,
                        "finish_reason": (
                            "length" if total + count >= max_tokens else None),
                    }
                    if first_batch:
                        choice["prompt_token_ids"] = prompt_ids
                        first_batch = False
                    self._sse({
                        "id": response_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": MODEL,
                        "choices": [choice],
                    })
                    total += count
                    if delay:
                        time.sleep(min(delay, 0.002))
                usage = {
                    "prompt_tokens": len(prompt_ids),
                    "completion_tokens": max_tokens,
                    "total_tokens": len(prompt_ids) + max_tokens,
                }
                self._sse({
                    "id": response_id,
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": MODEL,
                    "choices": [],
                    "usage": usage,
                })
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                finished = time.monotonic()
                event = {
                    "schema": ENGINE_EVENT_SCHEMA,
                    "event": "finished",
                    # Match live vLLM: engine stats append choice index 0;
                    # the SSE response keeps the base ID.
                    "request_id": response_id + "-0",
                    "finish_reason": "length",
                    "num_prompt_tokens": len(prompt_ids),
                    "num_generation_tokens": max_tokens,
                    "num_cached_tokens": 0,
                    "num_preemptions": 0,
                    "prefill_time_s": (
                        first - scheduled if first is not None else None),
                    "decode_time_s": (
                        last - first if first is not None and last is not None
                        else 0.0),
                    "queue_time_s": 0.0,
                    "e2e_latency_s": finished - scheduled,
                    "scheduled_ts": scheduled,
                    "first_token_ts": first,
                    "last_token_ts": last,
                    "finished_ts": finished,
                    "finished_iso": now_iso(),
                    "timestamp_source": "req_stats",
                    "clock": "time.monotonic",
                    "metric_definition": ENGINE_PREFILL_DEFINITION,
                }
                with state.lock:
                    with open(state.event_path, "a") as handle:
                        handle.write(json.dumps(
                            event, separators=(",", ":")) + "\n")
                        handle.flush()
                    state.finishes.append(finished)
            except (BrokenPipeError, ConnectionResetError) as exc:
                with state.lock:
                    state.errors.append(f"{type(exc).__name__}: {exc}")
            finally:
                with state.lock:
                    state.active -= 1

    return Handler


def assert_smoke(summary: dict, state: SmokeState, groups: list) -> list:
    problems = []
    totals = summary["totals"]
    if totals["measured_groups"] != 4:
        problems.append(
            f"expected 4 measured groups, got {totals['measured_groups']}")
    if totals["measured_requests"] != 6:
        problems.append(
            f"expected 6 measured requests, got {totals['measured_requests']}")
    if totals["invalid_requests"]:
        problems.append(
            f"invalid measured requests: {summary['invalid_reasons']}")
    if state.max_active != 2:
        problems.append(
            f"fake engine max concurrency {state.max_active} != 2")
    max_spread = max(
        (group["request_start_spread_ms"] for group in groups), default=0.0)
    if max_spread > 500.0:
        problems.append(f"group start spread {max_spread:.1f}ms > 500ms")
    if not any(
            (group.get("engine_events_all_present") for group in groups)):
        problems.append("no group joined engine request events")
    joined_ids = [
        str(row.get("engine_request_id"))
        for row in summary.get("_rows", [])
        if row.get("phase") == "measured" and row.get("engine_request_id")]
    if not joined_ids or not all(
            rid.endswith("-0") for rid in joined_ids):
        problems.append(
            "engine join did not exercise trailing choice-index event ID")
    first_chunks = [
        row["first_chunk_tokens"] for row in summary.get("_rows", [])
        if row.get("first_chunk_tokens") is not None]
    if not any(count > 1 for count in first_chunks):
        problems.append("no multi-token first chunk exercised")
    if not summary["by_type_concurrency"]:
        problems.append("empty by_type_concurrency summary")
    if not summary["by_type_shape_concurrency"]:
        problems.append("empty by_type_shape_concurrency summary")
    for entry in summary["by_type_concurrency"].values():
        if entry.get("engine_prefill_tokens_per_s_median") is None:
            problems.append("missing engine prefill median in by_type_concurrency")
            break
    decode_rows = [
        row for row in summary.get("_rows", [])
        if row.get("shape") == "decode"]
    for row in decode_rows:
        expected = (row["completion_tokens_usage"]
                    - row["first_chunk_tokens"])
        if row["decode_tokens_after_first"] != expected:
            problems.append(
                f"{row['request_id']}: decode_after_first "
                f"{row['decode_tokens_after_first']} != {expected}")
            break
    if not summary.get("by_shape_concurrency"):
        problems.append("empty by_shape_concurrency summary")
    else:
        shape_entries = summary["by_shape_concurrency"]
        if set(shape_entries) != {
                "prefill|1", "prefill|2", "decode|1", "decode|2"}:
            problems.append(
                f"unexpected shape/concurrency keys: {sorted(shape_entries)}")
        for key, entry in shape_entries.items():
            shape = key.split("|", 1)[0]
            if shape == "prefill":
                if entry.get("group_decode_tokens_per_s_median") is not None:
                    problems.append(
                        f"{key}: decode rate leaked into prefill shape")
                if entry.get("group_prefill_tokens_per_s_median") is None:
                    problems.append(f"{key}: missing prefill group rate")
            elif entry.get("group_prefill_tokens_per_s_median") is not None:
                problems.append(
                    f"{key}: prefill rate pooled into decode shape")
    for key, entry in summary.get("by_type_shape_concurrency", {}).items():
        if "|prefill|" in key and (
                entry.get("engine_decode_tokens_per_s_median") is not None
                or entry.get("decode_tok_s_median") is not None):
            problems.append(f"{key}: decode metrics in prefill shape")
            break
    overlap = summary.get("actual_overlap") or {}
    if not overlap.get("by_shape_concurrency"):
        problems.append("missing actual interval overlap evidence")
    ga = overlap.get("post_group_gauges") or {}
    if ga.get("proves_active_overlap") is not False:
        problems.append("post-group gauges not explicitly non-proof")
    rendered = markdown_summary(summary)
    for required in (
            "512 prompt / 2048 output",
            "post-group observations and never prove active overlap",
            "admission cap only",
            "not C simultaneous prefill GPU kernels",
            "GPU sampler evidence: not_provided"):
        if required not in rendered:
            problems.append(f"markdown missing caveat: {required}")
    cache_formula = summary.get("cache_capacity", {}).get("formula") or {}
    if any("3gib" in key for key in cache_formula):
        problems.append("stale nominal 3GiB cache fields in summary")
    return problems


def run_self_smoke(base_args) -> int:
    import_httpx()
    state = SmokeState()
    delay = max(base_args.smoke_delay, 0.0)
    handler = make_smoke_handler(state, delay)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    port = server.server_address[1]
    out_root = Path(tempfile.mkdtemp(prefix="cwb-smoke-"))
    state.event_path = str(out_root / "engine_events.jsonl")
    Path(state.event_path).touch()
    args = argparse.Namespace(**vars(base_args))
    args.base_url = f"http://127.0.0.1:{port}"
    args.out = str(out_root / "run")
    args.overwrite = True
    args.types = ["prose"]
    args.shapes = ["prefill", "decode"]
    args.concurrency = [1, 2]
    args.repeats = 1
    args.warmup = 1
    args.prefill_tokens = SHAPE_SPECS["prefill"]["prompt_tokens"]
    args.prefill_output = SHAPE_SPECS["prefill"]["completion_tokens"]
    args.decode_tokens = SHAPE_SPECS["decode"]["prompt_tokens"]
    args.decode_output = SHAPE_SPECS["decode"]["completion_tokens"]
    args.engine_event_file = state.event_path
    args.expected_native_context = 1048576
    args.gpu_csv = None
    args.gpu_label = None
    args.cache_capacity_json = None
    args.metrics = True
    args.metrics_interval = 0.02
    args.metrics_raw = False
    args.timeline = True
    args.strict = True
    args.require_engine_events = True
    args.idle_timeout = 10.0
    args.engine_event_timeout = 10.0
    args.timeout = 60.0
    args.tokenizer_timeout = 30.0
    args.seed = 1234
    args.label = "self-smoke"
    try:
        rc = asyncio.run(run_live_async(args))
        summary_path = Path(args.out) / "summary.json"
        if not summary_path.exists():
            print("SMOKE FAIL: no summary.json")
            print(f"artifacts: {out_root}")
            return 1
        summary = json.loads(summary_path.read_text())
        rows_path = Path(args.out) / "rows.jsonl"
        summary["_rows"] = [
            json.loads(line) for line in rows_path.read_text().splitlines()
            if line.strip()]
        groups_path = Path(args.out) / "groups.jsonl"
        groups = [
            json.loads(line) for line in groups_path.read_text().splitlines()
            if line.strip()]
        problems = assert_smoke(summary, state, groups)
        if rc != 0:
            problems.append(f"strict run exit code {rc}")
        if problems:
            print("SMOKE FAIL")
            for problem in problems:
                print(f"  {problem}")
            print(f"artifacts: {out_root}")
            return 1
        print(
            "SMOKE PASS "
            f"groups={summary['totals']['measured_groups']} "
            f"requests={summary['totals']['measured_requests']} "
            f"max_concurrent={state.max_active} "
            f"max_spread_ms={max((g['request_start_spread_ms'] for g in groups), default=0):.1f} "
            f"engine_events={summary['engine_prefill']['per_request_events']}"
            f"/{summary['engine_prefill']['events_expected']} "
            f"first_chunk_multi_token=True "
            f"summary={summary_path}")
        return 0
    finally:
        server.shutdown()
        server.server_close()


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize_from_dir(source_text, event_file=None,
                       cache_capacity_json=None, gpu_csv=None,
                       gpu_label=None) -> int:
    """Replay a completed campaign's raw artifacts without a live engine.

    The engine event snapshot is read from offset 0 through the frozen
    ``EngineEvents`` reader.  Rows are rejoined by response id / completion
    request id / request id, engine fields are repaired, and group rollups
    and actual interval overlap are recomputed.  Original summary evidence
    is never overwritten.
    """
    source = Path(source_text).expanduser()
    if not source.is_dir():
        raise SystemExit(f"--summarize-from is not a directory: {source}")
    required = ("run_meta.json", "plan.json", "prompts.json", "rows.jsonl",
                "groups.jsonl")
    missing = [name for name in required
               if not (source / name).exists()]
    if missing:
        raise SystemExit(
            f"{source} missing required artifacts: {', '.join(missing)}")

    def load_json(name, default=None):
        path = source / name
        if not path.exists():
            if default is not None:
                return default
            raise SystemExit(f"missing {path}")
        return json.loads(path.read_text())

    def load_jsonl(name):
        path = source / name
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines()
                if line.strip()]

    run_meta = load_json("run_meta.json")
    plan = load_json("plan.json")
    prompts_doc = load_json("prompts.json")
    rows = load_jsonl("rows.jsonl")
    warmup_rows = load_jsonl("warmup_rows.jsonl")
    groups = load_jsonl("groups.jsonl")
    phases_doc = load_json("phases.json", {"phases": []})
    phases = phases_doc.get("phases", []) if isinstance(phases_doc, dict) \
        else phases_doc
    gpu = (ingest_gpu_csv(gpu_csv, gpu_label, groups, phases)
           if gpu_csv else load_json("gpu_headroom.json", {}))
    prompts = {}
    prompt_items = prompts_doc.get("prompts", prompts_doc) \
        if isinstance(prompts_doc, dict) else prompts_doc
    for prompt in prompt_items or []:
        prompts[(prompt.get("context_type"), prompt.get("shape"))] = prompt

    saved_args = run_meta.get("args") or {}
    post_args = parse_args(["--out", str(source)])
    for key, value in saved_args.items():
        if hasattr(post_args, key) and key not in (
                "out", "engine_event_file", "self_smoke", "list",
                "summarize_from"):
            setattr(post_args, key, value)
    post_args.out = str(source)
    post_args.self_smoke = False
    post_args.list = False
    post_args.summarize_from = None
    post_args.expected_native_context = 1048576
    post_args.allow_non_native_context = False
    post_args.strict = True
    post_args.require_engine_events = True
    if cache_capacity_json:
        post_args.cache_capacity_json = cache_capacity_json
    if not post_args.cache_capacity_json or not Path(
            post_args.cache_capacity_json).exists():
        candidates = [
            source / "cache_capacity_c8_kv2g.json",
            source.parent / "cache_capacity_c8_kv2g.json",
        ]
        post_args.cache_capacity_json = next(
            (str(path) for path in candidates if path.exists()), None)
    for shape in ("prefill", "decode"):
        if shape in SHAPE_SPECS:
            prompt = next(
                (item for (_, item_shape), item in prompts.items()
                 if item_shape == shape), None)
            if prompt:
                SHAPE_SPECS[shape] = {
                    "prompt_tokens": prompt.get("target_prompt_tokens"),
                    "completion_tokens": prompt.get("completion_tokens"),
                }

    event_candidates = []
    if event_file and event_file != DEFAULT_ENGINE_EVENT_FILE:
        event_candidates.append(Path(event_file).expanduser())
    event_candidates.extend([
        source / "engine_events_snapshot.jsonl",
        source / "engine_events.jsonl",
        Path(saved_args.get("engine_event_file", "")).expanduser()
        if saved_args.get("engine_event_file") else None,
        Path(run_meta.get("engine_event_file", "")).expanduser()
        if run_meta.get("engine_event_file") else None,
    ])
    event_path = next(
        (path for path in event_candidates if path and path.exists()), None)
    post_args.engine_event_file = str(event_path) if event_path else ""
    events = EngineEvents(event_path)
    events.offset = 0
    events.buf = ""
    events.by_id.clear()
    events.alias_candidates.clear()
    events.bad_lines = 0
    events.lines = 0
    events.refresh()

    groups_by_id = {group.get("group_id"): group for group in groups}
    for row in rows:
        group = groups_by_id.get(row.get("group_id")) or {}
        expected_prompt = (
            group.get("prompt_tokens")
            if group.get("prompt_tokens") is not None
            else row.get("prompt_tokens_target"))
        expected_completion = (
            group.get("completion_tokens")
            if group.get("completion_tokens") is not None
            else row.get("completion_tokens_target"))
        request_id = row.get("request_id") or ""
        event = events.get(
            row.get("response_id"), "cmpl-" + request_id, request_id)
        apply_engine_event(
            row, event, expected_prompt, expected_completion, True)
    for group in groups:
        group["validation_errors"] = [
            error for error in group.get("validation_errors", [])
            if not str(error).startswith("engine_")]

    cache_config = run_meta.get("cache_config") or {}
    max_model_len = run_meta.get("max_model_len")
    summary = summarize(
        run_meta, plan, rows, warmup_rows, groups, prompts, cache_config,
        max_model_len, events, gpu, phases, post_args)
    summary["postprocess"] = {
        "schema": "cwb-postprocess-v1",
        "reprocessed_at": now_iso(),
        "source_dir": str(source),
        "source_summary_sha256": _file_sha256(source / "summary.json"),
        "original_summary_preserved": True,
        "event_file": str(event_path) if event_path else None,
        "event_start_offset": 0,
        "event_replay": "full_file_from_offset_zero",
        "event_lines_seen": events.lines,
        "bad_event_lines": events.bad_lines,
        "gpu_file": str(gpu.get("path")) if gpu_csv else None,
        "gpu_min_free_mib": gpu.get("min_free_mib"),
        "rejoined_rows": sum(
            1 for row in rows if row.get("engine_event_present")),
        "expected_rows": len(rows),
        "command": " ".join([
            "python3 eval/concurrency_workload_eval.py",
            f"--summarize-from {source}",
            f"--engine-event-file {event_path}" if event_path else "",
            f"--cache-capacity-json {post_args.cache_capacity_json}"
            if post_args.cache_capacity_json else "",
            f"--gpu-csv {gpu_csv}" if gpu_csv else "",
            f"--gpu-label {gpu_label}" if gpu_label else "",
        ]).replace("  ", " ").strip(),
        "notes": (
            "Offline replay repairs engine fields from finished-request "
            "events; it does not rerun HTTP requests or change client "
            "measurements. The original summary.json/summary.md are "
            "preserved byte-for-byte."),
    }
    out_json = source / "summary_reprocessed.json"
    out_md = source / "summary_reprocessed.md"
    if out_json.resolve() == (source / "summary.json").resolve() or \
            out_md.resolve() == (source / "summary.md").resolve():
        raise SystemExit("refusing to overwrite original summary evidence")
    write_json(out_json, summary)
    out_md.write_text(markdown_summary(summary))
    print(f"wrote {out_json}")
    print(f"wrote {out_md}")
    print(
        f"reprocessed valid groups={summary['totals']['valid_groups']}/"
        f"{summary['totals']['measured_groups']} "
        f"valid requests={summary['totals']['valid_requests']}/"
        f"{summary['totals']['measured_requests']} "
        f"engine events={summary['engine_prefill']['per_request_events']}/"
        f"{summary['engine_prefill']['events_expected']} "
        f"event lines={events.lines} bad={events.bad_lines}")
    return 0 if summary["validation"]["all_groups_valid"] else 2


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_int_list(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        value = int(part)
        if value < 1:
            raise argparse.ArgumentTypeError("concurrency must be >= 1")
        if value not in out:
            out.append(value)
    return out


def parse_shape_list(text: str) -> list[str]:
    out = []
    for part in text.split(","):
        part = part.strip().lower()
        if not part:
            continue
        if part not in SHAPE_SPECS:
            raise argparse.ArgumentTypeError(
                f"unknown shape {part!r}; choose prefill,decode")
        if part not in out:
            out.append(part)
    return out


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8001")
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--out", default="eval-out/concurrency_workload")
    parser.add_argument("--label", default="concurrency")
    parser.add_argument("--types", default="code,prose,json,toolagent",
                        help="comma list: code,prose,json,toolagent")
    parser.add_argument("--shapes", default="prefill,decode",
                        type=parse_shape_list)
    parser.add_argument("--concurrency", default="1,2,4,8",
                        type=parse_int_list)
    parser.add_argument("--repeats", default=DEFAULT_REPEATS, type=int)
    parser.add_argument("--warmup", default=DEFAULT_WARMUP, type=int,
                        help="warmup requests per execution configuration")
    parser.add_argument("--prefill-tokens", default=2048, type=int)
    parser.add_argument("--prefill-output", default=1, type=int)
    parser.add_argument("--decode-tokens", default=512, type=int)
    parser.add_argument("--decode-output", default=2048, type=int)
    parser.add_argument("--seed", default=20261022, type=int)
    parser.add_argument("--engine-event-file",
                        default=os.environ.get(
                            "DSPARK_ENGINE_EVENT_FILE",
                            DEFAULT_ENGINE_EVENT_FILE))
    parser.add_argument("--require-engine-events", action="store_true",
                        help="fail closed if request-scoped engine events "
                             "are missing (needed for per-request median/peak)")
    parser.add_argument("--engine-event-timeout", default=30.0, type=float)
    parser.add_argument("--idle-timeout", default=120.0, type=float)
    parser.add_argument("--timeout", default=600.0, type=float,
                        help="per-request read timeout seconds")
    parser.add_argument("--tokenizer-timeout", default=120.0, type=float)
    parser.add_argument("--metrics", dest="metrics", action="store_true",
                        default=True)
    parser.add_argument("--no-metrics", dest="metrics", action="store_false")
    parser.add_argument("--metrics-interval", default=0.2, type=float)
    parser.add_argument("--metrics-raw", action="store_true")
    parser.add_argument("--timeline", dest="timeline", action="store_true",
                        default=True)
    parser.add_argument("--no-timeline", dest="timeline", action="store_false")
    parser.add_argument("--gpu-csv", default=None,
                        help="external GPU0 sampler CSV (timestamp + free MiB)")
    parser.add_argument("--gpu-label", default=None,
                        help="optional phase/label filter in GPU CSV")
    parser.add_argument("--cache-capacity-json", default=None,
                        help="optional source-backed cache capacity evidence")
    parser.add_argument("--expected-native-context", default=1048576,
                        type=int)
    parser.add_argument(
        "--allow-non-native-context", action="store_true",
        help="allow a bootable non-1M admission cap (e.g. 8192) while still "
             "requiring the C8 block floor, prefix-cache OFF and engine events")
    parser.add_argument("--strict", dest="strict", action="store_true",
                        default=True)
    parser.add_argument("--no-strict", dest="strict", action="store_false")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list", action="store_true",
                        help="print the offline campaign plan")
    parser.add_argument(
        "--summarize-from", default=None, metavar="DIR",
        help="offline: reprocess an existing campaign directory and write "
             "summary_reprocessed.json/md without a live engine")
    parser.add_argument("--self-smoke", action="store_true",
                        help="run a fake HTTP engine and verify the harness")
    parser.add_argument("--smoke-delay", default=0.01, type=float)
    args = parser.parse_args(argv)
    args.types = normalize_types(args.types.split(","))
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    SHAPE_SPECS["prefill"] = {
        "prompt_tokens": args.prefill_tokens,
        "completion_tokens": args.prefill_output,
    }
    SHAPE_SPECS["decode"] = {
        "prompt_tokens": args.decode_tokens,
        "completion_tokens": args.decode_output,
    }
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_smoke:
        return run_self_smoke(args)
    if args.summarize_from:
        return summarize_from_dir(
            args.summarize_from, args.engine_event_file,
            args.cache_capacity_json, args.gpu_csv, args.gpu_label)
    plan = build_plan(args.types, args.shapes, args.concurrency,
                      args.repeats, args.warmup)
    if args.list:
        print_plan(plan)
        return 0
    return asyncio.run(run_live_async(args))


if __name__ == "__main__":
    sys.exit(main())
