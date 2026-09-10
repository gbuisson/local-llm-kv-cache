#!/usr/bin/env python3
"""Session-aware proxy adding disk-backed llama.cpp slot snapshots.

Pi/Zed requests get one serialized operation, an optional slot restore, explicit
prompt caching, and a post-response snapshot. Requests without an explicit
session header use a stable-prefix-derived local affinity; media requests skip
disk snapshots because their prompt prefix is not reusable.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import os
import signal
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from cache_core import build_prefix_payload, cache_filename, cache_key, with_slot_cache


LOGGER = logging.getLogger("local-llm-kv-cache")
DEFAULT_CACHE_DIR = str(Path.home() / ".llama-slot-cache")
HOP_BY_HOP_HEADERS = {
    "connection",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
}
MAX_METADATA_BYTES = 1024 * 1024


def _session_ref(session_id: str) -> str:
    """Return a pseudonymous, log-safe reference for a client session."""
    return hashlib.sha256(session_id.encode("utf-8")).hexdigest()[:16]


def _log_event(event: str, *, level: int = logging.INFO, **fields: Any) -> None:
    payload = {"event": event, **{key: value for key, value in fields.items() if value is not None}}
    LOGGER.log(level, "%s", json.dumps(payload, sort_keys=True, separators=(",", ":")))


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean")


def _is_slot_admin_path(path: str) -> bool:
    normalized = path.partition("?")[0].rstrip("/")
    return normalized == "/slots" or normalized.startswith("/slots/")


_ROUTE_ALIASES = {
    ("GET", "/v1/props"): "/props",
    ("HEAD", "/v1/props"): "/props",
    ("POST", "/v1/tokenize"): "/tokenize",
    ("POST", "/v1/detokenize"): "/detokenize",
    ("POST", "/v1/apply-template"): "/apply-template",
}
_NON_EVICTING_POST_PATHS = frozenset(
    {
        "/tokenize",
        "/detokenize",
        "/apply-template",
        "/v1/responses/input_tokens",
        "/v1/chat/completions/input_tokens",
        "/v1/messages/count_tokens",
        "/v1/chat/completions/control",
    }
)
_UNMANAGED_INFERENCE_POST_PATHS = frozenset(
    {
        "/completion",
        "/v1/completions",
        "/infill",
        "/embedding",
        "/embeddings",
        "/v1/embeddings",
        "/rerank",
        "/reranking",
        "/v1/rerank",
        "/v1/responses",
        "/v1/messages",
    }
)
_ADMIN_PATHS = frozenset({"/tools", "/lora-adapters"})


def _upstream_path(method: str, path: str) -> str:
    raw_path, separator, query = path.partition("?")
    alias = _ROUTE_ALIASES.get((method, raw_path.rstrip("/")))
    if alias is None:
        return path
    return alias + (f"?{query}" if separator else "")


def _normalized_path(path: str) -> str:
    return path.partition("?")[0].rstrip("/")


def _is_non_evicting_post_path(path: str) -> bool:
    normalized = _normalized_path(_upstream_path("POST", path))
    return normalized in _NON_EVICTING_POST_PATHS


def _is_known_post_path(path: str) -> bool:
    normalized = _normalized_path(_upstream_path("POST", path))
    return (
        normalized == "/v1/chat/completions"
        or normalized in _NON_EVICTING_POST_PATHS
        or normalized in _UNMANAGED_INFERENCE_POST_PATHS
    )


def _is_blocked_path(method: str, path: str) -> bool:
    normalized = _normalized_path(path)
    if _is_slot_admin_path(path) or normalized in _ADMIN_PATHS:
        return True
    if method in {"POST", "DELETE"} and (
        normalized == "/props" or normalized == "/models" or normalized.startswith("/models/")
    ):
        return True
    return method == "DELETE"


@dataclass(frozen=True)
class CompletionMetadata:
    finish_reason: str | None = None
    cached_tokens: int | None = None


@dataclass(frozen=True)
class ForwardResult:
    status: int
    metadata: CompletionMetadata = field(default_factory=CompletionMetadata)


def _metadata_from_object(value: Any) -> CompletionMetadata:
    if not isinstance(value, dict):
        return CompletionMetadata()
    finish_reason = None
    choices = value.get("choices")
    if not isinstance(choices, list):
        choices = []
    for choice in choices:
        if isinstance(choice, dict) and isinstance(choice.get("finish_reason"), str):
            finish_reason = choice["finish_reason"]
    cached_tokens = None
    usage = value.get("usage")
    if isinstance(usage, dict):
        details = usage.get("prompt_tokens_details")
        if isinstance(details, dict) and isinstance(details.get("cached_tokens"), int):
            cached_tokens = details["cached_tokens"]
    return CompletionMetadata(finish_reason, cached_tokens)


def _completion_metadata(chunks: list[bytes]) -> CompletionMetadata:
    raw = b"".join(chunks)
    try:
        return _metadata_from_object(json.loads(raw or b"{}"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        pass

    result = CompletionMetadata()
    for line in raw.splitlines():
        if not line.startswith(b"data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == b"[DONE]":
            continue
        try:
            current = _metadata_from_object(json.loads(payload))
        except (json.JSONDecodeError, UnicodeDecodeError):
            continue
        result = CompletionMetadata(
            current.finish_reason or result.finish_reason,
            current.cached_tokens if current.cached_tokens is not None else result.cached_tokens,
        )
    return result


@dataclass(frozen=True)
class SnapshotPlan:
    session_id: str
    slot_id: int | None
    prefix_key: str
    session_file: Path
    prefix_file: Path
    prefix_payload: dict[str, Any]
    prefix_was_present: bool
    candidate_slot_id: int | None = None
    slot_task_ids: dict[int, int] = field(default_factory=dict)


@dataclass(frozen=True)
class SlotState:
    slot_id: int
    prefix_key: str
    n_tokens: int
    session_file: Path | None = None
    dirty: bool = False


class LlamaCacheProxy:
    def __init__(
        self,
        upstream: str = "http://127.0.0.1:8080",
        cache_dir: str | None = None,
        max_cache_gib: float = 12.0,
        wait_seconds: float = 120.0,
        enable_prefix_seeding: bool = True,
        prefix_seed_delay_seconds: float = 2.0,
        save_policy: str = "all",
        require_session_id: bool = False,
    ) -> None:
        parsed = urlsplit(upstream)
        if parsed.scheme != "http" or not parsed.hostname:
            raise ValueError("upstream must be an http URL")
        self.upstream_host = parsed.hostname
        self.upstream_port = parsed.port or 80
        self.upstream_prefix = parsed.path.rstrip("/")
        self.cache_dir = Path(cache_dir or DEFAULT_CACHE_DIR)
        self.cache_dir.mkdir(parents=True, mode=0o700, exist_ok=True)
        self.cache_dir.chmod(0o700)
        self.max_cache_bytes = int(max_cache_gib * 1024**3)
        self.wait_seconds = wait_seconds
        self.enable_prefix_seeding = enable_prefix_seeding
        self.prefix_seed_delay_seconds = prefix_seed_delay_seconds
        if save_policy not in {"all", "terminal"}:
            raise ValueError("save_policy must be 'all' or 'terminal'")
        self.save_policy = save_policy
        self.require_session_id = require_session_id
        self.operation_lock = threading.Lock()
        self.session_states: dict[str, SlotState] = {}
        self.prefix_seed_lock = threading.Lock()
        self.prefix_seeds_in_flight: set[Path] = set()
        self.foreground_waiters = 0
        self.foreground_waiters_lock = threading.Lock()
        self._prune()

    def prepare(self, body: dict[str, Any], session_id: str) -> tuple[dict[str, Any], SnapshotPlan]:
        prefix_key = cache_key(body)
        session_file = self.cache_dir / cache_filename(session_id, body, "session")
        prefix_file = self.cache_dir / cache_filename("prefix", body, "prefix")
        prefix_payload = build_prefix_payload(body)
        session_ref = _session_ref(session_id)
        hot_state = self._hot_state(self.session_states.get(session_id), prefix_key)
        restored_source = None
        candidate_slot_id = None
        slot_task_ids: dict[int, int] = {}
        if hot_state is not None:
            slot_id = hot_state.slot_id
            self._touch_snapshot(session_file)
            _log_event("cache_hit", layer="hot", session_ref=session_ref, slot_id=slot_id)
        else:
            hot_state = None
            # A native cold request may choose any idle slot. Persist every dirty
            # owner before allowing llama.cpp to overwrite one.
            self._flush_dirty_states("before_eviction")
            candidate_slot_id = self._wait_for_idle_slot()
            sources = tuple(path for path in (session_file, prefix_file) if path.exists())
            if sources:
                self._forget_slot(candidate_slot_id)
                restored_source = self._restore_first_available(
                    candidate_slot_id, sources, session_ref=session_ref
                )
            if restored_source is None:
                _log_event("cache_miss", session_ref=session_ref, slot_id=candidate_slot_id)
            else:
                _log_event(
                    "cache_hit",
                    layer="session" if restored_source == session_file else "prefix",
                    session_ref=session_ref,
                    slot_id=candidate_slot_id,
                )
            slot_id = candidate_slot_id if restored_source is not None else None
            if slot_id is None:
                slot_task_ids = self._slot_task_ids()
        plan = SnapshotPlan(
            session_id=session_id,
            slot_id=slot_id,
            prefix_key=prefix_key,
            session_file=session_file,
            prefix_file=prefix_file,
            prefix_payload=prefix_payload,
            prefix_was_present=prefix_file.exists() and (restored_source is not None or hot_state is not None),
            candidate_slot_id=candidate_slot_id,
            slot_task_ids=slot_task_ids,
        )
        return with_slot_cache(body, slot_id), plan

    @staticmethod
    def _touch_snapshot(source: Path) -> int | None:
        if not source.exists():
            return None
        try:
            source.touch()
            return source.stat().st_size
        except OSError as error:
            _log_event(
                "snapshot_touch_failed",
                level=logging.WARNING,
                filename=source.name,
                error_type=type(error).__name__,
            )
            return None

    def prepare_uncached(self, reason: str = "before_uncached") -> None:
        """Protect dirty slots before a request that bypasses snapshot tracking."""
        self._flush_dirty_states(reason)
        invalidated = len(self.session_states)
        self.session_states.clear()
        _log_event("slot_ownership_reset", reason=reason, invalidated_sessions=invalidated)

    def _restore_first_available(
        self,
        slot_id: int,
        sources: tuple[Path, ...],
        session_ref: str | None = None,
    ) -> Path | None:
        for source in sources:
            if not source.exists():
                continue
            started = time.monotonic()
            try:
                n_restored = self._restore(slot_id, source)
            except (RuntimeError, TimeoutError, OSError) as error:
                _log_event(
                    "snapshot_restore_failed",
                    level=logging.WARNING,
                    session_ref=session_ref,
                    filename=source.name,
                    slot_id=slot_id,
                    error_type=type(error).__name__,
                )
                continue
            snapshot_bytes = self._touch_snapshot(source)
            _log_event(
                "snapshot_restore",
                session_ref=session_ref,
                filename=source.name,
                slot_id=slot_id,
                tokens=n_restored,
                duration_ms=round((time.monotonic() - started) * 1000, 3),
                snapshot_bytes=snapshot_bytes,
            )
            return source
        return None

    def finish(
        self,
        plan: SnapshotPlan,
        status: int,
        finish_reason: str | None = None,
        cached_tokens: int | None = None,
    ) -> None:
        if status < 200 or status >= 300:
            return
        slot_id = self._resolve_slot(plan)
        if plan.slot_id is None:
            self._forget_slot(slot_id)

        session_ref = _session_ref(plan.session_id)
        defer = self.save_policy == "terminal" and finish_reason == "tool_calls"
        did_save = False
        if defer:
            n_tokens = self._slot_token_count(slot_id)
            if n_tokens > 0:
                state = SlotState(slot_id, plan.prefix_key, n_tokens, plan.session_file, True)
                _log_event(
                    "snapshot_deferred",
                    session_ref=session_ref,
                    slot_id=slot_id,
                    tokens=n_tokens,
                    finish_reason=finish_reason,
                    cached_tokens=cached_tokens,
                )
            else:
                # Unknown slot metadata must fail safe: persist instead of risking
                # eviction of an uncheckpointed conversation.
                n_saved = self._save(
                    slot_id,
                    plan.session_file,
                    reason="metadata_fallback",
                    session_ref=session_ref,
                )
                state = SlotState(slot_id, plan.prefix_key, n_saved, plan.session_file, False)
                did_save = True
        else:
            n_saved = self._save(
                slot_id,
                plan.session_file,
                reason="response_complete",
                session_ref=session_ref,
            )
            state = SlotState(slot_id, plan.prefix_key, n_saved, plan.session_file, False)
            did_save = True
        self.session_states[plan.session_id] = state
        _log_event(
            "request_complete",
            session_ref=session_ref,
            slot_id=slot_id,
            finish_reason=finish_reason,
            cached_tokens=cached_tokens,
            snapshot_saved=did_save,
        )
        if not plan.prefix_was_present:
            self._schedule_prefix_seed(replace(plan, slot_id=slot_id))
        if did_save:
            self._prune()

    def _slot_token_count(self, slot_id: int) -> int:
        try:
            slots = self._slots()
        except (RuntimeError, TimeoutError, OSError, TypeError, ValueError) as error:
            _log_event(
                "slot_metadata_unavailable",
                level=logging.WARNING,
                slot_id=slot_id,
                error_type=type(error).__name__,
            )
            return 0
        for slot in slots:
            if int(slot.get("id", -1)) == slot_id and not slot.get("is_processing"):
                return int(slot.get("n_prompt_tokens") or 0)
        return 0

    def _flush_dirty_states(self, reason: str) -> None:
        for session_id, state in list(self.session_states.items()):
            if not state.dirty or state.session_file is None:
                continue
            session_ref = _session_ref(session_id)
            n_saved = self._save(
                state.slot_id,
                state.session_file,
                reason=reason,
                session_ref=session_ref,
            )
            self.session_states[session_id] = replace(state, n_tokens=n_saved, dirty=False)
            _log_event(
                "snapshot_flush",
                session_ref=session_ref,
                slot_id=state.slot_id,
                filename=state.session_file.name,
                reason=reason,
                tokens=n_saved,
            )

    def flush_dirty(self, reason: str = "shutdown") -> None:
        with self.foreground_operation():
            self._flush_dirty_states(reason)
            self._prune()

    def _hot_state(self, state: SlotState | None, prefix_key: str) -> SlotState | None:
        if state is None or state.prefix_key != prefix_key:
            return None
        for slot in self._slots():
            if int(slot.get("id", -1)) != state.slot_id:
                continue
            if slot.get("is_processing"):
                return None
            if int(slot.get("n_prompt_tokens") or 0) != state.n_tokens:
                return None
            return state
        return None

    def _forget_slot(self, slot_id: int) -> None:
        self.session_states = {
            key: state for key, state in self.session_states.items() if state.slot_id != slot_id
        }

    @staticmethod
    def _prefix_seed_payload(body: dict[str, Any]) -> dict[str, Any]:
        payload = build_prefix_payload(body)
        payload.update(
            {
                "add_generation_prompt": False,
                "cache_prompt": False,
                "n_predict": 0,
                "stream": False,
            }
        )
        return payload

    def _schedule_prefix_seed(self, plan: SnapshotPlan) -> None:
        if not self.enable_prefix_seeding:
            return
        if not plan.prefix_payload.get("messages") and not plan.prefix_payload.get("tools"):
            return
        with self.prefix_seed_lock:
            if plan.prefix_file in self.prefix_seeds_in_flight:
                return
            self.prefix_seeds_in_flight.add(plan.prefix_file)
        threading.Thread(
            target=self._seed_prefix,
            args=(plan.prefix_payload, plan.prefix_file, plan.slot_id),
            name="local-llm-kv-prefix-seed",
            daemon=True,
        ).start()

    def _seed_prefix(self, prefix_payload: dict[str, Any], prefix_file: Path, excluded_slot_id: int) -> None:
        try:
            time.sleep(self.prefix_seed_delay_seconds)
            if self._has_foreground_waiters():
                LOGGER.info("prefix seed skipped while foreground request is waiting: %s", prefix_file.name)
                return
            if not self.operation_lock.acquire(blocking=False):
                LOGGER.info("prefix seed skipped while proxy is busy: %s", prefix_file.name)
                return
            try:
                idle_slots = [
                    slot
                    for slot in self._idle_slots({excluded_slot_id})
                    if int(slot.get("id", -1)) not in {state.slot_id for state in self.session_states.values()}
                ]
                if not idle_slots:
                    LOGGER.info("prefix seed skipped: no safe idle slot: %s", prefix_file.name)
                    return
                slot_id = min(idle_slots, key=lambda slot: int(slot.get("n_prompt_tokens") or 0)).get("id", 0)
                request = self._prefix_seed_payload(prefix_payload)
                request["id_slot"] = int(slot_id)
                self._json_request("POST", "/v1/chat/completions", request)
                self._save(int(slot_id), prefix_file)
                self._forget_slot(int(slot_id))
                self._prune()
                LOGGER.info("seeded stable prefix -> %s", prefix_file.name)
            finally:
                self.operation_lock.release()
        except Exception:
            LOGGER.exception("prefix seed failed: %s", prefix_file.name)
        finally:
            with self.prefix_seed_lock:
                self.prefix_seeds_in_flight.discard(prefix_file)

    def _has_foreground_waiters(self) -> bool:
        with self.foreground_waiters_lock:
            return self.foreground_waiters > 0

    def _slot_task_ids(self) -> dict[int, int]:
        return {
            int(slot.get("id", -1)): int(slot.get("id_task") or -1)
            for slot in self._slots()
        }

    def _resolve_slot(self, plan: SnapshotPlan) -> int:
        if plan.slot_id is not None:
            return plan.slot_id
        try:
            slots = [slot for slot in self._slots() if not slot.get("is_processing")]
        except (RuntimeError, TimeoutError, OSError, TypeError, ValueError) as error:
            if plan.candidate_slot_id is None:
                raise
            _log_event(
                "slot_resolution_fallback",
                level=logging.WARNING,
                slot_id=plan.candidate_slot_id,
                error_type=type(error).__name__,
            )
            return plan.candidate_slot_id
        changed = [
            slot
            for slot in slots
            if int(slot.get("id_task") or -1) != plan.slot_task_ids.get(int(slot.get("id", -1)), -1)
        ]
        candidates = changed or slots
        if candidates:
            return int(max(candidates, key=lambda slot: int(slot.get("id_task") or -1)).get("id", 0))
        if plan.candidate_slot_id is not None:
            return plan.candidate_slot_id
        return self._wait_for_idle_slot()

    def _idle_slots(self, excluded_slot_ids: set[int] | None = None) -> list[dict[str, Any]]:
        excluded = excluded_slot_ids or set()
        return [
            slot
            for slot in self._slots()
            if not slot.get("is_processing") and int(slot.get("id", -1)) not in excluded
        ]

    @contextmanager
    def foreground_operation(self):
        with self.foreground_waiters_lock:
            self.foreground_waiters += 1
        try:
            self.operation_lock.acquire()
        except BaseException:
            with self.foreground_waiters_lock:
                self.foreground_waiters -= 1
            raise
        with self.foreground_waiters_lock:
            self.foreground_waiters -= 1
        try:
            yield
        finally:
            self.operation_lock.release()

    def forward(
        self,
        handler: BaseHTTPRequestHandler,
        method: str,
        path: str,
        body: bytes,
    ) -> ForwardResult:
        upstream = HTTPConnection(self.upstream_host, self.upstream_port, timeout=1200)
        headers = self._forward_headers(handler)
        is_head = method == "HEAD"
        upstream.request(method, f"{self.upstream_prefix}{path}", body=body, headers=headers)
        response = upstream.getresponse()
        setattr(handler, "_proxy_response_started", True)
        handler.send_response(response.status, response.reason)
        for key, value in response.getheaders():
            normalized_key = key.lower()
            if normalized_key in HOP_BY_HOP_HEADERS and not (
                is_head and normalized_key == "content-length"
            ):
                continue
            handler.send_header(key, value)
        if not is_head:
            handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        metadata_buffer = bytearray()
        try:
            while True:
                chunk = response.read(64 * 1024)
                if not chunk:
                    break
                metadata_buffer.extend(chunk)
                if len(metadata_buffer) > MAX_METADATA_BYTES:
                    del metadata_buffer[:-MAX_METADATA_BYTES]
                if not is_head:
                    handler.wfile.write(f"{len(chunk):X}\r\n".encode("ascii"))
                    handler.wfile.write(chunk)
                    handler.wfile.write(b"\r\n")
                    handler.wfile.flush()
            if not is_head:
                handler.wfile.write(b"0\r\n\r\n")
                handler.wfile.flush()
            return ForwardResult(response.status, _completion_metadata([bytes(metadata_buffer)]))
        except (BrokenPipeError, ConnectionResetError):
            LOGGER.debug("client disconnected while proxying %s %s", method, path)
            return ForwardResult(response.status, _completion_metadata([bytes(metadata_buffer)]))
        finally:
            upstream.close()

    def _forward_headers(self, handler: BaseHTTPRequestHandler) -> dict[str, str]:
        headers = {"Content-Type": handler.headers.get("Content-Type", "application/json")}
        for name in (
            "Authorization",
            "Accept",
            "User-Agent",
            "Origin",
            "Access-Control-Request-Method",
            "Access-Control-Request-Headers",
            "X-Api-Key",
            "Anthropic-Version",
            "Anthropic-Beta",
            "OpenAI-Organization",
            "OpenAI-Project",
            "Idempotency-Key",
        ):
            value = handler.headers.get(name)
            if value:
                headers[name] = value
        return headers

    def _json_request(self, method: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        connection = HTTPConnection(self.upstream_host, self.upstream_port, timeout=120)
        try:
            connection.request(
                method,
                f"{self.upstream_prefix}{path}",
                body=body,
                headers={"Content-Type": "application/json"},
            )
            response = connection.getresponse()
            raw = response.read()
            if response.status < 200 or response.status >= 300:
                raise RuntimeError(f"llama {method} {path} returned {response.status}: {raw[:240]!r}")
            return json.loads(raw or b"{}")
        finally:
            connection.close()

    def _slots(self) -> list[dict[str, Any]]:
        connection = HTTPConnection(self.upstream_host, self.upstream_port, timeout=10)
        try:
            connection.request("GET", f"{self.upstream_prefix}/slots")
            response = connection.getresponse()
            if response.status != 200:
                raise RuntimeError(f"llama GET /slots returned {response.status}")
            data = json.loads(response.read())
            return data if isinstance(data, list) else []
        finally:
            connection.close()

    def _wait_for_idle_slot(self) -> int:
        deadline = time.monotonic() + self.wait_seconds
        while time.monotonic() < deadline:
            slots = self._idle_slots()
            if slots:
                owned_slots = {state.slot_id for state in self.session_states.values()}
                unowned_slots = [slot for slot in slots if int(slot.get("id", -1)) not in owned_slots]
                candidates = unowned_slots or slots
                return min(candidates, key=lambda slot: int(slot.get("n_prompt_tokens") or 0)).get("id", 0)
            time.sleep(0.5)
        raise TimeoutError("no idle llama slot became available")

    def _restore(self, slot_id: int, filename: Path) -> int:
        result = self._json_request(
            "POST",
            f"/slots/{slot_id}?action=restore",
            {"filename": filename.name},
        )
        n_restored = int(result.get("n_restored") or 0)
        if n_restored <= 0:
            raise RuntimeError(f"llama restored no tokens from {filename.name}")
        return n_restored

    def _save(
        self,
        slot_id: int,
        target: Path,
        reason: str | None = None,
        session_ref: str | None = None,
    ) -> int:
        started = time.monotonic()
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.unlink(missing_ok=True)
        try:
            result = self._json_request(
                "POST",
                f"/slots/{slot_id}?action=save",
                {"filename": temporary.name},
            )
            if int(result.get("n_saved") or 0) <= 0 or not temporary.exists():
                raise RuntimeError(f"llama saved no tokens for slot {slot_id}")
            temporary.replace(target)
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError as error:
                _log_event(
                    "snapshot_temp_cleanup_failed",
                    level=logging.WARNING,
                    filename=temporary.name,
                    error_type=type(error).__name__,
                )
        n_saved = int(result["n_saved"])
        _log_event(
            "snapshot_save",
            filename=target.name,
            slot_id=slot_id,
            tokens=n_saved,
            duration_ms=round((time.monotonic() - started) * 1000, 3),
            snapshot_bytes=target.stat().st_size,
            reason=reason,
            session_ref=session_ref,
        )
        return n_saved

    def _prune(self) -> None:
        files = sorted(
            self.cache_dir.glob("local-llm-*.bin"),
            key=lambda path: path.stat().st_mtime,
        )
        total = sum(path.stat().st_size for path in files)
        while total > self.max_cache_bytes and files:
            victim = files.pop(0)
            size = victim.stat().st_size
            victim.unlink(missing_ok=True)
            total -= size
            _log_event(
                "snapshot_prune",
                filename=victim.name,
                snapshot_bytes=size,
                cache_bytes=total,
            )


def _session_id(handler: BaseHTTPRequestHandler) -> str | None:
    for header in (
        "X-Session-Affinity",
        "X-Session-Id",
        "X-Conversation-Id",
        "X-Pi-Session-Id",
        "X-OpenCode-Session",
        "X-Client-Request-Id",
    ):
        value = handler.headers.get(header)
        if value:
            return value.strip()
    return None


def _body_session_id(body: dict[str, Any]) -> str | None:
    containers = [body]
    extra_body = body.get("extra_body")
    if isinstance(extra_body, dict):
        containers.append(extra_body)
    for container in containers:
        for affinity_field in ("session_id", "conversation_id", "prompt_cache_key"):
            value = container.get(affinity_field)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _without_proxy_affinity_fields(body: dict[str, Any]) -> dict[str, Any]:
    request = copy.deepcopy(body)
    for affinity_field in ("session_id", "conversation_id", "prompt_cache_key"):
        request.pop(affinity_field, None)
    extra_body = request.get("extra_body")
    if isinstance(extra_body, dict):
        for affinity_field in ("session_id", "conversation_id", "prompt_cache_key"):
            extra_body.pop(affinity_field, None)
        if not extra_body:
            request.pop("extra_body")
    return request


def _anonymous_session_id(body: dict[str, Any]) -> str:
    return f"anonymous-{cache_key(body)}"


def _has_media(body: dict[str, Any]) -> bool:
    if body.get("images"):
        return True
    for message in body.get("messages") or []:
        content = message.get("content")
        if isinstance(content, list) and any(isinstance(item, dict) and item.get("type") != "text" for item in content):
            return True
    return False


class ProxyHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    proxy: LlamaCacheProxy

    def do_GET(self) -> None:
        if _is_blocked_path("GET", self.path):
            self.send_error(404)
            return
        try:
            self.proxy.forward(self, "GET", _upstream_path("GET", self.path), b"")
        except (RuntimeError, TimeoutError, OSError) as error:
            self._send_upstream_error(error)

    def do_HEAD(self) -> None:
        if _is_blocked_path("HEAD", self.path):
            self.send_error(404)
            return
        try:
            self.proxy.forward(self, "HEAD", _upstream_path("HEAD", self.path), b"")
        except (RuntimeError, TimeoutError, OSError) as error:
            self._send_upstream_error(error)

    def do_OPTIONS(self) -> None:
        if _is_blocked_path("OPTIONS", self.path):
            self.send_error(404)
            return
        try:
            self.proxy.forward(self, "OPTIONS", self.path, b"")
        except (RuntimeError, TimeoutError, OSError) as error:
            self._send_upstream_error(error)

    def _send_upstream_error(self, error: Exception) -> None:
        LOGGER.exception("upstream request failed")
        if not getattr(self, "_proxy_response_started", False) and not self.wfile.closed:
            self.send_error(503, f"upstream unavailable: {error}")

    def do_POST(self) -> None:
        if _is_blocked_path("POST", self.path):
            self.close_connection = True
            self.send_error(404)
            return
        if not _is_known_post_path(self.path):
            self.close_connection = True
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            self.close_connection = True
            self.send_error(400, "Content-Length must be an integer")
            return
        if length < 0:
            self.close_connection = True
            self.send_error(400, "Content-Length must not be negative")
            return
        raw = self.rfile.read(length)
        upstream_path = _upstream_path("POST", self.path)
        if _is_non_evicting_post_path(self.path):
            try:
                self.proxy.forward(self, "POST", upstream_path, raw)
            except (RuntimeError, TimeoutError, OSError) as error:
                self._send_upstream_error(error)
            return
        if _normalized_path(self.path) != "/v1/chat/completions":
            try:
                with self.proxy.foreground_operation():
                    self.proxy.prepare_uncached("before_unmanaged_post")
                    self.proxy.forward(self, "POST", upstream_path, raw)
            except (RuntimeError, TimeoutError, OSError) as error:
                self._send_upstream_error(error)
            return
        try:
            body = json.loads(raw or b"{}")
        except json.JSONDecodeError:
            self.send_error(400, "request body must be JSON")
            return
        if not isinstance(body, dict):
            self.send_error(400, "request body must be a JSON object")
            return
        header_session_id = _session_id(self)
        body_session_id = _body_session_id(body)
        body = _without_proxy_affinity_fields(body)
        if _has_media(body):
            try:
                with self.proxy.foreground_operation():
                    self.proxy.prepare_uncached("before_media")
                    self.proxy.forward(self, "POST", self.path, json.dumps(body).encode("utf-8"))
            except (RuntimeError, TimeoutError, OSError) as error:
                self._send_upstream_error(error)
            return
        explicit_session_id = header_session_id or body_session_id
        if explicit_session_id is None and self.proxy.require_session_id:
            self.send_error(400, "a stable session ID is required")
            return
        session_id = explicit_session_id or _anonymous_session_id(body)
        _log_event(
            "session_resolve",
            session_ref=_session_ref(session_id),
            source="header" if header_session_id else "body" if body_session_id else "anonymous",
        )
        try:
            with self.proxy.foreground_operation():
                request, plan = self.proxy.prepare(body, session_id)
                result = self.proxy.forward(
                    self,
                    "POST",
                    self.path,
                    json.dumps(request).encode("utf-8"),
                )
                try:
                    self.proxy.finish(
                        plan,
                        result.status,
                        finish_reason=result.metadata.finish_reason,
                        cached_tokens=result.metadata.cached_tokens,
                    )
                except Exception:
                    LOGGER.exception("response succeeded but snapshot finalization failed")
        except (RuntimeError, TimeoutError, OSError) as error:
            self._send_upstream_error(error)

    def do_DELETE(self) -> None:
        self.close_connection = True
        self.send_error(404)

    def log_message(self, format: str, *args: Any) -> None:
        LOGGER.info("%s - %s", self.address_string(), format % args)


def main() -> None:
    logging.basicConfig(
        level=os.environ.get("PI_LLAMA_CACHE_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(message)s",
    )
    proxy = LlamaCacheProxy(
        upstream=os.environ.get("PI_LLAMA_UPSTREAM", "http://127.0.0.1:8080"),
        cache_dir=os.environ.get("PI_LLAMA_CACHE_DIR", DEFAULT_CACHE_DIR),
        max_cache_gib=float(os.environ.get("PI_LLAMA_CACHE_MAX_GIB", "12")),
        wait_seconds=float(os.environ.get("PI_LLAMA_CACHE_WAIT_SECONDS", "120")),
        enable_prefix_seeding=_env_bool("PI_LLAMA_CACHE_ENABLE_PREFIX_SEEDING", True),
        prefix_seed_delay_seconds=float(os.environ.get("PI_LLAMA_CACHE_PREFIX_SEED_DELAY", "2")),
        save_policy=os.environ.get("PI_LLAMA_CACHE_SAVE_POLICY", "all").strip().lower(),
        require_session_id=_env_bool("PI_LLAMA_CACHE_REQUIRE_SESSION_ID", False),
    )
    ProxyHandler.proxy = proxy
    host = os.environ.get("PI_LLAMA_CACHE_HOST", "127.0.0.1")
    port = int(os.environ.get("PI_LLAMA_CACHE_PORT", "8081"))
    server = ThreadingHTTPServer((host, port), ProxyHandler)
    LOGGER.info("listening on http://%s:%d -> http://%s:%d", host, port, proxy.upstream_host, proxy.upstream_port)
    previous_sigterm = signal.getsignal(signal.SIGTERM)

    def stop_on_sigterm(_signum: int, _frame: Any) -> None:
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop_on_sigterm)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        try:
            proxy.flush_dirty("shutdown")
        except Exception:
            LOGGER.exception("failed to flush dirty snapshots during shutdown")
        signal.signal(signal.SIGTERM, previous_sigterm)
        server.server_close()


if __name__ == "__main__":
    main()
