"""Pure cache-key and request helpers for the Pi llama proxy."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any


_PREFIX_FIELDS = (
    "model",
    "tools",
    "tool_choice",
    "chat_template_kwargs",
    "chat_template_args",
    "enable_thinking",
    "reasoning_effort",
    "reasoning_format",
    "response_format",
    "json_schema",
    "grammar",
    "add_generation_prompt",
    "continue_final_message",
    "parallel_tool_calls",
)
_CACHE_FORMAT_VERSION = "3"
_SYSTEM_ROLES = {"system", "developer"}
MAX_MANIFEST_TOKENS = 1_000_000
MAX_TOKEN_ID = 2**31 - 1
_MANIFEST_KEYS = {"version", "namespace", "scope", "snapshot", "tokens"}


def normalize_tokens(value: Any) -> tuple[int, ...]:
    """Validate llama token IDs and return an immutable, bounded sequence."""
    if not isinstance(value, (list, tuple)) or len(value) > MAX_MANIFEST_TOKENS:
        raise ValueError("tokens must be a bounded array")
    tokens: list[int] = []
    for token in value:
        if isinstance(token, bool) or not isinstance(token, int) or not 0 <= token <= MAX_TOKEN_ID:
            raise ValueError("token IDs must be non-negative 32-bit integers")
        tokens.append(token)
    if not tokens:
        raise ValueError("tokens must not be empty")
    return tuple(tokens)


def longest_common_prefix(left: tuple[int, ...], right: tuple[int, ...]) -> int:
    """Return the number of byte-for-byte equal token positions."""
    for index, (left_token, right_token) in enumerate(zip(left, right)):
        if left_token != right_token:
            return index
    return min(len(left), len(right))


def manifest_filename(snapshot: str) -> str:
    """Return the deterministic private metadata companion name."""
    if not isinstance(snapshot, str) or not snapshot or os.path.basename(snapshot) != snapshot:
        raise ValueError("snapshot must be a basename")
    return f"{snapshot}.manifest.json"


@dataclass(frozen=True)
class PrefixManifest:
    namespace: str
    scope: str
    snapshot: str
    tokens: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.namespace, str) or not self.namespace.strip():
            raise ValueError("namespace must not be empty")
        if not isinstance(self.scope, str) or not self.scope.strip():
            raise ValueError("scope must not be empty")
        manifest_filename(self.snapshot)
        object.__setattr__(self, "tokens", normalize_tokens(self.tokens))

    def to_json(self) -> str:
        return json.dumps(
            {
                "version": 1,
                "namespace": self.namespace,
                "scope": self.scope,
                "snapshot": self.snapshot,
                "tokens": list(self.tokens),
            },
            sort_keys=True,
            separators=(",", ":"),
        )

    @classmethod
    def from_json(cls, raw: str | bytes) -> PrefixManifest:
        try:
            value = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as error:
            raise ValueError("manifest must be valid JSON") from error
        if (
            not isinstance(value, dict)
            or set(value) != _MANIFEST_KEYS
            or type(value.get("version")) is not int
            or value.get("version") != 1
        ):
            raise ValueError("unsupported manifest schema")
        return cls(value.get("namespace"), value.get("scope"), value.get("snapshot"), value.get("tokens"))


def best_prefix_manifest(
    candidates: list[PrefixManifest],
    request_tokens: tuple[int, ...],
    namespace: str,
    scope: str,
    minimum_lcp: int,
) -> tuple[PrefixManifest, int] | None:
    """Select the in-scope candidate with the longest exact common prefix."""
    request_tokens = normalize_tokens(request_tokens)
    if isinstance(minimum_lcp, bool) or not isinstance(minimum_lcp, int) or minimum_lcp < 1:
        raise ValueError("minimum_lcp must be a positive integer")
    best: tuple[PrefixManifest, int] | None = None
    for candidate in candidates:
        if candidate.namespace != namespace or candidate.scope != scope:
            continue
        lcp = longest_common_prefix(candidate.tokens, request_tokens)
        if lcp < minimum_lcp:
            continue
        if best is None or lcp > best[1]:
            best = (candidate, lcp)
    return best


def build_prefix_payload(body: dict[str, Any]) -> dict[str, Any]:
    """Return request fields that affect the stable prompt prefix."""
    messages = []
    for message in body.get("messages") or []:
        if message.get("role") not in _SYSTEM_ROLES:
            break
        messages.append(copy.deepcopy(message))

    prefix = {"messages": messages}
    for field in _PREFIX_FIELDS:
        if field in body:
            prefix[field] = copy.deepcopy(body[field])
    return prefix


def cache_key(body: dict[str, Any]) -> str:
    """Hash a canonical stable prefix so project changes invalidate it."""
    encoded = json.dumps(
        build_prefix_payload(body),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_filename(identity: str, body: dict[str, Any], kind: str) -> str:
    """Build a filesystem-safe filename scoped to an identity, namespace, and prefix."""
    namespace = os.environ.get("PI_LLAMA_CACHE_NAMESPACE", "default").strip() or "default"
    material = f"{_CACHE_FORMAT_VERSION}\0{namespace}\0{kind}\0{identity}\0{cache_key(body)}".encode("utf-8")
    digest = hashlib.sha256(material).hexdigest()
    return f"local-llm-{kind}-{digest}.bin"


def with_slot_cache(body: dict[str, Any], slot_id: int | None) -> dict[str, Any]:
    """Copy a request and optionally pin it to a llama slot."""
    request = copy.deepcopy(body)
    if slot_id is not None:
        request["id_slot"] = int(slot_id)
    return request
