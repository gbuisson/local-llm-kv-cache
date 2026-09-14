import os
import unittest
from unittest.mock import patch

from cache_core import (
    PrefixManifest,
    best_prefix_manifest,
    build_prefix_payload,
    cache_key,
    cache_filename,
    longest_common_prefix,
    manifest_filename,
    normalize_tokens,
    with_slot_cache,
)


class CacheCoreTests(unittest.TestCase):
    def setUp(self):
        self.body = {
            "model": "qwen3.8:27b",
            "messages": [
                {"role": "system", "content": "project rules"},
                {"role": "user", "content": "first request"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "read",
                        "description": "Read a file",
                        "parameters": {"type": "object"},
                    },
                }
            ],
            "chat_template_kwargs": {
                "enable_thinking": True,
                "preserve_thinking": True,
            },
        }

    def test_user_message_does_not_change_project_prefix_key(self):
        other = {**self.body, "messages": [
            self.body["messages"][0],
            {"role": "user", "content": "different request"},
        ]}

        self.assertEqual(cache_key(self.body), cache_key(other))

    def test_project_rules_change_invalidates_prefix_key(self):
        other = {**self.body, "messages": [
            {"role": "system", "content": "changed project rules"},
            self.body["messages"][1],
        ]}

        self.assertNotEqual(cache_key(self.body), cache_key(other))

    def test_tool_schema_change_invalidates_prefix_key(self):
        other = {**self.body, "tools": [
            {
                "type": "function",
                "function": {
                    "name": "read",
                    "description": "Read a file with a changed contract",
                    "parameters": {"type": "object"},
                },
            }
        ]}

        self.assertNotEqual(cache_key(self.body), cache_key(other))

    def test_prefix_payload_keeps_only_stable_leading_messages(self):
        prefix = build_prefix_payload(self.body)

        self.assertEqual(prefix["messages"], [self.body["messages"][0]])
        self.assertEqual(prefix["tools"], self.body["tools"])
        self.assertNotIn("first request", str(prefix))

    def test_slot_request_only_adds_slot_affinity(self):
        request = with_slot_cache(self.body, 1)

        self.assertEqual(request["id_slot"], 1)
        self.assertNotIn("cache_prompt", request)
        self.assertEqual(request["messages"], self.body["messages"])

    def test_slot_request_can_leave_slot_selection_to_llama(self):
        request = with_slot_cache(self.body, None)

        self.assertNotIn("id_slot", request)
        self.assertNotIn("cache_prompt", request)

    def test_cache_filename_is_safe_and_stable(self):
        name = cache_filename("session/with spaces", self.body, "session")

        self.assertRegex(name, r"^local-llm-session-[0-9a-f]{64}\.bin$")
        self.assertNotIn("/", name)
        self.assertEqual(name, cache_filename("session/with spaces", self.body, "session"))

    def test_cache_namespace_invalidates_snapshot_filename(self):
        with patch.dict(os.environ, {"PI_LLAMA_CACHE_NAMESPACE": "model-a"}):
            model_a = cache_filename("session", self.body, "session")
        with patch.dict(os.environ, {"PI_LLAMA_CACHE_NAMESPACE": "model-b"}):
            model_b = cache_filename("session", self.body, "session")

        self.assertNotEqual(model_a, model_b)

    def test_token_normalization_and_exact_lcp(self):
        self.assertEqual(normalize_tokens([1, 2, 0]), (1, 2, 0))
        self.assertEqual(longest_common_prefix((1, 2, 3), (1, 2, 4, 5)), 2)
        self.assertEqual(longest_common_prefix((1, 2), (1, 2)), 2)
        for malformed in ([], [1, True], [1, -1], [1, 2**31], "1,2", None):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                normalize_tokens(malformed)

    def test_manifest_round_trip_is_immutable_and_deterministic(self):
        manifest = PrefixManifest("private", "runtime-a", "prefix.bin", (11, 12, 13))
        encoded = manifest.to_json()

        self.assertEqual(PrefixManifest.from_json(encoded), manifest)
        self.assertEqual(manifest_filename("prefix.bin"), "prefix.bin.manifest.json")
        self.assertNotIn("prompt", encoded)
        with self.assertRaises(AttributeError):
            manifest.tokens = (1,)

    def test_manifest_rejects_malformed_scope_namespace_and_tokens(self):
        valid = {"version": 1, "namespace": "private", "scope": "runtime-a", "snapshot": "prefix.bin", "tokens": [1, 2]}
        malformed = [
            {},
            {**valid, "version": 2},
            {**valid, "namespace": ""},
            {**valid, "scope": ""},
            {**valid, "snapshot": "../prefix.bin"},
            {**valid, "tokens": [1, "2"]},
            {**valid, "extra": True},
        ]
        for value in malformed:
            with self.subTest(value=value), self.assertRaises(ValueError):
                PrefixManifest.from_json(__import__("json").dumps(value))
        for raw in (b"\xff", None):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                PrefixManifest.from_json(raw)

        for snapshot in ("", "../prefix.bin", 42):
            with self.subTest(snapshot=snapshot), self.assertRaises(ValueError):
                manifest_filename(snapshot)

    def test_best_manifest_requires_scope_and_selects_longest_exact_candidate(self):
        candidates = [
            PrefixManifest("private", "runtime-a", "short.bin", (1, 2)),
            PrefixManifest("other", "runtime-a", "wrong-namespace.bin", (1, 2, 3, 4)),
            PrefixManifest("private", "runtime-b", "wrong-scope.bin", (1, 2, 3, 4)),
            PrefixManifest("private", "runtime-a", "changed-skill.bin", (1, 2, 9, 4)),
            PrefixManifest("private", "runtime-a", "long.bin", (1, 2, 3, 4)),
        ]

        selected = best_prefix_manifest(candidates, (1, 2, 3, 8), "private", "runtime-a", minimum_lcp=2)

        self.assertEqual(selected, (candidates[-1], 3))
        self.assertIsNone(best_prefix_manifest(candidates, (7, 8), "private", "runtime-a", minimum_lcp=2))
        for minimum in (True, 0, 1.5):
            with self.subTest(minimum=minimum), self.assertRaises(ValueError):
                best_prefix_manifest(candidates, (1, 2), "private", "runtime-a", minimum)

    def test_best_manifest_prefers_complete_candidate_when_lcp_ties(self):
        divergent = PrefixManifest("private", "runtime-a", "divergent.bin", (1, 2, 3, 4))
        complete = PrefixManifest("private", "runtime-a", "complete.bin", (1, 2, 3))

        selected = best_prefix_manifest(
            [divergent, complete],
            (1, 2, 3, 9),
            "private",
            "runtime-a",
            minimum_lcp=2,
        )

        self.assertEqual(selected, (complete, 3))


if __name__ == "__main__":
    unittest.main()
