# Hermes Golden Prefix Cache Implementation Plan

> **For Hermes:** Use subagent-driven-development skill to implement this plan task-by-task.

**Goal:** Reuse the longest byte-for-byte token prefix of a previous Hermes bootstrap across new sessions, while failing closed whenever model/runtime scope or prompt tokens differ.

**Architecture:** Render and tokenize each incoming chat prompt through llama.cpp's local `/apply-template` and `/tokenize` APIs. Store a private JSON manifest beside each pure-prefix snapshot containing the exact token IDs and deployment scope. For a new session, select only a manifest in the same namespace/scope whose token sequence has the longest exact common prefix with the incoming rendered prompt; restore its snapshot and let patched llama.cpp native LCP discard every state after the first divergence. Extend background seeding to transactionally swap the sole `np=1` slot only after its owning session is cleanly persisted, and restore that session before releasing the operation lock.

**Tech Stack:** Python 3 stdlib, llama.cpp HTTP slot API, `unittest`, NixOS/systemd.

---

### Task 1: Add token-manifest primitives

**Objective:** Represent and validate private prefix manifests and select the longest exact token prefix without relying on prompt text or heuristic metadata.

**Files:**
- Modify: `cache_core.py`
- Modify: `test_cache_core.py`

**Steps:**
1. Add failing tests for exact token normalization, longest-common-prefix length, scope/namespace mismatch, malformed manifests, changed skill/tool token sequences, and longest valid candidate selection.
2. Run the focused tests and verify RED.
3. Implement immutable manifest serialization/validation helpers with bounded integer token arrays and deterministic companion filenames.
4. Require exact token equality for every reused position; never accept a candidate on hash or ratio alone.
5. Run focused tests and the full suite.

### Task 2: Render/tokenize and restore a safe shared candidate

**Objective:** Make `prepare()` compute the actual rendered request tokens and restore only the longest validated shared candidate in the same deployment scope.

**Files:**
- Modify: `cache_proxy.py`
- Modify: `test_cache_proxy.py`

**Steps:**
1. Add failing tests for `/apply-template` then `/tokenize`, exact snapshot preference, partial LCP selection, changed-skill divergence, zero/short-LCP rejection, malformed/orphan manifest rejection, API failure fallback, and no raw prompt/session leakage.
2. Verify RED.
3. Implement local rendering/tokenization with strict response validation and configurable minimum shared prefix length.
4. Preserve hit order: hot session → exact session snapshot → best validated shared prefix → exact legacy prefix → cold.
5. Pin a successfully restored slot so native llama.cpp LCP runs on that state; if discovery or restore fails, perform a normal cold prefill.
6. Emit structured telemetry with candidate token count and verified LCP only, never prompt/token contents.
7. Run focused and full tests.

### Task 3: Add transactional one-slot seeding

**Objective:** Create golden prefix snapshots safely on an `np=1` server without losing or corrupting the current session.

**Files:**
- Modify: `cache_proxy.py`
- Modify: `test_cache_proxy.py`

**Steps:**
1. Add failing tests for clean-owner save → seed → atomic manifest → owner restore ordering; dirty-owner skip; missing snapshot skip; foreground waiter skip; seed/save failure rollback; owner-restore failure invalidating only hot ownership; and spare-slot behavior remaining unchanged.
2. Verify RED.
3. Use a spare unowned slot when available. Otherwise use the excluded owner slot only when it is idle, clean, token-count-consistent, and its session snapshot exists.
4. Prefill through `/completion` with the exact token array returned by `/tokenize`; token-count equality alone is not identity proof.
5. Keep the operation lock for the complete swap. Save the owner to a separate validated guard and always attempt owner restoration in `finally` after the seed request may have changed the slot.
6. Write snapshots and manifests atomically with private permissions. A manifest must never name a missing, partially written, or token-count-mismatched snapshot.
7. Restore/update ownership only when `n_restored` equals the expected owner token count; otherwise forget the hot slot and leave the original persisted session available for the next request.
8. Run focused and full tests.

### Task 4: Document the correctness and operational contract

**Objective:** Explain cross-session golden reuse, skill/config invalidation, privacy, one-slot maintenance cost, and rollback.

**Files:**
- Modify: `README.md`
- Modify: `DESIGN.md`

**Steps:**
1. Document exact token-prefix validation and examples for changed skills, tools, memory, dates, templates, and reasoning modes.
2. Document `PI_LLAMA_CACHE_SHARED_PREFIX_SCOPE`, minimum LCP, seed delay, and `np=1` swap semantics.
3. State that manifests contain reversible token IDs and inherit the snapshot's private-data classification.
4. Document metrics/log events and A/B acceptance commands.
5. Update diagrams and remove the stale statement that prefix seeding must be disabled for `np=1`.

### Task 5: Quality gates and independent review

**Objective:** Prove implementation correctness before deployment.

**Files:** all changed files.

**Steps:**
1. Run `python3 -m py_compile cache_core.py cache_proxy.py`.
2. Run `python3 -m unittest -v`.
3. Run branch coverage and require 100% for `cache_core.py` and `cache_proxy.py` using the existing coverage installation.
4. Run `ruff check` if available.
5. Perform spec-compliance review, then code-quality/security review; fix all critical and important findings and re-run gates.
6. Review staged blobs for secrets and unintended files.

### Task 6: Nix integration and Cortex A/B

**Objective:** Deploy the immutable reviewed commit and prove the cold-prefill improvement without stale-skill reuse.

**Files (in the separate private dotfiles repository):**
- Modify only `packages/local-llm-kv-cache/default.nix`
- Modify only `hosts/nixos/cortex-1.nix`

**Steps:**
1. Commit/push the application repo and verify local/remote SHA parity.
2. Pin the exact commit and Nix hash; enable prefix seeding, set an explicit personal scope, conservative delay, and minimum LCP.
3. Stage only the two intended dotfiles paths/hunks while preserving all unrelated dirty work.
4. Build the target NixOS host from a staged synthetic checkout and inspect the generated unit.
5. Activate with rollback generation recorded.
6. Run first-session seed, wait for completion, then run a distinct new Hermes session and verify `cached_tokens`, LCP telemetry, TTFT, memory, services, and GPU/RPC faults.
7. Change a controlled system/skill-like suffix in a probe prompt and prove reuse stops exactly before divergence; change an early token and prove shorter/fallback behavior.
8. Run same-session, tool-call, compaction, restart restore, fail-closed sessionless text, and vision regression checks.
9. Commit/push only the intended dotfiles changes after all gates pass.
