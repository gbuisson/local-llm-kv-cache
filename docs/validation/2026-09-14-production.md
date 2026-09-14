# Production validation — September 14, 2026

## Scope

This report links the measurements cited in [README.md](../../README.md) and [DESIGN.md](../../DESIGN.md) to the validated working version of the proxy. It is intentionally sanitized: no prompt, rendered prompt, user content, tool result, token ID array, raw session ID, credential, or manifest body is retained here.

Raw production logs are not versioned because they contain correlatable pseudonymous references and private operational paths; only the sanitized production evidence below is versioned. The values below come from the proxy's structured counters, OpenAI metadata returned by llama.cpp, and verification commands run on the stated date.

## Validated version

```text
Repository  : local-llm-kv-cache
Branch      : feature/session-cache-production-hardening
Code commit : 382b469a8ebf46499c58c9a0b9ede11b92359eec
Commit date : 2026-09-14T10:35:58+02:00
```

SHA-256 hashes of the source and test files:

```text
3d5b61a7b3cd478c498e752656fc1d81e4d5c323cef38dfddf56002a0cbb0f5c  cache_core.py
6563b23fdc4b44ab442d26e9f2c086a3728c8d3ddbc639bd11271a677af076f1  cache_proxy.py
89ead62bc9614a9afd05975eb76cb56af6bea6d151fdf61f4eff0ad43d6b7862  test_cache_core.py
86436c74d038cada8e27ebaf7dacebc0a762253e86cebf1fe8f48961d3ef8ce7  test_cache_proxy.py
```

Validated Nix deployment:

```text
Proxy package : pinned to code commit 382b469a8ebf46499c58c9a0b9ede11b92359eec
Generation    : activated and verified; host-specific store path retained privately
Rollback      : previous generation retained and verified; store path retained privately
```

The `dry-activate`, activation, proxy restart, and health check all succeeded. After activation:

```text
proxy  = active
llama  = active
failed = 0
health = {"status":"ok"}
```

## Tests and coverage

Commands reproduced on the commit above:

```bash
python3 -m py_compile cache_core.py cache_proxy.py test_cache_core.py test_cache_proxy.py
python3 -m unittest -q

COVERAGE_FILE=/tmp/local-llm-kv-cache.coverage \
  nix shell nixpkgs#python3Packages.coverage --command sh -c \
  'coverage erase && coverage run --branch --source=. -m unittest discover -s . -p "test_*.py" && coverage report --include="cache_core.py,cache_proxy.py" --fail-under=100'

git diff --check
```

Results:

```text
Tests                  : 123 passed
cache_core.py          : 100% lines and branches
cache_proxy.py         : 100% lines and branches
Python compilation     : OK
git diff --check       : OK
```

The tests cover, among other cases: complete candidates only, rejection of a divergent snapshot, seeding of the exact tokenized LCP, preference for the complete candidate in the event of a tie, no legacy fallback across scopes, a bounded pending queue, preservation of the best seed, protection against a stale task, and fragmented streaming.

## OpenAI/SSE streaming

A real streaming call through the proxy produced the results below. The incremental-read fix was introduced by commit `9bcdb4d809c281ee0331a4f246157efe3dcfd00f`, an ancestor of the validated working commit `382b469a8ebf46499c58c9a0b9ede11b92359eec`; the hash of the final file is given above.

```text
HTTP status        : 200
Content type       : text/event-stream
SSE events         : 112
Observed duration  : 4.850 s
[DONE] marker      : present
Event JSON         : valid
```

The events arrived progressively during generation, rather than as a single terminal block. The automated tests also confirm the use of `HTTPResponse.read1()`, byte-for-byte forwarding of SSE payloads and `[DONE]`, filtering of the static list of hop-by-hop headers, and correct HTTP framing. They do not prove dynamic filtering of an extension header name advertised in the value of `Connection`; that filtering is not yet implemented.

## Real Hermes A/B validation

Two real variants of the Hermes prompt were exercised with new session affinities. For each one, a `shared_prefix_candidate_restored` event preceded a `shared_prefix_effective_hit`, and the response metadata confirmed a strictly positive `cached_tokens` value:

| Variant | `candidate_tokens` | `verified_lcp` | `cached_tokens` | Verdict |
| --- | ---: | ---: | ---: | --- |
| A | 24,464 | 24,464 | 24,464 | effective complete hit |
| B | 24,479 | 24,479 | 24,479 | effective complete hit |

No prompt body or token array was retained. These values are aggregate counters, not tokenized content.

An earlier real divergence had produced an LCP of approximately 19,411 tokens. This finding motivated the adaptive path. Deterministic evidence for this path on the final commit comes from the regression tests: a divergent candidate is never restored, the request remains cold, and only `request_tokens[:verified_lcp]` is offered for seeding. The two A/B validations above, for their part, prove complete hits for the two currently known variants; they do not claim to constitute a new internal-divergence trial.

## Contextualized historical measurements

These measurements come from earlier configurations in the same lineage and are provided only as performance context:

| Scenario | Observed result |
| --- | --- |
| 32,185-token prompt | 54.8 s cold prefill, then `cache_n=32151`, `prompt_n=34`, 0.37 s after restoration |
| 12,700-token snapshot after restart | `n_restored=12700`, then `cache_n=12662`, `prompt_n=38`, 0.356 s |
| Hermes measurement from September 12, 2026 | 25,960 tokens, `cached_tokens=0`, 233.94 s prompt evaluation, 110.97 tokens/s |
| Corresponding snapshot | 824,255,044 bytes, or 786.07 MiB |

These figures do not constitute a performance guarantee. The model, quantization, context, checkpoints, hardware, temperature, and load all affect the results.

## State after probe cleanup

Only artifacts explicitly created by the probes were deleted. No real Hermes golden or historical snapshot was removed:

```text
golden prefixes = 9
manifests       = 9
sessions        = 2
temporary       = 0
```

Any future deletion of a golden must follow the inventory, retention criteria, dry run, backup, and explicit confirmation described in the README.

## Evidence limitations

- A successful `/slots/{id}?action=restore` or `n_restored > 0` does not prove a KV hit; only `cached_tokens > 0` or `timings.cache_n > 0` confirms one.
- Raw logs are intentionally not included in this repository for privacy reasons.
- This report attests to the state observed on the stated date and version; it does not replace a new A/B test after a change to the model, template, tokenizer, namespace, or scope.
- Divergence measurements must never be reproduced by publishing the relevant tokens or prompt.
