# Local LLM KV Cache Design

## 1. Goals and Boundaries

This design aims to reduce first-message latency when Hermes Agent, Pi, pi-acp, or Zed uses the Qwen3.8-27B coding agent.

Goals:

- cache the prefill result for the stable system prompt, project rules, and tool schemas;
- reuse an in-memory slot when continuing the same session;
- restore a stable prefix from disk after the proxy or llama.cpp restarts;
- cache no responses and change neither model generation nor sampling;
- preserve the existing llama.cpp + GGUF + AMD execution path;
- fail safely: any incomplete or inconsistent evidence must lead to a colder path, never to reuse of stale state.

Prerequisite: disk restoration for Qwen3.8 depends on hybrid checkpoint persistence. A stock llama.cpp build may report a positive `n_restored` while the subsequent `cache_n` remains 0. The deployment must therefore use commit `862535a` from the local `local/kv-restore-checkpoints` branch, or an equivalent upstream fix.

Out of scope:

- no response or result cache;
- no mixing of KV states across projects, trust domains, models, or incompatible runtimes;
- no presenting a complete old conversation as a shared cross-session prefix;
- no changes to model weights, training, or the decoding algorithm.

## 2. Overall Architecture

~~~mermaid
flowchart LR
    C["Hermes Agent / Pi / pi-acp / Zed"] -->|OpenAI Chat Completions| P["Cache proxy<br/>127.0.0.1:18082"]

    subgraph CACHE["Cache proxy"]
        A["Session affinity"]
        K["Prefix key + native rendering/tokenization"]
        H["Hot slot map<br/>session_states"]
        R["Layer selection<br/>hot -> session -> shared LCP -> legacy -> cold"]
        S["Snapshot manager"]
        M["Private token manifests<br/>namespace + scope"]
        Q["Bounded seed queue<br/>pending_prefix_seeds <= 16"]
        A --> K --> R
        M --> R
        H --> R
        R --> S
        R --> Q --> S
    end

    P --> CACHE
    S -->|slot save / restore| L["llama.cpp llama-server<br/>127.0.0.1:8080"]
    S -->|private snapshots and manifests| D[("~/.llama-slot-cache")]
    L --> G["Qwen3.8-27B<br/>GGUF + MTP"]
~~~

The provided template listens on the loopback interface. Hermes, Pi, and Zed then target port 18082; llama.cpp continues to listen on port 8080. A deployment may choose an explicit LAN bind provided it is protected by a firewall or private network: the proxy adds no authentication of its own. The proxy serializes operations that may modify slots by means of `operation_lock`.

## 3. Cached Content

The stable prefix generally contains:

~~~text
system prompt
developer prompt
project rules and AGENTS.md content
tool definitions and JSON schemas
chat template / thinking parameters
~~~

The dynamic suffix generally contains:

~~~text
current user message
assistant history
tool results
temporary context for the current task
~~~

~~~mermaid
flowchart LR
    A["Stable prefix<br/>system / developer / tools"] --> B["KV + GDN state<br/>persistent"]
    B --> C["Dynamic suffix<br/>user / assistant / tool"]
    C --> D["Decoding<br/>generate the current response"]

    style A fill:#d9f2d9,stroke:#397a3c
    style B fill:#d9e8ff,stroke:#3569a8
    style C fill:#fff1cc,stroke:#a87900
    style D fill:#ffe0e0,stroke:#a33a3a
~~~

The cache stores the model state corresponding to the prefix, not the response produced in stage D.

## 4. Key and Manifest Design

### 4.1 Prefix Key

Starting at the beginning of `messages`, the proxy retains the contiguous sequence of messages whose role is `system` or `developer`. It stops at the first `user`, `assistant`, or `tool` message.

It adds the following stable fields when present:

~~~text
model
tools
tool_choice
chat_template_kwargs
chat_template_args
enable_thinking
reasoning_effort
reasoning_format
response_format
json_schema
grammar
add_generation_prompt
continue_final_message
parallel_tool_calls
~~~

The key is then computed as follows:

~~~text
prefix_key = SHA256(canonical_json(prefix_payload))
~~~

Canonical JSON uses `ensure_ascii=false`, `sort_keys=true`, compact separators, and UTF-8 encoding.

~~~mermaid
flowchart TD
    Q["Original request body"] --> M["Read messages from the beginning"]
    M --> T{"system/developer role?"}
    T -->|yes| KEEP["Retain message"]
    KEEP --> T
    T -->|no| STOP["Stop at first dynamic message"]
    KEEP --> F["Add stable request fields"]
    STOP --> F
    F --> J["Canonical JSON"]
    J --> H["SHA-256"]
    H --> K["prefix_key"]
~~~

### 4.2 Snapshot Names

~~~text
snapshot_key = SHA256("3" + namespace + kind + identity + prefix_key)
~~~

The material actually hashed uses NUL separators between these fields. Files use the following names:

~~~text
local-llm-session-<hash>.bin
local-llm-prefix-<hash>.bin
~~~

- for `kind=session`, `identity` is the Pi/Zed session identifier;
- for `kind=prefix`, `identity` is the fixed string `prefix`;
- format version `3` and the deployment namespace globally invalidate old formats or incompatible deployments.

A session change therefore misses the session-specific snapshot, but may still benefit from a compatible shared prefix.

### 4.3 Golden Shared-Prefix Manifest

The canonical `prefix_key` is suitable for exact session lookups and the legacy fallback, but it does not prove how much state two different Hermes bootstraps can share. The golden path uses the exact tokens actually produced by the current llama.cpp as its sole correctness evidence:

~~~text
incoming chat body
  -> POST /apply-template
  -> response.prompt (complete rendered prompt)
  -> POST /tokenize {content, add_special:false, parse_special:true}
  -> exact integer token IDs
~~~

During a seed, the proxy atomically publishes a private companion in mode `0600` alongside the pure-prefix snapshot:

~~~text
local-llm-prefix-<hash>.bin.manifest.json
{version, namespace, scope, snapshot, tokens[]}
~~~

The manifest schema is strict: exactly the keys `version`, `namespace`, `scope`, `snapshot`, and `tokens`, with `version=1`. `tokens[]` must be non-empty, limited to 1,000,000 elements, and contain only non-negative 32-bit integers. `snapshot` must be a basename; the manifest name must exactly match that of its companion snapshot, which must exist. A manifest larger than 1 MiB, malformed, or orphaned is ignored.

Candidate selection follows these steps:

1. strictly filter on the same `PI_LLAMA_CACHE_NAMESPACE` and the same `PI_LLAMA_CACHE_SHARED_PREFIX_SCOPE`;
2. ignore every malformed, oversized, or orphaned manifest;
3. compare each candidate with the request tokens, token by token from index 0;
4. select the candidate with the longest exact LCP, provided that LCP reaches `PI_LLAMA_CACHE_MIN_SHARED_PREFIX_TOKENS` (128 by default);
5. when LCP lengths tie, prefer the complete candidate, meaning one whose entire token sequence is contained at the start of the request, over a longer candidate that diverges at the next token.

No heuristic based on prompt text, hash similarity, or match ratio is allowed. `PI_LLAMA_CACHE_SHARED_PREFIX_SCOPE` represents an explicit data and trust boundary: `personal` and `work` environments must use different values. The namespace must also be renewed when the model, runtime, template, or a KV configuration becomes incompatible.

### 4.4 Restore Complete Candidates Only

A shared prefix is restored only if **all** candidate tokens are exactly the beginning of the current request, that is, `verified_lcp == len(candidate_tokens)`. The restore must then report exactly `n_restored == len(manifest.tokens)`; otherwise, the snapshot/manifest pair is considered corrupt, removed on a best-effort basis, and processing degrades to a safe path.

If the best candidate diverges within its own snapshot (`verified_lcp < len(candidate_tokens)`), the proxy **does not restore it** and does not rely on llama.cpp to restore and then truncate it. The request remains cold and unpinned. After it succeeds, the proxy asynchronously schedules a seed of only `request_tokens[:verified_lcp]`, whose identity has been strictly verified. A subsequent request can fully restore this shorter golden snapshot.

The proxy never concatenates KV states itself. A change to a skill, tool schema, `system`/`developer` prompt, memory, date, template, or reasoning mode can therefore never produce a hit beyond the first divergence.

## 5. Request Resolution Order

~~~mermaid
flowchart TD
    START["Chat completion request"] --> MEDIA{"Image or non-text media?"}
    MEDIA -->|yes| BYPASS["Flush dirty states, then forward without a disk snapshot"]
    MEDIA -->|no| ID["Resolve session affinity"]
    ID --> KEY["Build prefix_key"]
    KEY --> HOTS{"Compatible hot session?"}
    HOTS -->|yes| HS["Reuse the session hot slot"]
    HOTS -->|no| FLUSH["Save dirty owners"]
    FLUSH --> SLOT["Prefer an unowned idle slot"]
    SLOT --> DS{"Session snapshot on disk?"}
    DS -->|yes and restore succeeds| DSR["Restore session snapshot"]
    DS -->|no or failure| RT["Native /apply-template + /tokenize"]
    RT --> SP{"Best exact LCP in namespace+scope >= minimum?"}
    SP -->|complete candidate and verified restore| SPR["Restore shared golden snapshot"]
    SP -->|internal divergence| OL["Cold request; later seed the verified LCP"]
    SP -->|none / invalid / API or restore failure| DP{"scope=default, no manifest, and exact legacy prefix exists?"}
    DP -->|yes and restore succeeds| DPR["Restore exact legacy prefix"]
    DP -->|no, explicit scope, manifest present, or failure| MISS["Cold / full prefill"]

    HS --> SEND["Send with id_slot"]
    DSR --> PIN["Pin restored slot"]
    SPR --> PIN
    DPR --> PIN
    OL --> COLD["Do not provide id_slot"]
    MISS --> COLD
    PIN --> RESP
    COLD --> RESP
    SEND --> RESP
    RESP["Generate and forward the current response"] --> SAVE["Save or defer the session snapshot"]
    SAVE --> SEED{"Golden absent, rejected, or verified overlap?"}
    SEED -->|yes| BG["Enqueue the seed in the bounded queue and run it in the background"]
    SEED -->|no| END["Finish"]
    BG --> END
    BYPASS --> END
~~~

The complete order is:

~~~text
hot session > session snapshot > complete shared golden prefix > limited exact legacy prefix > cold
~~~

An error from `/apply-template` or `/tokenize`, an invalid manifest, an LCP that is too short, or an absent or inconsistent restore causes the shared layer to fail closed, then proceeds to the permitted legacy fallback or cold prefill. Compatibility is never inferred.

### 5.1 Hot Session

The proxy retains in memory:

~~~text
session_id -> slot_id + prefix_key + n_tokens + session_file + dirty
~~~

Direct reuse simultaneously requires:

- the same session identifier;
- the same `prefix_key`;
- a llama slot that is still `idle`;
- a current token count equal to the recorded count.

Under the `terminal` policy, a response with `finish_reason=tool_calls` may leave `dirty` state in the hot slot. Before any eviction, unmanaged request, media request, or clean shutdown, this state is saved. If the slot metadata cannot be verified, the proxy saves immediately rather than risk an unpersisted conversation.

### 5.2 Session Snapshot and Native LCP

The session snapshot remains specific to an affinity and a `prefix_key`. It may contain dynamic history longer than the next request, particularly after an in-place compaction. In this specific case, restoring the snapshot for the **same session** and then allowing llama.cpp's native LCP to truncate history after the first divergence is expected.

This rule must not be confused with **shared** snapshots: a divergent shared candidate is never restored and then truncated; it triggers a cold path and an asynchronous seed of the verified LCP.

### 5.3 Strictly Limited Legacy Fallback

The `local-llm-prefix-<hash>.bin` fallback without token evidence is retained only to migrate historical deployments when all of the following conditions hold:

- `PI_LLAMA_CACHE_SHARED_PREFIX_SCOPE == "default"`;
- the file matches the request's exact `prefix_key`;
- no companion manifest exists for this file;
- no shared candidate has already been examined and rejected during this `prepare`.

A non-`default` scope never accepts a snapshot without a valid manifest. A rejected manifested snapshot cannot be retried as legacy, even if the filesystem prevents its removal. A successful legacy restore remains marked for migration and schedules a token-proven golden snapshot.

### 5.4 Disk Restore

The proxy requests the following from llama.cpp:

~~~text
POST /slots/{id}?action=restore
{"filename": "local-llm-prefix-....bin"}
~~~

llama.cpp reads the KV binary from `--slot-save-path`. The proxy never parses this binary format; it validates only save/restore responses and private manifests.

## 6. Cold-Start Sequence

~~~mermaid
sequenceDiagram
    participant C as Hermes / Pi / Zed
    participant P as Proxy 18082
    participant L as llama.cpp 8080
    participant D as Disk cache

    C->>P: New session request
    P->>P: Resolve session_id and prefix_key
    P->>L: GET /slots
    L-->>P: Idle slots and work identities
    P->>P: Save dirty owners
    P->>P: Prefer an unowned idle slot
    P->>L: Restore the exact session snapshot if present
    alt no session restore
        P->>L: POST /apply-template
        L-->>P: Rendered prompt
        P->>L: POST /tokenize
        L-->>P: Exact token IDs
        P->>D: Validate manifests and select the longest LCP in scope
        alt complete candidate
            P->>L: Restore golden snapshot and verify n_restored
        else divergent candidate
            P->>P: Remain cold and retain request_tokens[:verified_lcp]
        else no shared candidate
            P->>D: Examine only the permitted legacy fallback
        end
    end
    alt restored slot
        P->>L: Request pinned to the restored slot
    else cold path
        P->>L: Request without id_slot, full prefill
    end
    L-->>P: Generated response fragments
    P-->>C: Forward and flush chunks immediately
    P->>L: Save the complete session snapshot
    L->>D: Atomically write local-llm-session-*.bin

    opt golden absent, rejected, or verified overlap
        P->>P: Retain seed in pending_prefix_seeds
        P->>P: Wait for a safe slot without blocking the foreground
        P->>L: /completion with exact token array and n_predict=0
        L-->>P: Prefix prefill complete
        P->>L: Atomically save the prefix snapshot
        P->>D: Atomically publish the private manifest
    end
~~~

Disk restoration incurs I/O cost but avoids recomputing the stable-prefix prefill. The new request's suffix is always processed normally.

## 7. Dynamic Appends Within the Same Session

~~~mermaid
sequenceDiagram
    participant C as Pi
    participant P as Proxy
    participant M as Hot slot map
    participant L as llama.cpp

    C->>P: Turn 1
    P->>L: Full prompt with default cache_prompt
    L-->>P: Response 1
    P->>L: Save session snapshot
    P->>M: session_id -> slot_id

    C->>P: Turn 2 with a new user message
    P->>M: Check session_id, prefix_key, and slot state
    M-->>P: Hot session hit
    P->>L: Full conversation + new suffix
    L->>L: Reuse common prefix and process suffix
    L-->>P: Response 2
    P->>L: Save updated session snapshot
    P->>M: Update token count
~~~

Turn 2 never directly returns the response from turn 1. It always performs prompt matching followed by a new decode.

## 8. Background Prefix Seed Safety

### 8.1 Bounded Queue and Work Identity

Each seed request is indexed by its `prefix_file` in `pending_prefix_seeds`. The in-memory queue contains at most **16** identities; beyond that limit, it evicts the oldest entry that is not currently running. An identity already present is replaced by the most recent work, but an already verified `stable_tokens` sequence takes precedence over a proposal without tokens and, between two verified sequences, the longer one is retained.

`prefix_seeds_in_flight` ensures that only one background task processes a given identity at a time. A seed deferred because no safe slot is available, because the lock is unavailable, or because a foreground request is waiting remains in `pending_prefix_seeds` and is retried after subsequent responses.

Removal of completed work is conditioned on its complete content identity: `_discard_pending_prefix_seed(prefix_file, stable_tokens)` removes the entry only if the `stable_tokens` still queued are exactly those of the task. An old task therefore cannot delete newer, longer work that replaced its entry while it was running. This protection makes stale tasks harmless; exact token deduplication then prevents creation of identical copies.

### 8.2 Seed Transaction, Including With `-np 1`

~~~mermaid
flowchart TD
    R["Successful response"] --> QUEUE["Insert/replace in the queue bounded at 16"]
    QUEUE --> WAIT["Wait PI_LLAMA_CACHE_PREFIX_SEED_DELAY<br/>2 seconds by default"]
    WAIT --> FRONT{"Foreground request waiting?"}
    FRONT -->|yes| DEFER["Keep pending; retry later"]
    FRONT -->|no| LOCK{"operation_lock available without waiting?"}
    LOCK -->|no| DEFER
    LOCK -->|yes, held for the entire operation| SLOT{"Unowned reserve idle slot?"}
    SLOT -->|yes| TOKENS["Render/tokenize or use the exact verified LCP"]
    SLOT -->|no, including np=1| OWNER{"Excluded owner idle and clean<br/>matching token count + persistent snapshot?"}
    OWNER -->|no| DEFERLOCK["Keep pending and release the lock"]
    OWNER -->|yes| GUARD["Save and validate a distinct owner guard"]
    GUARD --> TOKENS
    TOKENS --> DUP{"Same namespace, scope, and exact tokens already published?"}
    DUP -->|yes| DROPWORK["Remove only this exact work item"]
    DUP -->|no| SEED["/completion with exact token array"]
    SEED --> SAVE["Atomically save snapshot"]
    SAVE --> COUNT{"n_saved == len(tokens)?"}
    COUNT -->|no| DROP["Remove partial pair; fail closed"]
    COUNT -->|yes| MANIFEST["Atomically publish private manifest"]
    DROP --> RESTORE{"Was the owner replaced?"}
    MANIFEST --> RESTORE
    DROPWORK --> RESTORE
    RESTORE -->|yes| FINALLY["Restore owner in finally"]
    RESTORE -->|no| DONE["Release lock"]
    FINALLY -->|success and exact count| DONE
    FINALLY -->|failure or different count| FORGET["Forget hot ownership<br/>retain persistent session snapshot"]
    FORGET --> DONE
~~~

The seed request sent to `/completion` has this form:

~~~text
prompt       = request_tokens[:verified_lcp]  # integer array, content not logged
cache_prompt = false
n_predict    = 0
stream       = false
id_slot      = slot_id
~~~

When an unowned `idle` slot is available, the proxy uses the one containing the fewest tokens. Without a reserve slot, particularly with `-np 1`, it replaces the excluded slot only if its owner is `idle`, `clean`, associated with a persistent session snapshot, and its current token count matches `owner.n_tokens`.

Under `operation_lock`, the proxy first saves the owner to a distinct `*.seed-owner.tmp` guard file and requires `n_saved == owner.n_tokens`. Only then may it send the exact token array to `/completion`. After prefill, it requires `n_saved == len(tokens)`, publishes the snapshot by atomic rename, and then publishes the manifest through a temporary `0600` file, `flush`, `fsync`, and atomic rename. The published pair remains immutable while any replacement generation is being built.

As soon as the seed request may have modified the slot, the `finally` block restores the guard and requires `n_restored == owner.n_tokens`, whether or not the seed, save, or publication succeeded. On failure, only the slot's hot ownership is forgotten; the original persistent session snapshot remains available for a later restore. The temporary guard is removed on a best-effort basis.

The seed `/completion` request uses the dedicated `PI_LLAMA_CACHE_PREFIX_SEED_TIMEOUT` timeout (600 seconds by default), because a cold prefill of a long prefix may exceed the general internal timeout of 120 seconds. This setting does not change the timeouts for `/apply-template`, `/tokenize`, save, or restore. A timeout follows the same transactional rollback and publishes no manifest.

### 8.3 Deduplication and Degradation

After rendering/tokenization and while holding the operation lock, the task rescans valid manifests. If it finds the **same exact tokens** in the same namespace and scope, it emits `prefix_seed_duplicate_skipped` and avoids both prefill and creation of a duplicate randomly named snapshot. The comparison is neither an approximate fingerprint nor a simple token count.

A `dirty` owner, missing snapshot, inconsistent slot count, waiting foreground request, busy lock, rendering/tokenization error, timeout, inconsistent `n_saved`, or save, manifest, or guard-restore failure: all these cases defer, cancel, or fail the seed without compromising the foreground response. Seeding is a disposable, best-effort optimization; request correctness never depends on its success.

## 9. HTTP Contract, Routing, and Streaming

### 9.1 Allowlist and Administrative Denials

The public contract is explicit and closed by default for mutations:

| Class | Routes | Slot handling |
| --- | --- | --- |
| Session cache | `POST /v1/chat/completions` | Hot/disk KV lifecycle with session affinity |
| Utilities/control | `POST /tokenize`, `/detokenize`, `/apply-template`, token-counting routes, `/v1/chat/completions/control` | Forward without flushing or evicting KV state |
| Unmanaged inference | Allowed completions, responses, embeddings, infill, reranking, and Anthropic messages | Save `dirty` states, reset ownership, then forward under `operation_lock` |
| Monitoring/metadata | Ordinary `GET`, `HEAD`, and `OPTIONS` routes | Transparent forwarding without slot mutation |
| Administration | `/slots`, LoRA/tool administration, `/props` mutations, model load/unload/download, and every `DELETE` request | `404`, never forwarded |
| Unknown `POST` | Any route outside the allowlist | `404`, never forwarded |

The following compatible aliases are translated without altering the query string: `GET`/`HEAD /v1/props` to `/props`, and `POST /v1/tokenize`, `/v1/detokenize`, `/v1/apply-template` to the native unprefixed endpoints. `HEAD` forwards no body. `OPTIONS` and CORS preflight headers are relayed.

Media requests are serialized but do not participate in the disk cache: the proxy first saves `dirty` states, resets the slot ownership table, removes its private affinity fields, and forwards the request without a snapshot. When `PI_LLAMA_CACHE_REQUIRE_SESSION_ID=true`, textual chats without a stable identifier are rejected, but this unmanaged media path remains permitted.

### 9.2 Incremental Streaming

The proxy does not buffer the response until completion. It:

1. reads one immediately available fragment with `HTTPResponse.read1(64 * 1024)`;
2. writes its size and content as a valid HTTP/1.1 chunk;
3. immediately calls `handler.wfile.flush()`;
4. repeats until EOF, then emits the terminal chunk `0\r\n\r\n` and performs the final flush.

`HTTPResponse.read(size)` must not replace `read1`: `read(size)` may wait for `size` bytes or EOF and turn a token-by-token SSE stream into a response delivered as one block. Headers named in the static `HOP_BY_HOP_HEADERS` set are removed from non-HEAD downstream responses; `Content-Length` is preserved for HEAD responses. Downstream `Transfer-Encoding: chunked` is produced for non-HEAD responses, and SSE fragments — including `[DONE]` — remain byte-for-byte unchanged. Current limitation: extension header names declared in the `Connection` value are not parsed dynamically; the upstream must therefore remain a trusted llama.cpp instance that does not emit such headers.

A separate buffer, bounded to the most recent 1 MiB, is used only to extract `finish_reason` and `usage.prompt_tokens_details.cached_tokens` after forwarding; it does not delay chunks. A client disconnect cleanly stops forwarding and retains metadata already received.

## 10. The Cache Is Not a Response Cache

~~~text
cached: KV(prefix) + GDN/recurrent state
not cached: logits, assistant response, next RNG result
~~~

The response can be abstracted as:

~~~text
answer = Decode(KV(prefix), dynamic_suffix, sampling_parameters, RNG)
~~~

Consequently:

- the same prefix with different `user` messages can produce different responses;
- `temperature`, `top-p`, `top-k`, `seed`, and `max_tokens` do not participate in the `prefix_key`;
- even for an identical complete request, identical output is expected only if the model, sampling parameters, seed, and inference path are also identical;
- `cached_tokens > 0` only indicates that prompt tokens were reused, not that a response was cached.

## 11. Prompt Evolution, Invalidation, and Degradation

### 11.1 Evolution Scenarios

| Change | `prefix_key` | Hot/disk session | Shared golden snapshot |
| --- | --- | --- | --- |
| `system`/`developer` text, order, spaces, or line breaks | changes | new file, therefore miss | exact comparison; divergent candidate not restored, seed the verified LCP if long enough |
| Hermes skill, memory, or date injected late | may change | stable-key miss if included in the prefix | reuse possible only up to the divergence through a shorter complete golden snapshot |
| Tool or tool JSON schema | changes | miss | no state beyond the first different token |
| `model`, template, thinking/reasoning, grammar, `json_schema`, `response_format` | changes | miss | rendered tokens are authoritative; renew scope/namespace if runtime is incompatible |
| `user` message, `assistant`/`tool` history, or task suffix | unchanged | same session snapshot possible; native LCP truncates its dynamic suffix if needed | stable golden snapshot unchanged |
| `temperature`, `top-p`, `top-k`, `seed`, `max_tokens` | unchanged | prompt state reusable | prompt state reusable; response always decoded again |
| In-place Hermes compaction, stable prefix unchanged | unchanged | same affinity; session restore then native LCP over summary/protected tail | stable golden snapshot usable |
| Hermes rotation to a child session with `boundary_reason="compression"` | unchanged but new affinity | no parent snapshot | compatible golden snapshot possible, otherwise cold |
| Early bootstrap change | changes | miss | short LCP or below threshold, therefore cold |

In-place compaction deserves an explicit distinction: the session-specific snapshot may be restored, after which its old dynamic history is truncated by the native LCP. In contrast, the shared layer never performs a speculative restore of a divergent snapshot.

### 11.2 Fail Closed and Rollback

~~~mermaid
flowchart TD
    R["Restore requested"] --> OK{"Restore succeeded with exact count?"}
    OK -->|yes| HIT["Use restored state"]
    OK -->|no| WARN["Log metadata only"]
    WARN --> CLEAN["Remove corrupt shared pair on a best-effort basis"]
    CLEAN --> FALL{"Legacy fallback strictly permitted?"}
    FALL -->|yes| LEGACY["Try exact legacy snapshot"]
    FALL -->|no| FULL["Cold slot and full prefill"]
    LEGACY -->|failure| FULL
    LEGACY -->|success| SAVE
    FULL --> SAVE["Save new session snapshot"]
    SAVE --> RESEED["Schedule an exact golden snapshot on a best-effort basis"]
~~~

An unavailable or malformed `/apply-template`/`tokenize` response, an out-of-scope, malformed, oversized, or orphaned manifest, an LCP below the threshold, an I/O error, or an inconsistent restore count must not produce a shared hit. The proxy degrades to legacy only within its strict historical boundary, otherwise to cold prefill. Failure to finalize a snapshot after a successful response is logged but does not replace the client response with an error.

The current key does not include the GGUF hash, llama.cpp build hash, or every startup parameter. After changing the model file, quantization, chat template, or a material KV configuration, stop the service, retain the old cache directory for rollback, change the namespace if needed, and then regenerate snapshots. Safe operational rollback consists of reactivating the previous generation or package, or at minimum declaratively disabling `PI_LLAMA_CACHE_ENABLE_PREFIX_SEEDING`. Existing snapshots remain private data; cleaning them is a separate operation that requires an inventory, retention criteria, a dry run, and explicit confirmation.

## 12. Privacy and Observability

The cache directory is forced to mode `0700`; snapshots and manifests are published with mode `0600`. Token IDs in a manifest are reversible by anyone who has the corresponding tokenizer: they are therefore private data, not an anonymized hash. The `personal` and `work` scopes must be physically and logically separated.

Structured events are compact JSON objects. They must never contain the raw session identifier, raw prompt, rendered text, a token ID array, a manifest body, or secrets. `session_ref` is a SHA-256 value truncated to 16 hexadecimal characters: it provides pseudonymization useful for correlation, but not strong anonymization. A predictable source identifier remains exposed to a dictionary attack. Logs must therefore be private, access-controlled, and subject to short retention. Shared events record only metadata such as `layer`, `candidate_tokens`, `verified_lcp`, `cached_tokens`, `slot_id`, durations, sizes, controlled filenames, and error types.

`shared_prefix_candidate_restored` means that a **complete** candidate was restored and verified, but not yet that llama.cpp actually reused it. After reading response metadata:

- `shared_prefix_effective_hit` requires `cached_tokens > 0`;
- `shared_prefix_rejected_by_llama` corresponds to `cached_tokens == 0` and triggers an exact best-effort seed;
- `shared_prefix_effectiveness_unknown` means that the upstream did not provide this counter;
- `shared_prefix_overlap_discovered` means that a divergent candidate was not restored and a seed of the verified LCP was proposed.

## 13. Benefits, Costs, and Validation

### 13.1 Observed Measurements

The redacted, versioned report [docs/validation/2026-09-14-production.md](./docs/validation/2026-09-14-production.md) links the September 14, 2026 measurements to the commit, source fingerprints, validation commands, and deployment status. It distinguishes production evidence, deterministic tests, and historical measurements without publishing private logs or host-specific generation paths.

Historical and current measurements:

- a Pi request without tools contains approximately 5,218 prompt tokens;
- the corresponding seed produces approximately 5,198 prefix tokens;
- a cold restore for a new session takes approximately 2.6 to 3.4 seconds in the previous scenario;
- a hot session avoids disk reads;
- a prefix including the `read` tool schema reaches approximately 6,678 tokens and can be seeded;
- on the patched deployment, a 32,185-token snapshot produced a 54.8 s cold prefill followed by `cache_n=32151`, `prompt_n=34`, and 0.37 s after restoration;
- after a complete restart, a 12,700-token snapshot produced `n_restored=12700`, followed by `cache_n=12662`, `prompt_n=38`, and 0.356 s on a divergent prompt;
- the real Hermes baseline measurement from September 12, 2026 was 25,960 tokens, `cached_tokens=0`, 233.94 s of prompt evaluation, and 110.97 tokens/s, with an 824,255,044-byte (786.07 MiB) snapshot;
- the most recent real Hermes A/B trials produced effective hits of **24,464** and **24,479** tokens, not merely successful restore API calls.

The gain comes from avoiding prefill of the stable system prompt, project rules, and tool schemas. It is measured with `usage.prompt_tokens_details.cached_tokens`, `timings.cache_n`, and TTFT, not with `n_restored` alone.

### 13.2 Costs

- no reduction in video memory occupied by model weights;
- no direct improvement in decode tokens/s;
- snapshot size depends on the model, context, and recurrent checkpoints: approximately 150–240 MiB in earlier trials, approximately 786 MiB per snapshot in the validated 128K-checkpoint deployment;
- default 12 GiB LRU disk budget; restores and hot reuse refresh recency;
- the first golden snapshot incurs a prefill and additional I/O;
- mutation serialization through `operation_lock`, with priority given to foreground requests;
- seed queue deliberately bounded at 16 and processed on a best-effort basis;
- privacy cost of snapshots and reversible tokens.

### 13.3 A/B Acceptance and Tests

Production acceptance must compare real Hermes requests and distinguish two scenarios:

1. isolate old prefixes in a test scope and verify an A baseline measurement with a new affinity, seeding disabled, and `cached_tokens=0`;
2. enable seeding in an isolated test scope, send a first session to create the golden snapshot, and wait for it to be published;
3. send a second session with the same stable prefix and a distinct `user` suffix; require `shared_prefix_candidate_restored`, then `shared_prefix_effective_hit`, `cached_tokens > 0` or `timings.cache_n > 0`, and improved TTFT;
4. then make a controlled change to an element inside the manifested prefix; require `shared_prefix_overlap_discovered`, no restore of the divergent candidate, and a cold path;
5. wait for publication of the golden snapshot limited to the LCP, then send another variant for which this complete golden snapshot is an exact prefix; this time require `shared_prefix_candidate_restored` followed by `shared_prefix_effective_hit`;
6. finally, change an early token and verify a shorter LCP or a cold path;
7. restore the official declarative configuration for the `personal`/`work` scopes and remove every imperative test override.

At commit `382b469`, the suite contains **123 tests** and reaches **100% line and branch coverage** for `cache_core.py` and `cache_proxy.py`. The validation commands are:

~~~bash
python3 -m unittest -v test_cache_core.py test_cache_proxy.py
python3 -m py_compile cache_core.py cache_proxy.py
python3 -m coverage run --branch --source=. -m unittest discover -s . -p 'test_*.py' -v
python3 -m coverage report --include='cache_core.py,cache_proxy.py' --fail-under=100
~~~

## 14. Essential Files

| File | Role |
| --- | --- |
| [cache_core.py](./cache_core.py) | Prefix payload, SHA-256 key, exact manifest, LCP, and slot request helper |
| [cache_proxy.py](./cache_proxy.py) | HTTP proxy, routing, streaming, hot slots, save/restore, and transactional seeds |
| [test_cache_core.py](./test_cache_core.py) | Tests for keys, tokens, manifests, and LCP selection |
| [test_cache_proxy.py](./test_cache_proxy.py) | Tests for the proxy, restores, seeds, routes, logs, and streaming |
| [README.md](./README.md) | Operations, configuration, hit diagnostics, and controlled cleanup |
| [docs/validation/2026-09-14-production.md](./docs/validation/2026-09-14-production.md) | Redacted evidence for tests, streaming, Hermes A/B trials, Nix deployment, and post-probe state |
| [local-llm-kv-cache.service](./local-llm-kv-cache.service) | User-level systemd service template |

Operational checks:

~~~bash
curl -fsS http://127.0.0.1:18082/health
systemctl --user status local-llm-kv-cache.service
journalctl --user -u local-llm-kv-cache.service -n 100 --no-pager
~~~
