# Local LLM KV Cache Design

## 1. 目标与边界

本设计针对 Pi、pi-acp、Zed 使用 Qwen3.8-27B 编程 agent 时的首消息延迟。

目标：

- 缓存稳定 system prompt、项目规则和工具 schema 的 prefill 结果；
- 同一 session 继续对话时复用内存 slot；
- 代理或 llama.cpp 重启后，可以从磁盘恢复稳定 prefix；
- 不缓存答案，不改变模型生成和采样逻辑；
- 保持现有 llama.cpp + GGUF + AMD 运行链路。

前提：Qwen3.8 的磁盘 restore 依赖 hybrid checkpoint 持久化。stock llama.cpp
可能返回成功的 `n_restored`，但后续 `cache_n` 仍为 0；部署必须使用本机
`local/kv-restore-checkpoints` 分支的 `862535a` 或等价上游修复。

非目标：

- 不做 response/result cache；
- 不把不同项目或不同模型的 KV 状态混用；
- 不把完整旧对话伪装成跨 session prefix；
- 不改变模型权重、训练行为或 decode 算法。

## 2. 总体架构

~~~mermaid
flowchart LR
    C["Pi / pi-acp / Zed"] -->|OpenAI Chat Completions| P["Cache Proxy<br/>127.0.0.1:18082"]

    subgraph CACHE["Cache Proxy"]
        A["Session affinity"]
        K["Prefix key + native render/tokenize"]
        H["Hot slot map<br/>session_states"]
        R["Hit selection<br/>hot -> session -> shared LCP -> legacy -> cold"]
        S["Snapshot manager"]
        M["Private token manifests<br/>namespace + scope"]
        A --> K --> R
        M --> R
        H --> R
        R --> S
    end

    P --> CACHE
    S -->|slot save / restore| L["llama.cpp llama-server<br/>127.0.0.1:8080"]
    S -->|private snapshots + manifests| D[("~/.llama-slot-cache")]
    L --> G["Qwen3.8-27B<br/>GGUF + MTP"]
~~~

代理只监听 loopback。Pi 和 Zed 指向 18082；原来的 llama.cpp 8080 保持不变。

## 3. 缓存的是什么

稳定 prefix 通常包含：

~~~text
system prompt
developer prompt
项目规则和 AGENTS.md 内容
工具定义和 JSON schema
chat template / thinking 参数
~~~

动态 suffix 通常包含：

~~~text
当前 user 消息
assistant 历史
tool result
当前任务的临时上下文
~~~

~~~mermaid
flowchart LR
    A["Stable prefix<br/>system / developer / tools"] --> B["KV + GDN state<br/>可持久化"]
    B --> C["Dynamic suffix<br/>user / assistant / tool"]
    C --> D["Decode<br/>生成当前回答"]

    style A fill:#d9f2d9,stroke:#397a3c
    style B fill:#d9e8ff,stroke:#3569a8
    style C fill:#fff1cc,stroke:#a87900
    style D fill:#ffe0e0,stroke:#a33a3a
~~~

缓存的是 prefix 对应的模型状态，不是 D 阶段的答案。

## 4. Cache key 设计

### 4.1 Prefix key

代理从 messages 开头连续保留 role 为 system/developer 的消息，遇到第一个 user、assistant 或 tool 消息就停止。

同时加入这些字段：

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

然后计算：

~~~text
prefix_key = SHA256(canonical_json(prefix_payload))
~~~

canonical JSON 使用：

- ensure_ascii=false；
- sort_keys=true；
- 紧凑 separators；
- UTF-8 编码。

~~~mermaid
flowchart TD
    Q["Original request body"] --> M["Read messages from the beginning"]
    M --> T{"system/developer?"}
    T -->|yes| KEEP["Keep message"]
    KEEP --> T
    T -->|no| STOP["Stop at first dynamic message"]
    KEEP --> F["Add stable request fields"]
    STOP --> F
    F --> J["Canonical JSON"]
    J --> H["SHA-256"]
    H --> K["prefix_key"]
~~~

### 4.2 Snapshot 文件名

~~~text
snapshot_key = SHA256("3" + namespace + kind + identity + prefix_key)
~~~

文件格式：

~~~text
local-llm-session-<hash>.bin
local-llm-prefix-<hash>.bin
~~~

- session 的 identity 是 Pi/Zed session ID；
- prefix 的 identity 是固定字符串 prefix；
- 版本号 3 和 deployment namespace 用于让旧格式或不兼容部署的缓存整体失效。

因此，换 session 会错过具体 session 快照，但仍可能命中同项目的 prefix 快照。

### 4.3 Golden shared-prefix manifest

Canonical `prefix_key` 适合 exact session/legacy lookup，但不能证明两个不同 Hermes bootstrap 可以共享多少状态。Golden 路径以 **当前 llama.cpp 实际产生的 token** 为唯一正确性依据：

~~~text
incoming chat body
  -> POST /apply-template
  -> response.prompt（完整 rendered prompt）
  -> POST /tokenize {content, add_special:false, parse_special:true}
  -> exact integer token IDs
~~~

seed 时，纯 prefix snapshot 旁边原子发布 mode `0600` 的 companion：

~~~text
local-llm-prefix-<hash>.bin.manifest.json
{version, namespace, scope, snapshot, tokens[]}
~~~

`tokens[]` 是有界、非空、逐项验证的非负 32-bit token ID。manifest basename 必须与 companion 文件名一致，且所指 snapshot 必须存在。候选选择流程是：

1. 严格过滤到相同 `PI_LLAMA_CACHE_NAMESPACE` 和 `PI_LLAMA_CACHE_SHARED_PREFIX_SCOPE`；
2. 忽略 malformed、过大或 orphan manifest；
3. 对每个候选与请求 token 从 index 0 开始逐项比较；
4. 选择 exact LCP 最长、且 LCP 至少为 `PI_LLAMA_CACHE_MIN_SHARED_PREFIX_TOKENS`（默认 `128`）的候选；LCP 相同时优先选择自身完整包含在请求开头的较短候选。

不使用 prompt 文本启发式、hash 相似度或匹配百分比。`PI_LLAMA_CACHE_SHARED_PREFIX_SCOPE` 必须按信任域显式配置；personal 与 work 必须使用不同值。模型/runtime/template 部署不兼容时还需轮换 namespace。

只有候选的全部 token 都是当前请求前缀时才允许 restore，并把请求 pin 到该 slot。候选在内部发生分歧时，代理不会依赖 llama.cpp 截断已恢复 snapshot：它保持 cold 路径，并在请求结束后 seed 严格相等的 `request_tokens[:verified_lcp]`。后续请求可完整 restore 这个较短的稳定 golden。代理不会自行拼接 KV；skill、tool schema、system/developer prompt、memory、日期、template 或 reasoning mode 的改变绝不允许跨过第一处分歧产生 stale hit。

## 5. 请求命中顺序

~~~mermaid
flowchart TD
    START["Chat completion request"] --> MEDIA{"Has image or non-text media?"}
    MEDIA -->|yes| BYPASS["forward without disk snapshot"]
    MEDIA -->|no| ID["Resolve session affinity"]
    ID --> KEY["Build prefix_key"]
    KEY --> HOTS{"Hot session matches?"}
    HOTS -->|yes| HS["Reuse hot session slot"]
    HOTS -->|no| SLOT["Choose unowned idle slot"]
    SLOT --> DS{"Disk session snapshot exists?"}
    DS -->|yes and restore succeeds| DSR["Restore session snapshot"]
    DS -->|no or restore fails| RT["Native /apply-template + /tokenize"]
    RT --> SP{"Best same namespace+scope exact LCP >= minimum?"}
    SP -->|complete candidate and restore succeeds| SPR["Restore shared golden prefix"]
    SP -->|internal divergence| OL["Cold request; seed verified LCP later"]
    SP -->|no / malformed / API or restore failure| DP{"scope=default, no manifest, legacy exact prefix exists?"}
    DP -->|yes and restore succeeds| DPR["Restore legacy exact prefix"]
    DP -->|no, scoped, manifested, or restore fails| MISS["Cold / full prefill"]

    HS --> SEND["id_slot only for hot session"]
    DSR --> NATIVE["Restored slot pinned; cold unpinned<br/>native LCP"]
    SPR --> NATIVE
    OL --> NATIVE
    DPR --> NATIVE
    MISS --> NATIVE
    NATIVE --> RESP
    SEND --> RESP
    RESP["Generate current response<br/>restored slot pinned; native LCP truncates at divergence"]
    RESP --> SAVE["Save current session snapshot"]
    SAVE --> SEED{"Prefix file missing?"}
    SEED -->|yes| BG["Background pure-prefix seed"]
    SEED -->|no| END["Finish"]
    BG --> END
    BYPASS --> END
~~~

### 5.1 Hot session

代理内存中保存：

~~~text
session_id -> slot_id + prefix_key + n_tokens + session_file + dirty
~~~

只有以下条件同时满足时才直接复用：

- session ID 相同；
- prefix key 相同；
- llama slot 仍然 idle；
- 当前 token 数等于上次保存的 token 数。

### 5.2 Prefix fallback

代理不维护跨 session 的 hot prefix 所有权。这样一个新 session 不会把另一个 session 的完整动态历史当成共享 prefix。代理先选择 idle slot，优先恢复该 session 的 snapshot；否则 render/tokenize 当前请求，选择同 namespace+scope 下 exact LCP 最长的 golden manifest；再否则尝试当前 `prefix_key` 的 legacy exact snapshot；最后 cold。成功 restore 后会固定该 slot；proxy 不覆盖 `cache_prompt`，由客户端或 llama-server 的默认 prompt cache 执行 native LCP。显式关闭 prompt cache 会失去复用收益，但不会允许跨越 token 分歧的 stale hit。

完整顺序是：

~~~text
hot session > session snapshot > shared golden prefix > legacy exact prefix > cold
~~~

`/apply-template` 或 `/tokenize` 不可用/返回 malformed 数据、manifest malformed/orphan、LCP 过短或 restore 失败时，shared 层 fail closed 并继续 legacy/cold，不猜测兼容性。

### 5.3 Disk prefix

代理重启后内存映射会消失，但 prefix 文件仍然存在。代理调用 llama.cpp：

~~~text
POST /slots/{id}?action=restore
{"filename": "local-llm-prefix-....bin"}
~~~

KV 二进制由 llama.cpp 从 --slot-save-path 读取，代理不解析 KV 文件。

## 6. 冷启动时序

~~~mermaid
sequenceDiagram
    participant C as Pi / Zed
    participant P as Proxy 18082
    participant L as llama.cpp 8080
    participant D as Disk cache

    C->>P: New session chat request
    P->>P: Build session_id and legacy prefix_key
    P->>L: GET /slots
    L-->>P: Find idle slot
    P->>L: Choose unowned idle slot
    P->>L: Restore exact session snapshot if present
    alt no session restore
        P->>L: POST /apply-template
        L-->>P: rendered prompt
        P->>L: POST /tokenize
        L-->>P: exact token IDs
        P->>D: Validate manifests and choose longest in-scope LCP
        P->>L: Restore shared snapshot; otherwise legacy exact
    end
    L->>D: Read selected snapshot
    D-->>L: KV + recurrent state
    L-->>P: Restore complete
    P->>L: Chat request pinned to restored slot (cold has no id_slot)
    L->>L: Native LCP truncates all state after first divergence
    L-->>P: Stream generated answer
    P-->>C: Forward answer
    P->>L: Save full session snapshot
    L->>D: Write local-llm-session snapshot

    opt Prefix file was missing
        P->>P: Wait for a safe idle slot
        P->>L: Pure prefix request, n_predict=0
        L-->>P: Prefix prefill complete
        P->>L: Atomically save prefix snapshot
        L->>D: Write local-llm-prefix snapshot
        P->>D: Atomically publish private token manifest
    end
~~~

磁盘恢复有 I/O 成本，但避免重新执行完整 prefix prefill。新请求的 suffix 仍然正常执行。

## 7. 同 session 动态追加

~~~mermaid
sequenceDiagram
    participant C as Pi
    participant P as Proxy
    participant M as Hot slot map
    participant L as llama.cpp

    C->>P: Turn 1
    P->>L: Full prompt (cache_prompt default)
    L-->>P: Answer 1
    P->>L: Save session snapshot
    P->>M: session_id -> slot_id

    C->>P: Turn 2 with appended user message
    P->>M: Check session_id, prefix_key, slot state
    M-->>P: Hot session hit
    P->>L: Full conversation + new suffix
    L->>L: Reuse common prefix, process suffix
    L-->>P: Answer 2
    P->>L: Save updated session snapshot
    P->>M: Update token count
~~~

Turn 2 不会直接返回 Turn 1 的答案。它仍然经过 prompt matching 和新的 decode。

## 8. 后台 prefix seed 的安全边界

~~~mermaid
flowchart TD
    R["Successful response"] --> WAIT["Wait PI_LLAMA_CACHE_PREFIX_SEED_DELAY<br/>default 2 seconds"]
    WAIT --> FRONT{"Foreground request waiting?"}
    FRONT -->|yes| SKIP["Skip seed"]
    FRONT -->|no| LOCK{"Proxy operation lock available?"}
    LOCK -->|no| SKIP
    LOCK -->|yes, held through whole operation| SLOT{"Spare unowned idle slot?"}
    SLOT -->|yes| SEED["Render/tokenize + /completion exact token array"]
    SLOT -->|no, including np=1| OWNER{"Excluded owner idle + clean<br/>token count matches + persisted snapshot?"}
    OWNER -->|no| SKIPLOCK["Skip and release lock"]
    OWNER -->|yes| OWNERSAVE["Save + validate separate owner guard"]
    OWNERSAVE --> SEED
    SEED --> SAVE["Atomic prefix snapshot save"]
    SAVE --> COUNT{"n_saved equals rendered token count?"}
    COUNT -->|no| DROP["Delete shared pair; fail closed"]
    COUNT -->|yes| MANIFEST["Atomic private manifest publish"]
    DROP --> RESTORE{"Owner was swapped?"}
    MANIFEST --> RESTORE
    RESTORE -->|yes| FINALLY["Restore owner in finally"]
    RESTORE -->|no| DONE["Release lock"]
    FINALLY -->|success| DONE
    FINALLY -->|failure| FORGET["Forget hot ownership;<br/>persisted session remains"]
    FORGET --> DONE
~~~

prefix seed 使用：

~~~json
{
  "prompt": [151644, 8948, 198],
  "cache_prompt": false,
  "n_predict": 0,
  "stream": false
}
~~~

有 spare unowned idle slot 时使用 spare。没有 spare（尤其 `-np 1`）时，只有 excluded owner 已 idle、clean、其 slot token 数与记录一致且完整 session snapshot 已存在，才允许事务式 swap。代理持有 operation lock，先把 owner 保存到独立 guard snapshot 并验证 `n_saved`，再用 `/completion` 的 token-array prompt 执行 seed 和 atomic snapshot save。token identity 由传给 `/completion` 的 exact IDs 保证；`n_saved == len(tokens)` 只是额外完整性检查。两者成立后才原子发布 manifest；计数不一致或发布失败会删除 shared pair。shared restore 同样要求 `n_restored == len(manifest.tokens)`，否则删除损坏候选并 fallback。manifest 永远不应指向缺失、部分写入或未经验证的 snapshot。一旦 seed 请求可能改变 slot，无论 seed/save 是否成功，都在 `finally` 从 guard 恢复 owner，并验证 `n_restored == owner.n_tokens` 后才释放 lock。恢复失败或计数不符只清除该 slot 的 hot ownership，原始持久化 session snapshot 保留供后续恢复。

`/completion` seed 使用独立的 `PI_LLAMA_CACHE_PREFIX_SEED_TIMEOUT`（默认 600 秒），因为完整 prefix cold prefill 可以显著超过通用内部 API 的 120 秒 timeout。该值只作用于 seed completion；apply-template、tokenize、slot save/restore 等调用仍使用通用 timeout。超时仍走同一事务 rollback，且不会发布 manifest。

dirty owner、缺失 snapshot、slot token mismatch、seed `n_saved` mismatch、前台等待、busy lock、render/tokenize/seed/save/manifest 失败都跳过或回滚，不影响前台正确性。已经按 shared 规则尝试并拒绝的文件，在同一次 `prepare` 中不得降级为无 token-count 校验的 legacy exact restore；即使 read-only filesystem 阻止物理删除，也必须继续 cold。成功恢复一个 shared golden 后，该请求不会因为日期等尾部小差异再创建一个近重复 exact snapshot。每个后台 seed 在 render/tokenize 后还必须重新扫描有效 manifest；相同 namespace、scope 和 exact token IDs 已经发布时跳过 seed，防止多个请求在首个 golden 发布前排队并随后依次生成重复副本。`PI_LLAMA_CACHE_ENABLE_PREFIX_SEEDING=true` 可用于 `-np 1`；不再要求因为只有一个 slot 而禁用。首次 seed 仍支付一次额外 prefix prefill、snapshot I/O 和 manifest 写入成本。

## 9. 缓存不是答案缓存

~~~text
缓存：KV(prefix) + GDN/recurrent state
不缓存：logits、assistant answer、下一次 RNG 结果
~~~

回答可以抽象为：

~~~text
answer = Decode(KV(prefix), dynamic_suffix, sampling_parameters, RNG)
~~~

因此：

- 相同 prefix、不同 user 消息，答案可以不同；
- temperature、top-p、top-k、seed、max_tokens 不参与 prefix key；
- 相同完整请求也只有在模型、采样参数、seed 和推理路径都一致时，才适合期待完全一致；
- cached_tokens > 0 只说明 prompt token 被复用，不说明回答被缓存。

## 10. 失效和降级

~~~mermaid
flowchart TD
    R["Restore requested"] --> OK{"Restore succeeded?"}
    OK -->|yes| HIT["Use restored state"]
    OK -->|no| WARN["Log warning"]
    WARN --> FULL["Use idle slot and full prefill"]
    FULL --> SAVE["Save new session snapshot"]
    SAVE --> RESEED["Regenerate prefix snapshot"]
~~~

会生成新的 legacy prefix key，并在 golden 路径中让 exact LCP 停在实际 rendered token 第一处分歧处的变化：

- system/developer prompt 文本、顺序、空格或换行变化；
- tools 或工具 JSON schema 变化；
- model ID 变化；
- thinking 或 chat-template 参数变化；
- grammar、JSON schema、response format 变化。
- Hermes skills、system/developer 指令、memory 或注入日期变化；
- llama.cpp template 输出或 reasoning/thinking mode 变化。

不会改变 prefix key、但会改变最终回答的变化：

- user 消息变化；
- assistant/tool 历史变化；
- temperature、top-p、top-k、seed、max_tokens 变化。

shared discovery 的 `/apply-template`/`/tokenize` API 失败或 malformed response、malformed/过大/orphan manifest、namespace/scope 不匹配、LCP 低于阈值以及 restore 失败，只会按既定顺序降级到 legacy exact 或普通 cold prefill，不会直接让用户请求失败。因为 native LCP 会拒绝/截断第一处分歧之后的旧状态，上述配置变化只能造成较短命中或 cold，不能造成 stale hit。

当前 key 没有包含 GGUF 文件 hash、llama.cpp build hash 和完整启动参数。更换模型文件、量化版本、chat template 或关键 KV 配置后，应先备份并换名旧缓存目录，再重新生成 snapshot。

## 11. 收益与成本

实测：

- 无工具 Pi 请求约 5218 prompt tokens；
- 后台生成约 5198-token prefix snapshot；
- 新 session 冷恢复 wall time 约 2.6–3.4 秒；
- 同 session 后续请求使用 hot session，不读磁盘；
- 带 read 工具 schema 的 prefix 约 6678 tokens，也能正常 seed；
- direct smoke 已返回过 cached_tokens > 0。

收益来自跳过稳定 system prompt、项目规则和工具 schema 的 prefill。收益大小可以直接从 cached_tokens 和 timings.cache_n 观察。

成本：

- 不减少模型权重显存；
- 不提高 decode tokens/s；
- snapshot 大小依赖模型、context 和 recurrent checkpoints：旧验证约 150–240 MB，当前 Dirk 128K checkpointed 实测约 786 MiB/个；缓存上限 12 GiB；
- prefix seed 是可丢弃的 best-effort 优化，第一次建立 golden snapshot 会多一次 prefix prefill/save；没有安全 spare 或可事务交换的 clean owner 时立即跳过；
- 缓存请求使用全局 operation lock，优先保证 snapshot 不互相覆盖；prefix seed 不等待前台请求，竞争到前台请求时跳过；
- snapshot 与 manifest 都是私有数据。manifest 的 token IDs 可由同一 tokenizer detokenize，具有可逆性，不是匿名化 hash；目录必须保持 `0700`，snapshot/manifest 保持 `0600`，且 personal/work 使用不同 shared scope；
- 结构化日志不得包含 prompt、rendered text、token IDs 或 manifest body，只记录 `session_ref`、layer、`candidate_tokens`、`verified_lcp`、时延和错误类型等元数据。

运行验收必须包含 cold-vs-golden A/B：清除 prefix 文件，以禁用 seeding 的全新 session 测 cold；再启用 seeding，等待首个 seed 完成，并以不同 affinity 发出仅在较晚 skill/date/memory/suffix 处分歧的请求。要求先有 `shared_prefix_candidate_restored`，再由响应 metadata 产生 `shared_prefix_effective_hit` 且 API `cached_tokens`/`timings.cache_n > 0`、TTFT 改善；`shared_prefix_rejected_by_llama` 表示 snapshot restore 成功但 native LCP 没有形成实际 KV hit，不能计为命中，并触发当前 exact prefix 的 best-effort seed。改变早期 token 时必须观察到更短 LCP、rejected 或 cold。运维 purge、只显示计数而不泄露 token 的 manifest 诊断和具体 systemd 命令见 [README.md](./README.md#运行和排查)。

## 12. 关键文件

| 文件 | 作用 |
|---|---|
| [cache_core.py](./cache_core.py) | prefix payload、SHA-256 key、slot request helper |
| [cache_proxy.py](./cache_proxy.py) | HTTP proxy、hot slot、save/restore、prefix seed |
| [test_cache_core.py](./test_cache_core.py) | key 和 request helper 测试 |
| [test_cache_proxy.py](./test_cache_proxy.py) | proxy save/restore/hot slot 测试 |
| [README.md](./README.md) | 操作说明、命中判断和收益摘要 |
| [local-llm-kv-cache.service](./local-llm-kv-cache.service) | 用户级 systemd 服务模板 |

运行检查：

~~~bash
curl -fsS http://127.0.0.1:18082/health
systemctl --user status local-llm-kv-cache.service
journalctl --user -u local-llm-kv-cache.service -n 100 --no-pager
~~~
