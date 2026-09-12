# Local LLM KV Cache

这套方案为本机的 Pi、Zed 编程 agent 增加两级缓存：

1. 同一会话的内存热缓存，继续对话时直接复用 llama.cpp slot。
2. 项目稳定前缀的磁盘冷缓存，包括可跨新会话复用的 golden shared prefix。代理或 llama 重启后，新会话可以恢复 system prompt、工具 schema 等仍然完全相同的 token 前缀。

当前入口是 `127.0.0.1:18082`，上游是 `127.0.0.1:8080`。prompt cache 的实际计算和 checkpoint restore 都由 llama.cpp 完成，proxy 不伪造命中。

重要：Qwen3.8 这类 hybrid/recurrent 模型必须使用带 checkpoint 持久化修复的 llama.cpp。本机使用分支 `local/kv-restore-checkpoints`、提交 `862535a`；未修复的 llama.cpp 可能返回 `n_restored > 0`，但下一条请求仍然 `cache_n=0`。对应的上游修复讨论见 [llama.cpp PR #26004](https://github.com/ggml-org/llama.cpp/pull/26004)。

详细架构和 Mermaid 设计图见：[DESIGN.md](./DESIGN.md)。

## Quick start

要求：Python 3.10+，以及已经运行并开启 slot save/restore、包含 hybrid checkpoint 持久化修复的 llama.cpp server。

```bash
mkdir -p ~/server-ops
git clone https://github.com/gbuisson/local-llm-kv-cache.git ~/server-ops/local-llm-kv-cache
mkdir -p ~/.config/systemd/user
cp ~/server-ops/local-llm-kv-cache/local-llm-kv-cache.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now local-llm-kv-cache.service
curl -fsS http://127.0.0.1:18082/health
```

如果 llama.cpp 不在 `127.0.0.1:8080`，修改用户 unit 中的 `PI_LLAMA_UPSTREAM`。Pi 和 Zed 的 provider URL 需要指向 `http://127.0.0.1:18082/v1`。

### Production hardening options

| Variable | Default | Purpose |
|---|---:|---|
| `PI_LLAMA_CACHE_NAMESPACE` | `default` | Invalidates snapshots across model, llama.cpp, context, or template deployments. Use a stable deployment fingerprint. |
| `PI_LLAMA_CACHE_ENABLE_PREFIX_SEEDING` | `true` | Allows best-effort golden-prefix seeding. `-np 1` 也受支持，见下文的事务式 slot swap。 |
| `PI_LLAMA_CACHE_PREFIX_SEED_DELAY` | `2` | 成功响应后，后台 seed 开始前的秒数；有前台等待者时 seed 会跳过。 |
| `PI_LLAMA_CACHE_PREFIX_SEED_TIMEOUT` | `600` | `/completion` seed 的专用 HTTP timeout（秒）。应高于目标机上完整稳定 prefix 的 cold prefill 时间；不改变其他内部 API 的 120 秒 timeout。必须为正数。 |
| `PI_LLAMA_CACHE_SHARED_PREFIX_SCOPE` | `default` | Shared-prefix 隔离域。个人与工作环境必须使用不同的显式值，例如 `personal` 与 `work`。 |
| `PI_LLAMA_CACHE_MIN_SHARED_PREFIX_TOKENS` | `128` | 接受 golden 候选所需的最小 exact LCP token 数；必须为正整数。 |
| `PI_LLAMA_CACHE_SAVE_POLICY` | `all` | `all` saves every successful response. `terminal` defers `finish_reason=tool_calls`, then saves on a terminal response, before eviction, or during clean shutdown. |
| `PI_LLAMA_CACHE_REQUIRE_SESSION_ID` | `false` | Rejects anonymous cache-managed text chat requests. Media chat requests remain allowed without affinity because they run serialized and unmanaged after flushing dirty state. Recommended when conversation isolation is required. |
| `PI_LLAMA_CACHE_MAX_GIB` | `12` | LRU disk budget for snapshot files. Successful restores and hot reuse refresh recency. |

`terminal` is intended for tool-heavy agents. Deferred state remains in the hot llama.cpp slot and is flushed before any cold request can overwrite an owned slot. If slot metadata cannot be verified, the proxy fails safe and saves immediately.

Explicit affinity is accepted from `session_id`, `conversation_id`, or `prompt_cache_key` either at the JSON body root or in an `extra_body` object, and from `X-Session-Affinity`, `X-Session-Id`, `X-Conversation-Id`, `X-Pi-Session-Id`, `X-OpenCode-Session`, or `X-Client-Request-Id` headers. Affinity fields are removed before forwarding upstream.

The public route contract is explicit and fail-closed:

| Class | Routes | Slot behavior |
|---|---|---|
| Session cache | `POST /v1/chat/completions` | Session-affine hot/SSD KV lifecycle |
| Utility/control | `POST /tokenize`, `/detokenize`, `/apply-template`, token-count routes, and `/v1/chat/completions/control` | Forward without flushing or evicting KV |
| Unmanaged inference | completions, responses, embeddings, infill, reranking, and Anthropic messages routes | Flush dirty state and reset ownership before forwarding |
| Monitoring/model metadata | regular `GET`, `HEAD`, and `OPTIONS` routes | Transparent passthrough; no slot mutation |
| Administration | `/slots`, LoRA/tools administration, mutable `/props`, model load/unload/download, and all `DELETE` requests | `404`, never forwarded |
| Unknown `POST` | any route outside the allowlist | `404`, never forwarded |

For clients that consistently prefix llama.cpp utility routes with `/v1`, the proxy maps `GET`/`HEAD /v1/props` to `/props` and maps `POST /v1/tokenize`, `/v1/detokenize`, and `/v1/apply-template` to their native unprefixed endpoints. `HEAD` suppresses the upstream response body, while `OPTIONS` and its CORS preflight request headers pass through unchanged.

Operational events are emitted as compact JSON. Session identifiers、rendered prompt、token ID 数组和 manifest 内容都不会写入日志；`session_ref` 是可用于关联的截断 SHA-256。Shared-prefix 事件只记录候选 token 数和验证后的 LCP 长度等计数。

### Context compaction

A compaction boundary must not blindly continue the longer pre-compaction transcript. Hermes Agent compacts in place by default (`compression_in_place=True`), so its provider-derived affinity remains unchanged. The proxy may reuse the hot slot, or restore that session's latest disk snapshot after eviction; llama.cpp then performs longest-common-prefix matching against the compacted request, discards KV state after the divergence, and prefills the summary and protected tail. This preserves the stable system/developer prefix while replacing the old dynamic history.

Hermes also supports rotation to a child session at `boundary_reason="compression"`. In that mode a provider derived from the Hermes session ID produces a fresh proxy affinity, the parent snapshot remains isolated, and the compacted child starts cold unless a compatible prefix snapshot is available.

If compaction changes the stable system/developer prefix, the proxy flushes any dirty state and starts cold rather than restoring an incompatible session snapshot. Production validation should cover both hot in-place compaction and in-place compaction after the slot was evicted, because the latter exercises checkpoint restore followed by LCP truncation.

## 解决的问题

编程 agent 的第一条消息通常包含：

- coding-agent system prompt；
- 项目规则和 `AGENTS.md` 内容；
- 工具定义和 JSON schema；
- 当前模型的 chat template 参数。

这些内容在同一个项目中大部分稳定，但每次新建 session 时，传统请求会重新 prefill 全部 prompt。缓存代理把稳定前缀和会话尾部拆开处理，避免每次从零开始。

## 架构

```text
Pi / pi-acp / Zed
          |
          v
127.0.0.1:18082  cache_proxy.py
          |
          +-- 内存 session slot 映射
          +-- ~/.llama-slot-cache/*.bin
          +-- 私有 *.bin.manifest.json（scope + exact token IDs）
          |
          v
127.0.0.1:8080  llama.cpp llama-server
```

代理只监听 loopback，没有暴露到局域网或公网。

### Slot ownership

同一个 session 的 slot 状态是代理的内存 affinity 记录。请求不会为了复用另一个 session 的稳定 prefix 而抢占其完整上下文；没有可用的 hot session 时，代理优先选择没有 session 所有权的空闲 slot，再按以下顺序恢复：

```text
hot session → session snapshot → shared golden prefix → legacy exact prefix → cold/full prefill
```

这样多个 Pi、Zed session 交替使用时，失去内存 slot 的 session 仍能从自己的完整快照恢复，不会把另一个 session 的动态历史当成自己的缓存。

## Cache key

### Prefix key

代理首先构造稳定前缀对象：

```text
messages：从开头开始，连续保留 role=system/developer 的消息
```

遇到第一个 `user`、`assistant` 或 `tool` 消息就停止。除此之外还加入这些字段：

```text
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
```

然后使用排序后的 canonical JSON 计算：

```text
prefix_key = SHA256(canonical_json(prefix_payload))
```

实现位置：[cache_core.py](./cache_core.py)。

### Snapshot 文件名

磁盘文件名还会加入缓存版本、缓存类型和身份：

```text
snapshot_key = SHA256("3" + namespace + kind + identity + prefix_key)
```

文件格式：

```text
local-llm-session-<hash>.bin
local-llm-prefix-<hash>.bin
```

其中：

- `session` 的 identity 是 Pi/Zed 的 session ID；
- `prefix` 的 identity 是固定字符串 `prefix`，所以同一个项目的不同 session 可以共享 prefix 文件；
- 版本号 `3` 和 deployment namespace 用来让旧格式或不兼容部署的缓存整体失效。

用户消息、assistant 历史和 tool result 不参与 prefix key。这是有意设计：它们属于会话动态尾部，应该由 llama.cpp 的 common-prefix cache 处理。

### Golden shared prefix：以实际 token 为准

`prefix_key` 仍用于 session snapshot 和旧版 exact-prefix fallback；跨不同 bootstrap 的 golden 复用不把 canonical JSON hash 当作“足够相似”的证明。对于没有 hot/session snapshot 的文本请求，代理使用当前 llama.cpp 实例的原生 API 按固定顺序计算实际输入：

1. `POST /apply-template`，请求体是将要发送的 chat-completion body；
2. 取返回的完整 rendered prompt，再 `POST /tokenize`，参数为 `add_special=false`、`parse_special=true`；
3. 用返回的整数 token ID 序列与磁盘 manifest 比较。

每个 golden snapshot 有一个同名 companion manifest：

```text
local-llm-prefix-<hash>.bin
local-llm-prefix-<hash>.bin.manifest.json
```

manifest 包含 schema version、`PI_LLAMA_CACHE_NAMESPACE`、`PI_LLAMA_CACHE_SHARED_PREFIX_SCOPE`、snapshot basename 和完整 token ID 数组。只考虑 namespace 与 shared scope 都完全相同、manifest 合法且对应 `.bin` 确实存在的候选；其中选择与当前请求 **最长 exact common prefix (LCP)** 最大且达到 `PI_LLAMA_CACHE_MIN_SHARED_PREFIX_TOKENS` 的一个。比较逐 token 严格相等，不使用 hash 相似度、百分比或文本启发式。

restore 后代理把该 slot 固定给当前请求，让 llama.cpp 自己执行 native LCP。第一处 token 分歧后的旧 KV/GDN 状态全部被拒绝或截断，只重新使用分歧之前的状态。因此 skills、tools、system/developer prompt、memory、日期、chat template 或 reasoning/thinking mode 的变化只会让 LCP 变短（可能低于阈值而 cold），不会产生跨越分歧点的 stale hit。例如日期位于较晚位置时仍可复用它之前的 token；早期 system prompt 改动则可能几乎完全 cold。

`PI_LLAMA_CACHE_SHARED_PREFIX_SCOPE` 是显式数据与信任边界，不是性能标签。同一机器上的个人和工作 Hermes 必须分别配置，例如：

```ini
Environment=PI_LLAMA_CACHE_SHARED_PREFIX_SCOPE=personal
# 工作实例使用另一个 unit/cache namespace：
Environment=PI_LLAMA_CACHE_SHARED_PREFIX_SCOPE=work
```

不要让 personal/work 实例共享同一个 scope。部署模型、GGUF、llama.cpp、context 或 template 发生不兼容变化时，还必须轮换 `PI_LLAMA_CACHE_NAMESPACE`。

## 请求生命周期

### 1. 收到请求

代理识别 session affinity：

1. `X-Session-Affinity`, `X-Session-Id`, `X-Conversation-Id`
2. `X-Pi-Session-Id`, `X-OpenCode-Session`, `X-Client-Request-Id`
3. 请求体中的 `session_id`, `conversation_id` 或 `prompt_cache_key`
4. 没有显式 ID 时，默认使用 `anonymous-<prefix_key>`；`PI_LLAMA_CACHE_REQUIRE_SESSION_ID=true` 时拒绝请求

图片和其他非文本多模态请求不做磁盘快照，只转发请求。成功命中的 hot/session/shared/legacy slot 只由 proxy 固定 `id_slot`；`cache_prompt` 保留客户端或 llama-server 的默认行为。当前生产 llama-server 默认启用 prompt cache；客户端显式关闭时会失去 LCP 性能收益，但不会越过 token 分歧产生 stale hit。

### 2. 查找缓存

查找顺序如下：

| 层级 | 条件 | 动作 |
|---|---|---|
| 同 session 热缓存 | session ID、prefix key 相同，slot idle，token 数一致 | 直接复用内存 slot，不读盘 |
| 空闲 slot | hot session 不可用 | 优先选择没有 session 所有权的 slot，避免驱逐其他活跃会话 |
| 磁盘 session 快照 | `local-llm-session-*.bin` 存在且 restore 成功 | 恢复具体会话 |
| Golden shared prefix | 同 namespace+scope 的合法 manifest 中 exact LCP 最长且达到最小阈值，snapshot restore 成功 | 恢复候选并由 llama.cpp 在第一处分歧处截断 |
| Legacy exact prefix | 当前 `prefix_key` 的 `local-llm-prefix-*.bin` 存在且 restore 成功 | 兼容旧缓存格式的 exact fallback |
| 缓存未命中 | 上述条件都不满足 | cold/full prefill |

命中 hot session 或成功 restore session/shared/legacy snapshot 时，请求会固定到已准备的 slot：

```json
{
  "id_slot": 1
}
```

真正的 cold 请求不会添加 `id_slot`。proxy 不覆盖 `cache_prompt`；shared 性能复用依赖客户端或 llama-server 启用 native prompt cache。关闭它只会使当前完整 prompt 重新 prefill。`/apply-template`、`/tokenize` 失败或返回 malformed 数据，以及 malformed/orphan manifest、低于最小 LCP、scope/namespace 不匹配、shared restore 失败，都不会形成 shared hit：仅在 independently valid 的 legacy exact snapshot 存在时继续该 fallback，否则 cold；不会猜测 token，也不会把发现错误升级为用户请求失败。

### 3. 保存 session

默认情况下，成功响应返回后，代理通过 llama.cpp 的 slot save API 保存：

```text
当前 slot -> local-llm-session-<hash>.bin
```

同时更新内存中的 session slot 映射。

使用 `PI_LLAMA_CACHE_SAVE_POLICY=terminal` 时，`finish_reason=tool_calls` 不立即写盘。状态保持为 dirty hot slot；同一 session 的下一次工具回合直接复用它。代理会在最终响应、另一个 session 可能覆盖 slot 之前、或干净关闭时保存。

### 4. 后台生成纯 prefix

如果项目还没有纯 prefix 文件，代理先对纯 system/developer/tools body 做一次 `/apply-template` + `/tokenize`，然后在安全的空闲 slot 中用 `/completion` 直接提交这组 exact token IDs：

```json
{
  "prompt": [151644, 8948, 198],
  "cache_prompt": false,
  "n_predict": 0,
  "stream": false
}
```

这个请求的 slot 会被原子保存为 `local-llm-prefix-<hash>.bin`。token identity 的证明来自 `/completion` 收到的 exact token array；`n_saved == len(tokens)` 是额外的完整性检查，而不是单独的 identity 证明。两者都成立后，proxy 才原子发布 mode `0600` 的 companion manifest。shared restore 也要求 `n_restored == len(manifest.tokens)`；计数不一致会删除该 shared pair 并回退。缓存目录本身为 `0700`。有 spare unowned idle slot 时直接使用它。`-np 1` 没有 spare slot 时也可以安全 seed，但仅限唯一 owner 已 idle、clean、slot token 数一致，而且该 owner 的完整 session snapshot 已经落盘：代理在持有完整 operation lock 的情况下先把 owner 保存到独立 guard snapshot 并验证 token 数，执行 seed，依次原子保存 snapshot 和 manifest，再在 `finally` 中从 guard 恢复并验证 owner token 数后才释放锁。owner restore 失败或 token 数不符时只忘记 hot ownership，原始持久化 session snapshot 保留供下一次请求恢复。

dirty owner、缺失 session snapshot、slot/guard/seed/restore token 数不一致、前台请求正在等待、代理忙或任何 API/save/manifest 错误都会跳过或回滚为无 shared golden snapshot。损坏 pair 的物理删除是 best-effort；read-only/I/O 清理失败只记录结构化事件。已经被 shared 校验拒绝的文件在同一次请求内不会再被 legacy exact fallback 重试，因此清理失败仍继续 cold。legacy exact prefix 可以作为其他文件的兼容 fallback，但不会抑制 golden migration seed。prune 同时删除 orphan manifests 和临时 metadata。成功恢复任意兼容 shared golden 后不会为该次请求再 seed 一个仅日期不同的新 exact snapshot，避免磁盘中产生大量近重复文件。seed 是可丢弃的 best-effort 优化，不参与 session 正确性；第一个 session 仍需支付正常 prefill，以及后台 seed 的一次额外 prefill/save 成本。

纯 prefix 快照很重要。Qwen3.8 是混合 GDN 架构，不能可靠地把“包含旧 user/assistant 历史的完整快照”直接当作所有新 session 的 prefix。纯 prefix 恢复后，llama.cpp 才能把新用户消息作为 suffix 继续计算。

## 缓存不是答案缓存

这个系统缓存的是：

```text
KV(prefix) + 当前 slot 状态
```

不是：

```text
问题 -> 答案
```

因此，相同的 system prompt 或相同的第一个 prefix，只表示模型可以从相同的隐藏状态开始计算。不同的 user suffix、工具结果、采样参数或随机数，仍然可以产生不同答案。

即使完整请求完全相同，也只有在模型版本、推理参数、seed、采样器和硬件计算路径都一致时，才适合期待完全一致。当前 llama-server 没有固定 `--seed`，默认使用随机 seed；`temperature`、`top_p`、`top_k`、`seed` 和 `max_tokens` 不属于 prefix key，因为它们控制生成阶段，不改变已缓存的 prompt 状态。

需要做可重复性检查时，应显式固定相同的 `seed` 和全部采样参数，并比较完整请求的输出。`cached_tokens > 0` 只证明 prompt token 被复用，不证明回答一定相同。

## 什么时候会失效

以下变化会生成新的 legacy prefix key，并在 golden 路径中于实际 rendered token 的第一处分歧处缩短 LCP：

- system/developer prompt 文本、顺序、空格或换行变化；
- tools 或工具 JSON schema 变化；
- model ID 变化；
- thinking 或 chat template 参数变化；
- response format、grammar、JSON schema 变化；
- `add_generation_prompt`、`continue_final_message` 等模板行为变化。
- Hermes skills、system/developer 指令或 memory 内容变化；
- 注入日期或其他易变 bootstrap 内容变化；
- llama.cpp chat template 实际输出或 reasoning/thinking mode 变化。

以下情况也会导致磁盘缓存未命中：

- session ID 改变：具体 session 快照不同，但仍可能命中同一个 prefix 文件；
- 快照被 12 GiB 上限淘汰；
- 文件损坏或 llama.cpp restore 返回错误；
- 代理或 llama.cpp 的 slot 不可用；
- 多模态请求；
- 缓存版本变化。

render/tokenize API 失败、响应 malformed、manifest malformed、manifest 指向不存在的 snapshot（orphan）、seed 的 `n_saved` 计数不一致或 restore 失败都不会让请求失败。代理记录不含 prompt/token 内容的 warning，并 cold/full prefill；成功后可以重新生成 prefix 快照。严格 token LCP 保证配置变化只造成较短命中或 cold，不会造成 stale reuse。

### 模型更换注意事项

当前 prefix key 没有包含 GGUF 文件 hash、llama.cpp build hash 和完整启动参数。因此更换模型文件、量化版本、chat template 或关键 KV 配置后，应把旧缓存目录移到备份目录，再让代理重新生成缓存。不要把不同模型版本的 slot snapshot 混用。

## 实测收益

验证使用当前 Qwen3.8-27B IQ4_XS、llama.cpp、RX 7900 XTX 配置：

- patched llama.cpp 的 32,185-token 实测：冷 prefill `54.8s`；同进程 restore 后 `cache_n=32151`、`prompt_n=34`、`0.37s`；
- 完整重启 llama 后再 restore 12,700-token snapshot：`n_restored=12700`，下一条 divergent prompt 为 `cache_n=12662`、`prompt_n=38`、`0.356s`；
- 真实 proxy 链路中，session 被另一会话挤出后 restore 仍返回 `cached_tokens=12743/12793`；
- 未修复的 llama.cpp 在相同 restore 场景中会返回 `n_restored`，但 `cache_n=0`，因此不能只看 restore 日志判断命中。
- 一套 Dirk Qwen3.8 Q6_K、128K、KV q4_0、checkpointed 测试部署在 2026-09-12 的真实 Hermes cold baseline 为 `25,960` prompt tokens、`cached_tokens=0`、`233.94s` prompt eval、`110.97 tok/s`；对应 session snapshot 为 `824,255,044` bytes（`786.07 MiB`）。Golden-prefix B 结果见本次发布的 benchmark 记录。

对于更大的编程 agent prompt，收益主要来自跳过稳定 system prompt、项目规则和工具 schema 的 prefill。能节省多少时间取决于实际 token 数和当前 prompt processing throughput，但跳过的 token 数会直接体现在 `cached_tokens` / `timings.cache_n` 中。

## 成本和边界

- 这是 prompt prefill 优化，不会减少模型权重显存，也不会提高模型 decode 本身的 tokens/s；
- slot snapshot 大小强烈依赖模型、context 和 recurrent checkpoints：旧验证约 150–240 MB，而当前 Dirk 128K checkpointed 实测 session snapshot 为 786.07 MiB；12 GiB 在该尺寸下最多容纳 15 个完整 snapshot，实际还要给不同长度和 manifest 留余量；
- prefix seed 是可丢弃的 best-effort 优化；首次建立 golden snapshot 有一次 seed prefill/save 成本，有前台等待者或没有可安全交换的 slot 时立即跳过；
- 缓存请求通过全局 operation lock 串行化，优先保证 slot snapshot 不互相覆盖；高并发客户端会排队等待空闲 slot；
- snapshot 和 manifest 都是私有数据。manifest 的 token ID 可以用同一 tokenizer 逆向 detokenize，不能视为 hash 或匿名化数据；它继承 snapshot 的敏感级别。只应保留在 mode `0700` 的本机用户目录、以 `0600` 文件存储，不应同步/上传，也不应把 18082 暴露出去。

## 运行和排查

服务：

```bash
systemctl --user status local-llm-kv-cache.service
systemctl --user is-active local-llm-kv-cache.service
journalctl --user -u local-llm-kv-cache.service -n 100 --no-pager
```

健康检查：

```bash
curl -fsS http://127.0.0.1:18082/health
curl -fsS http://127.0.0.1:18082/v1/models
```

确认真正命中：

```text
代理日志：cache_hit(layer=hot/session/shared_prefix/prefix)、shared_prefix_restore(candidate_tokens, verified_lcp)
API 响应：usage.prompt_tokens_details.cached_tokens > 0
API timings：timings.cache_n > 0
```

`shared_prefix_restore` 中的 `verified_lcp` 是代理在 manifest 与当前实际 token 间验证的长度；最终真正复用量仍以 llama.cpp 返回的 `cached_tokens` / `timings.cache_n` 为准。诊断 manifest 时不要打印可逆的 token 数组；下面只显示隔离域、token 数和 orphan 状态：

```bash
python3 - <<'PY'
import json
from pathlib import Path
d = Path.home() / ".llama-slot-cache"
for p in sorted(d.glob("*.manifest.json")):
    try:
        m = json.loads(p.read_text())
        print(p.name, "namespace=", m.get("namespace"), "scope=", m.get("scope"),
              "tokens=", len(m.get("tokens", [])), "snapshot_exists=", (d / str(m.get("snapshot"))).is_file())
    except Exception as e:
        print(p.name, "MALFORMED", type(e).__name__)
PY
journalctl --user -u local-llm-kv-cache.service --since '10 minutes ago' --no-pager \
  | grep -E 'shared_prefix|cache_hit|cache_miss|prefix_seed'
```

只清除 golden/legacy prefix（保留 session snapshots）并让其重新 seed：

```bash
systemctl --user stop local-llm-kv-cache.service
rm -f ~/.llama-slot-cache/local-llm-prefix-*.bin \
      ~/.llama-slot-cache/local-llm-prefix-*.bin.manifest.json
systemctl --user start local-llm-kv-cache.service
```

### Golden cache A/B acceptance

使用真实 Hermes 请求做 A/B，不要只检查 restore API。两组都应使用相同模型、模板和 bootstrap，但使用全新的 session affinity；B 的第二个 session 应只在较晚的 skill/date/memory 或 user suffix 处做受控变化。

```bash
# A: 在声明式 service 配置中暂时设为 false，激活配置后发送一个全新 Hermes session。
PI_LLAMA_CACHE_ENABLE_PREFIX_SEEDING=false

# B: 在同一声明式配置中启用独立 scope，激活后发送第一个 seed session；
# 等待 seed 完成，再发送第二个全新 session。
PI_LLAMA_CACHE_ENABLE_PREFIX_SEEDING=true
PI_LLAMA_CACHE_SHARED_PREFIX_SCOPE=personal-ab-v1
PI_LLAMA_CACHE_MIN_SHARED_PREFIX_TOKENS=128
```

接受条件：先确认 A 的响应确实为 `cached_tokens=0`；如果 llama.cpp slot 仍有旧状态，先按部署方式重启/清空 llama.cpp 后再测。B 第一次支付 seed 成本，随后不同 session 出现 `layer=shared_prefix`、`verified_lcp` 达标，且 `cached_tokens`/`timings.cache_n` 和 TTFT 优于 A。受控变化只能把复用点停在第一处分歧之前；把早期 token 改掉应得到更短 LCP 或 cold。测试后恢复正式的 declarative personal/work scope 并重新激活配置；不要遗留 imperative service override。

测试：

```bash
cd ~/server-ops/local-llm-kv-cache
python3 -m unittest -v test_cache_core.py test_cache_proxy.py
python3 -m py_compile cache_core.py cache_proxy.py
python3 -m pip install coverage
python3 -m coverage run --branch --source=. -m unittest discover -s . -p 'test_*.py' -v
python3 -m coverage report --include='cache_core.py,cache_proxy.py' --fail-under=100
```

## 当前配置入口

- Pi：`~/.pi/agent/models.json`
- Zed：`~/.config/zed/settings.json`
- systemd 模板：[local-llm-kv-cache.service](./local-llm-kv-cache.service)；当前安装位置是 `~/.config/systemd/user/local-llm-kv-cache.service`
- 代理代码：[cache_proxy.py](./cache_proxy.py)
- key 逻辑：[cache_core.py](./cache_core.py)
- 磁盘缓存：`~/.llama-slot-cache`

## License

Apache-2.0，详见 [LICENSE](./LICENSE)。
