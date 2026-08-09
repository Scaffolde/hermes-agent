# Host-resolved subagent execution profiles

Execution profiles let trusted Hermes plugins request a named subagent policy without
supplying the protocol, role, or tool selection themselves. The host resolves the
symbolic profile from `config.yaml`, loads its protocol from `HERMES_HOME`, constructs
the child, verifies the effective policy, and returns an immutable launch receipt.

## Security boundary

The default `in_process` execution backend is policy pinning, not process isolation.

All profiles provide:

- host-owned protocol, role, toolset, transition, and deadline policy;
- fail-closed exact-tool verification after child construction;
- no ambient MCP inheritance for strict profile children;
- frozen build-time tool schemas and handler identities that registry or MCP refresh
  cannot replace or broaden;
- profile launch events published only after exact-contract enforcement succeeds;
- protocol, task, tool-schema, provider, model, role, and child-session evidence in
  an immutable launch receipt.

The default in-process backend does not provide:

- filesystem or workspace confinement;
- environment, credential, network, memory, or Python-module isolation;
- forced termination of a running Python child thread;
- descendant-process ownership or confirmed cleanup;
- proof that a provider-side model name represents a particular external agent
  implementation.

A profile receipt therefore labels cleanup as cooperative and unconfirmed. A
configured deadline requests a hard interrupt and abandons the worker thread after
the deadline; it does not prove that execution stopped. Use a separate-process
execution backend with an OS-owned containment boundary when those guarantees are
required.

## Configuration

Profiles are declared under `delegation.execution_profiles` in the active Hermes
`config.yaml`:

```yaml
delegation:
  execution_profiles:
    evo-verifier:
      protocol_file: protocols/evo/verifier.md
      role: leaf
      allowed_toolsets:
        - file
      expected_tool_names:
        - read_file
        - write_file
        - patch
        - search_files
      allow_root: true
      allowed_child_profiles: []
      timeout_seconds: 300
```

The profile object is strict. Unknown keys, duplicate list entries, invalid roles,
unknown toolsets, malformed profile IDs, and non-positive or excessive deadlines are
rejected.

| Key | Required | Contract |
| --- | --- | --- |
| `protocol_file` | yes | Literal relative path under `HERMES_HOME`; no empty, `.`, or `..` segments and no symlink components. The target must be a regular UTF-8 file no larger than 32,000 bytes. The host must support descriptor-relative `open`, `O_NOFOLLOW`, and `O_DIRECTORY`; resolution fails closed otherwise. |
| `role` | yes | `leaf` or `orchestrator`. The effective child role must match exactly; depth or kill-switch degradation fails the launch. Exact-profile orchestrators do not receive generic `delegate_task`; they coordinate through host plugin tools that submit named profile launches. |
| `allowed_toolsets` | yes | Non-empty list of registered toolset names. This is host policy, not a caller override. |
| `expected_tool_names` | yes | Non-empty exact set of post-build model-callable tools. Both missing and unexpected tools fail the launch. Only immutable-registry-dispatched tools are currently supported. Deferred bridges, `execute_code`, agent-local state tools (`todo`, `session_search`, `memory`, `clarify`, `read_terminal`), generic `delegate_task`, context-engine tools, and external-memory tools are rejected until they have frozen profile adapters. |
| `allow_root` | no | Whether a parent without an execution profile may launch this profile. Defaults to `false`. |
| `allowed_child_profiles` | no | Exact profile IDs this profile may launch when acting as an orchestrator. Defaults to none. |
| `timeout_seconds` | no | Cooperative deadline greater than zero and no more than 86,400 seconds. |

Profile launches disable tool-search progressive disclosure. The child receives
the literal selected tool schemas rather than `tool_search`, `tool_describe`, and
`tool_call` bridge tools. `resolved_tool_names` and `tool_schema_digest` therefore
describe the actual callable catalog instead of a bridge to hidden tool names.

Protocol files are opened relative to the canonical Hermes home through no-follow
directory descriptors on supported POSIX systems. The canonical root descriptor is
identity-checked and held through traversal and the file read. The validated literal
path components are opened from that descriptor without a pathname-resolution pass;
every intermediate and final symlink is rejected. Replacing the root with either a
symlink or an ordinary attacker directory cannot redirect the descriptor-relative
read. The protocol bytes are SHA-256 hashed into the launch receipt. Hosts without the
required descriptor-relative primitives are unsupported and fail profile resolution
closed.
The protocol text and caller context must also fit a combined 32,000-character
lifecycle ceiling.

The pinned protocol is installed at system authority. The caller-supplied `goal` and
`context` are sent as user-role task text; they are never concatenated into the system
prompt. Exact schema construction and handler capture must observe the same tool
registry generation or the launch is rejected. Each frozen dispatch entry carries the
canonical launch-time effective schema as immutable JSON; argument coercion never
consults the mutable live registry for a strict child.

MCP refresh publication and profile freezing share one lock. Refresh requests also
carry a monotonically increasing per-agent invocation epoch, so a slower older rebuild
cannot overwrite a newer restrictive selection at the same registry generation. Once
the strict marker is published, all refreshes are rejected. Provider recovery paths
that would sanitize system/protocol text or rewrite tool schemas (including Unicode and
llama.cpp grammar recovery) re-raise the provider error instead of mutating strict
launch state.

## Plugin API

A trusted plugin uses `PluginContext.subagent_lifecycle` and passes only the named
profile plus task data:

```python
from agent.subagent_lifecycle import SubagentLaunchRequest

handle = ctx.subagent_lifecycle.launch(
    SubagentLaunchRequest(
        goal="Verify experiment exp-17",
        context="Bounded experiment metadata",
        profile_id="evo-verifier",
    )
)
receipt = ctx.subagent_lifecycle.describe(handle)
```

For profile launches, callers must not pass `allowed_toolsets` or request a non-default
role. The host resolves those values. Nested launches are checked against the
parent's host-observed profile ID and `allowed_child_profiles`; model text cannot
select a different transition graph. A profiled parent cannot submit an unprofiled
legacy launch.

Legacy callers that omit `profile_id` keep the existing lifecycle behavior.

## Launch receipt

`describe(handle)` returns a frozen `SubagentLaunchReceipt` for profile-backed
launches. It records:

- profile and protocol identity;
- SHA-256 hashes of goal, caller context, and the pinned protocol bytes. The receipt
  deliberately does not claim to attest the complete Hermes system prompt, which is
  assembled by the conversation runtime after launch;
- sorted effective tool names and a canonical tool-schema digest;
- launch-time provider and model reported by the constructed child; later runtime
  failover is not reflected in the receipt;
- effective role, delegation depth, and child session ID;
- explicit ambient, MCP, deadline, and cleanup policy labels;
- host creation time.

The receipt is also attached to the terminal `SubagentResult` and included in its
canonical result hash. Within the direct lifecycle return path, object identity and
that binding make it host-observed evidence about launch state, not an OS sandbox
attestation. The receipt and its unkeyed canonical hash are not cryptographically
authentic after serialization: callers must not treat a copied or externally supplied
receipt as host evidence.

## Phase 2 process backends

`execution_backend` defaults to `in_process`, preserving the Phase 1 behavior and
receipt labels. A process profile selects `portable` or `linux_strict` and must
declare an absolute, existing, non-root `workspace_root`; the host resolves
symlinks and pins the canonical directory. `max_process_iterations` defaults to 8
and is restricted to 1–32.

Portable execution is labeled `portable-process-unconfined`. It owns and reaps a
separate POSIX process group and scrubs its environment, but it does not claim
filesystem or network confinement. Linux strict
execution is labeled `linux-strict-bwrap-cgroup-v2` and refuses to spawn unless
Linux, bubblewrap namespace support, and delegated cgroup v2 are available. A
Linux strict profile must declare an absolute, existing, non-root
`cgroup_parent` that the host service has already delegated; Hermes never falls
back to the cgroup-v2 root. Python `Popen(pass_fds=...)` admits only the
capability socket and cgroup-launcher descriptor. The launcher closes the
cgroup descriptor before `exec` and Bubblewrap passes the remaining inherited
capability socket to the sandbox command without a nonexistent compatibility
flag.

The worker receives the secret, capability id, and launch digest only through an
inherited Unix socket bootstrap. It has no provider credential or ambient tool
registry. `session.start` supplies the exact schemas, their digest, and host-owned
local-versus-brokered classifications. Evo v1 initializes only `terminal`,
`read_file`, `write_file`, `patch`, and `search_files` inside the worker, freezes
those handlers at startup, and executes them under the selected process boundary.
Only `scaffolde_evo_agent_dispatch` is host brokered; its frozen parent handler is
invoked with the strict child bound as lifecycle parent so nested profile transitions
remain enforced. Unclassified tools, including process, GUI, browser, network,
credential, and arbitrary plugin tools, fail before spawn. Provider calls remain in
the parent. Process profiles currently support only `api_mode: chat_completions`;
other modes fail before spawn. All exact profiles also reject xAI Responses because
that provider path rewrites tool schemas after the launch digest would otherwise be
frozen.

`describe_execution(handle)` returns a separate immutable process execution
receipt. Terminal process results bind both its canonical hash and the unchanged
Phase 1 launch receipt; launch receipts are never relabeled as containment proof.

## Rollout rule

Execution profiles are opt-in infrastructure for trusted plugins. Do not advertise a
capability as isolated or production-confined solely because it uses a profile. A
strict capability must separately prove its process boundary, tool-broker authority,
workspace enforcement, capability revocation, root-process reaping, and containment
emptiness.
