# Egress credential-injection proxy (iron-proxy)

When Hermes runs your agent inside a remote terminal sandbox — Docker, Modal, SSH — that sandbox normally holds your real upstream API keys (`OPENROUTER_API_KEY`, `OPENAI_API_KEY`, etc.). A prompt-injected agent in that sandbox can `cat ~/.config/openrouter/auth.json` or `printenv | grep -i key` and exfiltrate them.

The egress proxy fixes this: the sandbox holds opaque **proxy tokens**, never the real keys. All outbound traffic from the sandbox routes through a local [iron-proxy](https://github.com/ironsh/iron-proxy) daemon (Apache-2.0, Go) on the host, which terminates TLS and swaps the proxy token for the real credential before forwarding the request upstream. Compromise the sandbox and the attacker walks away with tokens that only work from behind the proxy.

This page covers the Docker backend, which is what v1 ships. Modal, Daytona, and SSH wiring will follow in later releases.

## What it is

- A managed `iron-proxy` subprocess on the host, lazy-installed into `~/.hermes/bin/iron-proxy`
- A local CA at `~/.hermes/proxy/ca.crt` that the sandbox trusts so iron-proxy can MITM TLS and rewrite headers
- A `proxy.yaml` config at `~/.hermes/proxy/proxy.yaml` listing the upstream hosts you allow and the secrets-transform mapping
- A `mappings.json` recording which proxy token corresponds to which real env var

The sandbox gets `HTTPS_PROXY=http://host.docker.internal:9090` plus a set of `HERMES_PROXY_TOKEN_<ENV_NAME>` env vars. The agent code reads those tokens instead of the real API keys. iron-proxy's `secrets` transform matches the token in the `Authorization` header and substitutes the real value sourced from its own environment.

## What it is not

- It is **not** the inbound `hermes proxy` command, which is an OAuth aggregator reverse proxy. Different command (`hermes egress`), different direction.
- It does **not** sit between your local terminal and providers — only between the sandbox and providers.
- It does **not** rewrite credentials for in-process LLM calls the host process makes. Those continue to use your `.env` keys directly. The threat model is the *sandbox*, not the host.

## Quick start

```bash
# 1. Install the iron-proxy binary (pinned version, SHA-256 verified)
hermes egress install

# 2. Run the wizard: generates CA, mints proxy tokens for every provider key
#    in your env, writes proxy.yaml.
hermes egress setup

# 3. Start the proxy daemon
hermes egress start

# 4. Check status
hermes egress status
```

Once running, the Docker terminal backend automatically:

- Mounts `~/.hermes/proxy/ca.crt` into the sandbox at `/etc/ssl/certs/hermes-egress-ca.crt`
- Sets `HTTPS_PROXY`, `HTTP_PROXY`, `REQUESTS_CA_BUNDLE`, `SSL_CERT_FILE`, `CURL_CA_BUNDLE`, `NODE_EXTRA_CA_CERTS` to make every common HTTP runtime route through the proxy and trust the CA
- Adds `--add-host=host.docker.internal:host-gateway` so the sandbox can reach the host-side proxy on Linux (Docker Desktop handles this automatically on macOS/Windows)
- Exports one `HERMES_PROXY_TOKEN_<ENV_NAME>` per minted mapping

## Configuration

The full config lives in `~/.hermes/config.yaml` under the `proxy:` section:

```yaml
proxy:
  # Master switch. When false the feature is a complete no-op — no
  # binaries downloaded, no docker mounts added, no subprocess started.
  enabled: false

  # Tunnel listener port. Sandboxes hit http://host.docker.internal:<port>.
  tunnel_port: 9090

  # Auto-download the pinned iron-proxy binary on first use.
  auto_install: true

  # Where iron-proxy looks up the real upstream secrets at egress time.
  #   env       — process env (default). Whatever is in your ~/.hermes/.env
  #               at proxy-start time is the source of truth.
  #   bitwarden — refetch from Bitwarden Secrets Manager on each proxy
  #               restart. Rotation in the BW web app propagates without
  #               touching .env. Requires `secrets.bitwarden.enabled: true`.
  credential_source: env

  # When true (default), the Docker backend refuses to start a sandbox if
  # the proxy is enabled but not running. Set to false to fall back to the
  # legacy "real credentials inside the sandbox" posture when the proxy
  # is unavailable.
  enforce_on_docker: true

  # Extra allowed upstream hosts beyond the bundled defaults.
  # Wildcards (`*.foo.com`) are supported. The defaults cover OpenRouter,
  # OpenAI, Anthropic, Google, xAI, Mistral, Groq, Together, DeepSeek,
  # and Nous Research.
  extra_allowed_hosts: []
```

## Bitwarden integration

If you already use Bitwarden Secrets Manager via [`hermes secrets bitwarden setup`](../secrets/bitwarden), the egress proxy can pull real credentials from there instead of `os.environ`:

```bash
hermes egress setup --from-bitwarden
```

This sets `proxy.credential_source: bitwarden` and discovers provider env names from your BW project. Rotating a key in the Bitwarden web app then propagates to your sandboxes on the next `hermes egress start` — no `.env` edits, no Hermes restart on the host.

## Slash commands

The CLI subcommand tree:

```
hermes egress install          # download the pinned iron-proxy binary
hermes egress setup            # interactive wizard
hermes egress setup --tunnel-port N
hermes egress setup --from-bitwarden
hermes egress start            # spawn the managed proxy daemon
hermes egress stop             # SIGTERM (then SIGKILL after 5s grace)
hermes egress status           # binary + config + pid + listening state + mappings
hermes egress status --show-tokens
hermes egress disable          # flip proxy.enabled = false
hermes egress config           # print the path to proxy.yaml for debugging
```

## How it works

```
┌──────────────┐                ┌──────────────┐                ┌─────────────┐
│ Docker       │ CONNECT /     │ iron-proxy    │ HTTPS w/       │ OpenRouter  │
│ sandbox      ├──────────────▶│ (host:9090)   ├───────────────▶│ / OpenAI /  │
│              │ HTTP forward  │               │ real API key   │ Anthropic …  │
│ has:         │ w/ proxy tok  │ mints leaf    │                │             │
│ - proxy tok  │ in Auth hdr   │ cert from CA  │                │             │
│ - CA cert    │               │ matches token │                │             │
│ - HTTPS_PROXY│               │ swaps secret  │                │             │
└──────────────┘               └──────────────┘                └─────────────┘
                                       │
                                       │ structured audit log
                                       ▼
                              ~/.hermes/proxy/iron-proxy.log
```

1. Sandbox makes an HTTPS request, e.g. `POST https://openrouter.ai/v1/chat/completions` with `Authorization: Bearer hermes-proxy-openrouter-…` (the proxy token, not the real key).
2. Because `HTTPS_PROXY` is set, the request goes to iron-proxy as a CONNECT tunnel.
3. iron-proxy checks the allowlist. `openrouter.ai` is allowed.
4. iron-proxy mints a leaf cert signed by our CA for `openrouter.ai`, terminates the TLS connection, inspects the request.
5. The `secrets` transform matches the proxy-token string in the `Authorization` header and substitutes the real `OPENROUTER_API_KEY` value, sourced from iron-proxy's own environment.
6. Request is re-encrypted and forwarded to OpenRouter.
7. Every request is logged as a structured JSON entry to `~/.hermes/proxy/iron-proxy.log`.

A request to a non-allowlisted host (e.g. `https://attacker.example.com/leak?key=...`) is rejected with HTTP 403 before any bytes leave the host.

## Security model

**What this protects against:**

- Prompt-injected agent in a Docker sandbox reading `printenv` / credential files and exfiltrating real keys.
- Compromised dependency in the sandbox phoning home to an arbitrary host — default-deny allowlist blocks unknown destinations.
- Agent dialing cloud metadata endpoints (`169.254.169.254`) — iron-proxy denies these by default via `upstream_deny_cidrs`.

**What it does NOT protect against:**

- A compromised host process. If the agent process itself is compromised, real keys in the host's `~/.hermes/.env` are exposed regardless. This is a defense-in-depth feature for *sandbox* compromise, not host compromise.
- Sandbox processes that bypass `HTTPS_PROXY` by using a raw socket or a non-standard env var. The proxy can't intercept what doesn't route to it.
- Allowlisted-host data exfiltration. If `api.openai.com` is allowed, an agent could embed exfil data in a request body to that host. The audit log captures this but doesn't prevent it.

## Failure modes

- **Binary not installed, `auto_install: true`** — first `hermes egress setup` or `hermes egress start` downloads it. SHA-256 verified against the upstream `checksums.txt`.
- **Binary not installed, `auto_install: false`** — `start` fails with a clear message pointing to manual install.
- **`enabled: true` but proxy not running** — with `enforce_on_docker: true` (default), Docker sandbox creation refuses to start with an explanatory error. With `enforce: false`, it falls back to direct outbound with real creds and logs a warning.
- **Port collision** — iron-proxy exits immediately; `hermes egress start` reports the last 20 log lines and fails with non-zero exit.
- **Upstream-host denied** — sandbox gets HTTP 403 from the proxy with a body explaining which host wasn't allowed. The agent sees the error and reports it.
- **Cloud metadata IP (169.254.169.254) requested** — refused by `upstream_deny_cidrs` regardless of allowlist.

## Limitations (v1)

- Docker backend only. Modal, Daytona, and SSH wiring will follow in separate PRs.
- Only bearer-token providers (OpenRouter, OpenAI, Anthropic-via-OR, etc.) are wired through the `secrets` transform out of the box. Providers with custom auth (x-api-key, query params, signatures) need per-provider rules.
- No native Windows binary upstream. Run on Linux/macOS or inside WSL.
- The CA is a 10-year self-signed cert on first generation. Rotation requires `openssl genrsa ...` by hand (or wait for a follow-up that adds `hermes egress rotate-ca`).

## See also

- Upstream project: <https://github.com/ironsh/iron-proxy>
- Upstream docs: <https://docs.iron.sh/>
- Bitwarden integration: [`hermes secrets bitwarden`](../secrets/bitwarden)
