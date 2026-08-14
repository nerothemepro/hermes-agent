# Telegram Control-Plane Router

The control-plane router is an optional, owner-only Telegram interceptor. It runs in the HerOrches gateway before pairing, plugin hooks, session creation, and LLM execution.

## Enablement

Add this to the HerOrches `config.yaml` only after the deployment PR is reviewed:

```yaml
control_plane_router:
  enabled: false
  marketing_video_ep2_enabled: false
  owner_telegram_user_env: HERMES_CONTROL_PLANE_OWNER_TELEGRAM_USER_ID
  command_timeout_seconds: 15
  project_path: /workspace/hermes-agent-plugin
  registry_dir: /opt/data/hermes/control-plane/runs
```

Set `HERMES_CONTROL_PLANE_OWNER_TELEGRAM_USER_ID` only in the HerOrches profile environment. Do not put its value in repository files, logs, evidence, or the run registry.

The configuration is snapshotted at gateway startup. Restart the HerOrches gateway after changing it.

## Rollback

Set `control_plane_router.enabled: false`, then perform the normal graceful HerOrches gateway restart. No code rollback or registry deletion is required. Existing SDTK runs remain owned by their canonical ledgers and the Phase B monitor.

## Exact Commands

The router accepts only these owner commands:

```text
/site-audit docs
/research-brief <topic>
/marketing-video ep2-usage
/status <run_id>
STATUS <run_id>
APPROVE DISPATCH <run_id>
APPROVE GATE <run_id> <gate_id>
CANCEL RUN <run_id>
```

Other senders are silently dropped before parsing. Invalid, partial, or natural-language control attempts receive a short exact-syntax refusal and cause no state change. Non-control owner messages pass through to the existing HerOrches conversation unchanged.

## Execution Contract

- Template preparation uses the Phase A `hermes-control-plane-prepare` helper.
- `/marketing-video ep2-usage` is disabled by default and, when enabled, can prepare only the fixed attended Episode 2 template. It never dispatches a worker.
- Dispatch, gate approval, and cancellation use audited `sdtk-agent` CLI invocations only.
- The router invokes commands as argv lists, with a bounded 1--30 second timeout and no automatic retry.
- Timeout or CLI errors produce a short fail-closed owner reply without exposing stderr, tokens, or command output.
- The router never polls progress, sends monitor notifications, retries tasks, archives cards, or decides approvals. The Phase B monitor remains responsible for progress observation and Telegram completion/gate notifications.
