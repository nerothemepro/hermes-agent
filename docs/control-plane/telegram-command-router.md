# Telegram Control-Plane Router

The control-plane router is an optional, owner-only Telegram interceptor. It runs in the HerOrches gateway before pairing, plugin hooks, session creation, and LLM execution.

## Enablement

Add this to the HerOrches `config.yaml` only after the deployment PR is reviewed:

```yaml
control_plane_router:
  enabled: false
  marketing_video_ep2_enabled: false
  owner_telegram_user_env: HERMES_CONTROL_PLANE_OWNER_TELEGRAM_USER_ID
  command_timeout_seconds: 30
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
/marketing-video prepare EP2|EP3|EP4
/marketing-video status <run_id>
APPROVE VIDEO KICKOFF <run_id> <manifest_sha256>
APPROVE VIDEO GATE <run_id> story_lock|picture_lock|publish <packet_sha256>
REJECT VIDEO GATE <run_id> story_lock|picture_lock|publish <REASON_CODE>
CANCEL VIDEO RUN <run_id>
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
- `marketing_video_self_service_enabled` is disabled by default. When enabled, the bounded EP2/EP3/EP4 prepare, kickoff, gate approve/reject, and cancel commands delegate to the manifest-driven controller. Publication remains outside the router.
- `/marketing-video ep2-usage` remains a deprecated compatibility alias; it never dispatches a worker.
- Dispatch, gate approval, and cancellation use audited `sdtk-agent` CLI invocations only.
- The router invokes commands as argv lists, with a bounded 1--30 second timeout and no automatic retry.
- Timeout or CLI errors produce a short fail-closed owner reply without exposing stderr, tokens, or command output. If a dispatch timeout occurs after durable external-task submission, the router reads the ledger once and reports that dispatch started; it never retries the command.
- The router never polls progress, sends monitor notifications, retries tasks, archives cards, or decides approvals. The Phase B monitor remains responsible for progress observation and Telegram completion/gate notifications.
