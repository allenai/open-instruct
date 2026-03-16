# Slack Alerts

Open Instruct can send Slack notifications during training. GRPO and DPO each have their own integration.

## GRPO

GRPO sends alerts on training failures and low disk space during checkpointing.

### Setup

1. Create a [Slack incoming webhook](https://api.slack.com/messaging/webhooks) for your channel.
2. Set the `SLACK_WEBHOOK` environment variable to the webhook URL.
3. Pass `--send_slack_alerts` to the training script.

```bash
export SLACK_WEBHOOK="https://hooks.slack.com/services/T.../B.../xxx"
python open_instruct/grpo_fast.py \
    --send_slack_alerts \
    ...
```

### What triggers alerts

- **Training failure**: if the training job raises an exception, posts `<!here> A RL job has died. Error message: ...` to the channel.
- **Low disk space**: before each checkpoint save, checks disk usage on `--checkpoint_state_dir`. If usage exceeds 85%, posts a warning with the percentage used and free space remaining.

If the `BEAKER_WORKLOAD_ID` environment variable is set, the Beaker experiment URL is automatically appended to every message.

## DPO

DPO uses OLMo-core's `SlackNotifierCallback`.

### Setup

Set the `SLACK_WEBHOOK_URL` environment variable (note: this is a different env var than GRPO's `SLACK_WEBHOOK`).

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/T.../B.../xxx"
python open_instruct/dpo.py \
    ...
```

No additional flags are needed — the callback is automatically registered when `SLACK_WEBHOOK_URL` is present.
