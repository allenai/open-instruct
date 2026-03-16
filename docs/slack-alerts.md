# Slack Alerts

Open Instruct can send Slack notifications during training. Both GRPO and DPO use the `SLACK_WEBHOOK_URL` environment variable.

Set `SLACK_WEBHOOK_URL` to a webhook URL for the appropriate channel. We have a bunch of [existing webhooks](https://api.slack.com/apps?new_app=1), and you can also [create a new one](https://api.slack.com/messaging/webhooks) if none of the existing ones suit you.

When running on Beaker, `mason.py` auto-injects the `SLACK_WEBHOOK_URL` Beaker secret as an environment variable (just like `HF_TOKEN` or `WANDB_API_KEY`). Create a Beaker secret named `SLACK_WEBHOOK_URL` (or `{username}_SLACK_WEBHOOK_URL`) and it will be available automatically.

## GRPO

GRPO sends alerts on training failures and low disk space during checkpointing.

Pass `--send_slack_alerts` to enable alerts:

```bash
python open_instruct/grpo_fast.py \
    --send_slack_alerts \
    ...
```

### What triggers alerts

- **Training failure**: if the training job raises an exception, posts `<!here> A RL job has died. Error message: ...` to the channel.
- **Low disk space**: before each checkpoint save, checks disk usage on `--checkpoint_state_dir`. If usage exceeds 85%, posts a warning with the percentage used and free space remaining.

If the `BEAKER_WORKLOAD_ID` environment variable is set, the Beaker experiment URL is automatically appended to every message.

## DPO

DPO uses OLMo-core's `SlackNotifierCallback`. Pass `--send_slack_alerts` to enable it:

```bash
python open_instruct/dpo.py \
    --send_slack_alerts \
    ...
```
