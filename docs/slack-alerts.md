# Slack Alerts

GRPO and DPO can send messages to Slack channels during training. To do so, set `SLACK_WEBHOOK_URL` to a webhook URL for the appropriate channel. 

If you're at Ai2, we have a bunch of [existing webhooks](https://api.slack.com/apps/A085QT5C5P0/incoming-webhooks?), and you can also [create a new one](https://api.slack.com/messaging/webhooks) if none of the existing ones suit you.

Once you've set the `SLACK_WEBHOOK_URL` env variable, set the `--send_slack_alerts` flag on your experiment and it'll start alerting you on Slack. 

### Beaker (Ai2 internal)

(This is only relevant if you work at Ai2.)

When running on Beaker, `mason.py` auto-injects the `SLACK_WEBHOOK_URL` Beaker secret as an environment variable (just like `HF_TOKEN` or `WANDB_API_KEY`). Create a per-user Beaker secret named `{username}_SLACK_WEBHOOK_URL` and it will be available automatically:

```bash
beaker secret write -w ai2/your-workspace {username}_SLACK_WEBHOOK_URL https://hooks.slack.com/services/T.../B.../xxx
```
