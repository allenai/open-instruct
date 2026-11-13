# Nemotron abstention QA

The Nemotron-RL-QA-Abstention-v1 dataset supplies factoid questions and factual references while asking the policy to abstain when uncertain. The `nemo_abstention_v1` source adapter preserves its original system/user messages and emits a structured question/reference target. Training needs a named correctness judge; plain answer matching does not reproduce the abstention objective.

The `nvidia/qa-abstention-v1` rubric ports NVIDIA-NeMo/Gym's abstention environment at `cf7f71a57dde0a69b14a2e919b60a01b17cc7abd`. It strips thinking traces, extracts the last balanced boxed answer (including nested braces), and falls back to the remaining output when no complete box exists. Normalized `[IDK]` receives 0.5 without contacting the judge. Other outputs use the original OMNISCIENCE_GRADER prompt and A/B/C parser: correct receives 1, incorrect 0, and NOT_ATTEMPTED 0.5. This version fixes the abstention reward at NVIDIA's default 0.5; it does not implement dynamic calibration.

```toml
[judges.qa]
mode = "external"
model = "YOUR_REVIEWED_JUDGE_MODEL"
endpoint = "http://YOUR_JUDGE_HOST:8000/v1"

[rubrics.abstention]
profile = "nvidia/qa-abstention-v1"
max_response_tokens = 64
temperature = 0.0

[judging.bindings.abstention]
judge = "qa"
rubric = "abstention"
```

Replace the verifier binding name with the actual name in the exported `verifiers.json`; Workbench namespaces names by their contract. Keep the profile paired with its structured `nemo_abstention_v1` target. Preparation emits a `NamedJudgeVerifier` registry entry and preserves the full input as `metadata.judge_query`.

The named-service probe requires correct/incorrect/explicit-abstention/implicit-abstention controls to score 1/0/0.5/0.5. Empty replies, transport errors and incomplete judge generations follow the existing failure policy: strict mode raises; zero mode records `status=judge_error` and a fallback zero, which the training bridge excludes as ungraded. Evaluation can continue with those explicit failure diagnostics. Diagnostics distinguish an explicit abstention from a judge-produced NOT_ATTEMPTED grade. Existing truncation/final-answer gates apply before judging; a capped response does not earn abstention credit.

This is the native permissive parsing contract, not strict boxed-format enforcement. In particular, nonempty judge output uses NVIDIA's substring fallback for A/B/C and otherwise defaults to incorrect. Preserve that fact in calibration; tightening the parser would be a separate contract change. The rubric also contains permissive numerical and name-equivalence examples. Test them with the selected judge before relying on training rewards.

A source-preparation or fixture test does not calibrate an actual judge or establish learning benefit. Qualify the intended model and endpoint with independently reviewed correct, wrong, equivalent, incomplete and abstaining responses across all four domains. The publisher provides no validation split. Preserve the card/row license discrepancy (CC BY 4.0 versus CC BY-SA 4.0), and run the intended release's protected-source hygiene before redistribution or training admission.

Upstream implementation: <https://github.com/NVIDIA-NeMo/Gym/blob/cf7f71a57dde0a69b14a2e919b60a01b17cc7abd/resources_servers/abstention/app.py>.
