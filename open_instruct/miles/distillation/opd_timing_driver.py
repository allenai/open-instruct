"""Checked AST instrumentation of the pinned async driver's awaited stages."""

import ast
from collections import Counter


class TimedAwaits(ast.NodeTransformer):
    def __init__(self):
        self.counts = Counter()

    def visit_Await(self, node):
        expression = ast.unparse(node.value)
        stage = None
        for prefix, label in (
            ("actor_model.train(", "learner_train"),
            ("critic_model.train(", "critic_train"),
            ("update_weights(", "weight_publication"),
            ("eval_dispatcher.dispatch(", "evaluation_dispatch"),
            ("eval_dispatcher.drain(", "evaluation_drain"),
            ("model.save_model(", "checkpoint_save"),
            ("rollout_executor.save.remote(", "rollout_state_save"),
            ("rollout_executor.dispose.remote(", "shutdown"),
        ):
            if expression.startswith(prefix):
                stage = label
                break
        if expression == "rollout_data_next_future":
            stage = "learner_batch_wait"
        elif expression == "x":
            stage = "learner_prefetch_wait"
        if stage is None:
            return node
        self.counts[stage] += 1
        rollout = ast.Call(
            func=ast.Attribute(
                value=ast.Call(func=ast.Name(id="locals", ctx=ast.Load()), args=[], keywords=[]),
                attr="get",
                ctx=ast.Load(),
            ),
            args=[ast.Constant("rollout_id")],
            keywords=[],
        )
        return ast.copy_location(
            ast.Await(
                value=ast.Call(
                    func=ast.Name(id="_opd_timed_await", ctx=ast.Load()),
                    args=[node.value, ast.Constant(stage), rollout],
                    keywords=[],
                )
            ),
            node,
        )


def transform(source):
    tree = ast.parse(source)
    visitor = TimedAwaits()
    tree = visitor.visit(tree)
    for stage, expected in {
        "learner_batch_wait": 1,
        "learner_prefetch_wait": 1,
        "learner_train": 2,
        "weight_publication": 2,
        "checkpoint_save": 1,
    }.items():
        if visitor.counts[stage] != expected:
            raise ValueError(
                f"Unsupported async driver timing boundary: {stage}={visitor.counts[stage]}, expected {expected}"
            )
    return ast.fix_missing_locations(tree)
