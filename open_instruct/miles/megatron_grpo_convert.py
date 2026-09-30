"""Check the pinned native switch that preserves PP1 during TP conversion."""

import ast
import hashlib


def verify_keep_pp1(source, filename):
    """Fail closed if the native converter does not honor CONVERT_KEEP_PP1."""
    tree = ast.parse(source, filename=filename)
    expected = ast.dump(
        ast.parse(
            'args.pipeline_model_parallel_size == 1 and world_size > 1 and not os.environ.get("CONVERT_KEEP_PP1")',
            mode="eval",
        ).body
    )
    matches = [
        node
        for function in tree.body
        if isinstance(function, ast.FunctionDef) and function.name == "get_args"
        for node in ast.walk(function)
        if isinstance(node, ast.If) and ast.dump(node.test) == expected
    ]
    if len(matches) != 1:
        raise ValueError("Native converter CONVERT_KEEP_PP1 guard differs; inspect the image source before conversion")
    return hashlib.sha256(source.encode()).hexdigest()
