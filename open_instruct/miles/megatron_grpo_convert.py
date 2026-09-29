"""Preserve TP-only conversion when the native converter auto-selects PP."""

import argparse
import ast
import hashlib
import sys
from pathlib import Path

from open_instruct import logger_utils

logger = logger_utils.setup_logger(__name__)


def converter_tree(source, filename):
    """Guard the one native auto-PP branch; reject an unfamiliar converter."""
    tree = ast.parse(source, filename=filename)
    expected = ast.dump(ast.parse("args.pipeline_model_parallel_size == 1 and world_size > 1", mode="eval").body)
    matches = []
    for function in tree.body:
        if isinstance(function, ast.FunctionDef) and function.name == "get_args":
            for node in ast.walk(function):
                if isinstance(node, ast.If) and ast.dump(node.test) == expected:
                    matches.append(node)
    if len(matches) != 1:
        raise ValueError("Native converter auto-PP guard changed; inspect the pinned source before conversion")
    node = matches[0]
    node.test = ast.BoolOp(
        op=ast.And(), values=[node.test, ast.parse("args.tensor_model_parallel_size == 1", mode="eval").body]
    )
    return ast.fix_missing_locations(tree)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--native-script", required=True)
    options, arguments = parser.parse_known_args()
    path = Path(options.native_script)
    source = path.read_text()
    tree = converter_tree(source, str(path))
    logger.info(
        "Native converter source SHA256=%s; auto-PP inference restricted to TP1",
        hashlib.sha256(source.encode()).hexdigest(),
    )
    sys.argv = [str(path), *arguments]
    sys.path.insert(0, str(path.parent))
    exec(compile(tree, str(path), "exec"), {"__name__": "__main__", "__file__": str(path)})


if __name__ == "__main__":
    main()
