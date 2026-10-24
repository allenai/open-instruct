"""Add diagnostics to the routing-logger-fixed pinned trainer; no update edits."""

import hashlib
from pathlib import Path

SOURCE_PATH = Path("/deps/src/maxtext/trainers/post_train/sft/train_sft.py")
SOURCE_SHA = "3343f418c06b88728dd5d1b31bc98d8c4f1707c724d05eeaf3167133f404dfa4"
REPLACEMENTS = (
    ("import jax\n", "import jax\nimport update_audit\n"),
    (
        "    # The scalars are replicated, so this host's copy is the global value.\n",
        '    if "h127_update_audit" in aux:\n'
        '      update_audit.emit(self._train_steps + 1, aux["h127_update_audit"])\n'
        "    # The scalars are replicated, so this host's copy is the global value.\n",
    ),
    (
        "      (out_val, (aux, new_rest)), grads = grad_fn(diff_params, rest, **inputs)\n",
        "      (out_val, (aux, new_rest)), grads = grad_fn(diff_params, rest, **inputs)\n"
        "      h127_audit = update_audit.before_update(diff_params, grads)\n",
    ),
    (
        "      aux_out = aux if has_aux else None\n      return out_val, aux_out, grad_norm\n",
        "      assert has_aux and isinstance(aux, dict)\n"
        '      h127_audit["rows_after"] = update_audit.selected_rows(nnx.state(model, wrt))\n'
        "      aux_out = dict(aux, h127_update_audit=h127_audit)\n"
        "      return out_val, aux_out, grad_norm\n",
    ),
)


def patch(source):
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA, "Unexpected runtime source"
    text = source.decode()
    for old, new in REPLACEMENTS:
        assert text.count(old) == 1, f"Patch target count: {old!r}"
        text = text.replace(old, new)
    compile(text, str(SOURCE_PATH), "exec")
    return text.encode()


def main():
    source = SOURCE_PATH.read_bytes()
    result = patch(source)
    SOURCE_PATH.write_bytes(result)
    print("H127-UPDATE-AUDIT-PATCH", hashlib.sha256(result).hexdigest(), flush=True)


if __name__ == "__main__":
    main()
