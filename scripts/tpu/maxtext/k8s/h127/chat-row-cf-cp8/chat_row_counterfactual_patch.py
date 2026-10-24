"""Add observational forwards after gradients, without changing the optimizer."""

import hashlib
from pathlib import Path

SOURCE_PATH = Path("/deps/src/maxtext/trainers/post_train/sft/train_sft.py")
SOURCE_SHA = "2d4853c7bcb0198d1e90a5394fc55fa09506f8fe143ca7d539ab6cfcb5735c8c"
REPLACEMENTS = (
    ("import update_audit\n", "import update_audit\nimport chat_row_counterfactual\n"),
    (
        "    loss_fn_ref = self.loss_fn\n",
        "    loss_fn_ref = self.loss_fn\n"
        "    h127_chat_donor = chat_row_counterfactual.load_donor(chat_row_counterfactual.DONOR_PATH)\n",
    ),
    (
        "      h127_audit = update_audit.before_update(diff_params, grads)\n",
        "      h127_audit = update_audit.before_update(diff_params, grads)\n"
        "      h127_cf = chat_row_counterfactual.measure(loss_wrapper, diff_params, rest, inputs, optimizer.step[...], h127_chat_donor)\n"
        "      aux = dict(aux, h127_chat_counterfactual=h127_cf)\n",
    ),
    (
        '      update_audit.emit(self._train_steps + 1, aux["h127_update_audit"])\n',
        '      update_audit.emit(self._train_steps + 1, aux["h127_update_audit"])\n'
        '    if "h127_chat_counterfactual" in aux:\n'
        '      chat_row_counterfactual.emit(self._train_steps + 1, aux["h127_chat_counterfactual"])\n',
    ),
)


def patch(source):
    assert hashlib.sha256(source).hexdigest() == SOURCE_SHA
    text = source.decode()
    for old, new in REPLACEMENTS:
        assert text.count(old) == 1, old
        text = text.replace(old, new)
    compile(text, str(SOURCE_PATH), "exec")
    return text.encode()


if __name__ == "__main__":
    result = patch(SOURCE_PATH.read_bytes())
    SOURCE_PATH.write_bytes(result)
    print("H127-CHAT-ROW-CF-PATCH", hashlib.sha256(result).hexdigest(), flush=True)
