"""Candidate numerical repair: cast gathered rows, not the whole master table.

Forward values/dtype are unchanged. With FP32 master parameters, gather's
backward scatter accumulates repeated-token cotangents in FP32. The iota
embedding path is preserved. Not authorized as qualified production by itself.
"""

import hashlib
from pathlib import Path

SOURCE_PATH = Path("/deps/src/maxtext/layers/embeddings.py")
SOURCE_SHA = "0f703ae4d338be52cd9015268306e0e6b4eeae1eaeb2ad1e58fbf6ef6008f00d"
REPLACEMENTS = (
    (
        "    embedding = jnp.asarray(\n"
        "        _maybe_move_embedding_to_device(self.embedding.get_value(), self.config),\n"
        "        self.dtype,\n"
        "    )\n",
        "    embedding = _maybe_move_embedding_to_device(self.embedding.get_value(), self.config)\n",
    ),
    (
        "      output = jnp.dot(one_hot, embedding, out_sharding=out_sharding)\n",
        "      output = jnp.dot(one_hot, jnp.asarray(embedding, self.dtype), out_sharding=out_sharding)\n",
    ),
    (
        "      output = embedding.at[inputs].get(out_sharding=out_sharding)\n",
        "      output = embedding.at[inputs].get(out_sharding=out_sharding).astype(self.dtype)\n",
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
    print("H127-EMBED-FP32-GATHER-PATCH", hashlib.sha256(result).hexdigest(), flush=True)
