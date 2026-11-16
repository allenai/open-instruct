"""Install the opt-in measurement call into a disposable H235 runtime image."""

import argparse
from pathlib import Path


def install(path):
    source = path.read_text()
    before = """                metrics = self.train_module.train_batch_with_loss(
                    step_batches + stopping_batches, objective, self._replay_context
                )"""
    after = """                if stopping_batches:
                    raise ValueError("Gradient-noise probe does not support stopping auxiliaries")
                metrics = gradient_noise.measure(self, step_batches, objective, self._replay_context)"""
    if source.count(before) != 1:
        raise ValueError("Pinned actor boundary changed; refusing an ambiguous runtime patch")
    source = source.replace(before, after)
    source = source.replace("import ray\n", "from scripts.miles.diagnostics import gradient_noise\n\nimport ray\n", 1)
    path.write_text(source)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("actor", type=Path)
    install(parser.parse_args().actor)


if __name__ == "__main__":
    main()
