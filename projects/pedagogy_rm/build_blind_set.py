"""Mix several policies' turns into one pool nobody can tell apart.

    python projects/pedagogy_rm/build_blind_set.py \
        --arm base=data/samples/base.json \
        --arm lora=data/samples/arm_a.json \
        --arm lora_z=data/samples/arm_b.json \
        --out-dir data/eval

WHY BLINDING IS THE WHOLE POINT AND NOT A FORMALITY. The question is whether a policy
trained against the probe is better or only scores better, and the person answering it
already has a belief about that. A pool that says `style: lora` on every trained turn
measures the belief. sample_policy.py stamps the arm into each unit deliberately - the
per-arm file is a record and should say what produced it - so the stripping happens here,
at the moment the turns stop being records and become questions.

WHAT IS WITHHELD: the arm, the sampling temperature, the sample index, and the probe's own
scores. What is kept is exactly what a labeller needs - question, choices, gold, the
student turn before, and the tutor turn - in the same shape as data/label_slices/*.json,
so label_agents.py, label_ui.py and agreement.py all read this pool unchanged.

PAIRED BY MOMENT, WHICH IS WHERE THE STATISTICAL POWER COMES FROM. Each arm answers the
same prompts, so every turn has siblings, and the comparison is within a prompt rather than
across the eval set. Question difficulty and the student's state are held fixed, and they
are the largest source of variance in an absolute rating. The pairs file exists for the
same reason in the preference task.

THE KEY IS A SEPARATE FILE, and it carries the probe's scores as well as the arm, so that
scoring the labels later is a join rather than a re-run of the encoder.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import random

# Fields that would tell a labeller which policy wrote the turn, or what the probe thought
# of it. `probe` is as disqualifying as `style`: a labeller who sees 1.9 has been anchored.
WITHHELD = ("style", "temperature", "sample_index", "probe", "policy", "adapter", "tag")


def moment(unit: dict) -> str:
    """What makes two turns comparable: the same question and the same student turn.

    Hashed rather than used raw so the grouping key is short in the key file, and derived
    from content rather than from any id, because ids are per-arm and a turn's siblings
    live in other files.
    """
    material = (unit.get("question") or "") + "\u241f" + (unit.get("student_before") or "")
    return hashlib.blake2b(material.encode(), digest_size=8).hexdigest()


def blind_id(arm: str, unit_id: str, salt: str) -> str:
    """An id that cannot be reversed into an arm by looking at it.

    Salted, because the arms are few and short: without a salt anyone with the script could
    hash 'base' against a known unit id and recover the mapping from the pool alone.
    """
    return "b" + hashlib.blake2b(f"{salt}{arm}{unit_id}".encode(), digest_size=7).hexdigest()


def load_arm(spec: str) -> tuple[str, list[dict]]:
    if "=" not in spec:
        raise SystemExit(f"--arm wants name=path, got {spec!r}")
    name, path = spec.split("=", 1)
    with open(path) as handle:
        blob = json.load(handle)
    return name, blob["units"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--arm", action="append", required=True, help="name=path to a sample_policy.py file")
    parser.add_argument("--out-dir", default="data/eval")
    parser.add_argument("--moments", type=int, default=80, help="how many prompts to keep, 0 for all")
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    salt = hashlib.blake2b(str(args.seed).encode(), digest_size=8).hexdigest()

    arms = dict(load_arm(spec) for spec in args.arm)
    if len(arms) < 2:
        raise SystemExit("blinding one arm against nothing is just relabelling; pass at least two")

    # One turn per arm per moment. sample_policy.py can draw several samples per prompt, and
    # keeping them all would weight a moment by how many times it happened to be sampled.
    by_moment: dict[str, dict[str, dict]] = {}
    for arm, units in arms.items():
        for unit in units:
            by_moment.setdefault(moment(unit), {}).setdefault(arm, unit)

    complete = sorted(key for key, got in by_moment.items() if len(got) == len(arms))
    dropped = len(by_moment) - len(complete)
    rng.shuffle(complete)
    if args.moments:
        complete = complete[: args.moments]

    pool, key, pairs = [], {}, []
    for m in complete:
        for arm, unit in by_moment[m].items():
            bid = blind_id(arm, unit["id"], salt)
            pool.append({k: v for k, v in unit.items() if k not in WITHHELD} | {"id": bid, "moment": m})
            key[bid] = {"arm": arm, "source_id": unit["id"], "moment": m, "probe": unit.get("probe")}
        # Every unordered pair of arms at this moment, with the sides drawn rather than
        # ordered: a labeller who learns that the left pane is always the base policy is
        # no longer blind, and arms would otherwise appear in dict order every time.
        ids = {arm: blind_id(arm, unit["id"], salt) for arm, unit in by_moment[m].items()}
        for one, two in itertools.combinations(sorted(ids), 2):
            left, right = (ids[one], ids[two]) if rng.random() < 0.5 else (ids[two], ids[one])
            pairs.append({"id": f"p{len(pairs):04d}", "moment": m, "left": left, "right": right})

    rng.shuffle(pool)
    rng.shuffle(pairs)

    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "pool.json"), "w") as handle:
        json.dump({"units": pool}, handle, indent=1)
    with open(os.path.join(args.out_dir, "pairs.json"), "w") as handle:
        json.dump({"pairs": pairs}, handle, indent=1)
    with open(os.path.join(args.out_dir, "key.json"), "w") as handle:
        json.dump({"salt": salt, "arms": sorted(arms), "key": key}, handle, indent=1)

    print(f"{len(complete)} moments x {len(arms)} arms -> {len(pool)} blinded turns, {len(pairs)} pairs")
    if dropped:
        print(f"  dropped {dropped} moments that not every arm answered")
    print(f"  pool  {args.out_dir}/pool.json   <- label this, and let the agents label it")
    print(f"  pairs {args.out_dir}/pairs.json  <- preference task over the same turns")
    print(f"  key   {args.out_dir}/key.json    <- do not open until the labels are in")


if __name__ == "__main__":
    main()
