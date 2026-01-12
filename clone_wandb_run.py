#!/usr/bin/env python3
"""
Clone a Weights & Biases run into another project by:
- copying config / tags / notes / name (optional)
- replaying full history (metrics)
- copying run files (optional)
- re-logging artifacts (optional, caveats)

Usage:
  python clone_wandb_run.py \
    --src ai2-llm/open_instruct_internal/25yfin0f \
    --dst-entity ai2-llm \
    --dst-project open_instruct_public \
    --copy-files \
    --copy-artifacts --artifacts-mode link
"""
from __future__ import annotations

import argparse
from typing import Any, Dict, Optional

import wandb

# This is the parser that creates the AST type expected by W&B's vendored graphql-core-1.1 printer.
from wandb.vendor.graphql_core_1_1.wandb_graphql.language.parser import parse  # type: ignore


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="ENTITY/PROJECT/RUN_ID")
    p.add_argument("--dst-entity", required=True)
    p.add_argument("--dst-project", required=True)
    p.add_argument("--dst-name", default=None)
    p.add_argument("--page-size", type=int, default=5000)
    p.add_argument("--max-pages", type=int, default=0, help="0 = no limit")
    p.add_argument("--min-step", type=int, default=0)
    return p.parse_args()


def safe_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in cfg.items() if not (isinstance(k, str) and k.startswith("_"))}


def to_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        return None


def main() -> int:
    args = parse_args()
    api = wandb.Api()
    src = api.run(args.src)

    dst = wandb.init(
        entity=args.dst_entity,
        project=args.dst_project,
        name=args.dst_name or src.name,
        config=safe_config(dict(src.config)),
        tags=list(getattr(src, "tags", []) or []),
        notes=getattr(src, "notes", None),
        reinit=True,
    )

    client = api.client

    query_ast = parse("""
      query RunHistory($entity: String!, $project: String!, $run: String!, $samples: Int!, $minStep: Int!) {
        project(name: $project, entityName: $entity) {
          run(name: $run) {
            history(samples: $samples, minStep: $minStep)
          }
        }
      }
    """)

    min_step = args.min_step
    total_rows = 0
    pages = 0
    unique_steps = set()
    last_max_step = -1

    print("Pulling raw history pages…")

    while True:
        res = client.execute(
            query_ast,
            variable_values={
                "entity": src.entity,
                "project": src.project,
                "run": src.id,  # short id like 25yfin0f
                "samples": args.page_size,
                "minStep": int(min_step),
            },
        )

        hist = res["project"]["run"]["history"]
        if not hist:
            print("No more history.")
            break

        max_step = -1
        for row in hist:
            s = to_int(row.get("_step"))
            if s is None:
                continue
            payload = {k: v for k, v in row.items()
                       if not (isinstance(k, str) and k.startswith("_"))}
            if not payload:
                continue
            wandb.log(payload, step=s)
            total_rows += 1
            unique_steps.add(s)
            if s > max_step:
                max_step = s

        pages += 1
        print(f"  page={pages} rows={total_rows} unique_steps={len(unique_steps)} max_step={max_step}")

        if max_step < 0 or max_step <= last_max_step:
            print("Stopped paging (no forward progress).")
            break

        last_max_step = max_step
        min_step = max_step + 1

        if args.max_pages and pages >= args.max_pages:
            print("Stopped due to --max-pages.")
            break

    wandb.finish()
    print(f"Done. Logged rows={total_rows}, unique_steps={len(unique_steps)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())