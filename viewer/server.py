from __future__ import annotations

import argparse
import json
import mimetypes
import os
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from viewer.experiment_service import ExperimentService
from viewer.registry_index import RegistryIndex
from viewer.rollout_store import RolloutStore, RolloutStoreError
from viewer.training_registry import TrainingRegistry, TrainingRegistryError
from viewer.wandb_evals import WandbEvalIndex

STATIC_DIR = Path(__file__).resolve().parent / "static"


def integer(params: dict[str, list[str]], key: str, default: int) -> int:
    value = params.get(key, [str(default)])[0]
    try:
        return int(value)
    except ValueError as error:
        raise RolloutStoreError(f"{key} must be an integer") from error


def optional_integer(params: dict[str, list[str]], key: str) -> int | None:
    if key not in params:
        return None
    return integer(params, key, 0)


class ViewerHandler(BaseHTTPRequestHandler):
    server: ViewerServer

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/health":
                self.send_json({"ok": True, "registry": self.server.service is not None})
                return
            if parsed.path == "/api/trainings":
                if self.server.service is None:
                    self.send_json({"error": "No training registry is configured"}, status=HTTPStatus.NOT_FOUND)
                else:
                    self.send_json(self.server.service.list_trainings())
                return
            if parsed.path.startswith("/api/trainings/"):
                if self.server.service is None:
                    self.send_json({"error": "No training registry is configured"}, status=HTTPStatus.NOT_FOUND)
                else:
                    suffix = parsed.path[len("/api/trainings/") :]
                    parts = suffix.split("/")
                    if len(parts) >= 3 and parts[1] == "evaluations":
                        training_id = unquote(parts[0])
                        evaluation_id = unquote(parts[2])
                        params = parse_qs(parsed.query)
                        if len(parts) == 3:
                            self.send_json(
                                self.server.service.get_evaluation_records(
                                    training_id=training_id,
                                    evaluation_id=evaluation_id,
                                    outcome=params.get("outcome", ["all"])[0],
                                    search=params.get("search", [""])[0],
                                    sort=params.get("sort", ["id"])[0],
                                    page=integer(params, "page", 1),
                                    page_size=integer(params, "page_size", 40),
                                )
                            )
                        elif len(parts) == 6 and parts[3] == "records" and parts[5] == "matches":
                            self.send_json(
                                self.server.service.search_evaluation_record(
                                    training_id=training_id,
                                    evaluation_id=evaluation_id,
                                    query_id=unquote(parts[4]),
                                    query=params.get("query", [""])[0],
                                    response_index=optional_integer(params, "response_index"),
                                )
                            )
                        elif len(parts) == 7 and parts[3] == "records" and parts[5] == "segments":
                            try:
                                segment_index = int(unquote(parts[6]))
                            except ValueError as error:
                                raise TrainingRegistryError(f"Invalid segment index: {parts[6]}") from error
                            self.send_json(
                                self.server.service.get_evaluation_segment(
                                    training_id=training_id,
                                    evaluation_id=evaluation_id,
                                    query_id=unquote(parts[4]),
                                    segment_index=segment_index,
                                    response_index=optional_integer(params, "response_index"),
                                )
                            )
                        elif len(parts) == 5 and parts[3] == "records":
                            self.send_json(
                                self.server.service.get_evaluation_record(
                                    training_id=training_id,
                                    evaluation_id=evaluation_id,
                                    query_id=unquote(parts[4]),
                                    response_index=optional_integer(params, "response_index"),
                                )
                            )
                        else:
                            self.send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
                    elif suffix.endswith("/metrics"):
                        training_id = unquote(suffix[: -len("/metrics")])
                        self.send_json(self.server.service.get_training_metrics(training_id))
                    else:
                        self.send_json(self.server.service.get_training(unquote(suffix)))
                return
            if parsed.path == "/api/path":
                if self.server.service is None:
                    self.send_json({"error": "No training registry is configured"}, status=HTTPStatus.NOT_FOUND)
                else:
                    params = parse_qs(parsed.query)
                    path = params.get("path", [""])[0]
                    self.send_json(self.server.service.path_info(path))
                return
            if parsed.path == "/api/meta":
                self.send_json(self.server.store.meta())
                return
            if parsed.path == "/api/steps":
                params = parse_qs(parsed.query)
                run = params.get("run", [""])[0]
                self.send_json(self.server.store.steps(run))
                return
            if parsed.path == "/api/rollouts":
                params = parse_qs(parsed.query)
                meta = self.server.store.meta()
                run = params.get("run", [meta.get("default_run") or ""])[0]
                source = params.get("source", ["accepted"])[0]
                # steps() resolves this run's shard boundaries on demand.
                ranges = self.server.store.steps(run)
                source_range = ranges["source_ranges"].get(source) or ranges
                default_step = source_range["last_step"]
                self.send_json(
                    self.server.store.query(
                        run=run,
                        step=integer(params, "step", default_step),
                        source=source,
                        category=params.get("category", ["review"])[0],
                        search=params.get("search", [""])[0],
                        sort=params.get("sort", ["suspicion"])[0],
                        group_key=params.get("group", [""])[0],
                        page=integer(params, "page", 1),
                        page_size=integer(params, "page_size", 24),
                    )
                )
                return
            if parsed.path == "/api/groups":
                params = parse_qs(parsed.query)
                meta = self.server.store.meta()
                run = params.get("run", [meta.get("default_run") or ""])[0]
                source = params.get("source", ["accepted"])[0]
                ranges = self.server.store.steps(run)
                source_range = ranges["source_ranges"].get(source) or ranges
                default_step = source_range["last_step"]
                self.send_json(
                    self.server.store.groups(
                        run=run,
                        step=integer(params, "step", default_step),
                        source=source,
                        category=params.get("category", ["all_groups"])[0],
                        search=params.get("search", [""])[0],
                        sort=params.get("sort", ["reward"])[0],
                        page=integer(params, "page", 1),
                        page_size=integer(params, "page_size", 24),
                    )
                )
                return
            if parsed.path.startswith("/api/rollouts/"):
                suffix = parsed.path[len("/api/rollouts/") :]
                if suffix.endswith("/trace"):
                    record_id = unquote(suffix[: -len("/trace")])
                    params = parse_qs(parsed.query)
                    self.send_json(
                        self.server.store.trace(
                            record_id, offset=integer(params, "offset", 0), limit=integer(params, "limit", 50_000)
                        )
                    )
                elif suffix.endswith("/matches"):
                    record_id = unquote(suffix[: -len("/matches")])
                    params = parse_qs(parsed.query)
                    self.send_json(
                        self.server.store.matches(
                            record_id,
                            query=params.get("query", [""])[0],
                            max_chars_per_turn=integer(params, "max_chars_per_turn", 8_000),
                        )
                    )
                elif suffix.endswith("/turns"):
                    record_id = unquote(suffix[: -len("/turns")])
                    params = parse_qs(parsed.query)
                    self.send_json(
                        self.server.store.turns(
                            record_id, max_chars_per_turn=integer(params, "max_chars_per_turn", 8_000)
                        )
                    )
                else:
                    self.send_json(self.server.store.detail(unquote(suffix)))
                return
            self.send_static(parsed.path)
        except RolloutStoreError as error:
            self.send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        except TrainingRegistryError as error:
            self.send_json({"error": str(error)}, status=HTTPStatus.NOT_FOUND)
        except BrokenPipeError:
            return
        except Exception as error:
            traceback.print_exc()
            self.send_json({"error": f"Unexpected server error: {error}"}, status=HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/api/refresh":
                if self.server.service is None:
                    self.server.store.refresh()
                    self.send_json(self.server.store.meta())
                else:
                    self.server.service.refresh_catalog()
                    self.server.service.start_validation_refresh(force=True)
                    self.send_json(self.server.service.list_trainings())
                return
            self.send_json({"error": "Not found"}, status=HTTPStatus.NOT_FOUND)
        except RolloutStoreError as error:
            self.send_json({"error": str(error)}, status=HTTPStatus.BAD_REQUEST)
        except TrainingRegistryError as error:
            self.send_json({"error": str(error)}, status=HTTPStatus.NOT_FOUND)

    def send_json(self, payload: object, status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        self.send_response(status.value)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def send_static(self, request_path: str) -> None:
        relative = "index.html" if request_path in {"", "/"} else request_path.lstrip("/")
        target = (STATIC_DIR / relative).resolve()
        if STATIC_DIR.resolve() not in target.parents and target != STATIC_DIR.resolve():
            self.send_error(HTTPStatus.NOT_FOUND.value)
            return
        if not target.is_file():
            target = STATIC_DIR / "index.html"
        body = target.read_bytes()
        content_type, _ = mimetypes.guess_type(target.name)
        self.send_response(HTTPStatus.OK.value)
        self.send_header("Content-Type", content_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:
        if self.server.verbose:
            super().log_message(format, *args)


class ViewerServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(
        self,
        address: tuple[str, int],
        store: RolloutStore,
        *,
        service: ExperimentService | None = None,
        verbose: bool = False,
    ) -> None:
        super().__init__(address, ViewerHandler)
        self.store = store
        self.service = service
        self.verbose = verbose


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Browse large rollout JSONL files without loading them in the browser"
    )
    parser.add_argument(
        "--rollouts-dir",
        default=os.environ.get("ROLLOUTS_DIR"),
        help="Directory containing rollout shards; defaults to <registry repo>/rl_rollouts",
    )
    default_registry = Path(__file__).resolve().parent / "registry"
    parser.add_argument(
        "--registry",
        default=os.environ.get("TRAINING_REGISTRY") or (str(default_registry) if default_registry.is_dir() else None),
        help="Training registry directory or config.yaml (or set TRAINING_REGISTRY)",
    )
    parser.add_argument(
        "--no-registry", action="store_true", help="Disable registry mode and inspect every discovered rollout attempt"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--response-limit", type=int, default=131_072)
    parser.add_argument("--cache-steps", type=int, default=16)
    parser.add_argument("--tokenizer", help="Tokenizer name/path used for on-demand trace decoding")
    parser.add_argument(
        "--wandb-path",
        default=os.environ.get("WANDB_VIEWER_PATH"),
        help="ENTITY/PROJECT to read validation steps from (or set WANDB_VIEWER_PATH)",
    )
    parser.add_argument(
        "--wandb-run",
        action="append",
        default=[],
        metavar="RUN_OR_DIR=WANDB_ID",
        help="Force a run or rollout directory to a W&B run id; repeatable",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.no_registry:
        args.registry = None
    if not args.rollouts_dir and not args.registry:
        parser.error("--rollouts-dir is required when no --registry is configured")
    return args


def build_eval_index(path: str | None, overrides: list[str]) -> WandbEvalIndex | None:
    if not path:
        return None
    mapping = {}
    for item in overrides:
        key, separator, value = item.partition("=")
        if not separator:
            raise SystemExit(f"--wandb-run expects RUN_OR_DIR=WANDB_ID, got {item!r}")
        mapping[key] = value
    return WandbEvalIndex(path, mapping)


def main() -> None:
    args = parse_args()
    registry = TrainingRegistry(args.registry) if args.registry else None
    registry_index = RegistryIndex(registry) if registry else None
    rollouts_dir = args.rollouts_dir or str(registry.repo_root / "rl_rollouts")
    store = RolloutStore(
        rollouts_dir,
        response_limit=args.response_limit,
        cache_steps=args.cache_steps,
        tokenizer_name=args.tokenizer,
        eval_index=registry_index or build_eval_index(args.wandb_path, args.wandb_run),
    )
    service = ExperimentService(registry, store, registry_index) if registry and registry_index else None
    server = ViewerServer((args.host, args.port), store, service=service, verbose=args.verbose)
    meta = store.meta()
    # Off the critical path so the viewer answers requests immediately.
    threading.Thread(target=store.prewarm_tokenizer, daemon=True).start()
    if service:
        service.start_validation_refresh(force=True)
    print(f"Rollout viewer: http://{args.host}:{args.port}")
    print(f"Rollout root: {meta['root']}")
    if registry:
        print(f"Training registry: {registry.path} ({len(registry.trainings)} entries)")
    print(f"Discovered {len(meta['runs'])} run(s); press Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("Stopping rollout viewer")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
