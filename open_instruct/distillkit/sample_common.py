# Portions of this file were adapted from distillkit (github.com/arcee-ai/DistillKit)
# Copyright 2025 Arcee AI. Licensed under the Apache License, Version 2.0.

import os
import queue
import shutil
import tempfile
import threading
from typing import Any

import pyarrow
import pyarrow.parquet as pq


def compressed_logit_schema() -> pyarrow.Schema:
    return pyarrow.schema(
        [
            pyarrow.field("input_ids", pyarrow.list_(pyarrow.uint64())),
            pyarrow.field(
                "compressed_logprobs", pyarrow.list_(pyarrow.list_(pyarrow.uint8()))
            ),
            pyarrow.field(
                "bytepacked_indices", pyarrow.list_(pyarrow.list_(pyarrow.uint8()))
            ),
            pyarrow.field("messages", pyarrow.string()),
            pyarrow.field("text", pyarrow.string()),
        ]
    )


class StreamingParquetWriter:
    def __init__(
        self,
        output_path: str,
        schema: pyarrow.Schema,
        file_max_rows: int,
        write_batch_size: int = 1000,
        queue_maxsize: int | None = None,
    ):
        self.output_path = output_path
        os.makedirs(self.output_path, exist_ok=True)
        self.schema = schema
        self.file_max_rows = file_max_rows
        self.write_batch_size = write_batch_size
        self.queue = queue.Queue(maxsize=queue_maxsize)
        self.shutdown_event = threading.Event()
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.file_index = 0
        self.rows_in_file = 0
        self.pq_writer = None

    def _ensure_writer_open(self) -> None:
        if self.pq_writer is None:
            path = os.path.join(self.output_path, f"data_{self.file_index}.parquet")
            self.pq_writer = pq.ParquetWriter(path, schema=self.schema)
            self.rows_in_file = 0

    def _write_batch(self, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        self._ensure_writer_open()
        columnar_data = {name: [] for name in self.schema.names}
        for row in rows:
            for name in self.schema.names:
                columnar_data[name].append(row[name])
        arrays = [
            pyarrow.array(columnar_data[name], type=self.schema.field(name).type)
            for name in self.schema.names
        ]
        table = pyarrow.Table.from_arrays(arrays, schema=self.schema)
        self.pq_writer.write_table(table)
        self.rows_in_file += len(rows)
        if self.rows_in_file >= self.file_max_rows:
            self.pq_writer.close()
            self.pq_writer = None
            self.file_index += 1

    def _writer_loop(self) -> None:
        batch = []
        while not self.shutdown_event.is_set() or not self.queue.empty():
            try:
                row = self.queue.get(timeout=0.1)
                if row is None:
                    self.queue.task_done()
                    break
                batch.append(row)
                self.queue.task_done()
                if len(batch) >= self.write_batch_size:
                    self._write_batch(batch)
                    batch = []
            except queue.Empty:
                continue
        if batch:
            self._write_batch(batch)
        if self.pq_writer is not None:
            self.pq_writer.close()
            self.pq_writer = None

    def write(self, row_data: dict[str, Any]) -> None:
        self.queue.put(row_data)

    def __enter__(self) -> "StreamingParquetWriter":
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self.thread.is_alive():
            self.queue.put(None)
            self.shutdown_event.set()
            self.thread.join()
        return False


def is_s3_path(path: str) -> bool:
    return path.startswith("s3://")


def parse_s3_path(s3_path: str) -> tuple[str, str]:
    path_without_scheme = s3_path[5:]
    parts = path_without_scheme.split("/", 1)
    return parts[0], parts[1] if len(parts) > 1 else ""


def upload_directory_to_s3(local_dir: str, s3_path: str, logger=None) -> None:
    from s3fs import S3FileSystem

    bucket, prefix = parse_s3_path(s3_path)
    fs = S3FileSystem()
    for root, _dirs, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            rel_path = os.path.relpath(local_path, local_dir)
            s3_key = os.path.join(prefix, rel_path)
            s3_full_path = os.path.join(bucket, s3_key)
            if logger:
                logger.info(f"Uploading {local_path} to s3://{s3_full_path}")
            fs.put(local_path, s3_full_path)


class OutputHandler:
    def __init__(self, output_path: str, logger=None):
        self.final_output_path = output_path
        self.logger = logger
        self.is_s3 = is_s3_path(output_path)
        self.temp_dir = None

    @property
    def local_output_path(self) -> str:
        if self.is_s3:
            if self.temp_dir is None:
                raise RuntimeError("OutputHandler must be entered before use.")
            return self.temp_dir
        return self.final_output_path

    def __enter__(self) -> "OutputHandler":
        if self.is_s3:
            self.temp_dir = tempfile.mkdtemp(prefix="open_instruct_distill_")
            if self.logger:
                self.logger.info(f"Using temporary output directory: {self.temp_dir}")
        else:
            os.makedirs(self.final_output_path, exist_ok=True)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        if self.is_s3 and self.temp_dir is not None:
            if exc_type is None:
                upload_directory_to_s3(
                    self.temp_dir, self.final_output_path, self.logger
                )
            shutil.rmtree(self.temp_dir, ignore_errors=True)
            self.temp_dir = None
        return False
