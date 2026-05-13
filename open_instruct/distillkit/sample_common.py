# Portions of this file were adapted from distillkit (github.com/arcee-ai/DistillKit)
# Copyright 2025 Arcee AI. Licensed under the Apache License, Version 2.0.

import os
import queue
import threading
from typing import Any

import pyarrow
import pyarrow.parquet as pq


def compressed_logit_schema() -> pyarrow.Schema:
    """Return parquet schema for compressed teacher-logit samples."""
    return pyarrow.schema(
        [
            pyarrow.field("input_ids", pyarrow.list_(pyarrow.uint64())),
            pyarrow.field("compressed_logprobs", pyarrow.list_(pyarrow.list_(pyarrow.uint8()))),
            pyarrow.field("bytepacked_indices", pyarrow.list_(pyarrow.list_(pyarrow.uint8()))),
            pyarrow.field("messages", pyarrow.string()),
            pyarrow.field("text", pyarrow.string()),
        ]
    )


class StreamingParquetWriter:
    """Background parquet writer with batched row buffering and file rotation."""

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
        # queue.Queue expects an integer maxsize; 0 means unbounded.
        self.queue = queue.Queue(maxsize=0 if queue_maxsize is None else queue_maxsize)
        self.shutdown_event = threading.Event()
        self.thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.file_index = 0
        self.rows_in_file = 0
        self.pq_writer = None

    def _ensure_writer_open(self) -> None:
        """Open the current parquet shard lazily when first needed."""
        if self.pq_writer is None:
            self.pq_writer = pq.ParquetWriter(
                os.path.join(self.output_path, f"data_{self.file_index}.parquet"), schema=self.schema
            )
            self.rows_in_file = 0

    def _write_batch(self, rows: list[dict[str, Any]]) -> None:
        """Write a row batch and rotate shard once file_max_rows is reached."""
        if not rows:
            return
        self._ensure_writer_open()
        columnar_data = {name: [] for name in self.schema.names}
        for row in rows:
            for name in self.schema.names:
                columnar_data[name].append(row[name])
        arrays = [pyarrow.array(columnar_data[name], type=self.schema.field(name).type) for name in self.schema.names]
        table = pyarrow.Table.from_arrays(arrays, schema=self.schema)
        self.pq_writer.write_table(table)
        self.rows_in_file += len(rows)
        if self.rows_in_file >= self.file_max_rows:
            self.pq_writer.close()
            self.pq_writer = None
            self.file_index += 1

    def _writer_loop(self) -> None:
        """Drain the queue, flush batches, and close resources on shutdown."""
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
        """Enqueue one row for asynchronous parquet writing."""
        self.queue.put(row_data)

    def __enter__(self) -> "StreamingParquetWriter":
        """Start the writer thread."""
        self.thread.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Flush and join writer thread before exiting context.

        Returning None means any exception from the with-block is propagated
        (as opposed to being suppressed by returning True)
        """
        if self.thread.is_alive():
            self.queue.put(None)
            self.shutdown_event.set()
            self.thread.join()
