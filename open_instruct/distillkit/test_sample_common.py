import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from open_instruct.distillkit.sample_common import StreamingParquetWriter, compressed_logit_schema


class TestSampleCommon(unittest.TestCase):
    def test_streaming_parquet_writer_writes_and_rotates(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer_dir = Path(tmp) / "writer"
            with StreamingParquetWriter(
                output_path=str(writer_dir), schema=compressed_logit_schema(), file_max_rows=2, write_batch_size=1
            ) as writer:
                for i in range(3):
                    writer.write(
                        {
                            "input_ids": [i, i + 1],
                            "compressed_logprobs": [[1, 2, 3]],
                            "bytepacked_indices": [[4, 5, 6]],
                            "messages": "",
                            "text": f"sample-{i}",
                        }
                    )

            parquet_files = sorted(writer_dir.glob("*.parquet"))
            self.assertEqual(len(parquet_files), 2)
            row_counts = [pq.read_table(path).num_rows for path in parquet_files]
            self.assertEqual(sum(row_counts), 3)


if __name__ == "__main__":
    unittest.main()
