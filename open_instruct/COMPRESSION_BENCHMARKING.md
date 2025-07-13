# Compression Benchmarking Feature

This document describes the compression benchmarking functionality added to `benchmark_generators.py`.

## Overview

The compression benchmarking feature measures how compressible the generated tokens are by running multiple compression algorithms over the generated text and measuring both compression ratios and compression timing.

## Features

- **Multiple Compression Algorithms**: Supports gzip, zlib, and brotli (if available)
- **Per-Response Analysis**: Each generated response is individually analyzed
- **Performance Metrics**: Measures both compression ratio and compression time
- **Detailed Statistics**: Provides min, max, mean, and median compression ratios
- **Data Persistence**: Saves compression results to JSON files for further analysis

## How It Works

1. **Token Decoding**: Generated token sequences are decoded back to text using the model's tokenizer
2. **Compression Testing**: Each text response is compressed using multiple algorithms
3. **Metrics Collection**: For each algorithm, the system measures:
   - Original size (bytes)
   - Compressed size (bytes)
   - Compression ratio (compressed/original)
   - Compression time (seconds)
4. **Statistics Generation**: Aggregates results across all responses and algorithms

## Output Files

### Compression Results JSON
- **Filename**: `compression_results_{timestamp}.json`
- **Format**: JSON with detailed compression data for each batch and response
- **Structure**:
  ```json
  [
    {
      "batch_idx": 0,
      "responses": [
        {
          "response_index": 0,
          "token_count": 150,
          "text_length": 600,
          "compression_results": [
            {
              "algorithm": "gzip",
              "original_size": 600,
              "compressed_size": 180,
              "compression_ratio": 0.3,
              "compression_time": 0.001,
              "error": null
            }
          ]
        }
      ],
      "algorithms": ["gzip", "zlib", "brotli"]
    }
  ]
  ```

## Console Output

The benchmark summary now includes a "COMPRESSION STATISTICS" section showing:

```
COMPRESSION STATISTICS:

GZIP COMPRESSION:
  Average compression ratio: 0.2345 (23.45%)
  Median compression ratio: 0.2100 (21.00%)
  Min compression ratio: 0.0290 (2.90%)
  Max compression ratio: 1.3333 (133.33%)
  Average compression time: 0.123 ms
  Total samples: 100

ZLIB COMPRESSION:
  Average compression ratio: 0.1890 (18.90%)
  Median compression ratio: 0.1700 (17.00%)
  Min compression ratio: 0.0170 (1.70%)
  Max compression ratio: 1.1333 (113.33%)
  Average compression time: 0.089 ms
  Total samples: 100
```

## Dependencies

### Required
- `gzip` (built-in Python module)
- `zlib` (built-in Python module)

### Optional
- `brotli` (install with `pip install brotli`)

If brotli is not available, the system will automatically skip it and only use gzip and zlib.

## Usage

The compression benchmarking is automatically enabled when running `benchmark_generators.py`. No additional command-line arguments are required.

### Example Command
```bash
python benchmark_generators.py \
  --dataset_mixer_list "your_dataset" \
  --model_name_or_path "your_model" \
  --response_length 512 \
  --temperature 0.7
```

## Interpretation

### Compression Ratios
- **Lower ratios** (closer to 0) indicate better compression
- **Higher ratios** (closer to 1) indicate poor compression
- **Ratios > 1.0** indicate the compressed data is larger than the original

### Typical Results
- **Repetitive text**: Very low ratios (2-10%)
- **Natural language**: Moderate ratios (20-40%)
- **Random/encrypted data**: High ratios (80-120%)

### Performance Considerations
- **Compression time** is typically very fast (< 1ms per response)
- **Memory usage** is minimal as compression is done on individual responses
- **CPU impact** is negligible compared to the main generation workload

## Troubleshooting

### Brotli Not Available
If you see warnings about brotli not being available, install it with:
```bash
pip install brotli
```

### Large Output Files
If compression results files are very large, consider:
- Reducing the number of batches (`num_batches` parameter)
- Reducing the number of samples per prompt
- Processing results in smaller chunks

### Memory Issues
If you encounter memory issues with very large datasets:
- The compression is done per-response, so memory usage scales linearly
- Consider processing in smaller batches
- Monitor system memory usage during long runs