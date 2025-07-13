#!/usr/bin/env python3
"""
Compression utilities for benchmarking generated text compressibility.
"""

import abc
import gzip
import lzma
import time
import zlib
from typing import Dict, Union, Optional

# Try to import optional compression libraries
try:
    import lz4.frame
    LZ4_AVAILABLE = True
except ImportError:
    LZ4_AVAILABLE = False
    lz4 = None

try:
    import zstandard as zstd
    ZSTD_AVAILABLE = True
except ImportError:
    ZSTD_AVAILABLE = False
    zstd = None

try:
    import brotli
    BROTLI_AVAILABLE = True
except ImportError:
    BROTLI_AVAILABLE = False
    brotli = None


class Compressor(abc.ABC):
    """Abstract base class for compression algorithms."""
    
    @abc.abstractmethod
    def compress(self, data: bytes) -> bytes:
        """Compress the given data."""
        pass
    
    @abc.abstractmethod
    def get_name(self) -> str:
        """Get the name of the compression algorithm."""
        pass
    
    def compress_and_measure(self, data: bytes) -> Dict[str, Union[float, int, str, None]]:
        """
        Compress data and measure performance.
        
        Args:
            data: Bytes to compress
            
        Returns:
            Dictionary with compression metrics
        """
        original_size = len(data)
        start_time = time.time()
        
        compressed_data = self.compress(data)
        compression_time = time.time() - start_time
        compressed_size = len(compressed_data)
        compression_ratio = compressed_size / original_size if original_size > 0 else 1.0
        
        return {
            'algorithm': self.get_name(),
            'original_size': original_size,
            'compressed_size': compressed_size,
            'compression_ratio': compression_ratio,
            'compression_time': compression_time,
            'error': None
        }


class GzipCompressor(Compressor):
    """Gzip compression implementation."""
    
    def compress(self, data: bytes) -> bytes:
        return gzip.compress(data, compresslevel=6)
    
    def get_name(self) -> str:
        return 'gzip'


class ZlibCompressor(Compressor):
    """Zlib compression implementation."""
    
    def compress(self, data: bytes) -> bytes:
        return zlib.compress(data, level=6)
    
    def get_name(self) -> str:
        return 'zlib'


class LZ4Compressor(Compressor):
    """LZ4 compression implementation."""
    
    def __init__(self):
        if not LZ4_AVAILABLE or lz4 is None:
            raise ImportError("lz4 is not available. Install with: pip install lz4")
    
    def compress(self, data: bytes) -> bytes:
        return lz4.frame.compress(data, compression_level=1)
    
    def get_name(self) -> str:
        return 'lz4'


class ZstdCompressor(Compressor):
    """Zstandard compression implementation."""
    
    def __init__(self):
        if not ZSTD_AVAILABLE or zstd is None:
            raise ImportError("zstandard is not available. Install with: pip install zstandard")
        self.compressor = zstd.ZstdCompressor(level=3)
    
    def compress(self, data: bytes) -> bytes:
        return self.compressor.compress(data)
    
    def get_name(self) -> str:
        return 'zstd'


class LZMACompressor(Compressor):
    """LZMA compression implementation."""
    
    def compress(self, data: bytes) -> bytes:
        return lzma.compress(data, preset=6)
    
    def get_name(self) -> str:
        return 'lzma'


class BrotliCompressor(Compressor):
    """Brotli compression implementation."""
    
    def __init__(self):
        if not BROTLI_AVAILABLE or brotli is None:
            raise ImportError("brotli is not available. Install with: pip install brotli")
    
    def compress(self, data: bytes) -> bytes:
        return brotli.compress(data, quality=6)
    
    def get_name(self) -> str:
        return 'brotli'


def get_available_compressors() -> Dict[str, Compressor]:
    """
    Get all available compression algorithms.
    
    Returns:
        Dictionary mapping algorithm names to compressor instances
    """
    compressors = {}
    
    # Always available
    compressors['gzip'] = GzipCompressor()
    compressors['zlib'] = ZlibCompressor()
    compressors['lzma'] = LZMACompressor()
    
    # Optional - try to add if available
    try:
        compressors['lz4'] = LZ4Compressor()
    except ImportError:
        pass
    
    try:
        compressors['zstd'] = ZstdCompressor()
    except ImportError:
        pass
    
    try:
        compressors['brotli'] = BrotliCompressor()
    except ImportError:
        pass
    
    return compressors


def benchmark_compression_for_responses(
    response_ids: list[list[int]], 
    tokenizer,
    compressors: Optional[Dict[str, Compressor]] = None
) -> Dict[str, list[Dict[str, Union[float, int]]]]:
    """
    Benchmark compression for a list of response token sequences.
    
    Args:
        response_ids: List of token ID sequences
        tokenizer: Tokenizer to decode tokens to text
        compressors: Dictionary of compressors to use (defaults to all available)
        
    Returns:
        Dictionary with compression results for each response
    """
    if compressors is None:
        compressors = get_available_compressors()
    
    results = {
        'responses': [],
        'algorithms': list(compressors.keys())
    }
    
    for i, token_sequence in enumerate(response_ids):
        # Decode tokens to text
        text = tokenizer.decode(token_sequence, skip_special_tokens=True)
        text_bytes = text.encode('utf-8')
        
        # Benchmark each compression algorithm
        response_results = []
        for compressor in compressors.values():
            result = compressor.compress_and_measure(text_bytes)
            response_results.append(result)
        
        results['responses'].append({
            'response_index': i,
            'token_count': len(token_sequence),
            'text_length': len(text_bytes),
            'compression_results': response_results
        })
    
    return results