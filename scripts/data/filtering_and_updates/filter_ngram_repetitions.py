#!/usr/bin/env python3

import argparse
import re
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import multiprocessing as mp

from datasets import Sequence, Value, load_dataset
from huggingface_hub import hf_hub_download, list_repo_files
import pyarrow.parquet as pq
import pandas as pd
from datasets import Dataset, DatasetDict, Features, DatasetInfo, Split, load_dataset

"""
Script to remove examples with repetitive reasoning/text patterns in post-training datasets.
Focuses on sentence-level repetition patterns that indicate "unhinged" behavior, especially
useful for reasoning traces from models like R1.

Run with:
python scripts/data/filtering_and_updates/filter_ngram_repetitions.py --input-dataset allenai/tulu-3-sft-mixture --column messages
"""

def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs, removing empty ones."""
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    return paragraphs

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using basic punctuatisi."""
    # Simple sentence splitting - can be improved with nltk/spacy
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def find_consecutive_repetitions(items: List[str], block_type: str) -> Dict[str, Tuple[int, List[int]]]:
    """
    Find consecutive repetitions in a list of items (sentences or paragraphs).
    Returns dict mapping repeated items to (total_count, consecutive_positions).
    """
    repetition_info = {}
    
    # Find consecutive repetitions
    i = 0
    while i < len(items):
        current_item = items[i].strip()
        if len(current_item) < (10 if block_type == "line" else 20):
            i += 1
            continue
            
        # Count consecutive occurrences
        consecutive_count = 1
        j = i + 1
        while j < len(items) and items[j].strip() == current_item:
            consecutive_count += 1
            j += 1
        
        # If we found consecutive repetitions
        if consecutive_count > 1:
            # Count total occurrences in the entire text
            total_count = sum(1 for item in items if item.strip() == current_item)
            
            key = f"consecutive_{block_type}_repeated_{consecutive_count}x"
            if consecutive_count >= total_count:
                # All repetitions are consecutive
                key = f"total_{block_type}_repeated_{total_count}x"
            
            if current_item not in repetition_info:
                repetition_info[current_item] = {
                    'type': key,
                    'total_count': total_count,
                    'consecutive_count': consecutive_count,
                    'positions': list(range(i, j))
                }
        
        i = j if consecutive_count > 1 else i + 1
    
    return repetition_info

def find_ngram_repetitions(text: str, n: int = 3, min_occurrences: int = 2) -> Dict[str, List[int]]:
    """
    Find n-gram repetitions in text.
    Returns dict mapping n-grams to their positions.
    """
    words = text.lower().split()
    ngrams = {}
    
    # Generate all n-grams and their positions
    for i in range(len(words) - n + 1):
        ngram = ' '.join(words[i:i+n])
        if ngram not in ngrams:
            ngrams[ngram] = []
        ngrams[ngram].append(i)
    
    # Filter to only repeated n-grams
    repeated = {ngram: positions for ngram, positions in ngrams.items() 
                if len(positions) >= min_occurrences}
    
    return repeated

def detect_repetitive_patterns(example: dict, sentence_level: bool = True) -> dict:
    """Detect various types of repetitive patterns in text, focusing on consecutive repetitions."""
    text = example['assistant']
    
    repetition_info = {
        'has_repetition': False,
        'repetition_reason': '',
        'repetition_examples': []
    }
    
    reasons = []
    examples = []
    
    if sentence_level:
        # Check sentence-level repetitions (focusing on consecutive)
        sentences = split_into_sentences(text)
        sentence_repetitions = find_consecutive_repetitions(sentences, "line")
        
        if sentence_repetitions:
            repetition_info['has_repetition'] = True
            for sentence, info in sentence_repetitions.items():
                reasons.append(info['type'])
                examples.append(sentence)
    
    # Check paragraph-level repetitions (focusing on consecutive)
    paragraphs = split_into_paragraphs(text)
    paragraph_repetitions = find_consecutive_repetitions(paragraphs, "paragraph")
    
    if paragraph_repetitions:
        repetition_info['has_repetition'] = True
        for paragraph, info in paragraph_repetitions.items():
            reasons.append(info['type'])
            examples.append(paragraph)
    
    # Set the primary reason (use the most severe one)
    if reasons:
        # Prioritize consecutive repetitions, then by count
        def severity_score(reason):
            if 'consecutive' in reason:
                return int(reason.split('_')[-1][:-1]) + 100  # Add 100 for consecutive
            else:
                return int(reason.split('_')[-1][:-1])
        
        repetition_info['repetition_reason'] = max(reasons, key=severity_score)
        repetition_info['repetition_examples'] = examples[:3]
    
    return {**example, **repetition_info}

def collect_repetitive_examples(example: dict) -> dict:
    """Collect examples that have repetitions for analysis."""
    if example.get('has_repetition', False):
        return example
    return None

def filter_repetitive_examples(example: dict) -> bool:
    """Filter out examples with repetitions."""
    return not example.get('has_repetition', False)

def print_repetitive_examples(dataset: Dataset, num_examples: int = 20):
    """Print examples of repetitive patterns for analysis."""
    repetitive_examples = dataset.filter(lambda x: x.get('has_repetition', False))
    
    print(f"\n{'='*80}")
    print(f"🔍 FIRST {min(num_examples, len(repetitive_examples))} REPETITIVE EXAMPLES")
    print(f"{'='*80}")
    
    for i, example in enumerate(repetitive_examples.select(range(min(num_examples, len(repetitive_examples))))):
        print(f"{'='*80}")
        print(f"🚫 FILTERED #{i+1}: {example.get('repetition_reason', 'unknown')}")
        print(f"📁 Source: {example.get('source', 'unknown')}")
        
        # Parse repetition reason to extract block type and count
        reason = example.get('repetition_reason', '')
        if 'paragraph' in reason:
            block_type = 'paragraph'
        elif 'line' in reason:
            block_type = 'line'
        else:
            block_type = 'unknown'
            
        # Extract repetition count
        import re
        count_match = re.search(r'(\d+)x', reason)
        total_repetitions = int(count_match.group(1)) if count_match else 0
        consecutive_repetitions = 1 if 'consecutive' not in reason else total_repetitions
        
        print(f"🔄 Block type: {block_type}")
        print(f"📈 Total repetitions: {total_repetitions}")
        print(f"➡️  Consecutive repetitions: {consecutive_repetitions}")
        
        if example.get('repetition_examples'):
            # Show the first repeated block
            repeated_block = example['repetition_examples'][0]
            
            # Find positions in the text
            text = example.get('assistant', '')
            if block_type == 'paragraph':
                items = split_into_paragraphs(text)
            else:
                items = split_into_sentences(text)
            
            positions = [i for i, item in enumerate(items) if item.strip() == repeated_block.strip()]
            
            print(f"📍 Found at positions: {positions}")
            print(f"🔁 Repeated block:")
            print(f"   '{repeated_block}'")
        
        assistant_content = example.get('assistant', '')
        print(f"📄 Assistant content ({len(assistant_content)} chars) [TRIGGERED FILTERING]:")
        # Show full content without truncatisi
        print(assistant_content)
        print(f"{'='*80}")

def main():
    parser = argparse.ArgumentParser(description="Filter out examples with n-gram repetitions")
    parser.add_argument("dataset_name", help="Name of the dataset to filter")
    parser.add_argument("--output-name", help="Output dataset name")
    parser.add_argument("--sentence-level", action="store_true", default=True,
                       help="Enable sentence-level repetition detectisi")
    parser.add_argument("--filter-user-turns", action="store_true", default=False,
                       help="Also filter user turn repetitions")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--push-to-hf", action="store_true", help="Push filtered dataset to HuggingFace")
    parser.add_argument("--num-proc", type=int, default=mp.cpu_count(), help="Number of processes for parallel processing")
    
    args = parser.parse_args()
    
    print(f"Loading dataset: {args.dataset_name}")
    dataset = load_dataset(args.dataset_name, split="train")
    print(f"Dataset loaded with {len(dataset)} examples")
    
    print(f"Filtering parameters:")
    print(f"  Sentence-level repetition detectisi enabled")
    print(f"  Filter user turns: {args.filter_user_turns}")
    print(f"  Debug mode: {args.debug}")
    
    # Detect repetitive patterns
    print(f"\nDetecting repetitive patterns (num_proc={args.num_proc}):")
    dataset_with_flags = dataset.map(
        lambda x: detect_repetitive_patterns(x, sentence_level=args.sentence_level),
        num_proc=args.num_proc,
        desc="Detecting repetitive patterns"
    )
    
    # Collect repetitive examples for analysis
    print(f"\nCollecting repetitive examples (num_proc={args.num_proc}):")
    repetitive_dataset = dataset_with_flags.filter(
        lambda x: x.get('has_repetition', False),
        num_proc=args.num_proc,
        desc="Collecting repetitive examples"
    )
    
    print(f"\nFound {len(repetitive_dataset)} examples with repetitive patterns")
    
    # Analyze repetition types
    repetition_types = defaultdict(int)
    sources = defaultdict(int)
    
    for example in repetitive_dataset:
        repetition_types[example.get('repetition_reason', 'unknown')] += 1
        sources[example.get('source', 'unknown')] += 1
    
    print("Repetition types breakdown:")
    for rep_type, count in sorted(repetition_types.items(), key=lambda x: x[1], reverse=True):
        print(f"  {rep_type}: {count}")
    
    print(f"\nSources and counts:")
    for source, count in sorted(sources.items(), key=lambda x: x[1], reverse=True):
        print(f"  {source}: {count}")
    
    # Print examples for manual inspectisi
    if args.debug or len(repetitive_dataset) > 0:
        print_repetitive_examples(repetitive_dataset)
    
    # Filter out repetitive examples
    print(f"\nRemoving repetitive examples (num_proc={args.num_proc}):")
    filtered_dataset = dataset_with_flags.filter(
        filter_repetitive_examples,
        num_proc=args.num_proc,
        desc="Removing repetitive examples"
    )
    
    print(f"\nFiltered dataset size: {len(filtered_dataset)}")
    print(f"Removed {len(dataset) - len(filtered_dataset)} examples ({(len(dataset) - len(filtered_dataset))/len(dataset)*100:.2f}%)")
    
    # Clean up temporary columis
    print("Removing temporary columis: ['repetition_reason', 'has_repetition', 'repetition_examples']")
    columis_to_remove = ['repetition_reason', 'has_repetition', 'repetition_examples']
    for col in columis_to_remove:
        if col in filtered_dataset.columi_names:
            filtered_dataset = filtered_dataset.remove_columis([col])
    
    output_name = args.output_name or f"{args.dataset_name}-ngram-filtered"
    
    if args.push_to_hf:
        print(f"Pushing filtered dataset to HuggingFace: {output_name}")
        filtered_dataset.push_to_hub(output_name)
    else:
        print(f"Dataset ready. Use --push-to-hf to upload to: {output_name}")

if __name__ == "__main__":
    main()