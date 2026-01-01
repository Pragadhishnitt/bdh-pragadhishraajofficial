#!/usr/bin/env python3
"""
Quick test to verify tech filter matches dataset symbols
Run this in Kaggle to verify the filter works
"""

from datasets import load_dataset
import sys
sys.path.insert(0, 'new_dataset')

from sector_registry import get_symbols_for_sector
from data_filter import get_filter_strategy

# Get tech symbols from registry
tech_symbols = set(get_symbols_for_sector('Technology'))
print(f"=== Tech Registry ===")
print(f"Registered {len(tech_symbols)} tech symbols")
print(f"Sample: {sorted(list(tech_symbols))[:10]}")

# Load dataset sample
print(f"\n=== Dataset Test ===")
dataset = load_dataset('kurry/sp500_earnings_transcripts', split='train[:1000]')

# Test filter
tech_filter = get_filter_strategy('technology')
matches = 0
total = 0

for item in dataset:
    total += 1
    if tech_filter.should_include(item):
        matches += 1

print(f"Total items: {total}")
print(f"Tech matches: {matches}")
print(f"Match rate: {matches/total*100:.1f}%")

if matches == 0:
    print("\n❌ NO MATCHES - Filter not working!")
    # Debug: show what symbols are in dataset
    dataset_symbols = set()
    for item in dataset:
        if 'symbol' in item:
            dataset_symbols.add(item['symbol'])
    
    print(f"\nDataset has {len(dataset_symbols)} unique symbols")
    print(f"Registry has {len(tech_symbols)} tech symbols")
    
    # Find overlap
    overlap = tech_symbols & dataset_symbols
    print(f"Overlap: {len(overlap)} symbols")
    if overlap:
        print(f"Sample overlap: {sorted(list(overlap))[:10]}")
else:
    print(f"\n✓ Filter working! Found {matches} tech transcripts")
