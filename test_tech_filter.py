#!/usr/bin/env python3
"""
Simple test to check if tech symbols match dataset
Run in Kaggle to verify filter will work
"""

from datasets import load_dataset

# Tech symbols from our registry (copy of what's in sector_registry.py)
TECH_SYMBOLS = {
    "AAPL", "MSFT", "GOOGL", "GOOG", "META", "AMZN", "NVDA", "TSLA",
    "AMD", "INTC", "AVGO", "QCOM", "TXN", "MU", "AMAT", "LRCX", "KLAC",
    "ADI", "NXPI", "ON", "MCHP", "SWKS", "QRVO", "MPWR", "SNPS", "CDNS",
    "CRM", "ORCL", "ADBE", "NOW", "INTU", "WDAY", "PANW", "CRWD", "FTNT",
    "ANSS", "ADSK", "PTC", "TYL", "PAYC", "IBM", "ACN", "CSCO", "CTSH",
    "IT", "EPAM", "GDDY", "LDOS", "HPQ", "HPE", "DELL", "WDC", "STX",
    "NTAP", "ANET", "JNPR", "MSI", "ZBRA", "KEYS", "TER", "SMCI",
    "V", "MA", "PYPL", "FIS", "FI", "GPN", "ADP", "PAYX",
    "NFLX", "BKNG", "UBER", "DASH", "ETSY", "EBAY", "TRIP", "TTWO", "EA", "ATVI",
    "DLR", "EQIX", "AMT", "CCI", "PLTR", "ENPH"
}

print(f"=== Tech Registry ===")
print(f"Registered {len(TECH_SYMBOLS)} tech symbols")
print(f"Sample: {sorted(list(TECH_SYMBOLS))[:10]}")

# Load dataset sample
print(f"\n=== Dataset Test ===")
print("Loading dataset...")
dataset = load_dataset('kurry/sp500_earnings_transcripts', split='train[:1000]')

# Count matches
matches = 0
total = 0
matched_symbols = set()

for item in dataset:
    total += 1
    symbol = item.get('symbol', '').upper()
    if symbol in TECH_SYMBOLS:
        matches += 1
        matched_symbols.add(symbol)

print(f"Total items: {total}")
print(f"Tech matches: {matches}")
print(f"Match rate: {matches/total*100:.1f}%")
print(f"Unique tech symbols found: {len(matched_symbols)}")

if matches == 0:
    print("\n❌ NO MATCHES - Problem detected!")
    
    # Debug: show what symbols are in dataset
    dataset_symbols = set()
    for item in dataset:
        if 'symbol' in item:
            dataset_symbols.add(item['symbol'].upper())
    
    print(f"\nDataset has {len(dataset_symbols)} unique symbols")
    print(f"Sample dataset symbols: {sorted(list(dataset_symbols))[:20]}")
    
    # Find overlap
    overlap = TECH_SYMBOLS & dataset_symbols
    print(f"\nOverlap: {len(overlap)} symbols")
    if overlap:
        print(f"Overlapping symbols: {sorted(list(overlap))[:20]}")
    else:
        print("NO OVERLAP - symbols don't match!")
else:
    print(f"\n✓ Filter working! Found {matches} tech transcripts")
    print(f"Tech symbols in sample: {sorted(list(matched_symbols))[:20]}")
