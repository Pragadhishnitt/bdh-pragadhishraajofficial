"""
Finance Data Loader for BDH Competition

Implements chronological streaming from S&P 500 Earnings Transcripts
as specified in PDF Section 3.2 and 5.1.

Key features:
- Chronological ordering (essential for Hebbian learning evaluation)
- Sector filtering (Technology sector for maximum concept drift)
- Structure preservation (Presentation vs Q&A sections)
- GPT-2 BPE tokenization for compatibility
"""

import os
from typing import Iterator, Dict, Any, Optional, Tuple
from dataclasses import dataclass

import torch
from torch.utils.data import IterableDataset, DataLoader

try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False


@dataclass
class TranscriptSample:
    """A single earnings transcript sample."""
    input_ids: torch.Tensor
    labels: torch.Tensor
    date: str
    company: str
    section: str  # "presentation" or "qa"


class FinanceDataset(IterableDataset):
    """
    Chronological streaming dataset for S&P 500 earnings transcripts.
    
    Implements the data pipeline from PDF Section 3.2:
    1. Source: kurry/sp500_earnings_transcripts (HuggingFace)
    2. Filtering: Technology sector
    3. Chronological sequencing (NOT shuffled)
    4. Structure preservation (Presentation vs Q&A)
    """
    
    def __init__(
        self,
        start_year: int = 2005,
        end_year: int = 2024,
        sector: str = "Technology",
        block_size: int = 512,
        split: str = "train",
        tokenizer_name: str = "gpt2",
    ):
        self.start_year = start_year
        self.end_year = end_year
        self.sector = sector
        self.block_size = block_size
        self.split = split
        
        # Initialize tokenizer
        if TIKTOKEN_AVAILABLE:
            self.tokenizer = tiktoken.get_encoding("gpt2")
        else:
            # Fallback to byte-level encoding (as in original train.py)
            self.tokenizer = None
        
        # Load dataset
        self._data = None
        self._load_data()
    
    def _load_data(self) -> None:
        """Load and filter the dataset."""
        if not HF_AVAILABLE:
            print("Warning: 'datasets' library not available. Using fallback data.")
            self._data = self._create_fallback_data()
            return
        
        try:
            # Try to load from HuggingFace
            dataset = load_dataset(
                "kurry/sp500_earnings_transcripts",
                split=self.split,
                trust_remote_code=True,
            )
            
            # Filter by sector and date range
            filtered = []
            for item in dataset:
                year = self._extract_year(item)
                if year is None:
                    continue
                if self.start_year <= year <= self.end_year:
                    if self._matches_sector(item):
                        filtered.append(item)
            
            # Sort chronologically
            filtered.sort(key=lambda x: self._extract_date(x))
            self._data = filtered
            print(f"Loaded {len(filtered)} transcripts from {self.start_year}-{self.end_year}")
            
        except Exception as e:
            print(f"Warning: Could not load HuggingFace dataset: {e}")
            print("Using fallback data for development.")
            self._data = self._create_fallback_data()
    
    def _extract_year(self, item: Dict[str, Any]) -> Optional[int]:
        """Extract year from dataset item."""
        # Handle various date field names
        for field in ["date", "filing_date", "call_date", "year"]:
            if field in item:
                val = item[field]
                if isinstance(val, int):
                    return val
                if isinstance(val, str) and len(val) >= 4:
                    try:
                        return int(val[:4])
                    except ValueError:
                        continue
        return None
    
    def _extract_date(self, item: Dict[str, Any]) -> str:
        """Extract sortable date string."""
        for field in ["date", "filing_date", "call_date"]:
            if field in item and item[field]:
                return str(item[field])
        return "0000-00-00"
    
    def _matches_sector(self, item: Dict[str, Any]) -> bool:
        """Check if item matches the target sector."""
        if self.sector is None:
            return True
        for field in ["sector", "industry", "gics_sector"]:
            if field in item:
                if self.sector.lower() in str(item[field]).lower():
                    return True
        return False
    
    def _create_fallback_data(self) -> list:
        """Create minimal fallback data for development/testing."""
        # Synthetic financial text for when dataset unavailable
        samples = [
            {
                "text": "Good morning and thank you for joining our Q1 2023 earnings call. "
                        "Revenue increased 15% year-over-year driven by strong AI adoption. "
                        "EBITDA margin expanded to 28% as operational efficiencies improved. "
                        "We are raising our full-year guidance to reflect continued momentum.",
                "date": "2023-04-15",
                "company": "TECH_CORP",
                "section": "presentation",
            },
            {
                "text": "Analyst: Can you discuss the impact of inflation on your margins? "
                        "CEO: We've seen some headwinds from rising labor costs but our "
                        "pricing power has allowed us to largely offset these pressures. "
                        "The AI revenue stream has been particularly resilient.",
                "date": "2023-04-15", 
                "company": "TECH_CORP",
                "section": "qa",
            },
        ] * 100  # Replicate for training
        return samples
    
    def _tokenize(self, text: str) -> torch.Tensor:
        """Tokenize text to tensor."""
        if self.tokenizer is not None:
            tokens = self.tokenizer.encode(text)
        else:
            # Byte-level fallback (matches original train.py)
            tokens = list(text.encode("utf-8"))
        return torch.tensor(tokens, dtype=torch.long)
    
    def __iter__(self) -> Iterator[TranscriptSample]:
        """Yield transcript samples in chronological order."""
        for item in self._data:
            text = item.get("text", item.get("transcript", ""))
            if not text:
                continue
            
            tokens = self._tokenize(text)
            
            # Create overlapping chunks of block_size
            for i in range(0, len(tokens) - self.block_size, self.block_size // 2):
                chunk = tokens[i:i + self.block_size + 1]
                if len(chunk) < self.block_size + 1:
                    continue
                
                yield TranscriptSample(
                    input_ids=chunk[:-1],
                    labels=chunk[1:],
                    date=str(item.get("date", "")),
                    company=str(item.get("company", "")),
                    section=str(item.get("section", "unknown")),
                )
    
    def __len__(self) -> int:
        """Approximate length (not exact for IterableDataset)."""
        return len(self._data) * 10  # Approximate chunks per transcript


def create_data_loaders(
    pretrain_years: Tuple[int, int] = (2005, 2018),
    eval_years: Tuple[int, int] = (2019, 2024),
    sector: str = "Technology",
    block_size: int = 512,
    batch_size: int = 32,
    num_workers: int = 0,
    val_split: float = 0.2,  # NEW: 20% validation split within Stage A
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create training, validation, and evaluation data loaders.
    
    Implements the Two-Stage Protocol from PDF Section 5.2:
    - Stage A: Pre-training on 2005-2018 data (split into train/val)
    - Stage B: Hebbian evaluation on 2019-2024 data (test set)
    
    Returns:
        (train_loader, val_loader, eval_loader)
    """
    
    def collate_fn(samples):
        """Collate samples into batch tensors."""
        input_ids = torch.stack([s.input_ids for s in samples])
        labels = torch.stack([s.labels for s in samples])
        return {"input_ids": input_ids, "labels": labels}
    
    # Stage A: Load all pre-training data first
    full_dataset = FinanceDataset(
        start_year=pretrain_years[0],
        end_year=pretrain_years[1],
        sector=sector,
        block_size=block_size,
    )
    
    # Split into train/val (chronologically - earlier for train, later for val)
    all_data = list(full_dataset._data)
    split_idx = int(len(all_data) * (1 - val_split))
    
    # Create separate datasets with pre-split data
    train_dataset = FinanceDataset.__new__(FinanceDataset)
    train_dataset.block_size = block_size
    train_dataset.tokenizer = full_dataset.tokenizer
    train_dataset._data = all_data[:split_idx]
    
    val_dataset = FinanceDataset.__new__(FinanceDataset)
    val_dataset.block_size = block_size
    val_dataset.tokenizer = full_dataset.tokenizer
    val_dataset._data = all_data[split_idx:]
    
    print(f"Train: {len(train_dataset._data)} transcripts, Val: {len(val_dataset._data)} transcripts")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    
    # Stage B: Evaluation/Test data (2019-2024)
    eval_dataset = FinanceDataset(
        start_year=eval_years[0],
        end_year=eval_years[1],
        sector=sector,
        block_size=block_size,
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
    
    return train_loader, val_loader, eval_loader


if __name__ == "__main__":
    # Quick test
    print("Testing FinanceDataset...")
    dataset = FinanceDataset(start_year=2020, end_year=2021)
    sample = next(iter(dataset))
    print(f"Sample shape: input_ids={sample.input_ids.shape}, labels={sample.labels.shape}")
    print(f"Date: {sample.date}, Company: {sample.company}, Section: {sample.section}")
    print("Finance loader ready!")
