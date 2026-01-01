"""
Data Filter Strategy Pattern

Implements Open/Closed Principle for data filtering:
- CLOSED for modification: Base interface is stable
- OPEN for extension: Add new filter strategies via inheritance

Usage:
    from data_filter import get_filter_strategy, TechnologyFilter
    
    # Via factory (recommended)
    filter = get_filter_strategy("technology")
    
    # Direct instantiation
    filter = TechnologyFilter()
    
    # Custom sector filter
    filter = SectorFilter("Healthcare")
    
    # Check if item passes filter
    if filter.should_include({"symbol": "AAPL", "content": "..."}):
        # Process item
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List

from .sector_registry import get_symbols_for_sector, is_registered, get_sector


# =============================================================================
# BASE STRATEGY (Closed for Modification)
# =============================================================================

class DataFilterStrategy(ABC):
    """
    Abstract base class for data filtering strategies.
    
    This defines the contract that all filter strategies must follow.
    Do NOT modify this class - extend it instead.
    """
    
    @abstractmethod
    def should_include(self, item: Dict[str, Any]) -> bool:
        """
        Determine if an item should be included in the dataset.
        
        Args:
            item: A dataset item with fields like 'symbol', 'content', 'year', etc.
            
        Returns:
            True if the item should be included, False otherwise.
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable name for this filter strategy."""
        pass
    
    @property
    def description(self) -> str:
        """Optional description of what this filter does."""
        return f"Filter: {self.name}"


# =============================================================================
# CONCRETE STRATEGIES (Open for Extension)
# =============================================================================

class AllDataFilter(DataFilterStrategy):
    """
    No-op filter that includes all data.
    
    Use this as the default when no filtering is required.
    """
    
    def should_include(self, item: Dict[str, Any]) -> bool:
        return True
    
    @property
    def name(self) -> str:
        return "all"
    
    @property
    def description(self) -> str:
        return "Includes all data without filtering"


class SectorFilter(DataFilterStrategy):
    """
    Filter by sector based on stock symbol.
    
    Uses sector_registry to map symbols to sectors.
    Extendable: subclass for specific sector presets.
    """
    
    def __init__(self, sector: str):
        """
        Initialize filter for a specific sector.
        
        Args:
            sector: Sector name (e.g., "Technology", "Healthcare")
        """
        self.sector = sector
        self._valid_symbols = set(get_symbols_for_sector(sector))
        
        if not self._valid_symbols:
            raise ValueError(
                f"Sector '{sector}' not found or has no registered symbols. "
                f"Register it first using sector_registry.register_sector()"
            )
    
    def should_include(self, item: Dict[str, Any]) -> bool:
        """Include item if its symbol belongs to the target sector."""
        symbol = item.get("symbol", "").upper()
        return symbol in self._valid_symbols
    
    @property
    def name(self) -> str:
        return f"sector:{self.sector.lower()}"
    
    @property
    def description(self) -> str:
        return f"Filters for {self.sector} sector ({len(self._valid_symbols)} symbols)"
    
    @property
    def symbol_count(self) -> int:
        """Number of registered symbols for this sector."""
        return len(self._valid_symbols)


class TechnologyFilter(SectorFilter):
    """
    Pre-configured filter for Technology sector.
    
    Convenience class that doesn't require sector name as argument.
    Uses comprehensive tech symbol list from sector_registry.
    """
    
    def __init__(self):
        super().__init__("Technology")
    
    @property
    def name(self) -> str:
        return "technology"
    
    @property
    def description(self) -> str:
        return f"Technology sector filter ({self.symbol_count} symbols including FAANG+, semiconductors, enterprise software)"


class SymbolListFilter(DataFilterStrategy):
    """
    Filter by explicit list of symbols.
    
    Useful for custom subsets or ad-hoc analysis.
    """
    
    def __init__(self, symbols: List[str]):
        """
        Initialize with explicit symbol list.
        
        Args:
            symbols: List of stock ticker symbols to include
        """
        self._symbols = {s.upper() for s in symbols}
    
    def should_include(self, item: Dict[str, Any]) -> bool:
        symbol = item.get("symbol", "").upper()
        return symbol in self._symbols
    
    @property
    def name(self) -> str:
        return f"symbols:{len(self._symbols)}"
    
    @property
    def description(self) -> str:
        preview = list(self._symbols)[:5]
        if len(self._symbols) > 5:
            preview_str = ", ".join(preview) + f" +{len(self._symbols) - 5} more"
        else:
            preview_str = ", ".join(preview)
        return f"Custom symbol filter: {preview_str}"


class YearRangeFilter(DataFilterStrategy):
    """
    Filter by year range.
    
    Can be composed with other filters for complex filtering.
    """
    
    def __init__(self, start_year: int, end_year: int):
        self.start_year = start_year
        self.end_year = end_year
    
    def should_include(self, item: Dict[str, Any]) -> bool:
        year = item.get("year")
        if year is None:
            return False
        try:
            year = int(year)
            return self.start_year <= year <= self.end_year
        except (ValueError, TypeError):
            return False
    
    @property
    def name(self) -> str:
        return f"years:{self.start_year}-{self.end_year}"
    
    @property
    def description(self) -> str:
        return f"Year range filter: {self.start_year} to {self.end_year}"


class CompositeFilter(DataFilterStrategy):
    """
    Combine multiple filters with AND logic.
    
    All sub-filters must pass for an item to be included.
    """
    
    def __init__(self, filters: List[DataFilterStrategy]):
        self._filters = filters
    
    def should_include(self, item: Dict[str, Any]) -> bool:
        return all(f.should_include(item) for f in self._filters)
    
    @property
    def name(self) -> str:
        return f"composite[{'+'.join(f.name for f in self._filters)}]"
    
    @property
    def description(self) -> str:
        return f"Composite filter: {' AND '.join(f.name for f in self._filters)}"


# =============================================================================
# FILTER FACTORY
# =============================================================================

# Registry of available filter strategies
_FILTER_REGISTRY: Dict[str, type] = {
    "all": AllDataFilter,
    "technology": TechnologyFilter,
    "tech": TechnologyFilter,  # Alias
}


def register_filter(name: str, filter_class: type) -> None:
    """
    Register a new filter strategy.
    
    This is the extension point for adding new filter types.
    
    Args:
        name: Name to register the filter under (lowercase)
        filter_class: Filter class (must be DataFilterStrategy subclass)
    """
    if not issubclass(filter_class, DataFilterStrategy):
        raise TypeError(f"{filter_class} must be a DataFilterStrategy subclass")
    _FILTER_REGISTRY[name.lower()] = filter_class


def get_filter_strategy(name: str, **kwargs) -> DataFilterStrategy:
    """
    Factory function to get a filter strategy by name.
    
    Args:
        name: Filter name (e.g., "all", "technology", "sector:Healthcare")
        **kwargs: Additional arguments passed to filter constructor
        
    Returns:
        Instantiated filter strategy
        
    Examples:
        get_filter_strategy("all")           -> AllDataFilter()
        get_filter_strategy("technology")    -> TechnologyFilter()
        get_filter_strategy("sector:Healthcare") -> SectorFilter("Healthcare")
    """
    name_lower = name.lower()
    
    # Check for sector prefix
    if name_lower.startswith("sector:"):
        sector_name = name[7:]  # Preserve original case for sector name
        return SectorFilter(sector_name)
    
    # Check registry
    if name_lower in _FILTER_REGISTRY:
        filter_class = _FILTER_REGISTRY[name_lower]
        return filter_class(**kwargs)
    
    # Unknown filter
    available = list(_FILTER_REGISTRY.keys())
    raise ValueError(
        f"Unknown filter strategy: '{name}'. "
        f"Available: {available}. "
        f"Or use 'sector:<SectorName>' for sector filtering."
    )


def list_filters() -> List[str]:
    """List all registered filter strategy names."""
    return list(_FILTER_REGISTRY.keys())


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("Data Filter Strategy Test")
    print("=" * 50)
    
    # Test filter factory
    print("\nAvailable filters:", list_filters())
    
    # Test each filter type
    test_items = [
        {"symbol": "AAPL", "year": 2023, "content": "Apple Q1..."},
        {"symbol": "JNJ", "year": 2023, "content": "J&J Q1..."},
        {"symbol": "MSFT", "year": 2015, "content": "Microsoft Q1..."},
    ]
    
    filters_to_test = [
        get_filter_strategy("all"),
        get_filter_strategy("technology"),
    ]
    
    for filter_strat in filters_to_test:
        print(f"\n{filter_strat.description}")
        for item in test_items:
            result = "✓" if filter_strat.should_include(item) else "✗"
            print(f"  {result} {item['symbol']}")
    
    print("\n✓ Data filter strategies ready!")
