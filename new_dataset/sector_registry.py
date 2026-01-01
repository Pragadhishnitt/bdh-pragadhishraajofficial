"""
S&P 500 Sector Registry

Centralized symbol-to-sector mapping using GICS (Global Industry Classification Standard).
Designed for extensibility - add new sectors without modifying existing code.

Usage:
    from sector_registry import get_sector, get_symbols_for_sector, register_sector
    
    # Query
    sector = get_sector("AAPL")  # Returns "Technology"
    symbols = get_symbols_for_sector("Technology")  # Returns list of tech symbols
    
    # Extend (for future sectors)
    register_sector("Energy", ["XOM", "CVX", "COP", ...])
"""

from typing import Dict, List, Optional, Set


# =============================================================================
# SECTOR REGISTRY (Open for Extension, Closed for Modification)
# =============================================================================

_SECTOR_SYMBOLS: Dict[str, Set[str]] = {}
_SYMBOL_TO_SECTOR: Dict[str, str] = {}


def register_sector(sector_name: str, symbols: List[str]) -> None:
    """
    Register a new sector with its symbols.
    
    This is the extension point - add new sectors without modifying existing code.
    
    Args:
        sector_name: Name of the sector (e.g., "Technology", "Healthcare")
        symbols: List of stock ticker symbols belonging to this sector
    """
    _SECTOR_SYMBOLS[sector_name] = set(symbols)
    for symbol in symbols:
        _SYMBOL_TO_SECTOR[symbol] = sector_name


def get_sector(symbol: str) -> Optional[str]:
    """Get the sector for a given stock symbol."""
    return _SYMBOL_TO_SECTOR.get(symbol.upper())


def get_symbols_for_sector(sector_name: str) -> List[str]:
    """Get all symbols registered for a sector."""
    return list(_SECTOR_SYMBOLS.get(sector_name, set()))


def list_sectors() -> List[str]:
    """List all registered sectors."""
    return list(_SECTOR_SYMBOLS.keys())


def is_registered(symbol: str) -> bool:
    """Check if a symbol is registered in any sector."""
    return symbol.upper() in _SYMBOL_TO_SECTOR


# =============================================================================
# TECHNOLOGY SECTOR - Comprehensive S&P 500 Tech Companies
# =============================================================================

# Information Technology (GICS Sector 45)
# Includes: Software, Hardware, IT Services, Semiconductors, Tech Hardware

TECHNOLOGY_SYMBOLS = [
    # === MEGA CAP / FAANG+ ===
    "AAPL",   # Apple Inc.
    "MSFT",   # Microsoft Corporation
    "GOOGL",  # Alphabet Inc. Class A
    "GOOG",   # Alphabet Inc. Class C
    "META",   # Meta Platforms Inc.
    "AMZN",   # Amazon.com Inc. (often classified as Consumer, but tech-heavy)
    "NVDA",   # NVIDIA Corporation
    "TSLA",   # Tesla Inc. (tech-forward automotive)
    
    # === SEMICONDUCTORS ===
    "AMD",    # Advanced Micro Devices
    "INTC",   # Intel Corporation
    "AVGO",   # Broadcom Inc.
    "QCOM",   # Qualcomm Inc.
    "TXN",    # Texas Instruments
    "MU",     # Micron Technology
    "AMAT",   # Applied Materials
    "LRCX",   # Lam Research
    "KLAC",   # KLA Corporation
    "MRVL",   # Marvell Technology
    "ADI",    # Analog Devices
    "NXPI",   # NXP Semiconductors
    "ON",     # ON Semiconductor
    "MCHP",   # Microchip Technology
    "SWKS",   # Skyworks Solutions
    "QRVO",   # Qorvo Inc.
    "MPWR",   # Monolithic Power Systems
    "SNPS",   # Synopsys Inc.
    "CDNS",   # Cadence Design Systems
    "ASML",   # ASML Holding (ADR)
    "TSM",    # Taiwan Semiconductor (ADR)
    "ARM",    # Arm Holdings (ADR)
    "GFS",    # GlobalFoundries
    "WOLF",   # Wolfspeed Inc.
    "SLAB",   # Silicon Laboratories
    "RMBS",   # Rambus Inc.
    "SMTC",   # Semtech Corporation
    "DIOD",   # Diodes Incorporated
    "POWI",   # Power Integrations
    "SITM",   # SiTime Corporation
    
    # === SOFTWARE - ENTERPRISE ===
    "CRM",    # Salesforce Inc.
    "ORCL",   # Oracle Corporation
    "SAP",    # SAP SE (ADR)
    "ADBE",   # Adobe Inc.
    "NOW",    # ServiceNow Inc.
    "INTU",   # Intuit Inc.
    "WDAY",   # Workday Inc.
    "TEAM",   # Atlassian Corporation
    "SNOW",   # Snowflake Inc.
    "DDOG",   # Datadog Inc.
    "PLTR",   # Palantir Technologies
    "HUBS",   # HubSpot Inc.
    "ZS",     # Zscaler Inc.
    "PANW",   # Palo Alto Networks
    "CRWD",   # CrowdStrike Holdings
    "FTNT",   # Fortinet Inc.
    "SPLK",   # Splunk Inc.
    "VEEV",   # Veeva Systems Inc.
    "ANSS",   # ANSYS Inc.
    "ADSK",   # Autodesk Inc.
    "PTC",    # PTC Inc.
    "ROP",    # Roper Technologies
    "TYL",    # Tyler Technologies
    "PAYC",   # Paycom Software
    "PCTY",   # Paylocity Holding
    "MANH",   # Manhattan Associates
    "BILL",   # Bill.com Holdings
    "SMAR",   # Smartsheet Inc.
    "APPF",   # AppFolio Inc.
    "NCNO",   # nCino Inc.
    "CFLT",   # Confluent Inc.
    "MDB",    # MongoDB Inc.
    "ESTC",   # Elastic N.V.
    "NET",    # Cloudflare Inc.
    "OKTA",   # Okta Inc.
    "MNDY",   # monday.com Ltd.
    "ZI",     # ZoomInfo Technologies
    "DOCN",   # DigitalOcean Holdings
    "PATH",   # UiPath Inc.
    "GTLB",   # GitLab Inc.
    "DOCU",   # DocuSign Inc.
    "ZM",     # Zoom Video Communications
    "BOX",    # Box Inc.
    "DBX",    # Dropbox Inc.
    "FIVN",   # Five9 Inc.
    "TWLO",   # Twilio Inc.
    "U",      # Unity Software Inc.
    "RBLX",   # Roblox Corporation
    "AI",     # C3.ai Inc.
    "UPST",   # Upstart Holdings
    "S",      # SentinelOne Inc.
    "TENB",   # Tenable Holdings
    "RPD",    # Rapid7 Inc.
    "VRNS",   # Varonis Systems
    "QLYS",   # Qualys Inc.
    "CYBR",   # CyberArk Software
    
    # === IT SERVICES & CONSULTING ===
    "IBM",    # International Business Machines
    "ACN",    # Accenture plc
    "CSCO",   # Cisco Systems
    "CTSH",   # Cognizant Technology
    "IT",     # Gartner Inc.
    "EPAM",   # EPAM Systems
    "GDDY",   # GoDaddy Inc.
    "WEX",    # WEX Inc.
    "G",      # Genpact Limited
    "LDOS",   # Leidos Holdings
    "SAIC",   # Science Applications International
    "BAH",    # Booz Allen Hamilton
    "CACI",   # CACI International
    "CLVT",   # Clarivate Plc
    
    # === HARDWARE & EQUIPMENT ===
    "HPQ",    # HP Inc.
    "HPE",    # Hewlett Packard Enterprise
    "DELL",   # Dell Technologies
    "WDC",    # Western Digital
    "STX",    # Seagate Technology
    "NTAP",   # NetApp Inc.
    "PSTG",   # Pure Storage
    "ANET",   # Arista Networks
    "JNPR",   # Juniper Networks
    "MSI",    # Motorola Solutions
    "ZBRA",   # Zebra Technologies
    "KEYS",   # Keysight Technologies
    "TER",    # Teradyne Inc.
    "COHR",   # Coherent Corp.
    "IPGP",   # IPG Photonics
    "MKSI",   # MKS Instruments
    "LOGI",   # Logitech International
    "CRSR",   # Corsair Gaming
    "SMCI",   # Super Micro Computer
    
    # === FINTECH / PAYMENTS ===
    "V",      # Visa Inc.
    "MA",     # Mastercard Inc.
    "PYPL",   # PayPal Holdings
    "SQ",     # Block Inc. (Square)
    "FIS",    # Fidelity National Information Services
    "FISV",   # Fiserv Inc.
    "GPN",    # Global Payments
    "ADP",    # Automatic Data Processing
    "PAYX",   # Paychex Inc.
    "FI",     # Fiserv Inc.
    "COIN",   # Coinbase Global
    "HOOD",   # Robinhood Markets
    "SOFI",   # SoFi Technologies
    "AFRM",   # Affirm Holdings
    "SHOP",   # Shopify Inc.
    
    # === INTERNET & DIGITAL MEDIA ===
    "NFLX",   # Netflix Inc.
    "DIS",    # Walt Disney (streaming tech)
    "SPOT",   # Spotify Technology
    "SNAP",   # Snap Inc.
    "PINS",   # Pinterest Inc.
    "MTCH",   # Match Group
    "ABNB",   # Airbnb Inc.
    "BKNG",   # Booking Holdings
    "UBER",   # Uber Technologies
    "LYFT",   # Lyft Inc.
    "DASH",   # DoorDash Inc.
    "ETSY",   # Etsy Inc.
    "EBAY",   # eBay Inc.
    "W",      # Wayfair Inc.
    "CHWY",   # Chewy Inc.
    "YELP",   # Yelp Inc.
    "TRIP",   # Tripadvisor Inc.
    "TTWO",   # Take-Two Interactive
    "EA",     # Electronic Arts
    "ATVI",   # Activision Blizzard (now part of MSFT)
    
    # === CLOUD INFRASTRUCTURE ===
    # (Many already listed under enterprise software)
    "DLR",    # Digital Realty Trust (Data Centers)
    "EQIX",   # Equinix Inc. (Data Centers)
    "AMT",    # American Tower (Infrastructure)
    "CCI",    # Crown Castle (Infrastructure)
    
    # === AI / ML FOCUSED ===
    "GOOG",   # Google/Alphabet (DeepMind, etc.)
    "NVDA",   # NVIDIA (AI chips)
    "AMD",    # AMD (AI accelerators)
    "MSFT",   # Microsoft (OpenAI partnership)
    "META",   # Meta (LLaMA, AI research)
    "PLTR",   # Palantir (AI analytics)
    "AI",     # C3.ai
    "BBAI",   # BigBear.ai
    "SOUN",   # SoundHound AI
    
    # === EMERGING / HIGH-GROWTH TECH ===
    "IONQ",   # IonQ Inc. (Quantum Computing)
    "RGTI",   # Rigetti Computing
    "LAZR",   # Luminar Technologies (Lidar)
    "VLDR",   # Velodyne Lidar
    "OPEN",   # Opendoor Technologies
    "COUR",   # Coursera Inc.
    "DUOL",   # Duolingo Inc.
]

# Register Technology sector
register_sector("Technology", TECHNOLOGY_SYMBOLS)


# =============================================================================
# FUTURE SECTOR TEMPLATES (Commented out - uncomment to enable)
# =============================================================================

# Healthcare sector template (for future extension)
# HEALTHCARE_SYMBOLS = [
#     "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT", "DHR", "BMY",
#     "AMGN", "GILD", "CVS", "CI", "ELV", "HUM", "CNC", "MCK", "CAH", "ABC",
#     "ISRG", "SYK", "MDT", "BDX", "ZBH", "BSX", "EW", "HOLX", "ALGN", "DXCM",
#     "VRTX", "REGN", "BIIB", "MRNA", "BNTX", ...
# ]
# register_sector("Healthcare", HEALTHCARE_SYMBOLS)

# Financials sector template
# FINANCIALS_SYMBOLS = [
#     "JPM", "BAC", "WFC", "C", "GS", "MS", "BLK", "SCHW", "AXP", "USB",
#     "PNC", "TFC", "COF", "BK", "STT", "NTRS", "KEY", "RF", "CFG", "HBAN",
#     "AIG", "MET", "PRU", "ALL", "TRV", "CB", "AON", "MMC", "SPGI", "MCO",
#     ...
# ]
# register_sector("Financials", FINANCIALS_SYMBOLS)

# Energy sector template
# ENERGY_SYMBOLS = [
#     "XOM", "CVX", "COP", "EOG", "SLB", "PXD", "MPC", "PSX", "VLO", "OXY",
#     "WMB", "KMI", "OKE", "HAL", "BKR", "DVN", "FANG", "HES", "MRO", "APA",
#     ...
# ]
# register_sector("Energy", ENERGY_SYMBOLS)


# =============================================================================
# MODULE TEST
# =============================================================================

if __name__ == "__main__":
    print("Sector Registry Test")
    print("=" * 50)
    
    print(f"\nRegistered Sectors: {list_sectors()}")
    
    tech_symbols = get_symbols_for_sector("Technology")
    print(f"\nTechnology Sector: {len(tech_symbols)} symbols")
    
    # Test lookups
    test_symbols = ["AAPL", "MSFT", "NVDA", "JNJ", "XOM", "UNKNOWN"]
    for sym in test_symbols:
        sector = get_sector(sym)
        print(f"  {sym}: {sector or 'Not registered'}")
    
    print("\n✓ Sector registry ready!")
