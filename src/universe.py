"""
A starter training universe: US 10-Q/10-K filers across every sector,
deliberately including unprofitable and small-revenue companies (early
biotech, growth tech, EVs) so filings like theirs aren't far outside what
the model has seen.

It's today's listed companies, so it carries survivorship bias: firms that
went bankrupt or were acquired are missing. Tickers EDGAR no longer knows
are logged as warnings during training and skipped.

HOLDOUT_TICKERS are left out on purpose so batch tests on them are
out-of-sample.
"""

HOLDOUT_TICKERS = [
    "MRNA", "NVAX", "INTC", "EL", "NKE", "NVDA", "MSFT", "AMZN", "META", "TSLA",
    "JPM", "XOM", "PFE", "WMT", "KO", "DIS", "BA", "SBUX", "PYPL", "SNAP",
]

UNIVERSE_BY_SECTOR = {
    "software_internet": [
        "AAPL", "GOOGL", "ORCL", "ADBE", "CRM", "INTU", "NOW", "CSCO", "IBM", "ACN",
        "ADP", "PAYX", "CDNS", "SNPS", "ADSK", "FTNT", "PANW", "CRWD", "ZS", "DDOG",
        "SNOW", "NET", "MDB", "OKTA", "TWLO", "DOCU", "ZM", "WDAY", "HUBS", "VEEV",
        "PLTR", "U", "PATH", "RBLX", "DBX", "BOX", "AKAM", "EPAM", "CTSH", "FICO",
        "TYL", "PTC", "NFLX", "EBAY", "ETSY", "BKNG", "EXPE", "ABNB", "UBER", "LYFT",
        "DASH", "PINS", "ROKU", "CHWY", "W", "CVNA", "ZG",
    ],
    "semis_hardware": [
        "AMD", "QCOM", "TXN", "AVGO", "MU", "AMAT", "LRCX", "KLAC", "ADI", "MCHP",
        "ON", "MRVL", "SWKS", "QRVO", "TER", "MPWR", "WDC", "STX", "HPQ", "HPE",
        "DELL", "NTAP", "ANET", "GLW", "APH", "KEYS", "ZBRA", "TRMB", "SMCI",
    ],
    "biopharma": [
        "JNJ", "MRK", "LLY", "ABBV", "BMY", "AMGN", "GILD", "REGN", "VRTX", "BIIB",
        "ALNY", "INCY", "EXEL", "SRPT", "IONS", "BMRN", "NBIX", "UTHR", "JAZZ", "HALO",
        "ACAD", "ARWR", "CRSP", "BEAM", "NTLA", "EDIT", "IOVA", "VKTX", "INSM", "RARE",
        "FOLD", "OCGN", "TWST", "EXAS", "ILMN", "TXG", "PACB", "NTRA", "GH",
    ],
    "medtech_health": [
        "ABT", "TMO", "DHR", "MDT", "SYK", "BSX", "EW", "ISRG", "ZBH", "BDX",
        "BAX", "DXCM", "PODD", "ALGN", "IDXX", "RMD", "HOLX", "UNH", "ELV", "CI",
        "HUM", "CVS", "CNC", "MOH", "HCA", "UHS", "THC", "DVA", "LH", "DGX",
        "MCK", "CAH", "TDOC", "HIMS",
    ],
    "financials": [
        "BAC", "WFC", "C", "GS", "MS", "USB", "PNC", "TFC", "COF", "SCHW",
        "BK", "STT", "AXP", "BLK", "TROW", "IVZ", "BEN", "KEY", "RF", "FITB",
        "HBAN", "MTB", "CFG", "ZION", "ALLY", "SYF", "SOFI", "HOOD", "COIN", "V",
        "MA", "FIS", "GPN", "ICE", "CME", "NDAQ", "SPGI", "MCO", "MSCI",
        "AIG", "MET", "PRU", "AFL", "ALL", "TRV", "PGR", "CB", "HIG", "CINF", "UNM",
    ],
    "real_estate": [
        "AMT", "PLD", "CCI", "EQIX", "SPG", "O", "PSA", "WELL", "VTR", "AVB",
        "EQR", "DLR", "MAA", "ESS", "ARE", "BXP", "VNO",
    ],
    "energy_utilities": [
        "CVX", "COP", "EOG", "OXY", "DVN", "APA", "HAL", "SLB", "BKR", "MPC",
        "VLO", "PSX", "OKE", "KMI", "WMB", "FANG", "CTRA", "EQT", "AR", "RRC",
        "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "ED", "PCG",
        "EIX", "PEG", "WEC", "ES", "AES",
    ],
    "telecom_media": [
        "T", "VZ", "TMUS", "CMCSA", "CHTR", "WBD", "FOXA", "NWSA", "EA", "TTWO",
        "LYV", "OMC",
    ],
    "consumer": [
        "COST", "TGT", "HD", "LOW", "TJX", "ROST", "DG", "DLTR", "BBY", "KSS",
        "M", "ULTA", "LULU", "DECK", "CROX", "UAA", "VFC", "PVH", "RL", "TPR",
        "HAS", "MAT", "MCD", "CMG", "YUM", "DRI", "DPZ", "WEN", "PEP", "MDLZ",
        "KHC", "GIS", "HSY", "CPB", "CAG", "SJM", "HRL", "TSN", "KR", "SYY",
        "PG", "CL", "KMB", "CLX", "CHD", "MO", "PM", "STZ", "TAP", "KDP",
        "MNST", "CELH", "BYND", "PTON", "F", "GM", "RIVN", "LCID", "HOG", "BWA",
        "LVS", "WYNN", "MGM", "CZR", "MAR", "HLT", "H", "CCL", "RCL", "NCLH",
    ],
    "industrials_materials": [
        "LMT", "RTX", "NOC", "GD", "LHX", "GE", "HON", "MMM", "CAT", "DE",
        "EMR", "ETN", "ITW", "PH", "ROK", "CMI", "PCAR", "UNP", "CSX", "NSC",
        "UPS", "FDX", "DAL", "UAL", "AAL", "LUV", "JBLU", "WM", "RSG", "URI",
        "FAST", "GWW", "JCI", "CARR", "OTIS", "TT", "SWK", "DOV", "XYL", "AME",
        "PWR", "CHRW", "EXPD", "ODFL", "JBHT",
        "LIN", "APD", "SHW", "ECL", "DD", "DOW", "LYB", "NUE", "STLD", "FCX",
        "NEM", "ALB", "MOS", "CF", "IP", "PKG", "BALL", "VMC", "MLM", "AA", "CLF",
    ],
}

STARTER_UNIVERSE = [t for tickers in UNIVERSE_BY_SECTOR.values() for t in tickers]
