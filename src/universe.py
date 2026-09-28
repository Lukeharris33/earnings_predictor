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
import hashlib
import re
import shutil
from concurrent.futures import ThreadPoolExecutor

from .config import config
from .sec_client import SECAPIError, SECClient, normalize_symbol

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


# ------------------------------------------------------------------ auto-fill
# Beyond the starter universe, auto-fill draws on every NYSE/Nasdaq company
# EDGAR lists. About 1 in TEST_SHARE of those is reserved for batch tests and
# never auto-filled into training. The split hashes the ticker, so it doesn't
# move when EDGAR reorders its list.
LISTED_EXCHANGES = {"NYSE", "Nasdaq"}
# Common and class shares (BRK-B); skips preferreds (BAC-PL) and the like.
_COMMON_SHARE = re.compile(r"^[A-Z]{1,5}(-[A-Z])?$")
TEST_SHARE = 6
# Company facts + filing history + prices for one ticker not yet cached.
NEW_TICKER_MB = 5.0


def _reserved_for_tests(symbol: str) -> bool:
    return int(hashlib.sha1(symbol.encode("utf-8")).hexdigest(), 16) % TEST_SHARE == 0


def _listed(sec: SECClient) -> list[tuple[str, int]]:
    """(ticker, cik) for NYSE/Nasdaq common stock, one ticker per company."""
    seen_ciks, out = set(), []
    for row in sec.get_listed_companies():
        symbol = normalize_symbol(row.get("ticker") or "")
        if row.get("exchange") not in LISTED_EXCHANGES or not _COMMON_SHARE.match(symbol):
            continue
        if row["cik"] in seen_ciks:
            continue
        seen_ciks.add(row["cik"])
        out.append((symbol, int(row["cik"])))
    return out


def autofill(sec: SECClient, kind: str, exclude: list[str] = ()) -> dict:
    """As many tickers as this machine can handle, for training ("train") or
    for batch tests ("batch"). The two never overlap: training gets the
    starter universe then the unreserved listed companies; batch tests get
    the holdout tickers then the reserved ones, minus `exclude` (the model's
    own training tickers)."""
    listed = _listed(sec)
    cik_of = dict(listed)
    starter, holdout = set(STARTER_UNIVERSE), set(HOLDOUT_TICKERS)
    if kind == "train":
        limit = config.AUTOFILL_TRAIN_TICKERS
        pool = STARTER_UNIVERSE + [
            s for s, _ in listed if s not in holdout and not _reserved_for_tests(s)
        ]
    else:
        limit = config.AUTOFILL_BATCH_TICKERS
        pool = HOLDOUT_TICKERS + [s for s, _ in listed if s not in starter and _reserved_for_tests(s)]
    skip = {normalize_symbol(s) for s in exclude}

    def usable(symbol: str) -> bool:
        # The curated lists are known 10-Q filers; check everything else.
        if symbol in starter or symbol in holdout:
            return True
        try:
            return sec.files_10q(cik_of[symbol])
        except SECAPIError:
            return False

    free_mb = shutil.disk_usage(config.CACHE_DIR).free / 2**20 if config.CACHE_ENABLED else float("inf")
    budget_mb = free_mb - config.AUTOFILL_DISK_RESERVE_GB * 1024
    candidates = [s for s in dict.fromkeys(pool) if s not in skip]  # de-duplicated, order kept
    chosen, new, foreign, out_of_disk = [], 0, 0, False
    # Checked in chunks, a few at a time (SECClient keeps the overall rate
    # under EDGAR's limit), stopping once the list is full.
    with ThreadPoolExecutor(max_workers=4) as pool_exec:
        for start in range(0, len(candidates), 40):
            if len(chosen) >= limit or out_of_disk:
                break
            chunk = candidates[start:start + 40]
            for symbol, ok in zip(chunk, pool_exec.map(usable, chunk)):
                if len(chosen) >= limit:
                    break
                if not ok:
                    foreign += 1
                    continue
                cik = cik_of.get(symbol)
                if not (cik and sec.has_cached_facts(cik)):
                    if budget_mb < NEW_TICKER_MB:
                        out_of_disk = True
                        break
                    budget_mb -= NEW_TICKER_MB
                    new += 1
                chosen.append(symbol)

    what = "training" if kind == "train" else "batch testing"
    note = f"{len(chosen)} tickers for {what}; {len(chosen) - new} already cached, {new} to download (~{new * NEW_TICKER_MB / 1024:.1f} GB)."
    if out_of_disk:
        note += f" Stopped early to keep {config.AUTOFILL_DISK_RESERVE_GB:g} GB of disk free."
    elif len(chosen) >= limit:
        note += f" That's the cap for this machine ({limit})."
    if foreign:
        note += f" Skipped {foreign} that don't file 10-Q/10-K reports (mostly foreign companies)."
    if skip:
        note += f" Left out the model's {len(skip)} training tickers."
    return {"symbols": chosen, "note": note, "new_downloads": new}
