"""Portfolio configuration that scales with capital.

All portfolio construction parameters are derived from equity size.
As capital grows, the system automatically adjusts positions, weights,
universe filtering, and stock price limits.

Hybrid universe (Option C):
  - Sector ETFs (XLF, XLE, XLU, XLC, XLRE) for base diversification
  - Individual stocks under price threshold for alpha
  - Dynamic price filter at each rebalance
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Sector mapping ─────────────────────────────────────────

SECTOR_MAP: dict[str, str] = {
    # Sector ETFs
    "XLK": "Tech", "XLF": "Fin", "XLV": "Health", "XLE": "Energy",
    "XLI": "Industrial", "XLC": "Comm", "XLY": "ConsDisc", "XLP": "Staples",
    "XLU": "Utility", "XLRE": "RealEstate", "XLB": "Materials",
    # Individual stocks — Tech
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "AVGO": "Tech",
    "ADBE": "Tech", "CRM": "Tech", "AMD": "Tech", "QCOM": "Tech",
    "TXN": "Tech", "AMAT": "Tech", "ACN": "Tech", "INTC": "Tech",
    "PYPL": "Tech", "HPQ": "Tech",
    # Communication
    "GOOGL": "Comm", "META": "Comm", "T": "Comm", "VZ": "Comm",
    # Consumer Discretionary
    "AMZN": "ConsDisc", "TSLA": "ConsDisc", "HD": "ConsDisc",
    "MCD": "ConsDisc", "NKE": "ConsDisc", "DAL": "ConsDisc", "F": "ConsDisc",
    # Financials
    "JPM": "Fin", "V": "Fin", "MA": "Fin", "BRK-B": "Fin", "BAC": "Fin",
    # Healthcare
    "JNJ": "Health", "UNH": "Health", "LLY": "Health", "ABBV": "Health",
    "MRK": "Health", "TMO": "Health", "ISRG": "Health", "PFE": "Health",
    # Consumer Staples
    "PG": "Staples", "COST": "Staples", "PEP": "Staples",
    "KO": "Staples", "WMT": "Staples", "MO": "Staples",
    # Energy
    "XOM": "Energy", "OXY": "Energy", "SLB": "Energy",
    # Materials
    "LIN": "Materials", "FCX": "Materials",
    # Index
    "SPY": "Index", "QQQ": "Index", "IWM": "Index",
}


# ── Hybrid universe (Option C: ETFs + affordable stocks) ───

# Sector ETFs — all under $100, good for base diversification
SECTOR_ETFS = ["XLF", "XLE", "XLU", "XLC", "XLRE", "XLB", "XLI", "XLP", "XLY"]

# Affordable individual stocks (S&P 500, typically $10-$100)
# These provide alpha opportunities on top of ETF diversification
AFFORDABLE_STOCKS = [
    "BAC", "KO", "PFE", "T", "VZ", "PYPL", "OXY", "INTC",
    "DAL", "MO", "F", "FCX", "SLB", "HPQ", "NKE",
]

# Full universe — used when capital is large enough for any stock
FULL_STOCKS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK-B",
    "JPM", "V", "JNJ", "UNH", "HD", "PG", "MA", "XOM", "LLY", "ABBV",
    "MRK", "COST", "AVGO", "PEP", "KO", "TMO", "WMT", "ADBE", "CRM",
    "ACN", "MCD", "NKE", "LIN", "TXN", "AMD", "QCOM", "ISRG", "AMAT",
]


@dataclass
class PortfolioConfig:
    """Portfolio parameters that scale with capital."""

    equity: float
    max_positions: int
    max_position_pct: float
    max_sector_count: int
    cash_reserve_pct: float
    use_fractional: bool
    min_position_usd: float
    max_stock_price: float  # この価格以上の銘柄はフィルターアウト
    rebalance_interval_days: int

    @classmethod
    def from_equity(cls, equity: float) -> PortfolioConfig:
        """資本額からポートフォリオパラメータを自動決定。

        Tiers:
          < $5K    → Hybrid: ETF + 安い個別株 (max $100)
          $5-20K   → Hybrid: ETF + 個別株 (max $300)
          $20-50K  → Full universe (max $500)
          > $50K   → Full universe (制限なし)
        """
        if equity < 5_000:
            return cls(
                equity=equity,
                max_positions=5,
                max_position_pct=0.20,
                max_sector_count=2,
                cash_reserve_pct=0.25,
                use_fractional=True,
                min_position_usd=100,
                max_stock_price=100,
                rebalance_interval_days=14,
            )
        elif equity < 20_000:
            return cls(
                equity=equity,
                max_positions=8,
                max_position_pct=0.15,
                max_sector_count=2,
                cash_reserve_pct=0.20,
                use_fractional=True,
                min_position_usd=200,
                max_stock_price=300,
                rebalance_interval_days=21,
            )
        elif equity < 50_000:
            return cls(
                equity=equity,
                max_positions=12,
                max_position_pct=0.12,
                max_sector_count=3,
                cash_reserve_pct=0.15,
                use_fractional=True,
                min_position_usd=500,
                max_stock_price=500,
                rebalance_interval_days=21,
            )
        else:
            return cls(
                equity=equity,
                max_positions=15,
                max_position_pct=0.10,
                max_sector_count=3,
                cash_reserve_pct=0.10,
                use_fractional=False,
                min_position_usd=1000,
                max_stock_price=99999,  # 制限なし
                rebalance_interval_days=21,
            )

    @property
    def investable_equity(self) -> float:
        return self.equity * (1 - self.cash_reserve_pct)

    @property
    def target_position_usd(self) -> float:
        return self.investable_equity / self.max_positions

    def get_universe(self) -> list[str]:
        """資本額に応じたユニバースを返す。

        小資本: セクターETF + 安い個別株 (Hybrid C)
        大資本: フルS&P 500
        """
        if self.max_stock_price <= 100:
            # Hybrid C: ETF + affordable stocks
            return SECTOR_ETFS + AFFORDABLE_STOCKS
        elif self.max_stock_price <= 300:
            # ETF + affordable + some full stocks
            return SECTOR_ETFS + AFFORDABLE_STOCKS + [
                s for s in FULL_STOCKS if s not in AFFORDABLE_STOCKS
            ]
        else:
            return FULL_STOCKS + [s for s in SECTOR_ETFS if s not in FULL_STOCKS]

    def describe(self) -> str:
        universe = self.get_universe()
        return (
            f"PortfolioConfig(equity=${self.equity:,.0f})\n"
            f"  Positions: {self.max_positions} (max {self.max_position_pct:.0%} each)\n"
            f"  Sector cap: {self.max_sector_count}/sector\n"
            f"  Cash reserve: {self.cash_reserve_pct:.0%} (${self.equity * self.cash_reserve_pct:,.0f})\n"
            f"  Max stock price: ${self.max_stock_price:.0f}\n"
            f"  Universe: {len(universe)} symbols\n"
            f"  Fractional: {self.use_fractional}\n"
            f"  Rebalance: every {self.rebalance_interval_days}d"
        )


def filter_by_price(
    symbols: list[str],
    prices: dict[str, float],
    max_price: float,
) -> list[str]:
    """株価でフィルター。max_price以下の銘柄のみ返す。"""
    return [s for s in symbols if prices.get(s, 0) <= max_price and prices.get(s, 0) > 0]


def select_universe(
    candidates: list[str],
    scores: dict[str, float],
    config: PortfolioConfig,
    correlations: dict[tuple[str, str], float] | None = None,
    max_correlation: float = 0.70,
) -> list[str]:
    """スコア順に銘柄を選択、セクター制限と相関フィルター付き。"""
    sorted_syms = sorted(candidates, key=lambda s: scores.get(s, 0), reverse=True)

    selected: list[str] = []
    sector_count: dict[str, int] = {}

    for sym in sorted_syms:
        if len(selected) >= config.max_positions:
            break

        # セクター制限
        sector = SECTOR_MAP.get(sym, "Other")
        if sector != "Index" and sector_count.get(sector, 0) >= config.max_sector_count:
            continue

        # 相関フィルター
        if correlations and selected:
            too_correlated = False
            for existing in selected:
                key = (min(sym, existing), max(sym, existing))
                corr = correlations.get(key, 0)
                if abs(corr) > max_correlation:
                    too_correlated = True
                    break
            if too_correlated:
                continue

        selected.append(sym)
        sector_count[sector] = sector_count.get(sector, 0) + 1

    return selected
