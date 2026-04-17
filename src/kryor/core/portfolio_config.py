"""Portfolio configuration that scales with capital.

All portfolio construction parameters are derived from equity size.
As capital grows, the system automatically adjusts:
  - Number of active positions
  - Max position weight
  - Sector concentration limits
  - Cash reserve ratio
  - Universe selection method

Usage:
    config = PortfolioConfig.from_equity(2000)
    config.max_positions  # → 5
    config.max_position_pct  # → 0.20

    config = PortfolioConfig.from_equity(50000)
    config.max_positions  # → 15
    config.max_position_pct  # → 0.10
"""

from __future__ import annotations

from dataclasses import dataclass


# GICS sector mapping for US large-cap stocks
SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "Tech", "MSFT": "Tech", "NVDA": "Tech", "AVGO": "Tech",
    "ADBE": "Tech", "CRM": "Tech", "AMD": "Tech", "QCOM": "Tech",
    "TXN": "Tech", "AMAT": "Tech", "ACN": "Tech",
    # Communication
    "GOOGL": "Comm", "META": "Comm",
    # Consumer Discretionary
    "AMZN": "ConsDisc", "TSLA": "ConsDisc", "HD": "ConsDisc",
    "MCD": "ConsDisc", "NKE": "ConsDisc",
    # Financials
    "JPM": "Fin", "V": "Fin", "MA": "Fin", "BRK-B": "Fin",
    # Healthcare
    "JNJ": "Health", "UNH": "Health", "LLY": "Health", "ABBV": "Health",
    "MRK": "Health", "TMO": "Health", "ISRG": "Health",
    # Consumer Staples
    "PG": "Staples", "COST": "Staples", "PEP": "Staples",
    "KO": "Staples", "WMT": "Staples",
    # Energy
    "XOM": "Energy",
    # Materials
    "LIN": "Materials",
    # Index
    "SPY": "Index", "QQQ": "Index", "IWM": "Index",
}


@dataclass
class PortfolioConfig:
    """Portfolio parameters that scale with capital."""

    equity: float
    max_positions: int
    max_position_pct: float  # 1銘柄の最大ウェイト
    max_sector_count: int  # 同一セクター最大銘柄数
    cash_reserve_pct: float  # リバージョン用現金リザーブ
    use_fractional: bool  # 端株取引を使うか
    min_position_usd: float  # 最小ポジション額
    rebalance_interval_days: int

    @classmethod
    def from_equity(cls, equity: float) -> PortfolioConfig:
        """資本額からポートフォリオパラメータを自動決定。

        Tiers:
          < $5K    → 超集中 (5-6銘柄, 端株必須)
          $5-20K   → 集中   (8-10銘柄)
          $20-50K  → 標準   (10-15銘柄)
          > $50K   → 分散   (15-20銘柄)
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
                rebalance_interval_days=14,  # 2週間（小資本は回転速め）
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
                rebalance_interval_days=21,
            )
        else:
            return cls(
                equity=equity,
                max_positions=15,
                max_position_pct=0.10,
                max_sector_count=3,
                cash_reserve_pct=0.10,
                use_fractional=False,  # 資本十分なら整数株で問題ない
                min_position_usd=1000,
                rebalance_interval_days=21,
            )

    @property
    def investable_equity(self) -> float:
        """現金リザーブを除いた投資可能額"""
        return self.equity * (1 - self.cash_reserve_pct)

    @property
    def target_position_usd(self) -> float:
        """1ポジションの目標額"""
        return self.investable_equity / self.max_positions

    def describe(self) -> str:
        return (
            f"PortfolioConfig(equity=${self.equity:,.0f})\n"
            f"  Positions: {self.max_positions} (max {self.max_position_pct:.0%} each)\n"
            f"  Sector cap: {self.max_sector_count}/sector\n"
            f"  Cash reserve: {self.cash_reserve_pct:.0%} (${self.equity * self.cash_reserve_pct:,.0f})\n"
            f"  Investable: ${self.investable_equity:,.0f}\n"
            f"  Target/position: ${self.target_position_usd:,.0f}\n"
            f"  Fractional: {self.use_fractional}\n"
            f"  Rebalance: every {self.rebalance_interval_days}d"
        )


def select_universe(
    candidates: list[str],
    scores: dict[str, float],
    config: PortfolioConfig,
    correlations: dict[tuple[str, str], float] | None = None,
    max_correlation: float = 0.70,
) -> list[str]:
    """スコア順に銘柄を選択、セクター制限と相関フィルター付き。

    Args:
        candidates: スコアが算出された銘柄リスト
        scores: 銘柄→スコアのdict（高いほど良い）
        config: PortfolioConfig
        correlations: (sym_a, sym_b) → correlation のdict（省略可）
        max_correlation: 相関閾値

    Returns:
        選択された銘柄リスト（最大 config.max_positions 個）
    """
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
