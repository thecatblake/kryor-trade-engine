# KRYOR Trade Engine — アーキテクチャ解説

## 全体像

NautilusTrader（NT）を中核に据えたイベント駆動型トレーディングシステム。
バックテストとライブ取引で**同一の戦略コードが動く**設計。

```
┌─────────────────────────────────────────────────────────────┐
│  TradingNode (ライブ) / BacktestEngine (バックテスト)        │
│                                                             │
│  ┌──────────────┐    ┌──────────────┐                      │
│  │ Data Client  │    │ Exec Client  │                      │
│  │ (Alpaca)     │    │ (Alpaca)     │                      │
│  └──────┬───────┘    └──────┬───────┘                      │
│         │                   │                              │
│         ▼                   ▼                              │
│  ┌──────────────────────────────┐                          │
│  │  MessageBus (NT内部 Pub/Sub) │                          │
│  └──────────────┬───────────────┘                          │
│                 │                                          │
│  ┌──────┬───────┼────────┬───────┐                         │
│  ▼      ▼       ▼        ▼       ▼                         │
│ Strategy Strategy  Actor   Actor  Actor                    │
│ (MOM)    (MR/ML)   (Regime)(CB)   (Metrics/QuestDB)        │
│                                                            │
└─────────────────────────────────────────────────────────────┘
                  │                  │
                  ▼                  ▼
          ┌───────────────┐    ┌───────────────┐
          │ QuestDB (時系列)│    │ Prometheus    │
          │ PostgreSQL     │    │ → Grafana     │
          │ Redis          │    │ → Trade API   │
          └───────────────┘    └───────────────┘
```

## ディレクトリ構造

```
trade-engine/
├── src/kryor/
│   ├── adapters/alpaca/    ← Alpaca連携（NT Adapter自作）
│   ├── api/                ← FastAPI 取引コントロールAPI
│   ├── core/               ← 共通モデル、設定、カスタムデータ型
│   ├── data/               ← OHLCV取得、QuestDB書込、特徴量計算
│   ├── execution/          ← 執行（NT組込で実質空）
│   ├── ml/                 ← LightGBM 訓練・推論
│   ├── monitoring/         ← Prometheusメトリクス
│   ├── regime/             ← HMM レジーム検出（ライブ/BT別実装）
│   ├── risk/               ← サーキットブレーカー
│   └── strategy/           ← 戦略実装
├── scripts/
│   ├── run_paper_trade.py  ← ライブ起動エントリポイント
│   ├── run_backtest.py     ← バックテスト+チャート生成
│   ├── train_model.py      ← MLモデル訓練
│   └── fetch_historical.py ← データダウンロード
├── deploy/                 ← Ansibleプレイブック（Hetzner）
├── config/                 ← Prometheus/Grafana 設定
└── models/                 ← 訓練済みLightGBMモデル（.pkl）
```

## 主要モジュール解説

### 1. Adapters — Alpaca連携 (`adapters/alpaca/`)

NTにはAlpacaの公式アダプタが存在しないため**自作**している。

| ファイル | 役割 |
|---------|------|
| `config.py` | `AlpacaDataClientConfig` / `AlpacaExecClientConfig` (msgspec) |
| `constants.py` | `ALPACA_VENUE = Venue("ALPACA")` |
| `data.py` | `LiveMarketDataClient` 実装（WebSocketバー受信） |
| `execution.py` | `LiveExecutionClient` 実装（注文送信、約定通知） |
| `factories.py` | TradingNode に登録するファクトリ |
| `providers.py` | `InstrumentProvider`（銘柄情報を Alpaca から取得） |
| `history.py` | バックテスト/プリロード用ヒストリカルバー取得 |

**重要な設計判断**:
- WebSocket接続は**1つだけ**（Alpaca無料プランの制限）
- `_subscribe_bars` は銘柄を蓄積するだけ、実ストリーム起動は2秒のデバウンス後
- `connection limit exceeded` 時は**最大3回30秒間隔**でリトライ

### 2. Core — データモデル (`core/`)

| ファイル | 内容 |
|---------|------|
| `models.py` | `OrderRequest`, `OrderResult`, `Position`, `PortfolioState` 等のデータクラス |
| `settings.py` | Pydantic BaseSettings (`.env`から読み込み) |
| `events.py` | 旧自作 EventBus（NT MessageBus 採用後はほぼ未使用） |
| `custom_data.py` | `RegimeData`, `CircuitBreakerData`（msgbus で配信） |

**ポイント**: NTのCython製`Data`基底クラスは継承しにくいので、`@dataclass`で実装し`msgbus.publish("topic", obj)`で配信する設計。

### 3. Data — データレイヤー (`data/`)

| ファイル | 役割 |
|---------|------|
| `fetcher.py` | yfinance ラッパー（OHLCV、マクロデータ） |
| `indicators.py` | RSI/BB/ATR/EMA等のテクニカル指標 |
| `store.py` | PostgreSQL (SQLAlchemy) + Redis 接続 |
| `questdb_writer.py` | QuestDB Actor — ILPプロトコルで価格バー書込 |

**QuestDBへの書込**:
- ライブバーが届くたびに`_write_bar()` 呼び出し
- ILP形式: `bars,symbol=AAPL open=150,high=152,... timestamp_ns`
- Grafana が QuestDB を PgWire 経由で読みチャート表示

### 4. Strategy — 戦略 (`strategy/`)

| ファイル | 戦略 | 起点 |
|---------|------|------|
| `momentum.py` | 12-1月モメンタム + Risk Parity | 月次リバランス + ボラスパイク |
| `mean_reversion.py` | RSI<30 + BB下抜け + 出来高 | 日次バー |
| `ml_signal.py` | LightGBM 3クラス分類 | 日次バー |

**全戦略の共通インターフェース** (NT `Strategy`):
- `on_start()` — 起動時セットアップ（MLモデルロード、ヒストリカル取得）
- `on_bar(bar)` — バー受信時メイン処理
- `on_stop()` — 停止時クリーンアップ

**msgbus購読**:
- `regime.update` → BEARで全決済、Kelly分率変更
- `circuit_breaker.update` → L3+で全決済、L2で新規停止

### 5. Risk — リスク管理 (`risk/`)

`circuit_breaker.py` の `CircuitBreakerActor`:

| Level | 条件 | 動作 |
|-------|------|------|
| L1 | 1トレード > 2% | その注文を拒否 |
| L2 | 日次損失 > 3% | 当日新規停止 |
| L3 | 週次DD > 8% | 全戦略停止、ポジション縮小 |
| L4 | 月次DD > 15% | **全ポジション即時クローズ** |
| L5 | 注文レート異常 | 異常検知 |

**重要**: バックテストでは含み損で誤発動するので、**実現損益（PositionClosed）でのみ判定**するよう実装。

### 6. Regime — レジーム検出 (`regime/`)

| ファイル | 用途 |
|---------|------|
| `hmm.py` | ライブ用 RegimeActor（yfinance から定期取得） |
| `hmm_backtest.py` | バックテスト用（事前取得した全データから時刻順に推論） |

**HMM 5特徴量**:
1. VIX z-score（恐怖指数）
2. SPY 出来高変化率（20日/60日）
3. セクターETF分散（11セクター）
4. イールドカーブ勾配（10Y - 3M）
5. 信用スプレッド（VIX/10 で代用）

**3状態判定**:
- VIX z-score 最低 → BULL
- 中間 → NEUTRAL
- 最高 → BEAR

**ノイズ対策**:
- `min_confidence: 0.7` — 確信度70%未満は据え置き
- `min_persistence_days: 5` — 同じregimeが5日続いてから確定

### 7. ML — 機械学習 (`ml/`)

| ファイル | 役割 |
|---------|------|
| `features.py` | 14テクニカル特徴量 + 5日後リターンターゲット |
| `trainer.py` | LightGBM 3クラス分類器、Walk-forward CV |
| `predictor.py` | 推論インターフェース、確率付き予測 |

**ターゲット定義**:
```
2 (BUY)  if (close[+5d] / close - 1) > +1%
0 (SELL) if (close[+5d] / close - 1) < -1%
1 (HOLD) otherwise
```

**14特徴量**:
- RSI(14, 28), BB幅/%B, ATR/価格比
- EMA(5,20)cross, SMA(10,50)cross, 出来高比率, ROC(10)
- ボラ(20, 60), モメンタム(60, 120), 高安レンジ(5)

**訓練/テストの分離**（リーク防止）:
- 訓練: 2015-01-01 〜 2022-12-31
- バックテスト: 2023-01-01 〜現在
- `run_backtest.py` で自動チェック、重複時は警告

### 8. Monitoring — 監視 (`monitoring/`)

`metrics_actor.py` の `MetricsActor`:

**Prometheusメトリクス**:
- 口座: equity, cash, drawdown, daily/realized P&L
- ポジション: open_positions, total_exposure, unrealized_pnl
- 取引: fills_total{side, strategy}, win_rate, avg_win/loss
- レジーム: regime, regime_*_prob
- システム: uptime, cb_level

`:8000/metrics` で公開 → Prometheus が15秒ごとスクレイプ → Grafana表示

### 9. API — 取引制御 (`api/`)

`control.py` の FastAPI サーバー（`:8001`）:

| エンドポイント | 機能 |
|---------------|------|
| `POST /api/trade` | 手動売買 |
| `POST /api/close` | 銘柄指定決済 |
| `POST /api/close-all` | 全決済 |
| `POST /api/cancel-all` | 全注文キャンセル |
| `POST /api/pause` / `resume` | 自動取引一時停止/再開 |
| `GET /api/status` | アカウント+ポジション一覧 |
| `GET /api/positions` / `orders` | 詳細取得 |

Grafanaダッシュボードに直接ボタンとして埋め込み済み。

## データフロー

### ライブ取引時

```
Alpaca WebSocket
     │
     ▼ Bar
AlpacaDataClient._on_alpaca_bar()
     │
     ▼ self._handle_data(bar)
NT MessageBus
     │
     ├─→ MomentumStrategy.on_bar()
     ├─→ MeanReversionStrategy.on_bar()
     ├─→ MLSignalStrategy.on_bar()
     ├─→ RegimeActor.on_bar() (SPYのみ)
     └─→ QuestDBWriterActor.on_bar()

戦略内で submit_order() 呼ばれる
     │
     ▼
NT RiskEngine → ExecutionEngine
     │
     ▼
AlpacaExecutionClient._submit_order()
     │
     ▼
Alpaca REST API
     │
     ▼ OrderFilled イベント
MessageBus → MetricsActor / Strategies の on_event()
```

### バックテスト時

```
yfinance から事前取得したヒストリカルバー
     │
     ▼ engine.add_data(bars)
BacktestEngine.run()
     │
     ▼ ts_init 順にバーを再生
DataEngine → MessageBus
     │
     ├─→ Strategies (上と同じ)
     └─→ Actors (上と同じ)

注文 → SimulatedExchange (NT組込)
     │
     ▼ 即時/次バー約定
PositionClosed → MetricsActor で集計
     │
     ▼
matplotlib で4枚チャート出力 → data/backtest_*.png
```

## msgbus 通信規約

### Topic: `regime.update`

**発信元**: `RegimeActor` / `RegimeBacktestActor`
**ペイロード**: `RegimeData`
**購読者**: 全Strategy

```python
@dataclass
class RegimeData:
    regime: RegimeEnum  # BULL / NEUTRAL / BEAR
    probability: float
    bull_prob, neutral_prob, bear_prob: float
    duration_days: int
    ts_event, ts_init: int
```

### Topic: `circuit_breaker.update`

**発信元**: `CircuitBreakerActor`
**ペイロード**: `CircuitBreakerData(level: int, reason: str)`
**購読者**: 全Strategy + MetricsActor

## 起動シーケンス（ライブ）

```
1. .env から API キー読込
2. Prometheus HTTP server 起動（:8000）
3. FastAPI サーバー起動（:8001）
4. AlpacaInstrumentProvider で銘柄情報ロード
5. TradingNode 構築
   - DataClient/ExecClient ファクトリ登録
6. Strategies/Actors 追加
   - MetricsActor (1番目, 他のpublish前にsubscribe必要)
   - CircuitBreakerActor
   - RegimeActor → 起動時HMMフィット
   - QuestDBWriterActor → ヒストリカルプリロード
   - MomentumStrategy → ヒストリカルプリロード
   - MeanReversionStrategy
   - MLSignalStrategy → モデルファイルロード
7. node.build() → node.run()
8. Alpaca WebSocket 接続
9. バー受信ループ開始
```

## 戦略の追加方法

新戦略を追加するには `src/kryor/strategy/` に新ファイル作成:

```python
from nautilus_trader.config import StrategyConfig
from nautilus_trader.trading.strategy import Strategy
from kryor.core.custom_data import RegimeData, RegimeEnum


class MyStrategyConfig(StrategyConfig, frozen=True):
    symbols: list[str] = []
    # ... カスタムパラメータ


class MyStrategy(Strategy):
    def __init__(self, config: MyStrategyConfig) -> None:
        super().__init__(config=config)
        self._config = config

    def on_start(self) -> None:
        self.msgbus.subscribe("regime.update", self._on_regime)
        # 銘柄ごとにバー購読
        for sym in self._config.symbols:
            instrument_id = InstrumentId.from_str(f"{sym}.ALPACA")
            bar_type = BarType(instrument_id, BarSpecification(1, BarAggregation.DAY, PriceType.LAST), AggregationSource.EXTERNAL)
            self.subscribe_bars(bar_type)

    def on_bar(self, bar) -> None:
        # シグナル判定 → submit_order()
        ...

    def _on_regime(self, data: RegimeData) -> None:
        # レジーム変化に応じた処理
        ...
```

`run_paper_trade.py` と `run_backtest.py` で `node.trader.add_strategy(MyStrategy(...))` を追加。

## デプロイ

### ローカル開発
```bash
docker compose up -d
.venv/bin/python scripts/run_paper_trade.py
```

### 本番（Hetzner）
```bash
cd deploy/
ansible-playbook site.yml  # 全自動デプロイ
```

サーバー側はsystemdで自動起動・自動再起動。

## 監視

| URL | 用途 |
|-----|------|
| http://204.168.224.92:3000 | Grafana ダッシュボード |
| http://204.168.224.92:8001/docs | Trade API（Swagger UI） |
| http://204.168.224.92:9090 | Prometheus（直接） |

ログ: `journalctl -u kryor-engine -f`

## 既知の問題と今後の改善

### 既知の問題

1. **MLストラテジーがノイズに過剰反応**
   - 6,770トレード/3年 → 異常に多い
   - 対策: Triple Barrier Method、Meta-Labeling（Phase 1で実装予定）

2. **MomentumとMLの特徴量重複**
   - vol_60, momentum_60/120 が両方で使われる
   - 対策: 相関チェック、別sleeve化

3. **HMMが起動時のみフィット**
   - 6時間ごとに再評価するが再フィットはしない
   - 対策: 月次再フィット予定

4. **バックテストのDrawdown表示が異常**
   - 評価額（含み損益込み）の瞬時表示で振動
   - 実体は実現P&Lで判断する

### 改善ロードマップ

詳細は `docs/ROADMAP.md`（未作成）参照予定。

**Phase 1（実装中）**:
- Triple Barrier Method
- Sample Uniqueness Weights
- Probabilistic Bet Sizing
- Calibrated Probabilities

**Phase 2**:
- Meta-Labeling
- Purged K-Fold CV
- Combinatorial Purged CV

**Phase 3**:
- Regime-Aware Models
- Information Bars (Volume/Dollar Bars)
- Fractional Differentiation

## 参考文献

実装の理論的基礎:

1. **López de Prado (2018)** *Advances in Financial Machine Learning* — メイン参考書
2. **Moskowitz, Ooi, Pedersen (2012)** "Time Series Momentum"
3. **Asness, Moskowitz, Pedersen (2013)** "Value and Momentum Everywhere"
4. **Gu, Kelly, Xiu (2020)** "Empirical Asset Pricing via Machine Learning"

詳細は `docs/LEARNING.md` （学習ロードマップ）参照。
