# モジュール詳細リファレンス

ファイル単位での実装詳細。各クラス・関数の責務と依存関係。

## adapters/alpaca/

### `data.py` — `AlpacaDataClient`

NTの`LiveMarketDataClient`を継承した自作データクライアント。

#### コンストラクタ

```python
def __init__(self, loop, client_id, venue, msgbus, cache, clock,
             instrument_provider, config):
```
NTフレームワークから自動的に呼ばれる。`AlpacaLiveDataClientFactory.create()`経由。

#### 主要メソッド

| メソッド | 役割 |
|---------|------|
| `_connect()` | `StockDataStream` インスタンス作成 |
| `_disconnect()` | ストリーム停止、ソケットクローズ |
| `_subscribe_bars(command)` | 銘柄を `_subscribed_bars` に追加、デバウンスタイマー起動 |
| `_request_bars(request)` | REST API でヒストリカルバー取得、`_handle_bars()` で配信 |
| `_on_alpaca_bar(alpaca_bar)` | WebSocket受信 → NT `Bar` に変換 → `_handle_data(bar)` |
| `_schedule_stream_start()` | 2秒のデバウンスで `_start_stream` 起動 |
| `_run_stream()` | バックグラウンドで `stream.run()`、エラー時最大3回リトライ |

**設計上の制約**:
- NTのライブクライアントメソッドは**全てasync**でなければならない
- `_handle_data(data)` がNT MessageBusへの配信入口
- ストリームは1接続のみ（Alpaca無料プラン）

### `execution.py` — `AlpacaExecutionClient`

NTの`LiveExecutionClient`を継承。`alpaca-py`の`TradingClient`を内部で使う。

#### 主要メソッド

| メソッド | 役割 |
|---------|------|
| `_connect()` | アカウント情報取得、`generate_account_state()`で残高設定 |
| `_submit_order(command)` | NTの`SubmitOrder`コマンド → Alpaca注文へ変換 |
| `_submit_order_async(command)` | 実際の送信。成功時 `generate_order_submitted/accepted/filled` |
| `_cancel_order(command)` | 注文キャンセル |
| `_cancel_all_orders(command)` | 全キャンセル |
| `generate_*_reports()` | リコンサイル用（現状空実装） |

**重要**: `_set_account_id()` を `_connect` で必ず呼ぶ（呼ばないと `generate_account_state` がエラー）。

### `providers.py` — `AlpacaInstrumentProvider`

Alpaca APIから銘柄情報を取得し、NT `Equity` インスタンスを生成。

```python
provider.load_symbols(["AAPL", "MSFT"])  # 同期版
await provider.load_all_async()           # 非同期版（全銘柄）
```

### `factories.py`

NT `TradingNode` に登録するファクトリ。

```python
node.add_data_client_factory("ALPACA", AlpacaLiveDataClientFactory)
node.add_exec_client_factory("ALPACA", AlpacaLiveExecClientFactory)
```

## strategy/

### `momentum.py` — `MomentumStrategy`

12-1月リスク調整モメンタム + 相関考慮 + Risk Parity。

#### 内部状態

```python
self._regime: RegimeEnum                # 現在のレジーム
self._regime_kelly: float                # Kelly分率（regime由来）
self._cb_level: int                      # CBレベル
self._bars: dict[str, deque[Bar]]        # 銘柄ごとのバー履歴
self._stop_loss: dict[str, float]        # 銘柄ごとのストップ価格
self._target_weights: dict[str, float]   # Risk Parityで決まった目標ウェイト
self._last_rebalance_bar: int            # 最終リバランス時のbar_count
self._last_vol_avg: float                # 前回リバランス時の平均ATR/価格比
```

#### `_rebalance()` 処理フロー

```
1. 全銘柄のmomentum_scoreを計算
   (12-1月リターン) / (ATR/価格比)
2. 200日SMA上の銘柄のみ残す（トレンドフィルター）
3. 上位 top_pct (10%) を選出
4. 相関フィルター: 既選銘柄と相関 > 0.7 の銘柄を除外
5. Risk Parity ウェイト: w_i ∝ 1/vol_i, scaled by regime_kelly
6. 既存ポジションのうち selected に含まれない銘柄を決済
7. 新規エントリーのみ実行（既存はサイズ変更しない）
```

#### `_should_rebalance()` の3トリガー

```python
1. 時間: bars_since_rebalance >= 21
2. ボラスパイク: 平均ATR > 前回 × 2.0
3. ドリフト: ポジションウェイトが目標から75%以上乖離
```

最低間隔 `min_rebalance_interval_days: 10` で過剰リバランス抑止。

### `mean_reversion.py` — `MeanReversionStrategy`

短期平均回帰戦略。

#### エントリー条件（全て同時に満たす）

```python
RSI(14) < 30                # 売られすぎ
価格 < BB下限 (20日, 2σ)    # バンド下抜け
出来高 > 20日平均 × 1.5      # セリングクライマックス
価格 > 200日SMA              # 上昇トレンド内の押し目
```

#### エグジット

```python
RSI(14) > 50                 # 平均回帰完了
or 5営業日経過                # タイムアウト
or 価格 ≤ entry - 2*ATR      # ストップロス
```

### `ml_signal.py` — `MLSignalStrategy`

LightGBMモデルによる予測ベース戦略。

#### `on_start()`

```python
1. msgbus.subscribe で regime, CB を受信
2. モデルファイル (.pkl) をロード
3. ヒストリカルバーをプリロード（300日分）
4. 各銘柄のバー購読開始
```

#### `_check_ml_signal(sym)`

```python
1. 直近バーから DataFrame 作成
2. compute_features() で 14特徴量計算
3. predictor.predict_signal() で予測
4. confidence > buy_threshold (0.55) なら BUY
5. ATR から position size 計算
6. submit_order
```

## risk/circuit_breaker.py

### `CircuitBreakerActor`

5階層のリスク管理。**実現損益のみで判定**（含み損では発動しない）。

#### `on_start()`

```python
1. 口座エクイティを記録（peak/day/week/month start）
2. タイマー設定:
   - daily reset: 24時間ごと
   - weekly reset: 7日ごと
   - monthly reset: 30日ごと
```

#### `on_event(event)`

`PositionClosed` イベントのみで `_check_limits()` を呼ぶ。

#### `_check_limits()`

```python
if equity <= 0 or _start_of_day_equity <= 0:
    初期化のみ (誤発動防止)

L2: daily_return < -3%        → trigger(2, "Daily loss")
L2: daily_trades >= 20         → trigger(2, "Trade count")
L2: consecutive_losses >= 5    → trigger(2, "Consecutive losses")
L3: weekly_dd > 8%             → trigger(3, "Weekly DD")
L4: monthly_dd > 15%           → trigger(4, "Monthly DD") + halt
```

#### `_trigger(level, reason)`

```python
1. 既に同レベル以上ならスキップ
2. self._active_level 更新
3. msgbus.publish("circuit_breaker.update", data)
4. L4以上なら _halted = True
```

## regime/

### `hmm.py` — `RegimeActor`（ライブ用）

#### `on_start()`

```python
1. 5年分のマクロデータを yfinance から取得
   - VIX, SPY, セクターETF×11, 10Y/3M金利
2. 5特徴量にz-score正規化
3. GaussianHMM をフィット
4. 即座に推論し regime.update をpublish
5. タイマー設定: 6時間ごとに再推論
```

#### 5特徴量

```python
features = [
    vix_z,                      # VIX z-score (252日)
    volume_change_z,            # SPY 出来高 (20d/60d - 1)
    sector_dispersion_z,        # 11セクターETFの20日リターン標準偏差
    yield_curve_slope_z,        # 10Y - 3M
    credit_spread_z,            # VIX/10 (代用)
]
```

#### state → regime マッピング

訓練後、各HMM状態のVIX平均を見て:
- 最低 → BULL
- 中間 → NEUTRAL
- 最高 → BEAR

### `hmm_backtest.py` — `RegimeBacktestActor`

バックテスト用。違いは:
- `start_date` / `end_date` を config で指定
- 全期間のマクロデータを起動時に一括取得
- `_features_by_date` に事前計算してキャッシュ
- `on_bar` で `bar.ts_event` に対応する features を引いて推論
- 確信度フィルター + 持続性フィルター（5日連続で初めて切替）

## ml/

### `features.py`

#### `compute_features(df) -> DataFrame`

入力: `open, high, low, close, volume` カラムを持つ DataFrame
出力: 14特徴量カラム追加

| 特徴量 | 計算式 |
|-------|------|
| `rsi_14`, `rsi_28` | 標準RSI |
| `bb_width` | (upper - lower) / sma20 |
| `bb_pctb` | (close - lower) / (upper - lower) |
| `atr_14_norm` | ATR(14) / close |
| `ema_cross_5_20` | (EMA5 - EMA20) / EMA20 |
| `sma_cross_10_50` | (SMA10 - SMA50) / SMA50 |
| `volume_ratio` | volume / vol_ma20 |
| `roc_10` | close.pct_change(10) |
| `vol_20`, `vol_60` | returns.rolling(N).std() |
| `momentum_60`, `momentum_120` | close.pct_change(N) |
| `high_low_range_5` | (high.rolling5.max - low.rolling5.min) / close |

#### `make_target(df, horizon=5, threshold=0.01) -> Series`

```python
future_return = close[+horizon] / close - 1
target = 2 if future_return > threshold       # BUY
target = 0 if future_return < -threshold      # SELL
target = 1 otherwise                           # HOLD
```

### `trainer.py`

#### `fetch_training_data(symbols, start, end) -> DataFrame`

期間指定で複数銘柄の OHLCV を取得（連結）。

#### `build_dataset(raw, horizon, threshold) -> (X, y)`

各銘柄ごとに `compute_features` + `make_target` を実行、結果を連結。
NaN行は除外。

#### `walk_forward_train(X, y, n_splits=5)`

`TimeSeriesSplit` で 5fold walk-forward CV。各foldで:
1. 訓練→検証
2. early stopping (50ラウンド)
3. accuracy記録

最後に全データで final model 訓練。

#### `save_model(model, metrics, path, train_start, train_end)`

pickle bundle:
```python
{
    "model": LGBMClassifier,
    "feature_cols": [...],
    "metrics": {"cv_accuracy_mean": ..., ...},
    "trained_at": ISO8601,
    "train_start": "2015-01-01",
    "train_end": "2022-12-31",
    "version": "1.1",
}
```

### `predictor.py` — `MLPredictor`

```python
predictor = MLPredictor("models/lgbm_signal_v1.pkl")
predictor.train_end          # "2022-12-31" → リーク検出に使う
predictor.metrics            # CV結果
predictor.predict_proba(features)             # → ndarray (N, 3)
predictor.predict_signal(row, buy_threshold)  # → ("buy", 0.62)
```

## monitoring/metrics_actor.py

### Prometheus Gauge / Counter 一覧

```python
# 口座
equity, cash, unrealized_pnl, realized_pnl_total, daily_pnl, drawdown

# 取引統計
fills_total{side, strategy}     # Counter
orders_submitted_total{side}     # Counter
trades_won, trades_lost          # Counter
win_rate, avg_win, avg_loss      # Gauge
last_trade_pnl                   # Gauge
slippage_bps, fill_value_usd     # Histogram

# ポジション
open_positions, total_exposure

# レジーム/CB
regime, regime_*_prob (4種), cb_level

# システム
uptime_seconds, engine_info
```

### イベント処理

#### `on_event(event)`

```python
- OrderFilled        → fills_total++, fill_value記録
- PositionClosed     → 実現P&L累積、win/loss統計更新
- AccountState       → equity, cash, drawdown更新
```

#### `_on_timer()`

10秒ごと: `_sync_account()` + `_sync_positions()` + uptime更新

## api/control.py

### FastAPI エンドポイント実装

各エンドポイントは `_client: TradingClient` (alpaca-py) を介して直接Alpaca APIを叩く。
NT MessageBus とは独立（手動操作のため）。

#### `start_api_server(client, port)`

```python
threading.Thread でバックグラウンド起動。
daemon=True でメインプロセス終了時に自動終了。
```

## data/questdb_writer.py

### `QuestDBWriterActor`

#### ILP（InfluxDB Line Protocol）

```
bars,symbol=AAPL open=150.0,high=152.0,low=149.0,close=151.0,volume=1000000 1735689600000000000
```

`socket.sendall()` で直接 TCP 書込。HTTP より高速。

#### `_preload_history()`

起動時に各銘柄400日分を一括投入。Grafanaが即座にチャート表示できるようにするため。

## scripts/

### `run_paper_trade.py` フロー

```python
1. 環境変数読込（.env）
2. Prometheus サーバー起動（:8000）
3. FastAPI 起動（:8001）
4. AlpacaInstrumentProvider 銘柄ロード
5. TradingNodeConfig 構築
6. Strategies / Actors 追加（順序重要: Metrics → CB → Regime → ...）
7. node.build() → node.run()  # ブロッキング
```

### `run_backtest.py` フロー

```python
1. ML モデルの train_end と backtest start を比較（リーク検出）
2. BacktestEngine 構築
3. Venue 追加（cash account）
4. yfinance から各銘柄のバー取得 → engine.add_data()
5. RegimeBacktestActor、CB追加
6. Strategies 追加
7. engine.run()
8. レポート出力 + matplotlib チャート保存
```

### `train_model.py` フロー

```python
1. 引数解析（--start, --end, --horizon, --threshold）
2. fetch_training_data(symbols, start, end)
3. build_dataset(raw) → (X, y)
4. walk_forward_train(X, y) → (model, metrics)
5. save_model(model, metrics, path, train_start, train_end)
```

## デバッグ・ログ確認

### ログレベル

NTログは色付きで出力:
- `[INFO]` 通常情報
- `[WARN]` 黄色 — 注意
- `[ERROR]` 赤 — エラー
- `[CRITICAL]` 太赤 — 致命的

### よく見るログ

```bash
# レジーム変化
grep "Regime changed" /var/log/kryor.log

# 約定
grep "FILL\|OrderFilled" /var/log/kryor.log

# CB発動
grep "Circuit Breaker\|CB L" /var/log/kryor.log

# ストップロス
grep "STOP LOSS" /var/log/kryor.log

# ML予測
grep "ML BUY\|ML EXIT" /var/log/kryor.log
```

サーバー側:
```bash
ssh root@204.168.224.92 "journalctl -u kryor-engine -f"
```

## テスト

```bash
.venv/bin/python -m pytest tests/ -v
```

`tests/test_models.py` — データモデル基本テスト。

## 開発ワークフロー

```bash
# 変更 → ローカルテスト
.venv/bin/python scripts/run_backtest.py --years 1

# OK ならコミット
git add -A && git commit -m "..."
git push

# サーバーへデプロイ
ssh root@204.168.224.92 "cd /opt/trade-engine && git pull && .venv/bin/pip install -e . && systemctl restart kryor-engine"
```

または:
```bash
cd deploy && ansible-playbook site.yml
```
