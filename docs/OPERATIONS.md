# 運用ガイド

## サーバー情報

- **ホスト**: 204.168.224.92 (Hetzner Cloud, Ashburn US)
- **スペック**: 4vCPU / 8GB RAM / 80GB SSD / ARM64
- **OS**: Ubuntu 24.04
- **コスト**: €8/月

## アクセスURL

| サービス | URL | 認証 |
|---------|-----|------|
| Grafana | http://204.168.224.92:3000 | admin / 自動生成パスワード |
| Trade API | http://204.168.224.92:8001/docs | なし（オープン） |

Grafanaパスワード確認:
```bash
ssh root@204.168.224.92 "grep GRAFANA_PASSWORD /opt/trade-engine/.env"
```

## SSH接続

```bash
ssh root@204.168.224.92
```

## ログ

### リアルタイム追従
```bash
ssh root@204.168.224.92 "journalctl -u kryor-engine -f"
```

### 直近100行
```bash
ssh root@204.168.224.92 "journalctl -u kryor-engine -n 100 --no-pager"
```

### フィルター例
```bash
# 約定のみ
ssh root@204.168.224.92 "journalctl -u kryor-engine | grep -i 'fill\|order'"

# エラーのみ
ssh root@204.168.224.92 "journalctl -u kryor-engine | grep -i 'error'"

# レジーム変化
ssh root@204.168.224.92 "journalctl -u kryor-engine | grep -i 'regime'"
```

## エンジン操作

### 状態確認
```bash
ssh root@204.168.224.92 "systemctl status kryor-engine"
```

### 再起動
```bash
ssh root@204.168.224.92 "systemctl restart kryor-engine"
```

### 停止 / 起動
```bash
ssh root@204.168.224.92 "systemctl stop kryor-engine"
ssh root@204.168.224.92 "systemctl start kryor-engine"
```

### Docker サービス
```bash
ssh root@204.168.224.92 "cd /opt/trade-engine && docker compose ps"
ssh root@204.168.224.92 "cd /opt/trade-engine && docker compose restart grafana"
```

## デプロイ

### コード更新（手動）
```bash
ssh root@204.168.224.92 "cd /opt/trade-engine && git pull && .venv/bin/pip install -e . && systemctl restart kryor-engine"
```

### Ansibleで再デプロイ
```bash
cd ~/work/studio-kryor/trade-engine/deploy
ansible-playbook site.yml
```

### MLモデル更新

ローカルで再訓練 → サーバーにコピー:
```bash
# ローカル
.venv/bin/python scripts/train_model.py --start 2015-01-01 --end 2022-12-31
git add models/lgbm_signal_v1.pkl
git commit -m "Retrain ML model"
git push

# サーバー
ssh root@204.168.224.92 "cd /opt/trade-engine && git pull && systemctl restart kryor-engine"
```

## トラブルシューティング

### エンジンが起動しない
```bash
ssh root@204.168.224.92 "journalctl -u kryor-engine -n 50 --no-pager | grep -i error"
```

### Alpaca接続エラー（connection limit exceeded）
- 別プロセスがWebSocketを掴んでいる
- 1-2分待ってから再起動
- `systemctl restart kryor-engine`

### Grafanaにデータが出ない
1. メトリクスエンドポイント確認:
   ```bash
   curl http://204.168.224.92:8000/metrics | grep kryor_
   ```
2. Prometheusのターゲット確認:
   ```bash
   curl http://204.168.224.92:9090/api/v1/targets
   ```
3. Grafanaでデータソース接続テスト

### QuestDB接続エラー
```bash
ssh root@204.168.224.92 "docker compose -f /opt/trade-engine/docker-compose.yml restart questdb"
```

### サーバーディスク不足
```bash
ssh root@204.168.224.92 "df -h"
ssh root@204.168.224.92 "docker system prune -af"  # 不要イメージ削除
```

## モニタリングのポイント

### 毎日チェック
- Grafana 「Equity」「Daily P&L」
- Grafana 「Circuit Breaker」 → OK以外なら要対応

### 毎週チェック
- 「Win Rate」 → 50%以下が続くなら戦略見直し
- 「Drawdown」 → 5%超えで警戒
- 「Open Positions」 → 過剰に多くないか

### 毎月チェック
- ヒストリカルチャートで実績推移
- レジーム分布（BULL/NEUTRAL/BEAR の割合）
- MLモデルの予測精度（外で計算）

## 緊急時対応

### 全ポジション即時決済
```bash
curl -X POST http://204.168.224.92:8001/api/close-all
```

または Grafana ダッシュボード「CLOSE ALL」ボタン

### 取引一時停止（ポジションは維持）
```bash
curl -X POST http://204.168.224.92:8001/api/pause
```

### 完全停止
```bash
ssh root@204.168.224.92 "systemctl stop kryor-engine"
```

## コスト管理

| 項目 | 月額 |
|------|------|
| Hetzner CX32 | €8 (~¥1,300) |
| Alpaca Paper Trading | $0 |
| GitHub | $0 |
| **合計** | **約¥1,300/月** |

将来の追加可能性:
- Alpaca Pro Data: $9/月 (リアルタイムSIP)
- VectorBT PRO: $20/月

## バックアップ

### モデル
`models/*.pkl` は Git に含める（小さいので）

### データベース
QuestDB / PostgreSQL のデータは Docker volume に保存。
重要なら定期 `docker volume export` を検討（現状未実装）。

### コード
GitHub: https://github.com/thecatblake/kryor-trade-engine

## セキュリティ

- SSH鍵認証のみ（パスワードログイン無効化推奨）
- UFWで開放ポート: 22 (SSH), 3000 (Grafana), 8001 (Trade API)
- Trade API は**認証なし**で全公開 → 必要なら Basic認証追加

### Trade APIに認証を追加するには（推奨）

`api/control.py` に:
```python
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()
def auth(creds: HTTPBasicCredentials = Depends(security)):
    if creds.username != "admin" or creds.password != os.environ["API_PASSWORD"]:
        raise HTTPException(401)

@app.post("/api/trade", dependencies=[Depends(auth)])
def submit_trade(...): ...
```
