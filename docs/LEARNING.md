# 学習ロードマップ

このコードを完全に理解＆改善していくための推奨学習順序。

## 1. 最優先（Phase 1実装に必要）

### 「ファイナンス機械学習」マルコス・ロペス・デ・プラド
- **章**: 第1〜10章を順に
- **特に**: 第3章（Triple Barrier）、第4章（Sample Weights）、第7章（CV）、第10章（Bet Sizing）
- 日本語訳: 金融財政事情研究会
- 原著: *Advances in Financial Machine Learning* (Wiley 2018)

→ Phase 1で実装する全機能の理論。**他の本より先にこれ**。

## 2. 並行学習（実装感覚）

### "Algorithmic Trading" Ernie Chan
- 平均回帰、モメンタム、コインテグレーション
- Pythonコード豊富、すぐ手を動かせる
- 1冊目だけで十分

### "Machine Learning for Asset Managers" López de Prado
- AFMLの短縮版
- 2-3日で読める。AFMLの導入として良い

## 3. 基礎固め

### 「証券投資論」日本証券アナリスト協会
- CAPM、Sharpe、ファクターモデル
- 体系的にポートフォリオ理論を学ぶ

### 「ウォール街のランダム・ウォーカー」
- 効率市場仮説の歴史
- クオンツの「謙虚さ」を学ぶ

## 4. 必読論文（無料）

| 論文 | ジャーナル | 重要ポイント |
|------|-----------|------------|
| Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum" | JFE | モメンタム戦略の決定版 |
| Asness, Moskowitz, Pedersen (2013) "Value and Momentum Everywhere" | JF | 全資産クラスでmomentum/valueの存在証明 |
| Gu, Kelly, Xiu (2020) "Empirical Asset Pricing via Machine Learning" | RFS | ML for finance のベンチマーク |
| López de Prado (2018) "10 Reasons Most ML Funds Fail" | JPM | 失敗パターン |
| Joubert (2022) "Meta-Labeling: Theory and Framework" | JFDS | Meta-Labeling実装 |

検索: SSRN、Google Scholar

## 5. 統計・時系列の補強（必要なら）

### 「時系列解析: 状態空間モデル」北川源四郎
- HMM/カルマンフィルタの数学的基礎

### "The Elements of Statistical Learning" Hastie/Tibshirani/Friedman
- 無料PDF: https://hastie.su.domains/ElemStatLearn/

## 6. オンラインリソース

| サイト | 用途 |
|-------|------|
| https://www.aqr.com/Insights | AQRの無料リサーチ |
| https://www.ssrn.com/ | 論文DB（無料登録） |
| https://arxiv.org/list/q-fin/recent | 最新プレプリント |
| https://hudsonthames.org/ | AFMLの実装解説（無料記事多数） |
| https://github.com/quantopian/research_public | Quantopian Lectures |

## 推奨学習スケジュール（8週間）

```
Week 1-2: 「ファイナンス機械学習」第1〜4章
          → Triple Barrier、Sample Weights を実装

Week 3-4: AFML 第5-6章 + Ernie Chan 1冊目
          → Fractional Differentiation 理解、戦略の感覚

Week 5-6: AFML 第7-10章
          → Purged CV、Bet Sizing 実装

Week 7-8: AFML 第11-15章 + 論文4本読破
          → バックテスト統計、評価指標を強化
```

並行: AQR の無料記事を週1本

## このシステムで実装予定の文献対応

| 機能 | 出典 | 章 |
|------|------|---|
| Triple Barrier | AFML | §3 |
| Sample Weights | AFML | §4 |
| Fractional Differentiation | AFML | §5 |
| Purged K-Fold | AFML | §7 |
| Combinatorial Purged CV | AFML | §7-8 |
| Meta-Labeling | AFML / Joubert | §3.6 / JFDS 2022 |
| Bet Sizing | AFML | §10 |
| Backtest Statistics | AFML | §14 |
| Volume/Dollar Bars | AFML | §2 |

## 一冊だけ選ぶなら

**「ファイナンス機械学習」（López de Prado, 日本語訳）**

これを読まずに次のフェーズには進めない。Phase 1の実装議論にも必須。
