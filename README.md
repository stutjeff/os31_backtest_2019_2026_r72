# OS 3.1 + V11-Core R7.2 Backtest｜2019–2026

這是把正式實戰版 OS 3.1 與 V11-Core R7.2 雷達結合後的回測。

## 核心資產

- 00662.TW：Nasdaq 100 ETF / 核心趨勢倉
- 00670L.TW：台股正2 / 波動倉
- 00865B.TW：短天期美債 / 防守倉

## 模式

- 452 = 45:25:30，平常作戰 / 中性偏進攻底盤
- 514 = 50:10:40，危機升溫 / 防守避震
- 433 = 40:30:30，R 模式確認 / 防守反擊

## OS 3.1 回測規則

1. 每週五收盤判斷一次模式。
2. 模式切換時立即再平衡。
3. 模式不變時，若任一資產權重偏離目標超過 5 個百分點，才再平衡。
4. 預設不計交易成本、稅、滑價。
5. 初始資金預設 5,000,000 元。
6. 無現金流、無借貸、無生活費提領。

## 跑法

```bash
pip install -r requirements.txt
python os31_r72_2019_2026_backtest.py
```

可選參數：

```bash
python os31_r72_2019_2026_backtest.py --start 2019-01-01 --end 2026-12-31 --initial-capital 5000000 --tolerance 0.05 --fee-bps 0
```

注意：若 end 設到未來，yfinance 只會下載到目前可取得的最新交易日。

## 輸出

- output/os31_r72_2019_2026_summary.md
- output/os31_r72_2019_2026_weekly_modes.csv
- output/os31_r72_2019_2026_equity_curve.csv
- output/os31_r72_2019_2026_switch_log.csv
- output/os31_r72_2019_2026_trades.csv
- output/os31_r72_2019_2026_comparison_curves.csv

## 判讀重點

- 2020 是否能防守後快速回攻
- 2022 是否避免熊市反彈亂切
- 2023–2026 AI 趨勢中是否能維持參與
- OS 3.1 是否比固定 452 / 514 / 433 更平衡
