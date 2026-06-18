#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OS 3.1 + V11-Core R7.2 Backtest
Period default: 2019-01-01 to today / user-provided end

Core portfolio:
- 00662.TW  = Nasdaq 100 ETF / 核心趨勢倉
- 00670L.TW = 台股正2 / 波動倉
- 00865B.TW = 短天期美債 / 防守倉

Battle modes:
- 452 = 45:25:30
- 514 = 50:10:40
- 433 = 40:30:30

OS 3.1 practical rules in this backtest:
1. Weekly radar decision using Friday close.
2. Immediate rebalance on mode switch.
3. If mode is unchanged, rebalance only when any asset weight deviates from target by >= tolerance.
4. Default tolerance = 5 percentage points.
5. No leverage borrowing, no tax, no transaction cost by default.
6. Optional transaction cost can be set by --fee-bps.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple
import math
import numpy as np
import pandas as pd
import yfinance as yf


DEFAULT_START = "2019-01-01"
DEFAULT_END = pd.Timestamp.today().strftime("%Y-%m-%d")
INITIAL_CAPITAL = 5_000_000.0

ASSETS = ["00662.TW", "00670L.TW", "00865B.TW"]
ASSET_LABELS = {
    "00662.TW": "00662 Nasdaq ETF",
    "00670L.TW": "00670L Taiwan 2x ETF",
    "00865B.TW": "00865B Short US Treasury ETF",
}

# Radar proxies
RADAR_TICKERS = [
    "QQQ", "SOXX", "SMH",
    "HYG", "LQD",
    "SPY", "RSP", "IWM", "SHY",
    "^VIX",
]

TICKERS = sorted(set(ASSETS + RADAR_TICKERS))

MODE_TARGETS = {
    "452": {"00662.TW": 0.45, "00670L.TW": 0.25, "00865B.TW": 0.30},
    "514": {"00662.TW": 0.50, "00670L.TW": 0.10, "00865B.TW": 0.40},
    "433": {"00662.TW": 0.40, "00670L.TW": 0.30, "00865B.TW": 0.30},
}


def ma(s: pd.Series, window: int) -> pd.Series:
    return s.rolling(window=window, min_periods=max(5, window // 4)).mean()


def safe_pct_change(series: pd.Series, periods: int) -> pd.Series:
    return series.pct_change(periods=periods).replace([np.inf, -np.inf], np.nan)


def download_prices(start: str, end: str) -> pd.DataFrame:
    # Buffer for 200D MA and 252D drawdown.
    download_start = (pd.Timestamp(start) - pd.Timedelta(days=460)).strftime("%Y-%m-%d")
    download_end = (pd.Timestamp(end) + pd.Timedelta(days=5)).strftime("%Y-%m-%d")
    raw = yf.download(
        TICKERS,
        start=download_start,
        end=download_end,
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )
    if raw.empty:
        raise RuntimeError("yfinance returned no data. Check internet access or ticker availability.")

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"].copy()
        elif "Adj Close" in raw.columns.get_level_values(0):
            close = raw["Adj Close"].copy()
        else:
            raise RuntimeError(f"Could not find Close / Adj Close in columns: {raw.columns}")
    else:
        close = raw[["Close"]].copy()

    close = close.sort_index().ffill()

    # SOXX sometimes has sparse data issues; use SMH to fill SOXX when needed.
    if "SOXX" in close.columns and "SMH" in close.columns:
        close["SOXX"] = close["SOXX"].fillna(close["SMH"])

    missing_assets = [t for t in ASSETS if t not in close.columns or close[t].dropna().empty]
    if missing_assets:
        raise RuntimeError(f"Missing core ETF price data: {missing_assets}")

    missing_radar = [t for t in ["QQQ", "SOXX", "HYG", "LQD", "SPY", "RSP", "IWM", "SHY", "^VIX"] if t not in close.columns or close[t].dropna().empty]
    if missing_radar:
        raise RuntimeError(f"Missing radar proxy data: {missing_radar}")

    return close


def to_weekly(close: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    weekly = close.resample("W-FRI").last().ffill()
    weekly = weekly[(weekly.index >= pd.Timestamp(start)) & (weekly.index <= pd.Timestamp(end))]
    return weekly.dropna(subset=ASSETS, how="any")


def score_market_momentum(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    for ticker in ["QQQ", "SOXX"]:
        s = df[ticker]
        score += np.where(s < ma(s, 20), 5, 0)
        score += np.where(s < ma(s, 60), 8, 0)
        score += np.where(s < ma(s, 200), 12, 0)
    return pd.Series(score, index=df.index).clip(upper=25)


def score_credit_proxy(df: pd.DataFrame) -> pd.Series:
    ratio = df["HYG"] / df["LQD"]
    score = pd.Series(0.0, index=df.index)
    score += np.where(ratio < ma(ratio, 20), 5, 0)
    score += np.where(ratio < ma(ratio, 60), 8, 0)
    score += np.where(ratio < ma(ratio, 200), 12, 0)
    dd_90 = ratio / ratio.rolling(90, min_periods=20).max() - 1.0
    score += np.where(dd_90 < -0.08, 8, 0)
    return pd.Series(score, index=df.index).clip(upper=25)


def score_breadth(df: pd.DataFrame) -> pd.Series:
    score = pd.Series(0.0, index=df.index)
    # Equal-weight / broad market weakness.
    rsp_spy = df["RSP"] / df["SPY"]
    iwm_spy = df["IWM"] / df["SPY"]
    score += np.where(rsp_spy < ma(rsp_spy, 60), 8, 0)
    score += np.where(iwm_spy < ma(iwm_spy, 60), 8, 0)
    score += np.where(df["SPY"] < ma(df["SPY"], 200), 10, 0)
    return pd.Series(score, index=df.index).clip(upper=20)


def score_vix(df: pd.DataFrame) -> pd.Series:
    vix = df["^VIX"]
    score = pd.Series(0.0, index=df.index)
    score += np.where(vix > 20, 5, 0)
    score += np.where(vix > 25, 8, 0)
    score += np.where(vix > 35, 15, 0)
    score += np.where(vix > 50, 20, 0)
    return pd.Series(score, index=df.index).clip(upper=20)


def score_defensive_strength(df: pd.DataFrame) -> pd.Series:
    # Defensive asset strength: when SHY/Treasury trends outperform risk assets, risk regime is stronger.
    shy_spy = df["SHY"] / df["SPY"]
    score = pd.Series(0.0, index=df.index)
    score += np.where(shy_spy > ma(shy_spy, 60), 8, 0)
    score += np.where(shy_spy > ma(shy_spy, 200), 10, 0)
    return pd.Series(score, index=df.index).clip(upper=15)


def compute_components(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    out["market_momentum_score"] = score_market_momentum(df)
    out["credit_proxy_score"] = score_credit_proxy(df)
    out["breadth_score"] = score_breadth(df)
    out["vix_score"] = score_vix(df)
    out["defensive_strength_score"] = score_defensive_strength(df)
    # Same 0-100 style total score used in event backtests.
    out["total_score"] = (
        out["market_momentum_score"]
        + out["credit_proxy_score"]
        + out["breadth_score"]
        + out["vix_score"]
        + out["defensive_strength_score"]
    ).clip(upper=100)
    return out


def compute_r72_conditions(df: pd.DataFrame, components: pd.DataFrame) -> pd.DataFrame:
    conds = pd.DataFrame(index=df.index)

    qqq = df["QQQ"]
    soxx = df["SOXX"]
    credit = df["HYG"] / df["LQD"]
    vix = df["^VIX"]

    conds["r_qqq_above_ma20"] = qqq > ma(qqq, 20)
    conds["r_soxx_above_ma20"] = soxx > ma(soxx, 20)
    conds["r_credit_above_ma20"] = credit > ma(credit, 20)
    conds["r_vix_below_25"] = vix < 25
    conds["r_qqq_20d_return_positive"] = safe_pct_change(qqq, 4) > 0  # weekly bars: roughly 20 trading days
    conds["r_count"] = conds[[
        "r_qqq_above_ma20",
        "r_soxx_above_ma20",
        "r_credit_above_ma20",
        "r_vix_below_25",
        "r_qqq_20d_return_positive",
    ]].sum(axis=1)
    conds["r_credit_veto"] = components["credit_proxy_score"] >= 12
    conds["r_momentum_veto"] = components["market_momentum_score"] >= 20
    conds["r_score_veto"] = components["total_score"] > 35
    conds["r_watch"] = conds["r_count"] >= 4
    conds["r_confirm"] = (
        conds["r_watch"]
        & (~conds["r_credit_veto"])
        & (~conds["r_momentum_veto"])
        & (~conds["r_score_veto"])
    )

    conds["release_qqq_above_ma60"] = qqq > ma(qqq, 60)
    conds["release_soxx_above_ma60"] = soxx > ma(soxx, 60)
    conds["release_credit_above_ma60"] = credit > ma(credit, 60)
    conds["release_score_ok"] = components["total_score"] <= 55
    conds["release_momentum_ok"] = components["market_momentum_score"] < 20
    conds["release_credit_ok"] = components["credit_proxy_score"] < 12
    conds["release_vix_ok"] = vix < 30
    conds["release_confirm"] = (
        conds["release_qqq_above_ma60"]
        & conds["release_soxx_above_ma60"]
        & conds["release_credit_above_ma60"]
        & conds["release_score_ok"]
        & conds["release_momentum_ok"]
        & conds["release_credit_ok"]
        & conds["release_vix_ok"]
    )

    # True panic regime within last 90 weekly bars.
    max_vix_90 = vix.rolling(90, min_periods=10).max()
    max_score_90 = components["total_score"].rolling(90, min_periods=10).max()
    conds["fast_panic_regime"] = (max_vix_90 > 40) & (max_score_90 > 85)
    conds["fast_vix_cooldown"] = vix < max_vix_90 * 0.60
    conds["fast_qqq_above_ma20"] = qqq > ma(qqq, 20)
    conds["fast_soxx_above_ma20"] = soxx > ma(soxx, 20)
    conds["fast_credit_above_ma20"] = credit > ma(credit, 20)
    conds["fast_qqq_20d_return_positive"] = safe_pct_change(qqq, 4) > 0
    conds["fast_vix_below_35"] = vix < 35
    conds["fast_count"] = conds[[
        "fast_vix_cooldown",
        "fast_qqq_above_ma20",
        "fast_soxx_above_ma20",
        "fast_credit_above_ma20",
        "fast_qqq_20d_return_positive",
        "fast_vix_below_35",
    ]].sum(axis=1)
    conds["fast_release_confirm"] = conds["fast_panic_regime"] & (conds["fast_count"] >= 5)
    conds["fast_r_confirm"] = (
        conds["fast_panic_regime"]
        & (conds["fast_count"] >= 6)
        & (components["credit_proxy_score"] < 12)
    )

    # R7.2: credit-crisis-only anti-whipsaw gate.
    credit_252w_high = credit.rolling(52, min_periods=20).max()
    credit_drawdown_252w = credit / credit_252w_high - 1.0
    recent_deep_credit_drawdown = credit_drawdown_252w.rolling(18, min_periods=4).min() <= -0.18
    conds["credit_crisis_regime"] = conds["fast_panic_regime"] & recent_deep_credit_drawdown
    conds["fast_release_safe"] = (
        conds["fast_release_confirm"]
        & (
            (~conds["credit_crisis_regime"])
            | (
                (components["total_score"] < 75)
                & (components["credit_proxy_score"] < 12)
            )
        )
    )

    # Medium repair lane, stricter R7 version.
    conds["medium_repair_regime"] = (vix.rolling(26, min_periods=5).max() > 28) | (components["total_score"].rolling(26, min_periods=5).max() > 70)
    conds["medium_vix_cool"] = vix < 30
    conds["medium_score_ok"] = components["total_score"] <= 55
    conds["medium_momentum_ok"] = components["market_momentum_score"] < 20
    conds["medium_credit_ok"] = components["credit_proxy_score"] < 12
    conds["medium_qqq_above_ma60"] = qqq > ma(qqq, 60)
    conds["medium_soxx_above_ma60"] = soxx > ma(soxx, 60)
    conds["medium_credit_above_ma60"] = credit > ma(credit, 60)
    conds["medium_qqq_20d_return_positive"] = safe_pct_change(qqq, 4) > 0
    conds["medium_count"] = conds[[
        "medium_vix_cool",
        "medium_score_ok",
        "medium_momentum_ok",
        "medium_credit_ok",
        "medium_qqq_above_ma60",
        "medium_soxx_above_ma60",
        "medium_credit_above_ma60",
        "medium_qqq_20d_return_positive",
    ]].sum(axis=1)
    conds["medium_release_confirm"] = conds["medium_repair_regime"] & (conds["medium_count"] >= 7)
    return conds


def raw_mode_from_score(score: float) -> Tuple[str, str]:
    if score <= 55:
        return "452", "風險低或過熱觀察 → 452"
    if score <= 70:
        return "514", "危機升溫 → 514"
    return "514", "高風險防守 → 514"


def compute_weekly_modes(weekly: pd.DataFrame, cooldown_weeks: int = 3) -> pd.DataFrame:
    components = compute_components(weekly)
    conds = compute_r72_conditions(weekly, components)
    out = pd.concat([weekly, components, conds], axis=1)

    current = "452"
    pending = None
    pending_count = 0
    r_pending_count = 0
    release_pending_count = 0
    fast_release_pending_count = 0
    fast_r_pending_count = 0
    medium_release_pending_count = 0

    final_modes = []
    reasons = []
    raw_modes = []
    raw_comments = []

    for _, row in out.iterrows():
        score = float(row["total_score"])
        raw, raw_comment = raw_mode_from_score(score)
        raw_modes.append(raw)
        raw_comments.append(raw_comment)

        r_confirm = bool(row.get("r_confirm", False))
        release_confirm = bool(row.get("release_confirm", False))
        fast_release_safe = bool(row.get("fast_release_safe", False))
        fast_r_confirm = bool(row.get("fast_r_confirm", False))
        medium_release_confirm = bool(row.get("medium_release_confirm", False))

        reason = "維持原模式"

        if score >= 75 and current != "514":
            current = "514"
            pending = None
            pending_count = 0
            r_pending_count = 0
            release_pending_count = 0
            fast_release_pending_count = 0
            fast_r_pending_count = 0
            medium_release_pending_count = 0
            reason = "風險分數>=75，立即切514防守"

        elif current == "514" and fast_release_safe:
            pending = None
            pending_count = 0
            r_pending_count = 0
            fast_release_pending_count += 1
            if fast_release_pending_count >= 1:
                current = "452"
                release_pending_count = 0
                reason = "R7.2快速回攻安全條件成立，514→452"
            else:
                reason = f"快速回攻第{fast_release_pending_count}週觀察，暫維持514"

        elif current == "514" and medium_release_confirm:
            pending = None
            pending_count = 0
            r_pending_count = 0
            medium_release_pending_count += 1
            if medium_release_pending_count >= 2:
                current = "452"
                release_pending_count = 0
                fast_release_pending_count = 0
                reason = "R7中型修復通道連續2週成立，514→452"
            else:
                reason = f"R7中型修復通道第{medium_release_pending_count}週觀察，暫維持514"

        elif current == "514" and fast_r_confirm:
            fast_r_pending_count += 1
            if fast_r_pending_count >= 2:
                current = "433"
                reason = "V型急殺後反攻確認連續2週成立，514→433"
            else:
                reason = f"V型反攻第{fast_r_pending_count}週觀察，暫維持514"

        elif current != "514" and score >= 75:
            current = "514"
            reason = "風險再升，切回514"

        elif current == "452" and r_confirm:
            r_pending_count += 1
            if r_pending_count >= cooldown_weeks:
                current = "433"
                pending = None
                pending_count = 0
                reason = f"R模式連續{cooldown_weeks}週成立，切433反攻"
            else:
                reason = f"R模式第{r_pending_count}週觀察，尚未切433"

        elif current == "514" and release_confirm:
            release_pending_count += 1
            if release_pending_count >= cooldown_weeks:
                current = "452"
                reason = f"解除防守條件連續{cooldown_weeks}週成立，514→452"
            else:
                reason = f"解除防守第{release_pending_count}週觀察，暫維持514"

        elif current == "433" and score >= 70:
            current = "514"
            r_pending_count = 0
            reason = "反攻後風險再升，切回514"

        else:
            # Normal raw mode confirmation.
            if raw != current:
                if pending == raw:
                    pending_count += 1
                else:
                    pending = raw
                    pending_count = 1

                if pending_count >= cooldown_weeks:
                    current = raw
                    pending = None
                    pending_count = 0
                    reason = f"{raw}訊號連續{cooldown_weeks}週成立，正式切換"
                else:
                    reason = f"{raw}訊號第{pending_count}週觀察，尚未切換"
            else:
                pending = None
                pending_count = 0
                reason = "分數與目前模式一致"

        final_modes.append(current)
        reasons.append(reason)

    out["raw_mode"] = raw_modes
    out["raw_comment"] = raw_comments
    out["final_mode"] = final_modes
    out["mode_reason"] = reasons
    return out


def portfolio_value(holdings: Dict[str, float], prices: pd.Series, cash: float = 0.0) -> float:
    return cash + sum(holdings.get(t, 0.0) * float(prices[t]) for t in ASSETS)


def weights_from_holdings(holdings: Dict[str, float], prices: pd.Series, cash: float = 0.0) -> Dict[str, float]:
    total = portfolio_value(holdings, prices, cash)
    if total <= 0:
        return {t: 0.0 for t in ASSETS}
    return {t: holdings.get(t, 0.0) * float(prices[t]) / total for t in ASSETS}


def rebalance_to_target(value: float, prices: pd.Series, target: Dict[str, float], fee_bps: float, old_holdings: Dict[str, float] | None = None) -> Tuple[Dict[str, float], float]:
    # Simple one-pass fee estimate based on turnover.
    if old_holdings is None:
        old_holdings = {t: 0.0 for t in ASSETS}
    old_values = {t: old_holdings.get(t, 0.0) * float(prices[t]) for t in ASSETS}
    target_values_pre = {t: value * target[t] for t in ASSETS}
    turnover = sum(abs(target_values_pre[t] - old_values.get(t, 0.0)) for t in ASSETS) / max(value, 1e-9)
    fee = value * turnover * (fee_bps / 10000.0)
    investable = value - fee
    holdings = {t: (investable * target[t]) / float(prices[t]) for t in ASSETS}
    return holdings, fee


def simulate_os31(weekly_modes: pd.DataFrame, initial_capital: float, tolerance: float, fee_bps: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    holdings: Dict[str, float] = {t: 0.0 for t in ASSETS}
    cash = 0.0
    records: List[dict] = []
    trades: List[dict] = []

    prev_mode = None
    first = True
    total_fees = 0.0

    for dt, row in weekly_modes.iterrows():
        prices = row[ASSETS]
        mode = row["final_mode"]
        target = MODE_TARGETS[mode]

        value_before = initial_capital if first else portfolio_value(holdings, prices, cash)
        current_weights = {t: 0.0 for t in ASSETS} if first else weights_from_holdings(holdings, prices, cash)
        max_dev = max(abs(current_weights[t] - target[t]) for t in ASSETS)

        do_rebalance = first or (mode != prev_mode) or (max_dev >= tolerance)
        reason = "initial" if first else ("mode_switch" if mode != prev_mode else ("drift_rebalance" if max_dev >= tolerance else "hold"))

        fee = 0.0
        if do_rebalance:
            new_holdings, fee = rebalance_to_target(value_before, prices, target, fee_bps, holdings if not first else None)
            old_weights = current_weights.copy()
            holdings = new_holdings
            cash = 0.0
            total_fees += fee
            trades.append({
                "Date": dt.strftime("%Y-%m-%d"),
                "reason": reason,
                "mode": mode,
                "value_before": value_before,
                "fee": fee,
                "max_deviation": max_dev,
                **{f"old_w_{t}": old_weights[t] for t in ASSETS},
                **{f"target_w_{t}": target[t] for t in ASSETS},
            })

        value_after = portfolio_value(holdings, prices, cash)
        weights_after = weights_from_holdings(holdings, prices, cash)

        records.append({
            "Date": dt.strftime("%Y-%m-%d"),
            "mode": mode,
            "mode_reason": row.get("mode_reason", ""),
            "portfolio_value": value_after,
            "fee_paid": fee,
            "total_fees": total_fees,
            "rebalance": do_rebalance,
            "rebalance_reason": reason,
            "max_deviation_before": max_dev,
            **{f"price_{t}": float(prices[t]) for t in ASSETS},
            **{f"weight_{t}": weights_after[t] for t in ASSETS},
            **{f"shares_{t}": holdings[t] for t in ASSETS},
        })
        prev_mode = mode
        first = False

    return pd.DataFrame(records), pd.DataFrame(trades)


def simulate_fixed_mode(weekly: pd.DataFrame, mode: str, initial_capital: float, tolerance: float, fee_bps: float) -> pd.Series:
    tmp = weekly.copy()
    tmp["final_mode"] = mode
    tmp["mode_reason"] = "fixed"
    curve, _ = simulate_os31(tmp, initial_capital, tolerance, fee_bps)
    return pd.Series(curve["portfolio_value"].values, index=pd.to_datetime(curve["Date"]), name=f"fixed_{mode}")


def simulate_buy_hold(weekly: pd.DataFrame, ticker: str, initial_capital: float) -> pd.Series:
    s = weekly[ticker].dropna()
    shares = initial_capital / float(s.iloc[0])
    return pd.Series(shares * s.values, index=s.index, name=f"buy_hold_{ticker}")


def perf_stats(curve: pd.Series, initial_capital: float) -> Dict[str, float]:
    curve = curve.dropna()
    if curve.empty:
        return {"ending": np.nan, "return": np.nan, "max_drawdown": np.nan, "cagr": np.nan}
    ending = float(curve.iloc[-1])
    total_return = ending / initial_capital - 1.0
    peak = curve.cummax()
    dd = curve / peak - 1.0
    max_dd = float(dd.min())
    years = max((curve.index[-1] - curve.index[0]).days / 365.25, 1e-9)
    cagr = (ending / initial_capital) ** (1 / years) - 1
    return {"ending": ending, "return": total_return, "max_drawdown": max_dd, "cagr": cagr}


def money(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x:,.0f}"


def pct(x: float) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x*100:.1f}%"


def make_switch_log(weekly_modes: pd.DataFrame) -> pd.DataFrame:
    df = weekly_modes.copy()
    prev = df["final_mode"].shift(1)
    sw = df[(df["final_mode"] != prev)].copy()
    cols = [
        "total_score", "final_mode", "raw_mode", "r_count", "r_confirm",
        "fast_release_confirm", "fast_release_safe", "credit_crisis_regime",
        "fast_r_confirm", "medium_release_confirm",
        "market_momentum_score", "credit_proxy_score", "breadth_score", "vix_score",
        "defensive_strength_score", "mode_reason",
    ]
    cols = [c for c in cols if c in sw.columns]
    out = sw[cols].reset_index().rename(columns={"index": "Date"})
    out["Date"] = pd.to_datetime(out["Date"]).dt.strftime("%Y-%m-%d")
    return out


def build_summary(start: str, end: str, initial_capital: float, tolerance: float, fee_bps: float, weekly_modes: pd.DataFrame, curve: pd.DataFrame, trades: pd.DataFrame, comparisons: Dict[str, pd.Series]) -> str:
    os_curve = pd.Series(curve["portfolio_value"].values, index=pd.to_datetime(curve["Date"]), name="OS31_R72")
    stats = {"OS31_R72": perf_stats(os_curve, initial_capital)}
    for name, s in comparisons.items():
        stats[name] = perf_stats(s, initial_capital)

    lines = []
    lines.append("# OS 3.1 + V11-Core R7.2 Backtest Summary")
    lines.append("")
    lines.append(f"Period: {start} to {end}")
    lines.append(f"Initial capital: {money(initial_capital)} TWD")
    lines.append(f"Rebalance tolerance: {tolerance*100:.1f} percentage points")
    lines.append(f"Transaction fee: {fee_bps:.2f} bps per turnover")
    lines.append("")
    lines.append("## Mode Definitions")
    lines.append("- 452 = 平常作戰 / 中性偏進攻底盤 = 45:25:30")
    lines.append("- 514 = 危機升溫 / 防守避震 = 50:10:40")
    lines.append("- 433 = R模式確認 / 防守反擊 = 40:30:30")
    lines.append("")
    lines.append("## Mode Distribution")
    counts = weekly_modes["final_mode"].value_counts().reindex(["452","514","433"]).fillna(0).astype(int)
    for mode in ["452","514","433"]:
        lines.append(f"- {mode} weeks: {counts[mode]}")
    first_514 = weekly_modes.index[weekly_modes["final_mode"].eq("514")]
    first_433 = weekly_modes.index[weekly_modes["final_mode"].eq("433")]
    lines.append(f"- First 514 week: {first_514[0].strftime('%Y-%m-%d') if len(first_514) else 'None'}")
    lines.append(f"- First 433 week: {first_433[0].strftime('%Y-%m-%d') if len(first_433) else 'None'}")
    lines.append(f"- Highest risk score: {weekly_modes['total_score'].max():.1f} on {weekly_modes['total_score'].idxmax().strftime('%Y-%m-%d')}")
    lines.append(f"- Mode switches: {(weekly_modes['final_mode'] != weekly_modes['final_mode'].shift(1)).sum()-1}")
    lines.append(f"- Rebalances: {int(curve['rebalance'].sum())}")
    lines.append(f"- Total fees: {money(curve['total_fees'].iloc[-1])} TWD")
    lines.append("")
    lines.append("## Performance")
    lines.append("| Strategy | Ending Value | Total Return | CAGR | Max Drawdown |")
    lines.append("|---|---:|---:|---:|---:|")
    order = ["OS31_R72", "fixed_452", "fixed_514", "fixed_433", "buy_hold_00662.TW", "buy_hold_00670L.TW", "buy_hold_00865B.TW"]
    for name in order:
        if name in stats:
            st = stats[name]
            lines.append(f"| {name} | {money(st['ending'])} | {pct(st['return'])} | {pct(st['cagr'])} | {pct(st['max_drawdown'])} |")
    lines.append("")
    lines.append("## Switch Log")
    sw = make_switch_log(weekly_modes)
    for _, r in sw.iterrows():
        lines.append(f"- {r['Date']}: {r['final_mode']}, score={r['total_score']:.1f}, reason={r['mode_reason']}")
    lines.append("")
    lines.append("## Notes")
    lines.append("- This is an OS 3.1 practical backtest, not tax/accounting advice.")
    lines.append("- Uses yfinance adjusted close. Taiwan ETF data quality can vary by vendor.")
    lines.append("- No cashflow, no tax, no slippage unless --fee-bps is set.")
    lines.append("- Rebalance occurs on weekly close only.")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OS 3.1 + V11-Core R7.2 backtest from 2019 to 2026/current.")
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END)
    parser.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL)
    parser.add_argument("--tolerance", type=float, default=0.05, help="Rebalance tolerance as decimal. 0.05 = 5 percentage points.")
    parser.add_argument("--fee-bps", type=float, default=0.0, help="Transaction fee bps per turnover. Default 0.")
    args = parser.parse_args()

    close = download_prices(args.start, args.end)
    weekly = to_weekly(close, args.start, args.end)
    weekly_modes = compute_weekly_modes(weekly)
    curve, trades = simulate_os31(weekly_modes, args.initial_capital, args.tolerance, args.fee_bps)

    comparisons = {
        "fixed_452": simulate_fixed_mode(weekly, "452", args.initial_capital, args.tolerance, args.fee_bps),
        "fixed_514": simulate_fixed_mode(weekly, "514", args.initial_capital, args.tolerance, args.fee_bps),
        "fixed_433": simulate_fixed_mode(weekly, "433", args.initial_capital, args.tolerance, args.fee_bps),
        "buy_hold_00662.TW": simulate_buy_hold(weekly, "00662.TW", args.initial_capital),
        "buy_hold_00670L.TW": simulate_buy_hold(weekly, "00670L.TW", args.initial_capital),
        "buy_hold_00865B.TW": simulate_buy_hold(weekly, "00865B.TW", args.initial_capital),
    }

    outdir = Path("output")
    outdir.mkdir(exist_ok=True)
    weekly_modes.reset_index().rename(columns={"index": "Date"}).to_csv(outdir / "os31_r72_2019_2026_weekly_modes.csv", index=False, encoding="utf-8-sig")
    curve.to_csv(outdir / "os31_r72_2019_2026_equity_curve.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(outdir / "os31_r72_2019_2026_trades.csv", index=False, encoding="utf-8-sig")
    make_switch_log(weekly_modes).to_csv(outdir / "os31_r72_2019_2026_switch_log.csv", index=False, encoding="utf-8-sig")

    comparison_df = pd.DataFrame({name: s for name, s in comparisons.items()})
    comparison_df["OS31_R72"] = pd.Series(curve["portfolio_value"].values, index=pd.to_datetime(curve["Date"]))
    comparison_df.to_csv(outdir / "os31_r72_2019_2026_comparison_curves.csv", encoding="utf-8-sig")

    summary = build_summary(args.start, args.end, args.initial_capital, args.tolerance, args.fee_bps, weekly_modes, curve, trades, comparisons)
    (outdir / "os31_r72_2019_2026_summary.md").write_text(summary, encoding="utf-8")
    print(summary)


if __name__ == "__main__":
    main()
