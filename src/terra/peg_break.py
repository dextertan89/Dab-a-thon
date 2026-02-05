# src/terra/peg_break.py
from __future__ import annotations
import pandas as pd
from dataclasses import dataclass

@dataclass(frozen=True)
class PegBreakRule:
    thr_low: float = 0.995
    thr_recover: float = 0.998
    sustain_hours: int = 4     # X
    lookahead_hours: int = 6   # Y

def detect_peg_break_sustained(
    df: pd.DataFrame,
    time_col: str,
    price_col: str,
    rule: PegBreakRule = PegBreakRule(),
) -> pd.Timestamp | None:
    """
    Peg breaks at the first time t0 where:
      (A) price stays < thr_low for sustain_hours (approx), OR
      (B) price does not recover >= thr_recover within lookahead_hours.
    Returns UTC timestamp or None.
    """
    d = df[[time_col, price_col]].copy()
    d[time_col] = pd.to_datetime(d[time_col], utc=True, errors="coerce")
    d = d.dropna(subset=[time_col, price_col]).sort_values(time_col).set_index(time_col)

    if len(d) < 10:
        return None

    median_dt = d.index.to_series().diff().median()
    if pd.isna(median_dt) or median_dt.total_seconds() <= 0:
        return None

    steps_per_hour = max(1, int(round(pd.Timedelta(hours=1) / median_dt)))
    k_sustain = max(1, rule.sustain_hours * steps_per_hour)
    k_lookahead = max(1, rule.lookahead_hours * steps_per_hour)

    below = d[price_col] < rule.thr_low
    candidates = d.index[below & (~below.shift(1).fillna(False))]

    for t0 in candidates:
        window = d.loc[t0:].head(k_lookahead)

        below_series = (window[price_col] < rule.thr_low).astype(int)
        sustained = below_series.rolling(k_sustain).sum().eq(k_sustain).any()

        recovers = (window[price_col] >= rule.thr_recover).any()
        no_recover = not recovers

        if sustained or no_recover:
            return t0

    return None
