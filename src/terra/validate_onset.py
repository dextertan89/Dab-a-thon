# src/terra/validate_onset.py
from __future__ import annotations
import pandas as pd

def window_slice(df: pd.DataFrame, time_col: str, t0: pd.Timestamp, hours: int = 24) -> pd.DataFrame:
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], utc=True, errors="coerce")
    d = d.dropna(subset=[time_col]).sort_values(time_col)
    start = t0 - pd.Timedelta(hours=hours)
    end = t0 + pd.Timedelta(hours=hours)
    return d[(d[time_col] >= start) & (d[time_col] <= end)]

def hourly_count(df: pd.DataFrame, time_col: str) -> pd.Series:
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], utc=True, errors="coerce")
    d = d.dropna(subset=[time_col]).set_index(time_col)
    return d.resample("1H").size()

def hourly_net_flow(df: pd.DataFrame, time_col: str, flow_col: str) -> pd.Series:
    d = df.copy()
    d[time_col] = pd.to_datetime(d[time_col], utc=True, errors="coerce")
    d = d.dropna(subset=[time_col]).set_index(time_col)
    return d[flow_col].resample("1h").sum()
