"""
Data loader and Table 5 US Federal Holiday feature engineering for GEFCom2014-L.
"""
from __future__ import annotations

import csv
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def get_us_federal_holidays(year: int) -> Dict[date, str]:
    """
    Return the 10 US Federal Holidays listed in Table 5 of Hong et al. (2016):
    1. New Year's Day: January 1
    2. Birthday of Martin Luther King, Jr.: Third Monday in January
    3. Washington's Birthday (Presidents' Day): Third Monday in February
    4. Memorial Day: Last Monday in May
    5. Independence Day: July 4
    6. Labor Day: First Monday in September
    7. Columbus Day: Second Monday in October
    8. Veterans Day: November 11
    9. Thanksgiving Day: Fourth Thursday in November
    10. Christmas Day: December 25
    """
    holidays: Dict[date, str] = {}

    def nth_weekday(y: int, m: int, weekday: int, n: int) -> date:
        # weekday: 0=Monday .. 6=Sunday
        first = date(y, m, 1)
        offset = (weekday - first.weekday()) % 7
        return first + timedelta(days=offset + 7 * (n - 1))

    def last_weekday(y: int, m: int, weekday: int) -> date:
        if m == 12:
            next_month = date(y + 1, 1, 1)
        else:
            next_month = date(y, m + 1, 1)
        last_day = next_month - timedelta(days=1)
        offset = (last_day.weekday() - weekday) % 7
        return last_day - timedelta(days=offset)

    holidays[date(year, 1, 1)] = "New Year's Day"
    holidays[nth_weekday(year, 1, 0, 3)] = "Martin Luther King Jr. Day"
    holidays[nth_weekday(year, 2, 0, 3)] = "Washington's Birthday"
    holidays[last_weekday(year, 5, 0)] = "Memorial Day"
    holidays[date(year, 7, 4)] = "Independence Day"
    holidays[nth_weekday(year, 9, 0, 1)] = "Labor Day"
    holidays[nth_weekday(year, 10, 0, 2)] = "Columbus Day"
    holidays[date(year, 11, 11)] = "Veterans Day"
    holidays[nth_weekday(year, 11, 3, 4)] = "Thanksgiving Day"
    holidays[date(year, 12, 25)] = "Christmas Day"

    return holidays


def raw_timestamp_token(ts) -> str:
    ts = pd.Timestamp(ts)
    return f"{ts.month}{ts.day}{ts.year} {ts.hour}:00"


def validate_raw_timestamps(df):
    for row in df[["timestamp", "task_origin", "raw_ts"]].itertuples(index=False):
        if row.task_origin == 16:
            day = row.timestamp - pd.Timedelta(hours=1)
            expected = f"{day.month}/{day.day}/{day.year} H{day.hour + 1}"
        else:
            expected = raw_timestamp_token(row.timestamp)
        if row.raw_ts != expected:
            raise ValueError(f"Raw timestamp/order mismatch at {row.timestamp}: {row.raw_ts!r} != {expected!r}")


def load_full_gefcom2014_l(data_dir: str | Path = "Data/Load") -> pd.DataFrame:
    """
    Consolidate Task 1..15 training files and Solution to Task 15 into a single
    continuous hourly DataFrame spanning 2001-01-01 to 2011-12-31 (24 hours/day).

    Convention in GEFCom2014-L:
    - Each day has 24 hourly observations (hour_of_day = 1..24).
    - In L*-train.csv, hour 24 of day D is stamped as '0:00' of day D+1.
    - In solution15_L_temperature.csv, it is explicitly stamped as day D, hour 24.
    We store both:
    - `timestamp`: exact end-of-hour timestamp
    - `op_date`: operational date D (2001-01-01 .. 2011-12-31)
    - `hour`: operational hour 1..24 (where 1 is 00:00-01:00 and 24 is 23:00-24:00)
    """
    base = Path(data_dir)
    records: List[dict] = []

    weather_cols = [f"w{i}" for i in range(1, 26)]

    for task_id in range(1, 16):
        train_path = base / f"Task {task_id}" / f"L{task_id}-train.csv"
        with open(train_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                load_val = float(row["LOAD"]) if row["LOAD"] != "" else np.nan
                entry = {
                    "raw_ts": row["TIMESTAMP"].strip(),
                    "ZONEID": int(row["ZONEID"]),
                    "LOAD": load_val,
                    "task_origin": task_id,
                }
                for w in weather_cols:
                    entry[w] = float(row[w])
                records.append(entry)

    # Task 15 solution (December 2011 actual load and 25 weather station temperatures)
    sol_path = base / "Solution to Task 15" / "solution15_L_temperature.csv"
    with open(sol_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[2].strip().upper() == "LOAD":
                continue
            # Format: date, hour, load, w1..w25
            dt_str, hr_str, load_str = row[0].strip(), row[1].strip(), row[2].strip()
            entry = {
                "raw_ts": f"{dt_str} H{hr_str}",
                "ZONEID": 1,
                "LOAD": float(load_str),
                "task_origin": 16,  # Ground truth of Task 15 (Dec 2011)
            }
            for idx, w in enumerate(weather_cols):
                entry[w] = float(row[3 + idx])
            records.append(entry)

    df = pd.DataFrame(records)

    # Verify total rows = 4017 days (2001-01-01 to 2011-12-31 inclusive) * 24 hours = 96,408
    expected_hours = 4017 * 24
    if len(df) != expected_hours:
        raise ValueError(f"Expected {expected_hours} hourly rows, got {len(df)}")

    # Construct exact timestamps and operational date/hour
    start_ts = pd.Timestamp("2001-01-01 01:00:00")
    df["timestamp"] = pd.date_range(start=start_ts, periods=len(df), freq="h")
    # Raw dates have unpadded month/day tokens. Reconstructing without checking
    # would hide reordered or missing rows, including ambiguous 1112011 tokens.
    validate_raw_timestamps(df)
    if set(df["ZONEID"].unique()) != {1}:
        raise ValueError("Expected the single GEFCom2014-L zone (ZONEID=1)")
    if not np.isfinite(df[weather_cols].to_numpy(float)).all():
        raise ValueError("Missing/nonfinite weather measurements")
    # Operational time: subtract 1 hour so hours 1..24 of 2001-01-01 all map to 2001-01-01
    op_ts = df["timestamp"] - pd.Timedelta(hours=1)
    df["op_date"] = op_ts.dt.date
    df["year"] = op_ts.dt.year
    df["month"] = op_ts.dt.month
    df["day"] = op_ts.dt.day
    df["day_of_year"] = op_ts.dt.day_of_year
    df["dayofweek"] = op_ts.dt.day_of_week  # 0=Mon .. 6=Sun
    df["hour"] = op_ts.dt.hour + 1  # 1..24
    df["is_weekend"] = (df["dayofweek"] >= 5).astype(int)

    # Add Table 5 US Federal Holiday annotations
    all_holidays: Dict[date, str] = {}
    for y in range(2001, 2013):
        all_holidays.update(get_us_federal_holidays(y))

    df["holiday_name"] = df["op_date"].map(lambda d: all_holidays.get(d, ""))
    df["is_holiday"] = (df["holiday_name"] != "").astype(int)

    # Major holidays with severe industrial load collapse (New Year, Memorial, Independence, Labor, Thanksgiving, Christmas)
    major_holidays = {
        "New Year's Day",
        "Memorial Day",
        "Independence Day",
        "Labor Day",
        "Thanksgiving Day",
        "Christmas Day",
    }
    df["is_major_holiday"] = df["holiday_name"].isin(major_holidays).astype(int)

    # Bridge/adjacent holiday days (Day after Thanksgiving, Christmas Eve, Dec 26, New Year's Eve)
    def is_holiday_bridge(d: date) -> int:
        if d.month == 12 and d.day in (24, 26, 31):
            return 1
        # Day after Thanksgiving (Friday after 4th Thursday in Nov)
        if d.month == 11 and d.weekday() == 4:
            prev_day = d - timedelta(days=1)
            if all_holidays.get(prev_day, "") == "Thanksgiving Day":
                return 1
        # Observed federal holiday when July 4, Dec 25, or Jan 1 falls on weekend
        if d.weekday() == 0 and all_holidays.get(d - timedelta(days=1), "") in major_holidays:
            return 1
        if d.weekday() == 4 and all_holidays.get(d + timedelta(days=1), "") in major_holidays:
            return 1
        return 0

    df["is_holiday_bridge"] = df["op_date"].map(is_holiday_bridge)

    # Summary statistics of the 25 weather stations
    w_matrix = df[weather_cols].to_numpy(dtype=float)
    df["temp_mean25"] = w_matrix.mean(axis=1)
    df["temp_std25"] = w_matrix.std(axis=1)
    df["temp_min25"] = w_matrix.min(axis=1)
    df["temp_max25"] = w_matrix.max(axis=1)

    return df


def load_task_benchmark(task_id: int, data_dir: str | Path = "Data/Load") -> np.ndarray:
    """
    Load the official 99-quantile benchmark forecast for `task_id` (1..15).
    Returns array of shape (H, 99).
    """
    if not 1 <= task_id <= 15:
        raise ValueError("task_id must be in 1..15")
    base = Path(data_dir)
    bpath = base / f"Task {task_id}" / f"L{task_id}-benchmark.csv"
    df_b = pd.read_csv(bpath)
    q_cols = [c for c in df_b.columns if c not in ("ZONEID", "TIMESTAMP")]
    from src.metrics import QUANTILES_99
    if len(q_cols) != 99 or not np.allclose(np.array(q_cols, float), QUANTILES_99):
        raise ValueError("Benchmark quantile columns must be ordered 0.01..0.99")
    origin = pd.Timestamp("2010-10-01") + pd.DateOffset(months=task_id - 1)
    end = origin + pd.DateOffset(months=1)
    expected = pd.date_range(origin + pd.Timedelta(hours=1), end, freq="h")
    tokens = [raw_timestamp_token(t) for t in expected]
    if list(df_b["TIMESTAMP"]) != tokens or not (df_b["ZONEID"] == 1).all():
        raise ValueError("Benchmark dates/zone do not match the target operational month")
    result = df_b[q_cols].to_numpy(dtype=float)
    if not np.isfinite(result).all() or np.any(np.diff(result, axis=1) < 0):
        raise ValueError("Invalid benchmark quantiles")
    return result


if __name__ == "__main__":
    df = load_full_gefcom2014_l()
    print(f"Loaded {len(df)} rows from {df['op_date'].min()} to {df['op_date'].max()}")
    print(f"Non-null LOAD rows: {df['LOAD'].notna().sum()} (starts {df.loc[df['LOAD'].notna(), 'op_date'].min()})")
    print("Holidays in Dec 2011:", df[(df["year"] == 2011) & (df["month"] == 12) & (df["is_holiday"] == 1)][["op_date", "holiday_name"]].drop_duplicates())
