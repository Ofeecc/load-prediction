"""
Weather station selection and spatial aggregation experiments for GEFCom2014-L.

Evaluates single-station heterogeneity across the 25 anonymous weather stations (w1..w25)
and compares validation-based station subset selection, temperature weighting, and
probabilistic forecast combination against the 25-station equal-weight mean baseline.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd

from src.metrics import evaluate_all_metrics, pinball_loss
from src.models import HourlySplineGAM


def evaluate_all_25_stations(
    df_full: pd.DataFrame,
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    val_scen_years: Sequence[int],
    test_scen_years: Sequence[int],
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, np.ndarray]]]:
    """
    Evaluate each of the 25 anonymous weather stations (w1..w25) individually on:
    - Validation Oracle Point RMSE (under true validation temperature) & Validation Scenario Pinball Loss
    - Test Oracle Point RMSE (under true test temperature) & Test Scenario Pinball Loss

    Caches validation and test predictions (point and 99-quantile) for each station.
    """
    y_val = df_full.loc[val_mask, "LOAD"].to_numpy(dtype=float)
    y_test = df_full.loc[test_mask, "LOAD"].to_numpy(dtype=float)

    val_first = int(np.flatnonzero(val_mask)[0])
    fit_val_mask = np.zeros(len(df_full), dtype=bool)
    fit_val_mask[:val_first] = True

    rows = []
    cache: Dict[str, Dict[str, np.ndarray]] = {}

    for i in range(1, 26):
        w_col = f"w{i}"
        temp_w = df_full[w_col].to_numpy(dtype=float)

        # 1. Fit strictly before val_mask and evaluate on val_mask
        gam_val = HourlySplineGAM(
            use_holiday_correction=True,
            use_thermal_inertia=True,
            use_regime_calibration=True,
        ).fit(df_full, temp_w, fit_val_mask)

        val_oracle_pt = gam_val.predict_point(df_full, temp_w, val_mask)
        val_scen_q, val_scen_mu = gam_val.predict_quantiles_via_scenarios(
            df_full,
            temp_w,
            val_mask,
            scenario_years=val_scen_years,
            ar_half_life_hours=36.0,
            adaptive_ar_gating=True,
        )
        val_oracle_rmse = float(np.sqrt(np.mean((y_val - val_oracle_pt) ** 2)))
        val_metrics = evaluate_all_metrics(y_val, val_scen_q, val_scen_mu)

        # 2. Fit strictly before test_mask and evaluate on test_mask
        gam_test = HourlySplineGAM(
            use_holiday_correction=True,
            use_thermal_inertia=True,
            use_regime_calibration=True,
        ).fit(df_full, temp_w, train_mask)

        test_oracle_pt = gam_test.predict_point(df_full, temp_w, test_mask)
        test_scen_q, test_scen_mu = gam_test.predict_quantiles_via_scenarios(
            df_full,
            temp_w,
            test_mask,
            scenario_years=test_scen_years,
            ar_half_life_hours=36.0,
            adaptive_ar_gating=True,
        )
        test_oracle_rmse = float(np.sqrt(np.mean((y_test - test_oracle_pt) ** 2)))
        test_metrics = evaluate_all_metrics(y_test, test_scen_q, test_scen_mu)

        rows.append(
            {
                "Station": w_col,
                "Station_ID": i,
                "Val_Oracle_RMSE": val_oracle_rmse,
                "Val_Scen_RMSE": val_metrics["RMSE"],
                "Val_Pinball": val_metrics["Pinball"],
                "Val_Pinball_Tail": val_metrics["Pinball_Tail"],
                "Test_Oracle_RMSE": test_oracle_rmse,
                "Test_Scen_RMSE": test_metrics["RMSE"],
                "Test_Pinball": test_metrics["Pinball"],
                "Test_Pinball_Tail": test_metrics["Pinball_Tail"],
                "Test_PICP_90": test_metrics["PICP_90(%)"],
            }
        )

        cache[w_col] = {
            "val_oracle_pt": val_oracle_pt,
            "val_scen_q": val_scen_q,
            "val_scen_mu": val_scen_mu,
            "test_oracle_pt": test_oracle_pt,
            "test_scen_q": test_scen_q,
            "test_scen_mu": test_scen_mu,
        }

    df_stations = pd.DataFrame(rows)
    return df_stations, cache


def compare_station_selection_strategies(
    df_full: pd.DataFrame,
    df_stations: pd.DataFrame,
    station_cache: Dict[str, Dict[str, np.ndarray]],
    train_mask: np.ndarray,
    val_mask: np.ndarray,
    test_mask: np.ndarray,
    test_scen_years: Sequence[int],
) -> pd.DataFrame:
    """
    Compare our validation-based weather station selection and aggregation strategies
    (ensuring selection objective and evaluation pipeline are strictly consistent):
    1. All-25 Equal Mean Temp (`w1..w25` equal average baseline)
    2. Single Best Val-RMSE Station (top-1 station by `Val_Oracle_RMSE`)
    3. Top-2 Val-RMSE Mean Temp (average temperature of top-2 stations by `Val_Oracle_RMSE`)
    4. Forward Val-RMSE Temp Subset (greedy forward selection of 4 stations where each candidate subset
       is evaluated by fitting the model on the candidate subset's mean temperature on `val_mask`)
    5. Val-Loss Weighted Temp Combination (exponentially weighted station temperatures from top-7 pool)
    6. Val-Pinball EWA Forecast Combination (combining single-station quantile forecasts directly via
       exponential weights on validation Pinball Loss)
    """
    y_val = df_full.loc[val_mask, "LOAD"].to_numpy(dtype=float)
    y_test = df_full.loc[test_mask, "LOAD"].to_numpy(dtype=float)

    val_first = int(np.flatnonzero(val_mask)[0])
    fit_val_mask = np.zeros(len(df_full), dtype=bool)
    fit_val_mask[:val_first] = True

    results = []

    def record_strategy(name: str, stations_desc: str, test_q: np.ndarray, test_mu: np.ndarray, oracle_rmse: float):
        m = evaluate_all_metrics(y_test, test_q, test_mu)
        m["Strategy"] = name
        m["Selected_Stations"] = stations_desc
        m["Test_Oracle_RMSE"] = oracle_rmse
        results.append(m)

    # Strategy 1: Mean of all 25 stations
    temp_mean25 = df_full["temp_mean25"].to_numpy(dtype=float)
    gam_25 = HourlySplineGAM().fit(df_full, temp_mean25, train_mask)
    oracle_25 = float(np.sqrt(np.mean((y_test - gam_25.predict_point(df_full, temp_mean25, test_mask)) ** 2)))
    q_25, mu_25 = gam_25.predict_quantiles_via_scenarios(
        df_full, temp_mean25, test_mask, scenario_years=test_scen_years, ar_half_life_hours=36.0, adaptive_ar_gating=True
    )
    record_strategy("1. All-25 Equal Mean Temp", "w1..w25 (equal)", q_25, mu_25, oracle_25)

    # Sort stations by Validation Point (Oracle) RMSE
    sorted_by_point = df_stations.sort_values("Val_Oracle_RMSE")["Station"].tolist()

    # Strategy 2: Best-1 Validation RMSE Station
    best1 = sorted_by_point[0]
    c1 = station_cache[best1]
    oracle_1 = float(np.sqrt(np.mean((y_test - c1["test_oracle_pt"]) ** 2)))
    record_strategy("2. Best-1 Val-RMSE Station", best1, c1["test_scen_q"], c1["test_scen_mu"], oracle_1)

    # Strategy 3: Top-2 Val-RMSE Stations Mean Temp
    top2 = sorted_by_point[:2]
    temp_top2 = df_full[top2].mean(axis=1).to_numpy(dtype=float)
    gam_top2 = HourlySplineGAM().fit(df_full, temp_top2, train_mask)
    oracle_top2 = float(np.sqrt(np.mean((y_test - gam_top2.predict_point(df_full, temp_top2, test_mask)) ** 2)))
    q_top2, mu_top2 = gam_top2.predict_quantiles_via_scenarios(
        df_full, temp_top2, test_mask, scenario_years=test_scen_years, ar_half_life_hours=36.0, adaptive_ar_gating=True
    )
    record_strategy("3. Top-2 Val-RMSE Mean Temp", ", ".join(top2), q_top2, mu_top2, oracle_top2)

    # Strategy 4: Consistent Forward Val-RMSE Temp-Subset Selection (4 stations)
    # Selection and test evaluation both fit HourlySplineGAM on the subset mean temperature
    candidate_pool = sorted_by_point[:12]  # Top-12 individual stations for fast forward search
    selected_fwd: List[str] = [candidate_pool[0]]
    remaining = [s for s in candidate_pool[1:]]
    for _ in range(3):
        best_cand = None
        best_rmse = float("inf")
        for cand in remaining:
            trial = selected_fwd + [cand]
            temp_trial = df_full[trial].mean(axis=1).to_numpy(dtype=float)
            gam_trial = HourlySplineGAM().fit(df_full, temp_trial, fit_val_mask)
            pred_val = gam_trial.predict_point(df_full, temp_trial, val_mask)
            rmse_val = float(np.sqrt(np.mean((y_val - pred_val) ** 2)))
            if rmse_val < best_rmse:
                best_rmse = rmse_val
                best_cand = cand
        if best_cand is not None:
            selected_fwd.append(best_cand)
            remaining.remove(best_cand)

    temp_fwd4 = df_full[selected_fwd].mean(axis=1).to_numpy(dtype=float)
    gam_fwd4 = HourlySplineGAM().fit(df_full, temp_fwd4, train_mask)
    oracle_fwd4 = float(np.sqrt(np.mean((y_test - gam_fwd4.predict_point(df_full, temp_fwd4, test_mask)) ** 2)))
    q_fwd4, mu_fwd4 = gam_fwd4.predict_quantiles_via_scenarios(
        df_full, temp_fwd4, test_mask, scenario_years=test_scen_years, ar_half_life_hours=36.0, adaptive_ar_gating=True
    )
    record_strategy("4. Forward Val-RMSE Temp Subset (4-Stn)", ", ".join(selected_fwd), q_fwd4, mu_fwd4, oracle_fwd4)

    # Strategy 5: Validation Composite-Loss Temperature Weighting (Top-7 -> Top-3 Weighted Temp)
    pool7 = sorted_by_point[:7]
    val_pinball = np.array(
        [df_stations.loc[df_stations["Station"] == s, "Val_Pinball"].iloc[0] for s in pool7],
        dtype=float,
    )
    val_rmse = np.array(
        [df_stations.loc[df_stations["Station"] == s, "Val_Oracle_RMSE"].iloc[0] for s in pool7],
        dtype=float,
    )
    norm_pb = (val_pinball - np.min(val_pinball)) / max(1e-6, float(np.std(val_pinball)))
    norm_rmse = (val_rmse - np.min(val_rmse)) / max(1e-6, float(np.std(val_rmse)))
    comp_loss = 0.55 * norm_pb + 0.45 * norm_rmse

    order_prob = np.argsort(comp_loss)[:3]
    tw_stations = [pool7[idx] for idx in order_prob]
    tw_losses = comp_loss[order_prob]
    tw_weights = np.exp(-0.65 * (tw_losses - np.min(tw_losses)))
    tw_weights = tw_weights / np.sum(tw_weights)

    temp_weighted = np.zeros(len(df_full), dtype=float)
    for w_s, s_name in zip(tw_weights, tw_stations):
        temp_weighted += w_s * df_full[s_name].to_numpy(dtype=float)
    gam_tw = HourlySplineGAM().fit(df_full, temp_weighted, train_mask)
    q_tw, mu_tw = gam_tw.predict_quantiles_via_scenarios(
        df_full, temp_weighted, test_mask, scenario_years=test_scen_years, ar_half_life_hours=36.0, adaptive_ar_gating=True
    )
    oracle_tw = float(np.sqrt(np.mean((y_test - gam_tw.predict_point(df_full, temp_weighted, test_mask)) ** 2)))
    tw_desc = ", ".join(f"{s}({w:.2f})" for s, w in zip(tw_stations, tw_weights))
    record_strategy("5. Val-Loss Weighted Temp (3-Stn)", tw_desc, q_tw, mu_tw, oracle_tw)

    # Strategy 6: Validation Pinball EWA Quantile Forecast Combination (combining single-station quantile forecasts)
    sorted_by_pb = df_stations.sort_values("Val_Pinball")["Station"].tolist()[:5]
    pb_top5 = np.array(
        [df_stations.loc[df_stations["Station"] == s, "Val_Pinball"].iloc[0] for s in sorted_by_pb],
        dtype=float,
    )
    ewa_q_weights = np.exp(-2.0 * (pb_top5 - np.min(pb_top5)))
    ewa_q_weights = ewa_q_weights / np.sum(ewa_q_weights)
    q_comb = np.zeros_like(q_25)
    mu_comb = np.zeros_like(mu_25)
    ora_comb = np.zeros_like(mu_25)
    for w_s, s_name in zip(ewa_q_weights, sorted_by_pb):
        q_comb += w_s * station_cache[s_name]["test_scen_q"]
        mu_comb += w_s * station_cache[s_name]["test_scen_mu"]
        ora_comb += w_s * station_cache[s_name]["test_oracle_pt"]
    oracle_comb_rmse = float(np.sqrt(np.mean((y_test - ora_comb) ** 2)))
    comb_desc = ", ".join(f"{s}({w:.2f})" for s, w in zip(sorted_by_pb[:3], ewa_q_weights[:3])) + " (+2)"
    record_strategy("6. Val-Pinball EWA Forecast Combination", comb_desc, np.sort(q_comb, axis=1), mu_comb, oracle_comb_rmse)

    cols_order = [
        "Strategy",
        "Selected_Stations",
        "Pinball",
        "Pinball_Tail",
        "Pinball_Central",
        "Test_Oracle_RMSE",
        "RMSE",
        "MAE",
        "MAPE(%)",
        "PICP_80(%)",
        "PICP_90(%)",
        "MIW_90(MW)",
    ]
    return pd.DataFrame(results)[cols_order]

