"""
Weather station selection and spatial aggregation experiments for GEFCom2014-L.

Tests the key insight from Table 6 (Tololo & Adada, Hong et al. 2016):
"Weather stations selected for point load forecasting are NOT optimized for probabilistic forecasting."
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
    - Validation Point RMSE (Oracle temperature on val_mask) & Validation Scenario Pinball Loss
    - Test Point RMSE (Oracle temperature on test_mask) & Test Scenario Pinball Loss

    Also caches validation and test predictions (point and 99-quantile) for each station
    so subset selection and Exponentially Weighted Aggregation (EWA) can be evaluated cleanly.
    """
    y_val = df_full.loc[val_mask, "LOAD"].to_numpy(dtype=float)
    y_test = df_full.loc[test_mask, "LOAD"].to_numpy(dtype=float)

    # Fit mask for validation evaluation excludes val_mask and test_mask
    val_first = int(np.flatnonzero(val_mask)[0])
    fit_val_mask = np.zeros(len(df_full), dtype=bool)
    fit_val_mask[:val_first] = True

    rows = []
    cache: Dict[str, Dict[str, np.ndarray]] = {}

    for i in range(1, 26):
        w_col = f"w{i}"
        temp_w = df_full[w_col].to_numpy(dtype=float)

        # 1. Fit up to val_mask and evaluate on val_mask
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
        )
        val_oracle_rmse = float(np.sqrt(np.mean((y_val - val_oracle_pt) ** 2)))
        val_metrics = evaluate_all_metrics(y_val, val_scen_q, val_scen_mu)

        # 2. Fit on full train_mask and evaluate on test_mask
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
    Compare 6 representative weather station selection and aggregation strategies from Table 6:
    1. Equal-Weight 25 Stations Temperature Mean (OxMath / Mangalova / Bidong baseline)
    2. Single Best Point-Forecast Station (Selected by min Val_Oracle_RMSE)
    3. Ziel-Style Top-2 Goodness-of-Fit Stations (Top 2 by Val_Oracle_RMSE)
    4. Tololo-Style GCV Top-4 Stations (Greedy forward selection minimizing Val_Oracle_RMSE)
    5. Point-Optimal Weighted Ensemble (Inverse-MSE weighting on Val_Oracle_RMSE)
    6. Adada-Style Two-Stage GCV + Probabilistic EWA (Initial top-7 GCV pool refined via
       Exponentially Weighted Algorithm on Validation Pinball Loss)
    """
    y_val = df_full.loc[val_mask, "LOAD"].to_numpy(dtype=float)
    y_test = df_full.loc[test_mask, "LOAD"].to_numpy(dtype=float)

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
        df_full, temp_mean25, test_mask, scenario_years=test_scen_years, ar_half_life_hours=36.0
    )
    record_strategy("1. All-25 Equal Mean (OxMath/Bidong)", "w1..w25 (equal)", q_25, mu_25, oracle_25)

    # Sort stations by Validation Point (Oracle) RMSE
    sorted_by_point = df_stations.sort_values("Val_Oracle_RMSE")["Station"].tolist()

    # Strategy 2: Best-1 Point Station
    best1 = sorted_by_point[0]
    c1 = station_cache[best1]
    oracle_1 = float(np.sqrt(np.mean((y_test - c1["test_oracle_pt"]) ** 2)))
    record_strategy("2. Best-1 Point Station", best1, c1["test_scen_q"], c1["test_scen_mu"], oracle_1)

    # Strategy 3: Ziel-Style Top-2 Goodness-of-Fit Stations (Temperature Average of Top-2 Point Stations)
    top2 = sorted_by_point[:2]
    temp_top2 = df_full[top2].mean(axis=1).to_numpy(dtype=float)
    gam_top2 = HourlySplineGAM().fit(df_full, temp_top2, train_mask)
    oracle_top2 = float(np.sqrt(np.mean((y_test - gam_top2.predict_point(df_full, temp_top2, test_mask)) ** 2)))
    q_top2, mu_top2 = gam_top2.predict_quantiles_via_scenarios(
        df_full, temp_top2, test_mask, scenario_years=test_scen_years, ar_half_life_hours=36.0
    )
    record_strategy("3. Top-2 Point Stations (Ziel style)", ", ".join(top2), q_top2, mu_top2, oracle_top2)

    # Strategy 4: Tololo-Style Greedy Forward GCV Top-4 Stations (Minimizing Val Point RMSE)
    selected_gcv: List[str] = [sorted_by_point[0]]
    remaining = [s for s in sorted_by_point[1:]]
    for _ in range(3):
        best_cand = None
        best_rmse = float("inf")
        for cand in remaining:
            trial = selected_gcv + [cand]
            pred_val = np.mean([station_cache[s]["val_oracle_pt"] for s in trial], axis=0)
            rmse_val = float(np.sqrt(np.mean((y_val - pred_val) ** 2)))
            if rmse_val < best_rmse:
                best_rmse = rmse_val
                best_cand = cand
        if best_cand is not None:
            selected_gcv.append(best_cand)
            remaining.remove(best_cand)

    temp_gcv4 = df_full[selected_gcv].mean(axis=1).to_numpy(dtype=float)
    gam_gcv4 = HourlySplineGAM().fit(df_full, temp_gcv4, train_mask)
    oracle_gcv4 = float(np.sqrt(np.mean((y_test - gam_gcv4.predict_point(df_full, temp_gcv4, test_mask)) ** 2)))
    q_gcv4, mu_gcv4 = gam_gcv4.predict_quantiles_via_scenarios(
        df_full, temp_gcv4, test_mask, scenario_years=test_scen_years, ar_half_life_hours=36.0
    )
    record_strategy("4. GCV Top-4 Point Subset (Tololo style)", ", ".join(selected_gcv), q_gcv4, mu_gcv4, oracle_gcv4)

    # Strategy 5: Adada-Style Two-Stage GCV + Probabilistic EWA (Exponentially Weighted Algorithm on Pinball Loss)
    # Stage 1: Take initial pool of 7 stations based on GCV / Point accuracy (exact Adada Table 6 protocol)
    pool7 = sorted_by_point[:7]

    # Stage 2: Refine selection within the 7-station GCV pool using EWA on validation Pinball Loss + GCV
    val_pinball = np.array(
        [df_stations.loc[df_stations["Station"] == s, "Val_Pinball"].iloc[0] for s in pool7],
        dtype=float,
    )
    val_gcv = np.array(
        [df_stations.loc[df_stations["Station"] == s, "Val_Oracle_RMSE"].iloc[0] for s in pool7],
        dtype=float,
    )
    # Normalized composite probabilistic loss within the 7-station pool
    norm_pb = (val_pinball - np.min(val_pinball)) / max(1e-6, float(np.std(val_pinball)))
    norm_gcv = (val_gcv - np.min(val_gcv)) / max(1e-6, float(np.std(val_gcv)))
    comp_loss = 0.55 * norm_pb + 0.45 * norm_gcv

    # Adada Table 6 protocol: "Initially seven stations based on GCV. Refined selection using EWA, reaching three stations"
    order_prob = np.argsort(comp_loss)[:3]
    ewa_stations = [pool7[idx] for idx in order_prob]
    ewa_losses = comp_loss[order_prob]
    ewa_weights = np.exp(-0.65 * (ewa_losses - np.min(ewa_losses)))
    ewa_weights = ewa_weights / np.sum(ewa_weights)

    # Combine station temperatures via EWA weights and generate probabilistic scenarios
    temp_ewa = np.zeros(len(df_full), dtype=float)
    for w_s, s_name in zip(ewa_weights, ewa_stations):
        temp_ewa += w_s * df_full[s_name].to_numpy(dtype=float)
    gam_ewa = HourlySplineGAM().fit(df_full, temp_ewa, train_mask)
    q_ewa_ens, mu_ewa_ens = gam_ewa.predict_quantiles_via_scenarios(
        df_full, temp_ewa, test_mask, scenario_years=test_scen_years, ar_half_life_hours=36.0
    )
    oracle_ewa = float(np.sqrt(np.mean((y_test - gam_ewa.predict_point(df_full, temp_ewa, test_mask)) ** 2)))
    ewa_desc = ", ".join(f"{s}({w:.2f})" for s, w in zip(ewa_stations, ewa_weights))
    record_strategy("5. Probabilistic EWA 7->3 (Adada style)", ewa_desc, q_ewa_ens, mu_ewa_ens, oracle_ewa)

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

