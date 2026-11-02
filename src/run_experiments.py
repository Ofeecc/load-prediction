"""
Master experiment and visualization script for GEFCom2014-L Static 1-Month Ahead Load Forecasting.

Executes four structured empirical studies:
1. Exploratory Physical & Regime Analysis (`fig1_eda_physics.png`)
2. Experiment 1: Unbundled Scenario Diversity & Two-Stage Probabilistic Modeling (`exp1_scenario_generation.csv`, `fig2_exp1_scenarios.png`)
3. Experiment 2: Validation-Based Weather Station Selection & Weighting (`exp2_25stations_metrics.csv`, `exp2_station_strategies.csv`, `fig3_exp2_stations.png`)
4. Experiment 3: Short-Term Boundary Correction, Thermal Inertia Propagation (1-48h vs 49-120h), & Holiday Phase-Shift (`exp3_multi_horizon.csv`, `exp3_ablation_holiday.csv`, `fig4_exp3_horizon_holiday.png`)
5. Experiment 4: 15-Month Sequential Backtest (Oct 2010 - Dec 2011), Scenario Similarity Weighting, & Pre-Origin Adaptive Gating (`exp4_15month_backtest.csv`, `exp4_15month_summary.csv`, `exp4_scenario_weighting_and_gating.csv`, `fig5_15month_backtest_and_gating.png`)
6. Granular Hourly Predictions & Config Manifest (`results/predictions/`)
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from src.data_loader import load_full_gefcom2014_l, load_task_benchmark
from src.metrics import QUANTILES_99, evaluate_all_metrics, pinball_loss, pinball_loss_matrix
from src.models import DirectCalendarQuantileBaseline, HourlySplineGAM, compute_thermal_features
from src.station_selection import compare_station_selection_strategies, evaluate_all_25_stations

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 180,
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 9.5,
    "legend.fontsize": 8.2,
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "axes.grid": True,
    "grid.alpha": 0.28,
})

TASK_MONTHS_15: List[Tuple[int, int, int, str]] = [
    (1, 2010, 10, "2010-10"),
    (2, 2010, 11, "2010-11"),
    (3, 2010, 12, "2010-12"),
    (4, 2011, 1, "2011-01"),
    (5, 2011, 2, "2011-02"),
    (6, 2011, 3, "2011-03"),
    (7, 2011, 4, "2011-04"),
    (8, 2011, 5, "2011-05"),
    (9, 2011, 6, "2011-06"),
    (10, 2011, 7, "2011-07"),
    (11, 2011, 8, "2011-08"),
    (12, 2011, 9, "2011-09"),
    (13, 2011, 10, "2011-10"),
    (14, 2011, 11, "2011-11"),
    (15, 2011, 12, "2011-12"),
]


def run_all_experiments() -> None:
    out_dir = Path("results/exploratory_rerun")
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    pred_dir = out_dir / "predictions"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    print("[1/6] Loading full GEFCom2014-L dataset...")
    df = load_full_gefcom2014_l("Data/Load")
    temp_25 = df["temp_mean25"].to_numpy(dtype=float)

    # Exploratory case-study masks: Dec 2011 (Task 15), Nov 2011 (Task 14), Jul 2011 (Task 10)
    test_mask_dec = ((df["year"] == 2011) & (df["month"] == 12)).to_numpy()
    val_mask_nov = ((df["year"] == 2011) & (df["month"] == 11)).to_numpy()
    test_mask_jul = ((df["year"] == 2011) & (df["month"] == 7)).to_numpy()

    dec_first = int(np.flatnonzero(test_mask_dec)[0])
    nov_first = int(np.flatnonzero(val_mask_nov)[0])
    jul_first = int(np.flatnonzero(test_mask_jul)[0])

    train_mask_dec = np.zeros(len(df), dtype=bool)
    train_mask_dec[:dec_first] = True

    train_mask_nov = np.zeros(len(df), dtype=bool)
    train_mask_nov[:nov_first] = True

    train_mask_jul = np.zeros(len(df), dtype=bool)
    train_mask_jul[:jul_first] = True

    # =========================================================================
    # STEP 2: Exploratory Physical Analysis & Paper Figure Reproduction
    # =========================================================================
    print("[2/6] Generating EDA & Physical Coupling figure (fig1_eda_physics.png)...")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))

    # Panel A: Full Load & Temperature Timeline showing the 2001-2004 Unlabeled Weather Archive
    ax = axes[0, 0]
    daily_df = df.groupby("op_date").agg(
        load_mean=("LOAD", "mean"),
        temp_mean=("temp_mean25", "mean"),
        year=("year", "first"),
    ).reset_index()
    daily_df["op_date"] = pd.to_datetime(daily_df["op_date"])

    ax_t = ax.twinx()
    ax_t.plot(daily_df["op_date"], daily_df["temp_mean"], color="#e07a5f", alpha=0.45, lw=0.8, label="Daily Mean Temp (°F)")
    ax.plot(daily_df["op_date"], daily_df["load_mean"], color="#1d3557", lw=1.0, label="Daily Mean Load (MW)")
    ax.axvspan(pd.Timestamp("2001-01-01"), pd.Timestamp("2004-12-31"), color="#f4a261", alpha=0.22, label="2001-2004 Unlabeled Weather Archive")
    ax.axvspan(pd.Timestamp("2010-10-01"), pd.Timestamp("2011-12-31"), color="#2a9d8f", alpha=0.25, label="15-Month Backtest Window (Tasks 1–15)")
    ax.set_title("(a) Data Horizon Asymmetry: 11-Yr Weather vs 7-Yr Load")
    ax.set_ylabel("Hourly Load Daily Mean (MW)", color="#1d3557")
    ax_t.set_ylabel("25-Station Mean Temp (°F)", color="#e07a5f")
    ax_t.grid(False)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_t.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left", framealpha=0.9)

    # Panel B: Reproduce & Enhance Paper Fig 4 (2010 Load vs Temperature U-Curve by Day Type / Hour)
    ax = axes[0, 1]
    df_2010 = df[df["year"] == 2010]
    wkday_peak = df_2010[(df_2010["is_weekend"] == 0) & (df_2010["is_major_holiday"] == 0) & (df_2010["hour"].isin([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]))]
    wknd_peak = df_2010[(df_2010["is_weekend"] == 1) & (df_2010["hour"].isin([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]))]
    night_all = df_2010[df_2010["hour"].isin([2, 3, 4, 5])]
    hol_peak = df_2010[(df_2010["is_major_holiday"] == 1) & (df_2010["hour"].isin([10, 11, 12, 13, 14, 15, 16, 17, 18, 19]))]

    ax.scatter(night_all["temp_mean25"], night_all["LOAD"], s=6, alpha=0.25, color="#8d99ae", label="Night Off-Peak (H2–H5)")
    ax.scatter(wknd_peak["temp_mean25"], wknd_peak["LOAD"], s=8, alpha=0.35, color="#2a9d8f", label="Weekend Daytime (H10–H19)")
    ax.scatter(wkday_peak["temp_mean25"], wkday_peak["LOAD"], s=8, alpha=0.35, color="#e63946", label="Weekday Daytime (H10–H19)")
    ax.scatter(hol_peak["temp_mean25"], hol_peak["LOAD"], s=28, alpha=0.9, color="#ffb703", edgecolors="black", linewidth=0.5, label="Table 5 Federal Holidays (H10–H19)")
    ax.axvline(60.0, color="#1d3557", linestyle="--", lw=1.1, label="Comfort Inflection (~60°F)")
    ax.set_title("(b) Nonlinear U-Curve & Holiday Phase-Shift (Paper Fig. 4)")
    ax.set_xlabel("25-Station Mean Temperature (°F)")
    ax.set_ylabel("Hourly Load (MW)")
    ax.legend(loc="upper center", framealpha=0.9)

    # Panel C: Building Thermal Inertia Effect (Same current temperature, different 24h prior mean temperature)
    ax = axes[1, 0]
    df_labeled = df[df["LOAD"].notna()].copy()
    t_feats_25 = compute_thermal_features(temp_25)
    df_labeled["t_mean24"] = t_feats_25["t_mean24"][df["LOAD"].notna().to_numpy()]

    sub_h16 = df_labeled[(df_labeled["hour"] == 16) & (df_labeled["is_weekend"] == 0) & (df_labeled["is_major_holiday"] == 0)]
    bins = np.linspace(20, 92, 19)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    warm_prior_means, cold_prior_means = [], []
    for b0, b1 in zip(bins[:-1], bins[1:]):
        in_bin = sub_h16[(sub_h16["temp_mean25"] >= b0) & (sub_h16["temp_mean25"] < b1)]
        if len(in_bin) >= 10:
            med_m24 = in_bin["t_mean24"].median()
            warm_prior_means.append(in_bin[in_bin["t_mean24"] >= med_m24]["LOAD"].mean())
            cold_prior_means.append(in_bin[in_bin["t_mean24"] < med_m24]["LOAD"].mean())
        else:
            warm_prior_means.append(np.nan)
            cold_prior_means.append(np.nan)

    ax.plot(bin_centers, warm_prior_means, "o-", color="#d62828", lw=2.0, label="Preceding 24h Temp Above Median (Heat Accumulation)")
    ax.plot(bin_centers, cold_prior_means, "s--", color="#003049", lw=2.0, label="Preceding 24h Temp Below Median (Cold Accumulation)")
    ax.set_title("(c) Building Thermal Inertia Hysteresis at Weekday 16:00")
    ax.set_xlabel("Instantaneous Temperature $T_t$ (°F)")
    ax.set_ylabel("Mean Load at 16:00 (MW)")
    ax.legend(loc="upper center")

    # Panel D: December Climate Regimes (2001-2011)
    ax = axes[1, 1]
    dec_summary = df[df["month"] == 12].groupby("year").agg(
        temp_mean=("temp_mean25", "mean"),
        load_mean=("LOAD", "mean"),
    ).reset_index()
    colors_bar = ["#f4a261" if y <= 2004 else ("#e63946" if y == 2010 else ("#2a9d8f" if y == 2011 else "#457b9d")) for y in dec_summary["year"]]
    bars = ax.bar(dec_summary["year"].astype(str), dec_summary["temp_mean"], color=colors_bar, alpha=0.85, edgecolor="black", linewidth=0.6)
    ax.axhline(dec_summary.loc[dec_summary["year"] == 2011, "temp_mean"].iloc[0], color="#2a9d8f", linestyle="--", lw=1.5, label="Dec 2011 Actual Temp (48.5°F)")
    for b, row in zip(bars, dec_summary.itertuples()):
        lbl = f"{row.temp_mean:.1f}°F\n({row.load_mean:.0f}MW)" if not np.isnan(row.load_mean) else f"{row.temp_mean:.1f}°F\n(No Load)"
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.8, lbl, ha="center", va="bottom", fontsize=7.5)
    ax.set_ylim(0, 60)
    ax.set_title("(d) December Climate Regimes (2001–2011): Cold Snap vs Warm Analog")
    ax.set_xlabel("Year (Orange=2001–04 Unlabeled, Red=2010 Benchmark, Green=2011 Target)")
    ax.set_ylabel("December Mean Temperature (°F)")
    ax.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(fig_dir / "fig1_eda_physics.png", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # STEP 3: Experiment 1 — Unbundled Scenario Generation & Two-Stage Modeling
    # =========================================================================
    print("[3/6] Running Experiment 1: Unbundled Scenario Generation & Ablations...")
    exp1_rows = []
    exp1_predictions_dec = {}

    for task_id, yr, mo, m_name, mask_test, mask_train in [
        (15, 2011, 12, "Dec 2011 (Winter, Task 15)", test_mask_dec, train_mask_dec),
        (14, 2011, 11, "Nov 2011 (Autumn, Task 14)", val_mask_nov, train_mask_nov),
        (10, 2011, 7, "Jul 2011 (Summer, Task 10)", test_mask_jul, train_mask_jul),
    ]:
        y_true = df.loc[mask_test, "LOAD"].to_numpy(dtype=float)

        # M0: Official Naive Benchmark
        q_bench = load_task_benchmark(task_id)
        m0 = evaluate_all_metrics(y_true, q_bench)
        m0["Month"] = m_name
        m0["Method"] = "0. Official Benchmark (Same Month Last Year)"
        exp1_rows.append(m0)

        # M1: Direct Calendar Quantiles (No Weather Scenarios)
        cal_mod = DirectCalendarQuantileBaseline(use_holiday_correction=True).fit(df, mask_train)
        q_cal, mu_cal = cal_mod.predict_quantiles(df, mask_test)
        m1 = evaluate_all_metrics(y_true, q_cal, mu_cal)
        m1["Month"] = m_name
        m1["Method"] = "1. Direct Calendar Quantiles (No Weather Scenarios)"
        exp1_rows.append(m1)

        # Fit base HourlySplineGAM
        gam = HourlySplineGAM(
            use_holiday_correction=True,
            use_thermal_inertia=True,
            use_regime_calibration=True,
        ).fit(df, temp_25, mask_train)

        # M2: Weather Paths ONLY (2005..2010, d=0, no residual convolution)
        q_s6_nores, mu_s6_nores = gam.predict_quantiles_via_scenarios(
            df, temp_25, mask_test, scenario_years=list(range(2005, yr + 1)), day_shifts=(0,), include_residual_uncertainty=False
        )
        m2 = evaluate_all_metrics(y_true, q_s6_nores, mu_s6_nores)
        m2["Month"] = m_name
        m2["Method"] = "2. Weather Paths Only (2005–2010, 6 Scen, No Residual)"
        exp1_rows.append(m2)

        # M3: Labeled Weather Years (2005..2010, d=0) + Residual Convolution
        q_s6, mu_s6 = gam.predict_quantiles_via_scenarios(
            df, temp_25, mask_test, scenario_years=list(range(2005, yr + 1)), day_shifts=(0,), include_residual_uncertainty=True
        )
        m3 = evaluate_all_metrics(y_true, q_s6, mu_s6)
        m3["Month"] = m_name
        m3["Method"] = "3. Labeled Weather Years (2005–2010, 6 Scen) + Residual"
        exp1_rows.append(m3)

        # M4a: + Unlabeled Weather Archive (2001..2010, d=0, 10 Scen, No Shift, No Short-Term AR)
        q_s10, mu_s10 = gam.predict_quantiles_via_scenarios(
            df, temp_25, mask_test, scenario_years=list(range(2001, yr + 1)), day_shifts=(0,), include_residual_uncertainty=True, ar_half_life_hours=None
        )
        m4a = evaluate_all_metrics(y_true, q_s10, mu_s10)
        m4a["Month"] = m_name
        m4a["Method"] = "4a. + Unlabeled Weather Archive (2001–2010, 10 Scen, No Shift, No AR)"
        exp1_rows.append(m4a)

        # M4b: 10 Scenarios (d=0) + Unconditional 48h Short-Term Correction ONLY (No Window Shift)
        q_s10_ar, mu_s10_ar = gam.predict_quantiles_via_scenarios(
            df, temp_25, mask_test, scenario_years=list(range(2001, yr + 1)), day_shifts=(0,), include_residual_uncertainty=True, ar_half_life_hours=36.0, adaptive_ar_gating=False
        )
        m4b = evaluate_all_metrics(y_true, q_s10_ar, mu_s10_ar)
        m4b["Month"] = m_name
        m4b["Method"] = "4b. + Short-Term 48h Correction Only (10 Scen, No Shift + Uncond AR)"
        exp1_rows.append(m4b)

        # M5a: + Synoptic Window Shift ONLY (2001..2010, d=±6 -> 70 Scen, No Short-Term AR)
        q_s70_noar, mu_s70_noar = gam.predict_quantiles_via_scenarios(
            df, temp_25, mask_test, scenario_years=list(range(2001, yr + 1)), day_shifts=(-6, -4, -2, 0, 2, 4, 6), include_residual_uncertainty=True, ar_half_life_hours=None
        )
        m5a = evaluate_all_metrics(y_true, q_s70_noar, mu_s70_noar)
        m5a["Month"] = m_name
        m5a["Method"] = "5a. + Window Shift Only (70 Scen, ±6d Shift, No Short-Term AR)"
        exp1_rows.append(m5a)

        # M5b: 70 Scenarios (d=±6) + Unconditional 48h Short-Term Correction
        q_s70_unc, mu_s70_unc = gam.predict_quantiles_via_scenarios(
            df, temp_25, mask_test, scenario_years=list(range(2001, yr + 1)), day_shifts=(-6, -4, -2, 0, 2, 4, 6),
            include_residual_uncertainty=True, ar_half_life_hours=36.0, adaptive_ar_gating=False
        )
        m5b = evaluate_all_metrics(y_true, q_s70_unc, mu_s70_unc)
        m5b["Month"] = m_name
        m5b["Method"] = "5b. + Window Shift & Unconditional 48h AR (70 Scen + Uncond AR)"
        exp1_rows.append(m5b)

        # M5c: 70 Scenarios (d=±6) + Pre-Origin Adaptive Gated Correction & 48h Similarity
        q_s70_adp, mu_s70_adp = gam.predict_quantiles_via_scenarios(
            df, temp_25, mask_test, scenario_years=list(range(2001, yr + 1)), day_shifts=(-6, -4, -2, 0, 2, 4, 6),
            include_residual_uncertainty=True, ar_half_life_hours=36.0, adaptive_ar_gating=True,
            similarity_weight_lambda=0.35, similarity_decay_hours=48.0
        )
        m5c = evaluate_all_metrics(y_true, q_s70_adp, mu_s70_adp)
        m5c["Month"] = m_name
        m5c["Method"] = "5c. + Pre-Origin Adaptive Gated AR & 48h Similarity (70 Scen + Adaptive)"
        exp1_rows.append(m5c)

        # M6: Diagnostic Reference — Oracle True Future Weather + Conditional Residual
        mu_oracle = gam.predict_point(df, temp_25, mask_test)
        df_t = df.loc[mask_test]
        q_oracle = np.zeros((len(y_true), 99), dtype=float)
        for idx_row, (h_val, dow_val) in enumerate(zip(df_t["hour"].to_numpy(), gam.effective_dayofweek(df_t))):
            wk_val = 1 if dow_val >= 5 else 0
            q_oracle[idx_row, :] = mu_oracle[idx_row] + gam.residual_quantiles[(int(h_val), wk_val)] * 0.85
        m6 = evaluate_all_metrics(y_true, q_oracle, mu_oracle)
        m6["Month"] = m_name
        m6["Method"] = "6. [Diagnostic Reference] Oracle True Future Weather + Residual"
        exp1_rows.append(m6)

        if task_id == 15:
            exp1_predictions_dec = {
                "y_true": y_true,
                "q_bench": q_bench,
                "q_cal": q_cal,
                "mu_cal": mu_cal,
                "q_s6_nores": q_s6_nores,
                "q_s70": q_s70_adp,
                "mu_s70": mu_s70_adp,
                "q_oracle": q_oracle,
                "mu_oracle": mu_oracle,
            }

    df_exp1 = pd.DataFrame(exp1_rows)
    cols_exp1 = ["Month", "Method", "Pinball", "Pinball_Tail", "Pinball_Central", "RMSE", "MAE", "MAPE(%)", "PICP_80(%)", "PICP_90(%)", "MIW_90(MW)"]
    df_exp1 = df_exp1[cols_exp1]
    df_exp1.to_csv(tab_dir / "exp1_scenario_generation.csv", index=False)
    print("\n--- Experiment 1 Summary (Dec 2011, Nov 2011, Jul 2011) ---")
    print(df_exp1.to_string(index=False))

    # Plot Figure 2: Experiment 1 Probabilistic Fan Chart & Pinball Loss across Quantiles
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.2), gridspec_kw={"height_ratios": [1.35, 1.0]})

    ax = axes[0]
    hours_x = np.arange(1, len(exp1_predictions_dec["y_true"]) + 1)
    q_s70 = exp1_predictions_dec["q_s70"]
    y_true_dec = exp1_predictions_dec["y_true"]
    dec_m5c = evaluate_all_metrics(y_true_dec, q_s70, exp1_predictions_dec["mu_s70"])
    ax.fill_between(hours_x, q_s70[:, 4], q_s70[:, 94], color="#a8dadc", alpha=0.45, label="90% Prediction Interval ($q_{0.05}$–$q_{0.95}$)")
    ax.fill_between(hours_x, q_s70[:, 9], q_s70[:, 89], color="#457b9d", alpha=0.40, label="80% Prediction Interval ($q_{0.10}$–$q_{0.90}$)")
    ax.fill_between(hours_x, q_s70[:, 24], q_s70[:, 74], color="#1d3557", alpha=0.35, label="50% Prediction Interval ($q_{0.25}$–$q_{0.75}$)")
    ax.plot(hours_x, exp1_predictions_dec["q_bench"][:, 49], color="#e63946", linestyle=":", lw=1.1, alpha=0.75, label="Official Benchmark (Dec 2010 Load, Pinball=34.07)")
    ax.plot(hours_x, exp1_predictions_dec["mu_s70"], color="#1d3557", lw=1.4, label=f"Scenario GAM Mean Forecast (Pinball={dec_m5c['Pinball']:.2f})")
    ax.plot(hours_x, y_true_dec, color="#ffb703", lw=1.3, label=f"Actual Load (Dec 2011 Ground Truth, PICP90={dec_m5c['PICP_90(%)']:.1f}%)")
    ax.axvspan(23 * 24, 26 * 24, color="#f4a261", alpha=0.25, label="Christmas Holiday Window (Dec 24–26)")
    ax.set_title("(a) Static 1-Month Probabilistic Load Forecast on Dec 2011 (744 Hours): Historical Scenario Fan vs Actual Load")
    ax.set_xlabel("Forecast Horizon Hour in December 2011 (1 to 744)")
    ax.set_ylabel("Load (MW)")
    ax.set_xlim(1, 744)
    ax.legend(loc="upper right", ncol=3)

    ax2 = axes[1]
    pb_cal_q = np.mean(pinball_loss_matrix(y_true_dec, exp1_predictions_dec["q_cal"]), axis=0)
    pb_nores_q = np.mean(pinball_loss_matrix(y_true_dec, exp1_predictions_dec["q_s6_nores"]), axis=0)
    pb_s70_q = np.mean(pinball_loss_matrix(y_true_dec, exp1_predictions_dec["q_s70"]), axis=0)
    pb_ora_q = np.mean(pinball_loss_matrix(y_true_dec, exp1_predictions_dec["q_oracle"]), axis=0)

    ax2.plot(QUANTILES_99, pb_cal_q, "--", color="#e63946", lw=1.8, label=f"1. Direct Calendar QR (No Weather Scen, Mean={np.mean(pb_cal_q):.2f})")
    ax2.plot(QUANTILES_99, pb_nores_q, "-.", color="#f4a261", lw=1.8, label=f"2. Weather Paths Only (2005–2010, No Residual, Mean={np.mean(pb_nores_q):.2f})")
    ax2.plot(QUANTILES_99, pb_s70_q, "-", color="#1d3557", lw=2.2, label=f"5c. Two-Stage Scenario GAM (70 Scen + Residual + Adaptive, Mean={np.mean(pb_s70_q):.2f})")
    ax2.plot(QUANTILES_99, pb_ora_q, ":", color="#2a9d8f", lw=2.0, label=f"6. [Diagnostic Ref] Oracle True Future Weather + Residual (Mean={np.mean(pb_ora_q):.2f})")
    ax2.set_title("(b) Pinball Loss Curve Across All 99 Quantiles ($q=0.01\\dots 0.99$) on Dec 2011")
    ax2.set_xlabel("Quantile Level $q$")
    ax2.set_ylabel("Mean Pinball Loss (MW)")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.savefig(fig_dir / "fig2_exp1_scenarios.png")
    plt.close(fig)

    # =========================================================================
    # STEP 4: Experiment 2 — Weather Station Selection & Aggregation
    # =========================================================================
    print("[4/6] Running Experiment 2: Validation-Based Weather Station Selection across all 25 stations...")
    df_stations, station_cache = evaluate_all_25_stations(
        df_full=df,
        train_mask=train_mask_dec,
        val_mask=val_mask_nov,
        test_mask=test_mask_dec,
        val_scen_years=list(range(2001, 2011)),
        test_scen_years=list(range(2001, 2011)),
    )
    df_stations.to_csv(tab_dir / "exp2_25stations_metrics.csv", index=False)

    df_strat = compare_station_selection_strategies(
        df_full=df,
        df_stations=df_stations,
        station_cache=station_cache,
        train_mask=train_mask_dec,
        val_mask=val_mask_nov,
        test_mask=test_mask_dec,
        test_scen_years=list(range(2001, 2011)),
    )
    df_strat.to_csv(tab_dir / "exp2_station_strategies.csv", index=False)
    print("\n--- Experiment 2 Station Selection & Weighting Strategies (Dec 2011) ---")
    print(df_strat.to_string(index=False))

    rho_val, p_val = spearmanr(df_stations["Val_Oracle_RMSE"], df_stations["Val_Pinball"])
    rho_test, p_test = spearmanr(df_stations["Test_Oracle_RMSE"], df_stations["Test_Pinball"])
    print(f"Station Rank Correlation (Oracle Point RMSE vs Scenario Pinball Loss): Val Spearman rho={rho_val:.3f} (p={p_val:.3f}), Test Spearman rho={rho_test:.3f} (p={p_test:.3f})")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))
    ax = axes[0]
    sc = ax.scatter(
        df_stations["Test_Oracle_RMSE"],
        df_stations["Test_Pinball"],
        c=df_stations["Test_Pinball_Tail"],
        cmap="viridis",
        s=75,
        edgecolors="black",
        linewidth=0.7,
        zorder=3,
    )
    for row in df_stations.itertuples():
        ax.annotate(
            row.Station,
            (row.Test_Oracle_RMSE, row.Test_Pinball),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=8,
        )
    cb = plt.colorbar(sc, ax=ax)
    cb.set_label("Extreme Tail Pinball Loss ($q \\leq 0.1$ or $q \\geq 0.9$)")

    for row, marker, col in zip(
        df_strat.itertuples(),
        ["s", "^", "D", "P", "*", "X"],
        ["#8d99ae", "#e63946", "#f4a261", "#457b9d", "#2a9d8f", "#6d597a"],
    ):
        ax.scatter(
            row.Test_Oracle_RMSE,
            row.Pinball,
            marker=marker,
            s=155 if marker != "*" else 210,
            color=col,
            edgecolors="black",
            linewidth=1.1,
            label=row.Strategy,
            zorder=5,
        )
    ax.set_title(f"(a) 25 Stations: Point RMSE vs Probabilistic Pinball ($\\rho={rho_test:.2f}$)")
    ax.set_xlabel("Point Forecast RMSE under True Temp (MW) [Lower = Better Point]")
    ax.set_ylabel("1-Month Probabilistic Pinball Loss (MW)")
    ax.set_ylim(8.05, 9.52)
    ax.legend(loc="upper right", fontsize=7.5)

    ax2 = axes[1]
    strat_names = [
        "1. All-25\nEqual Mean",
        "2. Best-1\nVal-RMSE",
        "3. Top-2\nMean Temp",
        "4. Forward-4\nTemp Subset",
        "5. Val-Loss\nWeighted Temp",
        "6. Val-Pinball\nEWA Forecast",
    ]
    x_pos = np.arange(len(strat_names))
    width = 0.36
    b1 = ax2.bar(x_pos - width / 2, df_strat["Pinball"], width, color="#1d3557", label="Overall Pinball Loss (All 99 Quantiles)")
    b2 = ax2.bar(x_pos + width / 2, df_strat["Pinball_Tail"] * 2.0, width, color="#e76f51", label="2x Tail Pinball Loss ($q \\leq 0.10, q \\geq 0.90$)")
    for b in b1:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    for b, val in zip(b2, df_strat["Pinball_Tail"]):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12, f"{val:.2f}", ha="center", va="bottom", fontsize=7.5)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(strat_names, fontsize=8)
    ax2.set_ylim(0, max(df_strat["Pinball"]) * 1.24)
    ax2.set_title("(b) Validation-Based Station Selection & Weighting Comparison")
    ax2.set_ylabel("Pinball Loss (MW)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(fig_dir / "fig3_exp2_stations.png", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # STEP 5: Experiment 3 — Short-Term Correction, Thermal Propagation, & Holidays
    # =========================================================================
    print("[5/6] Running Experiment 3: Horizon Breakdown (Direct 1-48h vs 49-120h Thermal Propagation) & Holiday Ablations...")
    horizon_rows = []
    for m_label, mask_test, mask_train in [
        ("Dec 2011 (Task 15)", test_mask_dec, train_mask_dec),
        ("Nov 2011 (Task 14)", val_mask_nov, train_mask_nov),
    ]:
        y_t = df.loc[mask_test, "LOAD"].to_numpy(dtype=float)
        gam_m = HourlySplineGAM(use_holiday_correction=True, use_thermal_inertia=True, use_regime_calibration=True).fit(df, temp_25, mask_train)

        horizon_configs = [
            ("1. No Short-Term Correction (Pure 70 Scenarios)", None, False, False, 0.0, None),
            ("2. Unconditional 48h AR (Freeze 49–120h Thermal Propagation)", 36.0, False, True, 0.0, None),
            ("3. Unconditional 48h AR (With 49–120h Thermal Propagation)", 36.0, False, False, 0.0, None),
            ("4. Pre-Origin Adaptive Gated AR + 48h Similarity", 36.0, True, False, 0.35, 48.0),
        ]

        for ar_label, ar_hl, adp_gate, freeze_th, sim_lam, sim_dec in horizon_configs:
            q_pred, mu_pred = gam_m.predict_quantiles_via_scenarios(
                df, temp_25, mask_test, scenario_years=list(range(2001, 2011)),
                ar_half_life_hours=ar_hl, adaptive_ar_gating=adp_gate,
                freeze_thermal_inertia_after_cutoff=freeze_th,
                similarity_weight_lambda=sim_lam, similarity_decay_hours=sim_dec,
            )
            for h_slice_name, sl in [
                ("1–48h (Days 1–2 Direct Correction Window)", slice(0, 48)),
                ("49–120h (Days 3–5 Thermal Inertia Propagation Window)", slice(48, 120)),
                ("121–End (Days 6–31 Unmodified Medium-Term Window)", slice(120, None)),
                ("Full Month (1–End)", slice(0, None)),
            ]:
                m_sl = evaluate_all_metrics(y_t[sl], q_pred[sl], mu_pred[sl])
                horizon_rows.append({
                    "Month": m_label,
                    "Strategy": ar_label,
                    "Horizon_Window": h_slice_name,
                    "Pinball": m_sl["Pinball"],
                    "Pinball_Tail": m_sl["Pinball_Tail"],
                    "RMSE": m_sl["RMSE"],
                    "MAE": m_sl["MAE"],
                    "PICP_90(%)": m_sl["PICP_90(%)"],
                })

    df_horizon = pd.DataFrame(horizon_rows)
    df_horizon.to_csv(tab_dir / "exp3_multi_horizon.csv", index=False)
    print("\n--- Experiment 3A: Multi-Horizon Short-Term Correction & Thermal Propagation Breakdown ---")
    print(df_horizon.to_string(index=False))

    # Part B: Component Ablation & Table 5 Federal Holiday Phase-Shift Analysis
    ablation_rows = []
    holiday_window_curves = {}

    for m_label, mask_test, mask_train in [
        ("Nov 2011 (Thanksgiving Month)", val_mask_nov, train_mask_nov),
        ("Dec 2011 (Christmas/NYE Month)", test_mask_dec, train_mask_dec),
    ]:
        y_t = df.loc[mask_test, "LOAD"].to_numpy(dtype=float)
        df_t = df.loc[mask_test]
        hol_mask_local = ((df_t["is_holiday"] == 1) | (df_t["is_holiday_bridge"] == 1)).to_numpy()

        configs = [
            ("1. Base Spline GAM (Instantaneous Temp Only, No Holiday Fix)", False, False, False, None, False, 0.0),
            ("2. + Building Thermal Inertia (24h/72h Rolling & Lags)", False, True, False, None, False, 0.0),
            ("3. + Weather-Normalized Local Regime Calibration (Uncanceled)", False, True, True, None, False, 0.0),
            ("4. + Table 5 Holiday & Bridge Weekend Phase-Shift", True, True, True, None, False, 0.0),
            ("5. + Unconditional 48h Short-Term Persistence", True, True, True, 36.0, False, 0.0),
            ("6. + Pre-Origin Adaptive Gated AR & 48h Similarity", True, True, True, 36.0, True, 0.35),
        ]

        for cfg_name, use_hol, use_therm, use_reg, ar_hl, adp_gate, sim_lam in configs:
            g = HourlySplineGAM(
                use_holiday_correction=use_hol,
                use_thermal_inertia=use_therm,
                use_regime_calibration=use_reg,
            ).fit(df, temp_25, mask_train)

            q_p, mu_p = g.predict_quantiles_via_scenarios(
                df, temp_25, mask_test, scenario_years=list(range(2001, 2011)),
                ar_half_life_hours=ar_hl, adaptive_ar_gating=adp_gate,
                similarity_weight_lambda=sim_lam, similarity_decay_hours=48.0 if sim_lam > 0 else None,
            )
            y_ora = g.predict_point(df, temp_25, mask_test)

            m_all = evaluate_all_metrics(y_t, q_p, mu_p)
            m_hol = evaluate_all_metrics(y_t[hol_mask_local], q_p[hol_mask_local], mu_p[hol_mask_local])
            ora_hol_rmse = float(np.sqrt(np.mean((y_t[hol_mask_local] - y_ora[hol_mask_local]) ** 2)))
            ora_all_rmse = float(np.sqrt(np.mean((y_t - y_ora) ** 2)))

            ablation_rows.append({
                "Month": m_label,
                "Configuration": cfg_name,
                "Full_Pinball": m_all["Pinball"],
                "Full_Scen_RMSE": m_all["RMSE"],
                "Full_Oracle_RMSE": ora_all_rmse,
                "Holiday_Pinball": m_hol["Pinball"],
                "Holiday_Scen_RMSE": m_hol["RMSE"],
                "Holiday_Oracle_RMSE": ora_hol_rmse,
                "Full_PICP_90(%)": m_all["PICP_90(%)"],
            })

            if m_label.startswith("Nov") and cfg_name.startswith("3."):
                holiday_window_curves["nov_no_hol_ora"] = y_ora
                holiday_window_curves["nov_no_hol_mu"] = mu_p
                holiday_window_curves["nov_no_hol_q"] = q_p
            if m_label.startswith("Nov") and cfg_name.startswith("6."):
                holiday_window_curves["nov_full_ora"] = y_ora
                holiday_window_curves["nov_full_mu"] = mu_p
                holiday_window_curves["nov_full_q"] = q_p
                holiday_window_curves["nov_true"] = y_t
            if m_label.startswith("Dec") and cfg_name.startswith("3."):
                holiday_window_curves["dec_no_hol_ora"] = y_ora
                holiday_window_curves["dec_no_hol_mu"] = mu_p
                holiday_window_curves["dec_no_hol_q"] = q_p
            if m_label.startswith("Dec") and cfg_name.startswith("6."):
                holiday_window_curves["dec_full_ora"] = y_ora
                holiday_window_curves["dec_full_mu"] = mu_p
                holiday_window_curves["dec_full_q"] = q_p
                holiday_window_curves["dec_true"] = y_t

    df_ablation = pd.DataFrame(ablation_rows)
    df_ablation.to_csv(tab_dir / "exp3_ablation_holiday.csv", index=False)
    print("\n--- Experiment 3B: Progressive Heuristic Ablation & Holiday Effect ---")
    print(df_ablation.to_string(index=False))

    # Plot Figure 4: Multi-Horizon Short-Term Comparison (Dec vs Nov) & Holiday Curves (Real Forecast + Oracle)
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6))

    # Panel A: First 72 hours of Dec 2011 vs Nov 2011 showing why Unconditional AR helps Dec but hurts Nov, while Adaptive succeeds in both
    ax = axes[0, 0]
    gam_dec = HourlySplineGAM().fit(df, temp_25, train_mask_dec)
    _, mu_dec_no = gam_dec.predict_quantiles_via_scenarios(df, temp_25, test_mask_dec, list(range(2001, 2011)), ar_half_life_hours=None)
    _, mu_dec_unc = gam_dec.predict_quantiles_via_scenarios(df, temp_25, test_mask_dec, list(range(2001, 2011)), ar_half_life_hours=36.0, adaptive_ar_gating=False)
    _, mu_dec_adp = gam_dec.predict_quantiles_via_scenarios(df, temp_25, test_mask_dec, list(range(2001, 2011)), ar_half_life_hours=36.0, adaptive_ar_gating=True, similarity_weight_lambda=0.35, similarity_decay_hours=48.0)
    h72 = np.arange(1, 73)
    ax.plot(h72, y_true_dec[:72], "o-", color="#ffb703", markersize=3.2, lw=1.8, label="Actual Load (Dec 1–3, 2011)")
    ax.plot(h72, mu_dec_no[:72], "--", color="#e63946", lw=1.5, label=f"No Short-Term Correction (1–48h RMSE={np.sqrt(np.mean((y_true_dec[:48]-mu_dec_no[:48])**2)):.1f} MW)")
    ax.plot(h72, mu_dec_unc[:72], "-.", color="#457b9d", lw=1.6, label=f"Unconditional 48h AR (1–48h RMSE={np.sqrt(np.mean((y_true_dec[:48]-mu_dec_unc[:48])**2)):.1f} MW)")
    ax.plot(h72, mu_dec_adp[:72], "-", color="#1d3557", lw=2.0, label=f"Adaptive Gated AR + 48h Sim (1–48h RMSE={np.sqrt(np.mean((y_true_dec[:48]-mu_dec_adp[:48])**2)):.1f} MW)")
    ax.axvline(48, color="#2a9d8f", linestyle=":", lw=1.5, label="48h Direct Injection Cutoff")
    ax.set_title("(a) Dec 2011 (1–72h): Persistent Cold Front Favors Boundary Correction")
    ax.set_xlabel("Forecast Lead Time (Hours from 2011-12-01 01:00)")
    ax.set_ylabel("Load (MW)")
    ax.legend(loc="upper right", fontsize=7.8)

    # Panel B: Bar comparison of 1-48h (Direct) and 49-120h (Thermal Propagation) on Dec 2011 vs Nov 2011
    ax = axes[0, 1]
    sub_dec = df_horizon[df_horizon["Month"].str.startswith("Dec")]
    sub_nov = df_horizon[df_horizon["Month"].str.startswith("Nov")]
    dec_48_3 = [
        sub_dec[(sub_dec["Strategy"].str.startswith("1.")) & (sub_dec["Horizon_Window"].str.startswith("1–48h"))]["Pinball"].iloc[0],
        sub_dec[(sub_dec["Strategy"].str.startswith("3.")) & (sub_dec["Horizon_Window"].str.startswith("1–48h"))]["Pinball"].iloc[0],
        sub_dec[(sub_dec["Strategy"].str.startswith("4.")) & (sub_dec["Horizon_Window"].str.startswith("1–48h"))]["Pinball"].iloc[0],
    ]
    nov_48_3 = [
        sub_nov[(sub_nov["Strategy"].str.startswith("1.")) & (sub_nov["Horizon_Window"].str.startswith("1–48h"))]["Pinball"].iloc[0],
        sub_nov[(sub_nov["Strategy"].str.startswith("3.")) & (sub_nov["Horizon_Window"].str.startswith("1–48h"))]["Pinball"].iloc[0],
        sub_nov[(sub_nov["Strategy"].str.startswith("4.")) & (sub_nov["Horizon_Window"].str.startswith("1–48h"))]["Pinball"].iloc[0],
    ]
    nov_120_3 = [
        sub_nov[(sub_nov["Strategy"].str.startswith("2.")) & (sub_nov["Horizon_Window"].str.startswith("49–120h"))]["Pinball"].iloc[0],
        sub_nov[(sub_nov["Strategy"].str.startswith("3.")) & (sub_nov["Horizon_Window"].str.startswith("49–120h"))]["Pinball"].iloc[0],
        sub_nov[(sub_nov["Strategy"].str.startswith("4.")) & (sub_nov["Horizon_Window"].str.startswith("49–120h"))]["Pinball"].iloc[0],
    ]
    x_lbls = ["1. No Short-Term\nCorrection", "2. Unconditional 48h\nPersistence", "3. Pre-Origin Adaptive\nGated AR + 48h Sim"]
    xw = np.arange(len(x_lbls))
    wb = 0.25
    b_dec = ax.bar(xw - wb, dec_48_3, wb, color="#1d3557", alpha=0.92, label="Dec 2011: 1–48h Direct Window")
    b_nov = ax.bar(xw, nov_48_3, wb, color="#e76f51", alpha=0.92, label="Nov 2011: 1–48h Direct Window")
    b_n120 = ax.bar(xw + wb, nov_120_3, wb, color="#2a9d8f", alpha=0.85, label="Nov 2011: 49–120h Thermal Prop.")
    for bars_grp in (b_dec, b_nov, b_n120):
        for b in bars_grp:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.08, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    ax.set_xticks(xw)
    ax.set_xticklabels(x_lbls, fontsize=8.2)
    ax.set_ylim(0, max(dec_48_3) * 1.28)
    ax.set_title("(b) Direct (1–48h) vs Thermal Propagation (49–120h) Pinball Loss")
    ax.set_ylabel("Mean Pinball Loss (MW)")
    ax.legend(loc="upper right", fontsize=7.6)

    # Panel C: Thanksgiving Window Zoom-in (Nov 23 to Nov 27, 2011) showing BOTH Month-Ahead Scenario Forecast AND Oracle Diagnostic
    ax = axes[1, 0]
    s_nov, e_nov = 22 * 24, 27 * 24
    hrs_nov = np.arange(s_nov + 1, e_nov + 1)
    q_nov_full = holiday_window_curves["nov_full_q"][s_nov:e_nov]
    ax.fill_between(hrs_nov, q_nov_full[:, 4], q_nov_full[:, 94], color="#a8dadc", alpha=0.38, label="Month-Ahead 90% PI (With Holiday Fix)")
    ax.plot(hrs_nov, holiday_window_curves["nov_true"][s_nov:e_nov], "o-", color="#ffb703", markersize=2.8, lw=1.7, label="Actual Load (Nov 23–27, Thanksgiving)")
    ax.plot(hrs_nov, holiday_window_curves["nov_no_hol_mu"][s_nov:e_nov], "--", color="#e63946", lw=1.5, label="Month-Ahead Mean (No Holiday Fix)")
    ax.plot(hrs_nov, holiday_window_curves["nov_full_mu"][s_nov:e_nov], "-", color="#1d3557", lw=1.9, label="Month-Ahead Mean (With Holiday Phase-Shift)")
    ax.plot(hrs_nov, holiday_window_curves["nov_full_ora"][s_nov:e_nov], ":", color="#2a9d8f", lw=1.6, label="[Diagnostic] Oracle True-Temp Fit (With Holiday Fix)")
    ax.axvspan(23 * 24, 25 * 24, color="#f4a261", alpha=0.22, label="Thanksgiving + Black Friday")
    ax.set_ylim(70, 220)
    ax.set_title("(c) Thanksgiving Window (Nov 23–27): Month-Ahead Forecast vs Oracle Diagnostic")
    ax.set_xlabel("Hour of Month in November 2011")
    ax.set_ylabel("Load (MW)")
    ax.legend(loc="upper right", fontsize=7.4, ncol=2)

    # Panel D: Christmas Window Zoom-in (Dec 23 to Dec 27, 2011) showing BOTH Month-Ahead Scenario Forecast AND Oracle Diagnostic
    ax = axes[1, 1]
    s_dec, e_dec = 22 * 24, 27 * 24
    hrs_dec = np.arange(s_dec + 1, e_dec + 1)
    q_dec_full = holiday_window_curves["dec_full_q"][s_dec:e_dec]
    ax.fill_between(hrs_dec, q_dec_full[:, 4], q_dec_full[:, 94], color="#a8dadc", alpha=0.38, label="Month-Ahead 90% PI (With Holiday Fix)")
    ax.plot(hrs_dec, holiday_window_curves["dec_true"][s_dec:e_dec], "o-", color="#ffb703", markersize=2.8, lw=1.7, label="Actual Load (Dec 23–27, Christmas)")
    ax.plot(hrs_dec, holiday_window_curves["dec_no_hol_mu"][s_dec:e_dec], "--", color="#e63946", lw=1.5, label="Month-Ahead Mean (No Holiday Fix)")
    ax.plot(hrs_dec, holiday_window_curves["dec_full_mu"][s_dec:e_dec], "-", color="#1d3557", lw=1.9, label="Month-Ahead Mean (With Holiday Phase-Shift)")
    ax.plot(hrs_dec, holiday_window_curves["dec_full_ora"][s_dec:e_dec], ":", color="#2a9d8f", lw=1.6, label="[Diagnostic] Oracle True-Temp Fit (With Holiday Fix)")
    ax.axvspan(23 * 24, 26 * 24, color="#f4a261", alpha=0.22, label="Christmas + Observed Mon (Dec 24–26)")
    ax.set_ylim(65, 235)
    ax.set_title("(d) Christmas Window (Dec 23–27): Month-Ahead Forecast vs Oracle Diagnostic")
    ax.set_xlabel("Hour of Month in December 2011")
    ax.set_ylabel("Load (MW)")
    ax.legend(loc="upper right", fontsize=7.4, ncol=2)

    plt.tight_layout()
    plt.savefig(fig_dir / "fig4_exp3_horizon_holiday.png", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # STEP 6: Experiment 4 — 15-Month Sequential Backtest & Core Research Question
    # =========================================================================
    print("[6/6] Running Experiment 4: 15-Month Sequential Backtest (Tasks 1–15) & Saving Hourly Predictions...")
    backtest_rows = []
    gating_diag_rows = []
    hourly_records = []
    npz_q_noar_list = []
    npz_q_uncond_list = []
    npz_q_adapt_list = []
    npz_actual_list = []
    npz_task_list = []
    npz_ts_list = []

    cached_models_15 = {}

    for task_id, yr, mo, m_lbl in TASK_MONTHS_15:
        mask_test = ((df["year"] == yr) & (df["month"] == mo)).to_numpy()
        start_idx = int(np.flatnonzero(mask_test)[0])
        mask_train = np.zeros(len(df), dtype=bool)
        mask_train[:start_idx] = True
        y_true = df.loc[mask_test, "LOAD"].to_numpy(dtype=float)
        df_t = df.loc[mask_test]

        # Fit Calendar Baseline & Base HourlySplineGAM strictly on data prior to start_idx
        cal_m = DirectCalendarQuantileBaseline(use_holiday_correction=True).fit(df, mask_train)
        q_cal, mu_cal = cal_m.predict_quantiles(df, mask_test)

        gam = HourlySplineGAM(
            use_holiday_correction=True,
            use_thermal_inertia=True,
            use_regime_calibration=True,
        ).fit(df, temp_25, mask_train)

        yrs_6 = list(range(2005, yr + 1))
        yrs_10 = list(range(2001, yr + 1))

        diag = gam.compute_pre_origin_diagnostics(df, temp_25, mask_test, yrs_10)

        q_bench = load_task_benchmark(task_id)
        q_s6, mu_s6 = gam.predict_quantiles_via_scenarios(df, temp_25, mask_test, yrs_6, day_shifts=(0,), ar_half_life_hours=None)
        q_s10, mu_s10 = gam.predict_quantiles_via_scenarios(df, temp_25, mask_test, yrs_10, day_shifts=(0,), ar_half_life_hours=None)
        q_s10_ar, mu_s10_ar = gam.predict_quantiles_via_scenarios(df, temp_25, mask_test, yrs_10, day_shifts=(0,), ar_half_life_hours=36.0, adaptive_ar_gating=False)
        q_s70_noar, mu_s70_noar = gam.predict_quantiles_via_scenarios(df, temp_25, mask_test, yrs_10, day_shifts=(-6, -4, -2, 0, 2, 4, 6), ar_half_life_hours=None)
        q_s70_unc, mu_s70_unc = gam.predict_quantiles_via_scenarios(df, temp_25, mask_test, yrs_10, day_shifts=(-6, -4, -2, 0, 2, 4, 6), ar_half_life_hours=36.0, adaptive_ar_gating=False)
        q_s70_adp, mu_s70_adp = gam.predict_quantiles_via_scenarios(df, temp_25, mask_test, yrs_10, day_shifts=(-6, -4, -2, 0, 2, 4, 6), ar_half_life_hours=36.0, adaptive_ar_gating=True)
        q_s70_hyb, mu_s70_hyb = gam.predict_quantiles_via_scenarios(
            df, temp_25, mask_test, yrs_10, day_shifts=(-6, -4, -2, 0, 2, 4, 6),
            ar_half_life_hours=36.0, adaptive_ar_gating=True,
            similarity_weight_lambda=0.35, similarity_decay_hours=48.0,
        )

        mu_ora = gam.predict_point(df, temp_25, mask_test)
        q_ora = np.zeros((len(y_true), 99), dtype=float)
        for idx_row, (h_val, dow_val) in enumerate(zip(df_t["hour"].to_numpy(), gam.effective_dayofweek(df_t))):
            wk_val = 1 if dow_val >= 5 else 0
            q_ora[idx_row, :] = mu_ora[idx_row] + gam.residual_quantiles[(int(h_val), wk_val)] * 0.85

        cached_models_15[m_lbl] = (gam, mask_test, y_true, yrs_10)

        method_outputs = [
            ("0. Official Benchmark", q_bench, q_bench[:, 49]),
            ("1. Direct Calendar Quantiles", q_cal, mu_cal),
            ("2. 6-Yr Weather Scenarios (2005–Prior, d=0)", q_s6, mu_s6),
            ("3. 10-Yr Weather Scenarios (2001–Prior, d=0)", q_s10, mu_s10),
            ("4. 10-Yr Scenarios + Uncond 48h AR (d=0 + AR)", q_s10_ar, mu_s10_ar),
            ("5. 70 Scenarios Only (2001–Prior, d=±6, No AR)", q_s70_noar, mu_s70_noar),
            ("6. 70 Scenarios + Unconditional 48h AR", q_s70_unc, mu_s70_unc),
            ("7. 70 Scenarios + Pre-Origin Adaptive Gated AR", q_s70_adp, mu_s70_adp),
            ("8. 70 Scenarios + Adaptive AR + 48h Similarity", q_s70_hyb, mu_s70_hyb),
            ("9. [Diagnostic Ref] Oracle Future Temp + Residual", q_ora, mu_ora),
        ]

        for m_name, q_pred, mu_pred in method_outputs:
            m_full = evaluate_all_metrics(y_true, q_pred, mu_pred)
            m_48 = evaluate_all_metrics(y_true[:48], q_pred[:48], mu_pred[:48])
            backtest_rows.append({
                "Task_ID": task_id,
                "Month": m_lbl,
                "Method": m_name,
                "Pinball": m_full["Pinball"],
                "Pinball_Tail": m_full["Pinball_Tail"],
                "Pinball_Central": m_full["Pinball_Central"],
                "Pinball_1_48h": m_48["Pinball"],
                "RMSE_1_48h": m_48["RMSE"],
                "RMSE": m_full["RMSE"],
                "MAE": m_full["MAE"],
                "MAPE(%)": m_full["MAPE(%)"],
                "PICP_80(%)": m_full["PICP_80(%)"],
                "PICP_90(%)": m_full["PICP_90(%)"],
                "MIW_90(MW)": m_full["MIW_90(MW)"],
            })

        pb48_noar = float(pinball_loss(y_true[:48], q_s70_noar[:48]))
        pb48_unc = float(pinball_loss(y_true[:48], q_s70_unc[:48]))
        pb48_hyb = float(pinball_loss(y_true[:48], q_s70_hyb[:48]))
        gating_diag_rows.append({
            "Task_ID": task_id,
            "Month": m_lbl,
            "PreOrigin_dT_Abs(F)": round(diag["dt_day_abs"], 2),
            "PreOrigin_dT_Signed(F)": round(diag["dt_day_signed"], 2),
            "PreOrigin_AnomMag(F)": round(diag["anom_mag"], 2),
            "PreOrigin_AnomGrowth(F)": round(diag["anom_growth"], 2),
            "Temp_Gate_Weight": round(diag["temp_gate"], 3),
            "Err_Gate_Weight": round(diag["err_gate"], 3),
            "Pinball_1_48h_NoAR": round(pb48_noar, 3),
            "Pinball_1_48h_UncondAR": round(pb48_unc, 3),
            "Pinball_1_48h_AdaptiveHyb": round(pb48_hyb, 3),
            "Gain_UncondAR_vs_NoAR": round(pb48_noar - pb48_unc, 3),
            "Gain_Adaptive_vs_NoAR": round(pb48_noar - pb48_hyb, 3),
        })

        # Record granular hourly predictions for this month
        pb_hourly = np.mean(pinball_loss_matrix(y_true, q_s70_hyb), axis=1)
        for h_i in range(len(y_true)):
            row_df = df_t.iloc[h_i]
            hourly_records.append({
                "task_id": task_id,
                "timestamp": str(row_df["timestamp"]),
                "year": int(row_df["year"]),
                "month": int(row_df["month"]),
                "day": int(row_df["day"]),
                "hour": int(row_df["hour"]),
                "dayofweek": int(row_df["dayofweek"]),
                "is_holiday": int(row_df["is_holiday"]),
                "is_holiday_bridge": int(row_df["is_holiday_bridge"]),
                "temp_mean25": round(float(row_df["temp_mean25"]), 2),
                "actual_load": round(float(y_true[h_i]), 2),
                "pred_mean": round(float(mu_s70_hyb[h_i]), 2),
                "oracle_mean": round(float(mu_ora[h_i]), 2),
                "q01": round(float(q_s70_hyb[h_i, 0]), 2),
                "q05": round(float(q_s70_hyb[h_i, 4]), 2),
                "q10": round(float(q_s70_hyb[h_i, 9]), 2),
                "q25": round(float(q_s70_hyb[h_i, 24]), 2),
                "q50": round(float(q_s70_hyb[h_i, 49]), 2),
                "q75": round(float(q_s70_hyb[h_i, 74]), 2),
                "q90": round(float(q_s70_hyb[h_i, 89]), 2),
                "q95": round(float(q_s70_hyb[h_i, 94]), 2),
                "q99": round(float(q_s70_hyb[h_i, 98]), 2),
                "pinball_hour": round(float(pb_hourly[h_i]), 4),
            })

        npz_q_noar_list.append(np.round(q_s70_noar, 2).astype(np.float32))
        npz_q_uncond_list.append(np.round(q_s70_unc, 2).astype(np.float32))
        npz_q_adapt_list.append(np.round(q_s70_hyb, 2).astype(np.float32))
        npz_actual_list.append(np.round(y_true, 2).astype(np.float32))
        npz_task_list.append(np.full(len(y_true), task_id, dtype=np.int16))
        npz_ts_list.append(df_t["timestamp"].astype(str).to_numpy(dtype="U19"))

    df_bt = pd.DataFrame(backtest_rows)
    df_bt.to_csv(tab_dir / "exp4_15month_backtest.csv", index=False)

    df_bt_summary = (
        df_bt.groupby("Method", sort=False)[
            ["Pinball", "Pinball_Tail", "Pinball_Central", "Pinball_1_48h", "RMSE_1_48h", "RMSE", "MAE", "MAPE(%)", "PICP_80(%)", "PICP_90(%)", "MIW_90(MW)"]
        ]
        .mean()
        .reset_index()
    )
    df_bt_summary.to_csv(tab_dir / "exp4_15month_summary.csv", index=False)
    print("\n--- Experiment 4A: 15-Month Sequential Backtest Summary (Tasks 1–15, Oct 2010 – Dec 2011) ---")
    print(df_bt_summary.to_string(index=False))

    df_gating_diag = pd.DataFrame(gating_diag_rows)
    df_gating_diag.to_csv(tab_dir / "exp4_preorigin_gating_diagnostics.csv", index=False)
    print("\n--- Experiment 4B: Pre-Origin Boundary Diagnostics & 1-48h Gain Across 15 Months ---")
    print(df_gating_diag.to_string(index=False))

    # Scenario Weighting & Cutoff Duration Sensitivity Table across 15 Months
    weighting_rows = []
    weighting_configs = [
        ("1. Equal-Weight Scenarios (lambda=0.0, Uniform Diversity)", 0.0, 5.0, None, None, False, 48.0),
        ("2. Static Similarity Hybrid (lambda=0.25, tau=5F, All Horizons)", 0.25, 5.0, None, None, False, 48.0),
        ("3. Static Similarity Hybrid (lambda=0.50, tau=5F, All Horizons)", 0.50, 5.0, None, None, False, 48.0),
        ("4. Static Similarity High (lambda=0.80, tau=4F, All Horizons)", 0.80, 4.0, None, None, False, 48.0),
        ("5. Pure Recent Similarity (lambda=1.00, tau=2.5F, All Horizons)", 1.00, 2.5, None, None, False, 48.0),
        ("6. Horizon-Decaying Similarity Only (lambda0=0.35, H_sim=48h, No AR)", 0.35, 5.0, 48.0, None, False, 48.0),
        ("7. Unconditional Short-Term AR (Cutoff=24h, H_1/2=18h)", 0.0, 5.0, None, 18.0, False, 24.0),
        ("8. Unconditional Short-Term AR (Cutoff=48h, H_1/2=36h)", 0.0, 5.0, None, 36.0, False, 48.0),
        ("9. Unconditional Short-Term AR (Cutoff=96h, H_1/2=48h)", 0.0, 5.0, None, 48.0, False, 96.0),
        ("10. Pre-Origin Adaptive Gated AR + 48h Decaying Similarity", 0.35, 5.0, 48.0, 36.0, True, 48.0),
    ]

    for w_name, lam, tau, sim_dec, ar_hl, adp_g, cut_h in weighting_configs:
        m_list = []
        for _, yr, mo, m_lbl in TASK_MONTHS_15:
            gam, mask_test, y_true, yrs_10 = cached_models_15[m_lbl]
            q_w, mu_w = gam.predict_quantiles_via_scenarios(
                df, temp_25, mask_test, yrs_10, day_shifts=(-6, -4, -2, 0, 2, 4, 6),
                ar_half_life_hours=ar_hl, ar_cutoff_hours=cut_h, adaptive_ar_gating=adp_g,
                similarity_weight_lambda=lam, similarity_weight_tau=tau, similarity_decay_hours=sim_dec,
            )
            mf = evaluate_all_metrics(y_true, q_w, mu_w)
            m48 = evaluate_all_metrics(y_true[:48], q_w[:48], mu_w[:48])
            m_list.append({
                "Full_Pinball": mf["Pinball"],
                "Tail_Pinball": mf["Pinball_Tail"],
                "Central_Pinball": mf["Pinball_Central"],
                "Pinball_1_48h": m48["Pinball"],
                "RMSE_1_48h": m48["RMSE"],
                "Full_RMSE": mf["RMSE"],
                "PICP_90(%)": mf["PICP_90(%)"],
                "MIW_90(MW)": mf["MIW_90(MW)"],
            })
        df_m_tmp = pd.DataFrame(m_list).mean().to_dict()
        df_m_tmp["Scheme"] = w_name
        weighting_rows.append(df_m_tmp)

    df_weighting = pd.DataFrame(weighting_rows)[
        ["Scheme", "Full_Pinball", "Tail_Pinball", "Central_Pinball", "Pinball_1_48h", "RMSE_1_48h", "Full_RMSE", "PICP_90(%)", "MIW_90(MW)"]
    ]
    df_weighting.to_csv(tab_dir / "exp4_scenario_weighting_and_gating.csv", index=False)
    print("\n--- Experiment 4C: Scenario Similarity Weighting & Cutoff Duration Tradeoffs (15-Month Mean) ---")
    print(df_weighting.to_string(index=False))

    # Plot Figure 5: 15-Month Sequential Backtest, Unbundled Ablation, Pre-Origin Gating Physics, & Weighting Tradeoff
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0))

    # Panel A: 15-Month Sequential Pinball Loss curve across Tasks 1-15
    ax = axes[0, 0]
    m_labels = [x[3] for x in TASK_MONTHS_15]
    x_m = np.arange(len(m_labels))
    pb_s6_15 = df_bt[df_bt["Method"].str.startswith("2.")]["Pinball"].to_numpy()
    pb_s10_15 = df_bt[df_bt["Method"].str.startswith("3.")]["Pinball"].to_numpy()
    pb_s70_15 = df_bt[df_bt["Method"].str.startswith("8.")]["Pinball"].to_numpy()
    picp_s70_15 = df_bt[df_bt["Method"].str.startswith("8.")]["PICP_90(%)"].to_numpy()

    ax.plot(x_m, pb_s6_15, "s--", color="#e63946", lw=1.5, markersize=4.5, label=f"6-Yr Scenarios (2005–Prior, Mean={pb_s6_15.mean():.2f})")
    ax.plot(x_m, pb_s10_15, "^-.", color="#f4a261", lw=1.6, markersize=4.5, label=f"10-Yr Scenarios (+2001–04 Archive, Mean={pb_s10_15.mean():.2f})")
    ax.plot(x_m, pb_s70_15, "o-", color="#1d3557", lw=2.0, markersize=5.0, label=f"70 Scen + Adaptive Gated AR (Mean={pb_s70_15.mean():.2f}, PICP90={picp_s70_15.mean():.1f}%)")
    ax.set_xticks(x_m)
    ax.set_xticklabels(m_labels, rotation=40, ha="right", fontsize=8)
    ax.set_title("(a) 15-Month Sequential Backtest (Tasks 1–15: Oct 2010 – Dec 2011)")
    ax.set_ylabel("Monthly Mean Pinball Loss (MW)")
    ax.legend(loc="upper right", fontsize=7.8)

    # Panel B: Unbundled Factor Isolation on Full-Month vs 1-48h Pinball Loss (15-Month Mean)
    ax = axes[0, 1]
    unbundled_methods = [
        ("2. 6-Yr", "6-Yr\n(d=0)"),
        ("3. 10-Yr", "+2001–04\n(10 Scen, d=0)"),
        ("4. 10-Yr", "10 Scen\n+ Uncond AR"),
        ("5. 70", "+±6d Shift\n(70 Scen, No AR)"),
        ("6. 70", "70 Scen\n+ Uncond AR"),
        ("8. 70", "70 Scen\n+ Adaptive AR"),
    ]
    full_vals, h48_vals, u_names = [], [], []
    for prefix, short_lbl in unbundled_methods:
        row_s = df_bt_summary[df_bt_summary["Method"].str.startswith(prefix)].iloc[0]
        full_vals.append(row_s["Pinball"])
        h48_vals.append(row_s["Pinball_1_48h"])
        u_names.append(short_lbl)
    xu = np.arange(len(u_names))
    w_u = 0.36
    b_f = ax.bar(xu - w_u / 2, full_vals, w_u, color="#1d3557", label="15-Month Mean Full-Month Pinball ↓")
    b_h = ax.bar(xu + w_u / 2, h48_vals, w_u, color="#2a9d8f", label="15-Month Mean 1–48h Pinball ↓")
    for b in b_f:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7.8, fontweight="bold")
    for b in b_h:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=7.8)
    ax.set_xticks(xu)
    ax.set_xticklabels(u_names, fontsize=8)
    ax.set_ylim(0, max(full_vals) * 1.24)
    ax.set_title("(b) Unbundled Factor Isolation (15-Month Average: Window Shift vs Short-Term AR)")
    ax.set_ylabel("Pinball Loss (MW)")
    ax.legend(loc="upper right", fontsize=8)

    # Panel C: Physical Mechanism of 1-48h Persistence Gain vs Pre-Origin Weather Transition across 15 Months
    ax = axes[1, 0]
    dt_vals = df_gating_diag["PreOrigin_dT_Abs(F)"].to_numpy()
    gain_unc = df_gating_diag["Gain_UncondAR_vs_NoAR"].to_numpy()
    anom_g_vals = df_gating_diag["PreOrigin_AnomGrowth(F)"].to_numpy()
    sc_g = ax.scatter(
        dt_vals,
        gain_unc,
        c=anom_g_vals,
        cmap="coolwarm",
        s=85,
        edgecolors="black",
        linewidth=0.8,
        zorder=4,
    )
    ax.axhline(0.0, color="black", linestyle="--", lw=1.1, label="Zero Gain Threshold (Above = AR Helps, Below = AR Hurts)")
    ax.axvline(3.0, color="#2a9d8f", linestyle=":", lw=1.3, label="Quasi-Stationary Boundary ($|\\Delta T_{-24\\mathrm{h}}| = 3^\\circ$F)")
    for row in df_gating_diag.itertuples():
        ax.annotate(
            row.Month[2:],
            (row._3, row.Gain_UncondAR_vs_NoAR),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7.8,
            fontweight="bold" if row.Month in ("2011-11", "2011-12") else "normal",
        )
    cb_g = plt.colorbar(sc_g, ax=ax)
    cb_g.set_label("Pre-Origin Anomaly Growth $\\Delta |A|_{-1}$ (°F) [Neg = Mean Reverting]")
    r_p, p_p = pearsonr(dt_vals, gain_unc)
    ax.set_ylim(-2.6, 6.2)
    ax.set_title(f"(c) Why 48h AR Helps or Hurts Across 15 Months ($r={r_p:.2f}, p={p_p:.3f}$)")
    ax.set_xlabel("Pre-Origin 24h Day-over-Day Temp Jump $|\\bar{T}_{-1} - \\bar{T}_{-2}|$ (°F)")
    ax.set_ylabel("1–48h Pinball Gain of Unconditional AR (MW)")
    ax.legend(loc="upper right", fontsize=7.6)

    # Panel D: Scenario Diversity vs Recent Similarity Weighting Tradeoff (1-48h Pinball vs Tail Risk)
    ax = axes[1, 1]
    static_rows = df_weighting.iloc[:5]
    lam_vals = [0.0, 0.25, 0.50, 0.80, 1.00]
    ax_tail = ax.twinx()
    l1 = ax.plot(lam_vals, static_rows["Pinball_1_48h"], "o-", color="#1d3557", lw=2.0, label="Static Weighting: 1–48h Pinball Loss (MW)")
    l2 = ax_tail.plot(lam_vals, static_rows["Tail_Pinball"], "s--", color="#e63946", lw=2.0, label="Static Weighting: Full-Month Tail Pinball Loss (MW)")
    dec_row = df_weighting.iloc[9]
    l3 = ax.scatter([0.35], [dec_row["Pinball_1_48h"]], marker="*", s=220, color="#2a9d8f", edgecolors="black", zorder=5, label=f"Adaptive Gated AR + 48h Decaying Sim (1–48h={dec_row['Pinball_1_48h']:.2f}, Tail={dec_row['Tail_Pinball']:.2f})")
    ax.set_title("(d) Scenario Diversity vs Recent Similarity Tradeoff ($\\lambda \\in [0, 1]$)")
    ax.set_xlabel("Recent-Similarity Weight $\\lambda$ (0 = Equal Historical Weight, 1 = Pure Nearest)")
    ax.set_ylabel("1–48h Mean Pinball Loss (MW)", color="#1d3557")
    ax_tail.set_ylabel("Full-Month Extreme Tail Pinball Loss (MW)", color="#e63946")
    ax_tail.grid(False)
    ax.legend(l1 + l2 + [l3], [h.get_label() for h in l1 + l2 + [l3]], loc="upper center", fontsize=7.6)

    plt.tight_layout()
    plt.savefig(fig_dir / "fig5_15month_backtest_and_gating.png", bbox_inches="tight")
    plt.close(fig)

    # Save granular hourly predictions CSV, NPZ archive, and JSON manifest
    df_hourly = pd.DataFrame(hourly_records)
    df_hourly.to_csv(pred_dir / "hourly_summary_15months.csv", index=False)

    np.savez_compressed(
        pred_dir / "quantiles_99_15months.npz",
        quantiles_levels=QUANTILES_99.astype(np.float32),
        q_70scen_noar=np.vstack(npz_q_noar_list),
        q_70scen_uncond_ar=np.vstack(npz_q_uncond_list),
        q_70scen_adaptive_hyb=np.vstack(npz_q_adapt_list),
        actual_load=np.concatenate(npz_actual_list),
        task_id=np.concatenate(npz_task_list),
        timestamp=np.concatenate(npz_ts_list),
    )

    try:
        git_rev = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        git_rev = "unknown"

    manifest = {
        "project": "GEFCom2014-L Static 1-Month Ahead Probabilistic Load Forecasting",
        "git_base_revision": git_rev,
        "quantiles_count": 99,
        "quantiles_range": [0.01, 0.99],
        "total_backtest_tasks": 15,
        "total_backtest_hours": int(len(df_hourly)),
        "model_hyperparameters": {
            "recency_half_life_years": 3.5,
            "ridge_alphas": [0.1, 1.0, 5.0, 20.0, 100.0],
            "spline_knots_raw_temp": 7,
            "spline_knots_thermal_inertia": 5,
            "recent_regime_window_days": 45,
            "recent_regime_damping": 0.75,
            "residual_window_years": 3.0,
            "residual_centering": "zero-median (uncanceled by recent_offset)",
            "residual_scale": 0.85,
            "day_shifts": [-6, -4, -2, 0, 2, 4, 6],
            "short_term_ar_cutoff_hours": 48.0,
            "short_term_ar_half_life_hours": 36.0,
            "similarity_weight_lambda0": 0.35,
            "similarity_weight_tau_F": 5.0,
            "similarity_decay_half_life_hours": 48.0,
        },
        "tasks_diagnostics": gating_diag_rows,
    }
    with open(pred_dir / "experiment_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\nExploratory rerun saved in results/exploratory_rerun/; authoritative controlled results are in results/boundary_merged/.")


if __name__ == "__main__":
    from threadpoolctl import threadpool_limits
    with threadpool_limits(limits=1):
        run_all_experiments()
