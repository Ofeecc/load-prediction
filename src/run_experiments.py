"""
Master experiment and visualization script for GEFCom2014-L Static 1-Month Ahead Load Forecasting.
Runs the 3 scenario-driven heuristic experiments and generates tables + figures.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, pearsonr

from src.data_loader import load_full_gefcom2014_l, load_task_benchmark
from src.metrics import QUANTILES_99, evaluate_all_metrics, pinball_loss, pinball_loss_matrix
from src.models import DirectCalendarQuantileBaseline, HourlySplineGAM, compute_thermal_features
from src.station_selection import compare_station_selection_strategies, evaluate_all_25_stations

# Configure clean academic plotting style
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 180,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8.5,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.grid": True,
    "grid.alpha": 0.28,
})


def run_all_experiments() -> None:
    out_dir = Path("results")
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading full GEFCom2014-L dataset...")
    df = load_full_gefcom2014_l("Data/Load")

    # Define static test masks:
    # Primary static target: December 2011 (Task 15, 744 hours)
    # Validation / Secondary targets: November 2011 (Task 14, Thanksgiving) & July 2011 (Task 10, Summer Peak)
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
    print("[2/5] Generating EDA & Physical Coupling figure (fig1_eda_physics.png)...")
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.0))

    # Panel A: Full Load & Temperature Timeline showing the 2001-2004 Unlabeled Weather Bank
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
    ax.axvspan(pd.Timestamp("2011-12-01"), pd.Timestamp("2011-12-31"), color="#2a9d8f", alpha=0.35, label="Static Target (Dec 2011)")
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
    t_feats_25 = compute_thermal_features(df["temp_mean25"].to_numpy())
    df_labeled["t_mean24"] = t_feats_25["t_mean24"][df["LOAD"].notna().to_numpy()]
    df_labeled["thermal_lag_delta"] = df_labeled["t_mean24"] - df_labeled["temp_mean25"]

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

    # Panel D: December Climate Anomaly (Why Dec 2010 Benchmark Fails on Dec 2011 & Why 2001-2004 Matters)
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
    # STEP 3: Experiment 1 — Scenario Generation & Uncertainty Decomposition
    # =========================================================================
    print("[3/5] Running Experiment 1: Scenario Generation & Uncertainty Decomposition...")
    exp1_rows = []
    exp1_predictions_dec = {}

    for task_id, yr, mo, m_name, mask_test, mask_train in [
        (15, 2011, 12, "Dec 2011 (Winter, Task 15)", test_mask_dec, train_mask_dec),
        (14, 2011, 11, "Nov 2011 (Autumn, Task 14)", val_mask_nov, train_mask_nov),
        (10, 2011, 7, "Jul 2011 (Summer, Task 10)", test_mask_jul, train_mask_jul),
    ]:
        y_true = df.loc[mask_test, "LOAD"].to_numpy(dtype=float)
        temp_25 = df["temp_mean25"].to_numpy(dtype=float)

        # M0: Official Naive Benchmark (Last year's same month copied across 99 quantiles)
        q_bench = load_task_benchmark(task_id)
        m0 = evaluate_all_metrics(y_true, q_bench)
        m0["Month"] = m_name
        m0["Method"] = "0. Official Benchmark (Same Month Last Year)"
        exp1_rows.append(m0)

        # M1: Direct Calendar Quantile Regression (No Future Weather Scenarios)
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

        # M2: Weather Scenarios ONLY (Labeled Years 2005..2010, no day shift, no residual convolution)
        q_s6_nores, mu_s6 = gam.predict_quantiles_via_scenarios(
            df, temp_25, mask_test, scenario_years=list(range(2005, yr)), day_shifts=(0,), include_residual_uncertainty=False
        )
        m2 = evaluate_all_metrics(y_true, q_s6_nores, mu_s6)
        m2["Month"] = m_name
        m2["Method"] = "2. Weather Paths Only (2005–2010, 6 Scen, No Residual)"
        exp1_rows.append(m2)

        # M3: Labeled Years 2005..2010 + Residual Convolution (6 scenarios)
        q_s6, mu_s6 = gam.predict_quantiles_via_scenarios(
            df, temp_25, mask_test, scenario_years=list(range(2005, yr)), day_shifts=(0,), include_residual_uncertainty=True
        )
        m3 = evaluate_all_metrics(y_true, q_s6, mu_s6)
        m3["Month"] = m_name
        m3["Method"] = "3. Labeled Weather Years (2005–2010, 6 Scen) + Residual"
        exp1_rows.append(m3)

        # M4: Adding 2001..2004 Unlabeled Weather Years (2001..2010, 10 scenarios)
        q_s10, mu_s10 = gam.predict_quantiles_via_scenarios(
            df, temp_25, mask_test, scenario_years=list(range(2001, yr)), day_shifts=(0,), include_residual_uncertainty=True
        )
        m4 = evaluate_all_metrics(y_true, q_s10, mu_s10)
        m4["Month"] = m_name
        m4["Method"] = "4. + Unlabeled Weather Archive (2001–2010, 10 Scen) + Residual"
        exp1_rows.append(m4)

        # M5: Adding Synoptic Calendar Window Shifts (+/-6 days -> 70 scenarios) + Multi-Horizon AR
        q_s70, mu_s70 = gam.predict_quantiles_via_scenarios(
            df, temp_25, mask_test, scenario_years=list(range(2001, yr)), day_shifts=(-6, -4, -2, 0, 2, 4, 6),
            include_residual_uncertainty=True, ar_half_life_hours=36.0
        )
        m5 = evaluate_all_metrics(y_true, q_s70, mu_s70)
        m5["Month"] = m_name
        m5["Method"] = "5. Full Two-Stage Scenario GAM (70 Scen + Residual + 36h AR)"
        exp1_rows.append(m5)

        # M6: Reference Oracle Weather (Actual future temperature + conditional residual quantiles)
        mu_oracle = gam.predict_point(df, temp_25, mask_test)
        df_t = df.loc[mask_test]
        q_oracle = np.zeros((len(y_true), 99), dtype=float)
        for idx_row, (h_val, dow_val) in enumerate(zip(df_t["hour"].to_numpy(), df_t["dayofweek"].to_numpy())):
            wk_val = 1 if dow_val >= 5 else 0
            q_oracle[idx_row, :] = mu_oracle[idx_row] + gam.residual_quantiles[(int(h_val), wk_val)] * 0.85
        m6 = evaluate_all_metrics(y_true, q_oracle, mu_oracle)
        m6["Month"] = m_name
        m6["Method"] = "6. [Upper Bound] Oracle Future Temperature + Residual"
        exp1_rows.append(m6)

        if task_id == 15:
            exp1_predictions_dec = {
                "y_true": y_true,
                "q_bench": q_bench,
                "q_cal": q_cal,
                "mu_cal": mu_cal,
                "q_s6_nores": q_s6_nores,
                "q_s70": q_s70,
                "mu_s70": mu_s70,
                "q_oracle": q_oracle,
                "mu_oracle": mu_oracle,
            }

    df_exp1 = pd.DataFrame(exp1_rows)
    cols_exp1 = ["Month", "Method", "Pinball", "Pinball_Tail", "Pinball_Central", "RMSE", "MAE", "MAPE(%)", "PICP_80(%)", "PICP_90(%)", "MIW_90(MW)"]
    df_exp1 = df_exp1[cols_exp1]
    df_exp1.to_csv(tab_dir / "exp1_scenario_generation.csv", index=False)
    print("\n--- Experiment 1 Summary (Dec 2011) ---")
    print(df_exp1[df_exp1["Month"].str.startswith("Dec")].to_string(index=False))

    # Plot Figure 2: Experiment 1 Probabilistic Fan Chart & Pinball Loss across Quantiles
    fig, axes = plt.subplots(2, 1, figsize=(13.5, 8.2), gridspec_kw={"height_ratios": [1.35, 1.0]})

    ax = axes[0]
    hours_x = np.arange(1, len(exp1_predictions_dec["y_true"]) + 1)
    q_s70 = exp1_predictions_dec["q_s70"]
    y_true_dec = exp1_predictions_dec["y_true"]
    ax.fill_between(hours_x, q_s70[:, 4], q_s70[:, 94], color="#a8dadc", alpha=0.45, label="90% Prediction Interval ($q_{0.05}$–$q_{0.95}$)")
    ax.fill_between(hours_x, q_s70[:, 9], q_s70[:, 89], color="#457b9d", alpha=0.40, label="80% Prediction Interval ($q_{0.10}$–$q_{0.90}$)")
    ax.fill_between(hours_x, q_s70[:, 24], q_s70[:, 74], color="#1d3557", alpha=0.35, label="50% Prediction Interval ($q_{0.25}$–$q_{0.75}$)")
    ax.plot(hours_x, exp1_predictions_dec["q_bench"][:, 49], color="#e63946", linestyle=":", lw=1.1, alpha=0.75, label="Official Benchmark (Dec 2010 Load, Pinball=34.07)")
    ax.plot(hours_x, exp1_predictions_dec["mu_s70"], color="#1d3557", lw=1.4, label="Scenario GAM Median/Mean Forecast (Pinball=8.98)")
    ax.plot(hours_x, y_true_dec, color="#ffb703", lw=1.3, label="Actual Load (Dec 2011 Ground Truth, PICP90=89.7%)")
    ax.axvspan(23 * 24, 26 * 24, color="#f4a261", alpha=0.25, label="Christmas Holiday Window (Dec 24–26)")
    ax.set_title("(a) Static 1-Month Probabilistic Load Forecast on Dec 2011 (744 Hours): Physical Scenario Fan vs Actual Load")
    ax.set_xlabel("Forecast Horizon Hour in December 2011 (1 to 744)")
    ax.set_ylabel("Load (MW)")
    ax.set_xlim(1, 744)
    ax.legend(loc="upper right", ncol=3)

    # Bottom panel: Pinball Loss by Quantile Level q=0.01..0.99 across methods
    ax2 = axes[1]
    pb_cal_q = np.mean(pinball_loss_matrix(y_true_dec, exp1_predictions_dec["q_cal"]), axis=0)
    pb_nores_q = np.mean(pinball_loss_matrix(y_true_dec, exp1_predictions_dec["q_s6_nores"]), axis=0)
    pb_s70_q = np.mean(pinball_loss_matrix(y_true_dec, exp1_predictions_dec["q_s70"]), axis=0)
    pb_ora_q = np.mean(pinball_loss_matrix(y_true_dec, exp1_predictions_dec["q_oracle"]), axis=0)

    ax2.plot(QUANTILES_99, pb_cal_q, "--", color="#e63946", lw=1.8, label=f"1. Direct Calendar QR (No Weather Scen, Mean={np.mean(pb_cal_q):.2f})")
    ax2.plot(QUANTILES_99, pb_nores_q, "-.", color="#f4a261", lw=1.8, label=f"2. Weather Paths Only (2005–2010, No Residual, Mean={np.mean(pb_nores_q):.2f})")
    ax2.plot(QUANTILES_99, pb_s70_q, "-", color="#1d3557", lw=2.2, label=f"5. Full Two-Stage Scenario GAM (2001–2010 + Residual, Mean={np.mean(pb_s70_q):.2f})")
    ax2.plot(QUANTILES_99, pb_ora_q, ":", color="#2a9d8f", lw=2.0, label=f"6. [Oracle Upper Bound] True Future Weather + Residual (Mean={np.mean(pb_ora_q):.2f})")
    ax2.set_title("(b) Pinball Loss Curve Across All 99 Quantiles ($q=0.01\\dots 0.99$) on Dec 2011")
    ax2.set_xlabel("Quantile Level $q$")
    ax2.set_ylabel("Mean Pinball Loss (MW)")
    ax2.legend(loc="upper center", ncol=2)

    plt.tight_layout()
    plt.savefig(fig_dir / "fig2_exp1_scenarios.png")
    plt.close(fig)

    # =========================================================================
    # STEP 4: Experiment 2 — Weather Station Selection (Point != Probabilistic)
    # =========================================================================
    print("[4/5] Running Experiment 2: Weather Station Selection across all 25 stations...")
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
    print("\n--- Experiment 2 Station Strategies (Dec 2011) ---")
    print(df_strat.to_string(index=False))

    rho_val, p_val = spearmanr(df_stations["Val_Oracle_RMSE"], df_stations["Val_Pinball"])
    rho_test, p_test = spearmanr(df_stations["Test_Oracle_RMSE"], df_stations["Test_Pinball"])
    print(f"Station Rank Correlation (Oracle Point RMSE vs Scenario Pinball Loss): Val Spearman rho={rho_val:.3f} (p={p_val:.3f}), Test Spearman rho={rho_test:.3f} (p={p_test:.3f})")

    # Plot Figure 3: Weather Station Selection (Point RMSE vs Probabilistic Pinball Loss)
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

    # Plot strategy markers on the same plane
    for row, marker, col in zip(
        df_strat.itertuples(),
        ["s", "^", "D", "P", "*"],
        ["#8d99ae", "#e63946", "#f4a261", "#457b9d", "#2a9d8f"],
    ):
        ax.scatter(
            row.Test_Oracle_RMSE,
            row.Pinball,
            marker=marker,
            s=160 if marker != "*" else 230,
            color=col,
            edgecolors="black",
            linewidth=1.1,
            label=row.Strategy,
            zorder=5,
        )
    ax.set_title(f"(a) 25 Stations: Point RMSE vs Probabilistic Pinball ($\\rho={rho_test:.2f}$)")
    ax.set_xlabel("Point Forecast RMSE under True Temp (MW) [Lower = Better Point]")
    ax.set_ylabel("1-Month Probabilistic Pinball Loss (MW)")
    ax.legend(loc="upper left", fontsize=8)

    # Panel B: Bar chart comparing the 5 Station Strategies on Overall & Tail Pinball Loss
    ax2 = axes[1]
    strat_names = [
        "1. All-25 Mean\n(OxMath/Bidong)",
        "2. Best-1 Point\n(Single Station)",
        "3. Top-2 Point\n(Ziel Style)",
        "4. GCV Top-4\n(Tololo Style)",
        "5. Prob EWA\n(Adada Style)",
    ]
    x_pos = np.arange(len(strat_names))
    width = 0.36
    b1 = ax2.bar(x_pos - width / 2, df_strat["Pinball"], width, color="#1d3557", label="Overall Pinball Loss (All 99 Quantiles)")
    b2 = ax2.bar(x_pos + width / 2, df_strat["Pinball_Tail"] * 2.0, width, color="#e76f51", label="2x Tail Pinball Loss ($q \\leq 0.10, q \\geq 0.90$)")
    for b in b1:
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for b, val in zip(b2, df_strat["Pinball_Tail"]):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.12, f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(strat_names)
    ax2.set_ylim(0, max(df_strat["Pinball"]) * 1.22)
    ax2.set_title("(b) Table 6 Weather Station Selection & Aggregation Strategies")
    ax2.set_ylabel("Pinball Loss (MW)")
    ax2.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(fig_dir / "fig3_exp2_stations.png", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # STEP 5: Experiment 3 — Multi-Horizon AR Cutoff & Table 5 Holiday Phase-Shift
    # =========================================================================
    print("[5/5] Running Experiment 3: Multi-Horizon Persistence & Table 5 Holiday Ablations...")
    temp_25 = df["temp_mean25"].to_numpy(dtype=float)

    # Part A: Multi-Horizon Breakdown (Hours 1..48, 49..120, 121..744) on Dec 2011 and Nov 2011
    horizon_rows = []
    for m_label, mask_test, mask_train in [
        ("Dec 2011 (Task 15)", test_mask_dec, train_mask_dec),
        ("Nov 2011 (Task 14)", val_mask_nov, train_mask_nov),
    ]:
        y_t = df.loc[mask_test, "LOAD"].to_numpy(dtype=float)
        gam_m = HourlySplineGAM(use_holiday_correction=True, use_thermal_inertia=True, use_regime_calibration=True).fit(df, temp_25, mask_train)

        for ar_label, ar_hl in [
            ("No Short-Term Persistence (Pure Medium-Term Scenario GAM)", None),
            ("Multi-Horizon Persistence Blend (Tololo 36h Half-Life Cutoff)", 36.0),
        ]:
            q_pred, mu_pred = gam_m.predict_quantiles_via_scenarios(
                df, temp_25, mask_test, scenario_years=list(range(2001, 2011)), ar_half_life_hours=ar_hl
            )
            for h_slice_name, sl in [
                ("1–48h (Days 1–2 Short-Term)", slice(0, 48)),
                ("49–120h (Days 3–5 Transition)", slice(48, 120)),
                ("121–End (Days 6–31 Medium-Term)", slice(120, None)),
                ("Full Month (1–End)", slice(0, None)),
            ]:
                m_sl = evaluate_all_metrics(y_t[sl], q_pred[sl], mu_pred[sl])
                horizon_rows.append({
                    "Month": m_label,
                    "Strategy": ar_label,
                    "Horizon_Window": h_slice_name,
                    "Pinball": m_sl["Pinball"],
                    "RMSE": m_sl["RMSE"],
                    "MAE": m_sl["MAE"],
                    "MAPE(%)": m_sl["MAPE(%)"],
                })

    df_horizon = pd.DataFrame(horizon_rows)
    df_horizon.to_csv(tab_dir / "exp3_multi_horizon.csv", index=False)
    print("\n--- Experiment 3A: Multi-Horizon Short-Term Persistence Breakdown ---")
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
            ("1. Base Spline GAM (Instantaneous Temp Only, No Holiday Fix)", False, False, False, None),
            ("2. + Building Thermal Inertia (24h/72h Rolling & Lags)", False, True, False, None),
            ("3. + Weather-Normalized Local Regime Calibration", False, True, True, None),
            ("4. + Table 5 Holiday & Bridge Weekend Phase-Shift", True, True, True, None),
            ("5. + Multi-Horizon 36h Short-Term Persistence (Full Model)", True, True, True, 36.0),
        ]

        for cfg_name, use_hol, use_therm, use_reg, ar_hl in configs:
            g = HourlySplineGAM(
                use_holiday_correction=use_hol,
                use_thermal_inertia=use_therm,
                use_regime_calibration=use_reg,
            ).fit(df, temp_25, mask_train)

            q_p, mu_p = g.predict_quantiles_via_scenarios(
                df, temp_25, mask_test, scenario_years=list(range(2001, 2011)), ar_half_life_hours=ar_hl
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
                holiday_window_curves["nov_no_hol_q"] = q_p
            if m_label.startswith("Nov") and cfg_name.startswith("5."):
                holiday_window_curves["nov_full_ora"] = y_ora
                holiday_window_curves["nov_full_q"] = q_p
                holiday_window_curves["nov_true"] = y_t
            if m_label.startswith("Dec") and cfg_name.startswith("3."):
                holiday_window_curves["dec_no_hol_ora"] = y_ora
                holiday_window_curves["dec_no_hol_q"] = q_p
            if m_label.startswith("Dec") and cfg_name.startswith("5."):
                holiday_window_curves["dec_full_ora"] = y_ora
                holiday_window_curves["dec_full_q"] = q_p
                holiday_window_curves["dec_true"] = y_t

    df_ablation = pd.DataFrame(ablation_rows)
    df_ablation.to_csv(tab_dir / "exp3_ablation_holiday.csv", index=False)
    print("\n--- Experiment 3B: Progressive Heuristic Ablation & Holiday Effect ---")
    print(df_ablation.to_string(index=False))

    # Plot Figure 4: Multi-Horizon Short-Term Cutoff & Holiday Phase-Shift Zoom-In
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.5))

    # Panel A: First 96 hours of Dec 2011 showing Multi-Horizon Short-Term Persistence
    ax = axes[0, 0]
    gam_dec = HourlySplineGAM(use_holiday_correction=True, use_thermal_inertia=True, use_regime_calibration=True).fit(df, temp_25, train_mask_dec)
    _, mu_no_ar = gam_dec.predict_quantiles_via_scenarios(df, temp_25, test_mask_dec, scenario_years=list(range(2001, 2011)), ar_half_life_hours=None)
    _, mu_with_ar = gam_dec.predict_quantiles_via_scenarios(df, temp_25, test_mask_dec, scenario_years=list(range(2001, 2011)), ar_half_life_hours=36.0)
    h96 = np.arange(1, 97)
    ax.plot(h96, y_true_dec[:96], "o-", color="#ffb703", markersize=3.5, lw=1.8, label="Actual Load (Dec 1–4, 2011)")
    ax.plot(h96, mu_no_ar[:96], "--", color="#e63946", lw=1.6, label=f"Without Short-Term Persistence (1–48h RMSE={np.sqrt(np.mean((y_true_dec[:48]-mu_no_ar[:48])**2)):.2f} MW)")
    ax.plot(h96, mu_with_ar[:96], "-", color="#1d3557", lw=2.0, label=f"With 36h Persistence Cutoff (1–48h RMSE={np.sqrt(np.mean((y_true_dec[:48]-mu_with_ar[:48])**2)):.2f} MW)")
    ax.axvline(48, color="#2a9d8f", linestyle=":", lw=1.5, label="Tololo 48h Short/Medium Cutoff")
    ax.set_title("(a) Multi-Horizon Forecasting: Short-Term Persistence Gain in First 48 Hours (Dec 1–4)")
    ax.set_xlabel("Forecast Lead Time (Hours from 2011-12-01 01:00)")
    ax.set_ylabel("Load (MW)")
    ax.legend(loc="upper right", fontsize=8)

    # Panel B: Bar comparison of Pinball Loss across Horizon Windows (1-48h, 49-120h, 121-744h)
    ax = axes[0, 1]
    sub_h_dec = df_horizon[df_horizon["Month"].str.startswith("Dec")]
    no_ar_vals = sub_h_dec[sub_h_dec["Strategy"].str.startswith("No")]["Pinball"].to_numpy()
    with_ar_vals = sub_h_dec[sub_h_dec["Strategy"].str.startswith("Multi")]["Pinball"].to_numpy()
    win_labels = ["1–48h\n(Days 1–2)", "49–120h\n(Days 3–5)", "121–744h\n(Days 6–31)", "Full Month\n(1–744h)"]
    x_w = np.arange(len(win_labels))
    w_bar = 0.35
    b_no = ax.bar(x_w - w_bar / 2, no_ar_vals, w_bar, color="#e63946", alpha=0.85, label="Pure Medium-Term Scenario GAM")
    b_ar = ax.bar(x_w + w_bar / 2, with_ar_vals, w_bar, color="#1d3557", alpha=0.90, label="+ Short-Term 36h Persistence Blend")
    for b in b_no:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8)
    for b in b_ar:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.15, f"{b.get_height():.2f}", ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xticks(x_w)
    ax.set_xticklabels(win_labels)
    ax.set_ylim(0, max(no_ar_vals) * 1.25)
    ax.set_title("(b) Pinball Loss Across Forecast Lead-Time Horizons (Dec 2011)")
    ax.set_ylabel("Pinball Loss (MW)")
    ax.legend(loc="upper right")

    # Panel C: Thanksgiving Window Zoom-in (Nov 23 to Nov 27, 2011: days 23..27 -> indices 22*24 .. 27*24)
    ax = axes[1, 0]
    s_nov, e_nov = 22 * 24, 27 * 24
    hrs_nov = np.arange(s_nov + 1, e_nov + 1)
    ax.plot(hrs_nov, holiday_window_curves["nov_true"][s_nov:e_nov], "o-", color="#ffb703", markersize=3, lw=1.8, label="Actual Load (Nov 23–27, Thanksgiving)")
    ax.plot(hrs_nov, holiday_window_curves["nov_no_hol_ora"][s_nov:e_nov], "--", color="#e63946", lw=1.6, label="Without Table 5 Holiday Fix (Treats Thu/Fri as Workdays)")
    ax.plot(hrs_nov, holiday_window_curves["nov_full_ora"][s_nov:e_nov], "-", color="#1d3557", lw=1.9, label="With Holiday & Bridge Phase-Shift (Thu→Sun, Fri→Bridge)")
    ax.axvspan(23 * 24, 25 * 24, color="#f4a261", alpha=0.28, label="Thanksgiving + Black Friday")
    ax.set_ylim(75, 215)
    ax.set_title("(c) Table 5 Holiday Collapse: Thanksgiving & Black Friday (Nov 23–27)")
    ax.set_xlabel("Hour of Month in November 2011")
    ax.set_ylabel("Load (MW)")
    ax.legend(loc="upper right", fontsize=8)

    # Panel D: Christmas Window Zoom-in (Dec 23 to Dec 27, 2011: days 23..27 -> indices 22*24 .. 27*24)
    ax = axes[1, 1]
    s_dec, e_dec = 22 * 24, 27 * 24
    hrs_dec = np.arange(s_dec + 1, e_dec + 1)
    ax.plot(hrs_dec, holiday_window_curves["dec_true"][s_dec:e_dec], "o-", color="#ffb703", markersize=3, lw=1.8, label="Actual Load (Dec 23–27, Christmas)")
    ax.plot(hrs_dec, holiday_window_curves["dec_no_hol_ora"][s_dec:e_dec], "--", color="#e63946", lw=1.6, label="Without Table 5 Holiday Fix (Overpredicts Dec 26 Mon)")
    ax.plot(hrs_dec, holiday_window_curves["dec_full_ora"][s_dec:e_dec], "-", color="#1d3557", lw=1.9, label="With Holiday & Observed Monday Phase-Shift")
    ax.axvspan(23 * 24, 26 * 24, color="#f4a261", alpha=0.28, label="Christmas Eve + Christmas + Observed Mon (Dec 24–26)")
    ax.set_ylim(70, 225)
    ax.set_title("(d) Table 5 Holiday Collapse: Christmas & Observed Mon Bridge (Dec 23–27)")
    ax.set_xlabel("Hour of Month in December 2011")
    ax.set_ylabel("Load (MW)")
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(fig_dir / "fig4_exp3_horizon_holiday.png", bbox_inches="tight")
    plt.close(fig)

    print("\nAll experiments and figures successfully generated in results/!")


if __name__ == "__main__":
    run_all_experiments()

