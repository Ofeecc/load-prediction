"""
Scenario-driven heuristic models for GEFCom2014-L static 1-month ahead load forecasting.

Core philosophy (avoiding black-box model stacking):
1. Interpretable 24-Hour Penalized Cubic Spline Additive Model (GAM-style) capturing:
   - Non-linear U-shaped temperature response (heating & cooling degree splines)
   - Building thermal inertia (short-term lags + 24h/72h rolling mean splines + daily min/max)
   - Annual seasonal Fourier harmonics & post-2008 load plateau piece-wise trend
   - Table 5 US Federal Holiday & bridge-day phase-shift into weekend/holiday activity profiles
2. Two-Stage Probabilistic Modeling (Historical Weather Scenarios + Conditional Residuals):
   - Stage 1 (Weather Trajectory Ensemble): Resample historical multi-year temperature
     trajectories (including the 2001-2004 unlabeled weather archive) with calendar window shifts.
   - Stage 2 (Conditional Residual Dispersion): Convolve scenario load paths with zero-centered,
     hour-and-daytype-specific empirical residual quantiles.
3. Balancing Historical Scenario Diversity vs. Pre-Origin Recent Information:
   - Compare equal-weight historical scenarios, recent-similarity weighted scenarios, and hybrid mixtures.
   - Compare unconditional 48h short-term persistence against pre-origin regime-stability gated correction,
     distinguishing direct temperature injection (1..48h) from lagged thermal-inertia propagation (49..120h).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LassoCV, QuantileRegressor, RidgeCV
from sklearn.preprocessing import SplineTransformer

from src.metrics import QUANTILES_99, pinball_loss


def compute_thermal_features(temp_series: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Given a 1D hourly temperature array `temp_series` of length N,
    compute building thermal inertia features causal in time:
    - t_raw: current temperature T_t
    - t_lag1, t_lag2, t_lag3, t_lag6, t_lag12, t_lag24
    - t_mean24: causal 24-hour rolling mean temperature
    - t_mean72: causal 72-hour rolling mean temperature
    - t_max24, t_min24: causal 24-hour rolling max and min temperature
    - t_diff24: T_t - T_{t-24}
    """
    s = pd.Series(temp_series, dtype=float)
    feats = {
        "t_raw": s.to_numpy(),
        "t_lag1": s.shift(1).bfill().to_numpy(),
        "t_lag2": s.shift(2).bfill().to_numpy(),
        "t_lag3": s.shift(3).bfill().to_numpy(),
        "t_lag6": s.shift(6).bfill().to_numpy(),
        "t_lag12": s.shift(12).bfill().to_numpy(),
        "t_lag24": s.shift(24).bfill().to_numpy(),
        "t_mean24": s.rolling(24, min_periods=1).mean().to_numpy(),
        "t_mean72": s.rolling(72, min_periods=1).mean().to_numpy(),
        "t_max24": s.rolling(24, min_periods=1).max().to_numpy(),
        "t_min24": s.rolling(24, min_periods=1).min().to_numpy(),
    }
    feats["t_diff24"] = feats["t_raw"] - feats["t_lag24"]
    return feats


class HourlySplineGAM:
    """
    24 separate hourly Semi-Parametric Penalized Spline Additive Models (one per hour h=1..24),
    following the core GAM formulation of Tololo (Gaillard et al., 2016) and Adada (Dordonnat et al., 2016).
    """

    def __init__(
        self,
        use_holiday_correction: bool = True,
        use_thermal_inertia: bool = True,
        use_regime_calibration: bool = True,
        recency_half_life_years: float = 3.5,
        ridge_alphas: Tuple[float, ...] = (0.1, 1.0, 5.0, 20.0, 100.0),
    ):
        self.use_holiday_correction = use_holiday_correction
        self.use_thermal_inertia = use_thermal_inertia
        self.use_regime_calibration = use_regime_calibration
        self.recency_half_life_years = recency_half_life_years
        self.ridge_alphas = ridge_alphas

        self.spline_raw: Dict[int, SplineTransformer] = {}
        self.spline_m24: Dict[int, SplineTransformer] = {}
        self.spline_m72: Dict[int, SplineTransformer] = {}
        self.models: Dict[int, RidgeCV] = {}
        self.residual_quantiles: Dict[Tuple[int, int], np.ndarray] = {}  # (hour, is_wknd) -> 99 quantiles
        self.residual_std: Dict[int, float] = {}
        self.recent_offset: Dict[int, float] = {}
        self.t0_year: float = 2005.0
        self.holiday_damping: Dict[int, float] = {}
        self.bridge_damping: Dict[int, float] = {}

    def _build_calendar_matrix(self, df_h: pd.DataFrame) -> np.ndarray:
        """
        Build calendar, day-type (with optional holiday phase-shift), annual Fourier,
        and linear trend basis for a fixed hour `h`.
        """
        n = len(df_h)
        dow = df_h["dayofweek"].to_numpy().copy()

        if self.use_holiday_correction:
            # Phase-shift weekday major federal holidays to Sunday (dow=6) and minor/bridge weekday holidays to Saturday (dow=5)
            # Only apply additional indicator on Thanksgiving/weekday major holidays to avoid double-discounting Sunday Christmas
            is_wkday = dow < 5
            is_maj = df_h["is_major_holiday"].to_numpy() == 1
            is_hol = (df_h["is_holiday"].to_numpy() == 1) & (~is_maj)
            is_brg = df_h["is_holiday_bridge"].to_numpy() == 1
            dow[is_maj & is_wkday] = 6
            dow[is_hol & is_wkday] = 5
            dow[is_brg & is_wkday] = 5

        # One-hot day of week (0..6)
        dow_oh = np.eye(7)[dow]
        is_wknd = (dow >= 5).astype(float).reshape(-1, 1)

        # Annual Fourier terms (k=1..4 harmonics of day_of_year)
        doy = df_h["day_of_year"].to_numpy(dtype=float)
        omega = 2.0 * np.pi * doy / 365.25
        fourier_cols = []
        for k in range(1, 5):
            fourier_cols.append(np.sin(k * omega))
            fourier_cols.append(np.cos(k * omega))
        fourier_mat = np.column_stack(fourier_cols)

        # Interaction between weekend and annual seasonality (1st harmonic)
        wknd_season = is_wknd * fourier_mat[:, :2]

        # Post-2008 macro-economic plateau: ramp from 2005 to 2008, flat thereafter
        # (prevents confounding the 2010 extreme weather spike with secular linear trend)
        t_yr = np.clip(
            df_h["year"].to_numpy(dtype=float)
            + df_h["day_of_year"].to_numpy(dtype=float) / 365.25
            - 2005.0,
            0.0,
            3.0,
        ).reshape(-1, 1)

        cols = [dow_oh, is_wknd, fourier_mat, wknd_season, t_yr]

        if self.use_holiday_correction:
            is_wkday_orig = (df_h["dayofweek"].to_numpy() < 5).astype(float)
            cols.append((df_h["is_major_holiday"].to_numpy(dtype=float) * is_wkday_orig).reshape(-1, 1))
            cols.append((df_h["is_holiday_bridge"].to_numpy(dtype=float) * is_wkday_orig).reshape(-1, 1))

        return np.hstack(cols)

    def _build_design_matrix(
        self,
        df_h: pd.DataFrame,
        t_feats_h: Dict[str, np.ndarray],
        hour: int,
        fit_splines: bool = False,
    ) -> np.ndarray:
        cal_X = self._build_calendar_matrix(df_h)

        t_raw = t_feats_h["t_raw"].reshape(-1, 1)
        if fit_splines:
            sp_raw = SplineTransformer(n_knots=7, degree=3, include_bias=False, extrapolation="linear")
            B_raw = sp_raw.fit_transform(t_raw)
            self.spline_raw[hour] = sp_raw
        else:
            B_raw = self.spline_raw[hour].transform(t_raw)

        # Interaction of temperature spline with weekend/weekday & annual season (summer/winter sensitivity shift)
        dow = df_h["dayofweek"].to_numpy().copy()
        if self.use_holiday_correction:
            dow[df_h["is_major_holiday"].to_numpy() == 1] = 6
            dow[df_h["is_holiday_bridge"].to_numpy() == 1] = 5
        is_wknd = (dow >= 5).astype(float).reshape(-1, 1)

        doy = df_h["day_of_year"].to_numpy(dtype=float)
        omega = (2.0 * np.pi * doy / 365.25).reshape(-1, 1)
        sin1, cos1 = np.sin(omega), np.cos(omega)

        B_raw_wknd = B_raw * is_wknd
        B_raw_sin = B_raw * sin1
        B_raw_cos = B_raw * cos1

        parts = [cal_X, B_raw, B_raw_wknd, B_raw_sin, B_raw_cos]

        if self.use_thermal_inertia:
            t_m24 = t_feats_h["t_mean24"].reshape(-1, 1)
            t_m72 = t_feats_h["t_mean72"].reshape(-1, 1)
            if fit_splines:
                sp_m24 = SplineTransformer(n_knots=5, degree=3, include_bias=False, extrapolation="linear")
                sp_m72 = SplineTransformer(n_knots=5, degree=3, include_bias=False, extrapolation="linear")
                B_m24 = sp_m24.fit_transform(t_m24)
                B_m72 = sp_m72.fit_transform(t_m72)
                self.spline_m24[hour] = sp_m24
                self.spline_m72[hour] = sp_m72
            else:
                B_m24 = self.spline_m24[hour].transform(t_m24)
                B_m72 = self.spline_m72[hour].transform(t_m72)

            lag_mat = np.column_stack(
                [
                    t_feats_h["t_lag1"],
                    t_feats_h["t_lag2"],
                    t_feats_h["t_lag3"],
                    t_feats_h["t_lag6"],
                    t_feats_h["t_lag24"],
                    t_feats_h["t_max24"],
                    t_feats_h["t_min24"],
                    t_feats_h["t_diff24"],
                ]
            )
            parts.extend([B_m24, B_m72, lag_mat])

        return np.hstack(parts)

    def fit(self, df_full: pd.DataFrame, temp_series: np.ndarray, train_mask: np.ndarray) -> "HourlySplineGAM":
        """
        Fit 24 hourly penalized spline GAM models on `df_full[train_mask]` where LOAD is non-null.
        `temp_series` has the same length as `df_full` so causal rolling thermal features are exact.
        """
        valid_mask = train_mask & df_full["LOAD"].notna().to_numpy()
        valid_idx = np.flatnonzero(valid_mask)
        first_valid = max(0, int(valid_idx[0]) - 72)
        last_valid = int(valid_idx[-1]) + 1

        # Compute thermal features only on the slice that has valid LOAD (+72h lookback)
        t_feats_slice = compute_thermal_features(temp_series[first_valid:last_valid])
        rel_idx = valid_idx - first_valid
        t_feats_valid = {k: v[rel_idx] for k, v in t_feats_slice.items()}

        df_valid = df_full.iloc[valid_idx]
        hours_valid = df_valid["hour"].to_numpy()
        y_valid = df_valid["LOAD"].to_numpy(dtype=float)
        years_valid = df_valid["year"].to_numpy(dtype=float)
        doy_valid = df_valid["day_of_year"].to_numpy(dtype=float)
        dow_valid = df_valid["dayofweek"].to_numpy()
        max_year = float(years_valid.max())

        for h in range(1, 25):
            h_pos = np.flatnonzero(hours_valid == h)
            df_h = df_valid.iloc[h_pos]
            t_feats_h = {k: v[h_pos] for k, v in t_feats_valid.items()}
            y_h = y_valid[h_pos]

            X_h = self._build_design_matrix(df_h, t_feats_h, hour=h, fit_splines=True)

            # Exponential recency sample weights
            age_years = max_year - (years_valid[h_pos] + doy_valid[h_pos] / 365.25)
            weights = np.power(0.5, np.maximum(0.0, age_years) / self.recency_half_life_years)

            model = RidgeCV(alphas=self.ridge_alphas)
            model.fit(X_h, y_h, sample_weight=weights)
            self.models[h] = model

            # Estimate empirical conditional residual quantiles (split by weekday/weekend in recent 3 years)
            y_hat = model.predict(X_h)
            res = y_h - y_hat

            # Weather-normalized recent regime calibration (last 45 days before target month)
            # Shifts the conditional predictive center without shifting the zero-centered residual dispersion
            if self.use_regime_calibration and len(res) >= 45:
                self.recent_offset[h] = float(0.75 * np.median(res[-45:]))
            else:
                self.recent_offset[h] = 0.0

            recent_mask = age_years <= 3.0
            if np.sum(recent_mask) < 100:
                recent_mask = np.ones_like(res, dtype=bool)

            res_recent = res[recent_mask]
            self.residual_std[h] = float(np.std(res_recent))
            is_wknd_h = (dow_valid[h_pos] >= 5).astype(int)
            for wk in (0, 1):
                sub_res = res[recent_mask & (is_wknd_h == wk)]
                if len(sub_res) < 30:
                    sub_res = res_recent
                # Center the 3-year conditional error distribution around zero median so recent_offset
                # shifts the predictive center cleanly without being canceled by residual quantiles
                sub_res_centered = sub_res - float(np.median(sub_res))
                self.residual_quantiles[(h, wk)] = np.quantile(sub_res_centered, QUANTILES_99)

        return self

    def predict_point(
        self,
        df_full: pd.DataFrame,
        temp_series: np.ndarray,
        target_mask: np.ndarray,
    ) -> np.ndarray:
        """
        Predict conditional mean load for `df_full[target_mask]` given a full temperature array `temp_series`.
        Uses a local slice with 72-hour lookback when `target_mask` is a contiguous window for 100x speedup.
        """
        target_indices = np.flatnonzero(target_mask)
        if len(target_indices) == 0:
            return np.array([], dtype=float)

        start_idx = int(target_indices[0])
        end_idx = int(target_indices[-1]) + 1
        if end_idx - start_idx == len(target_indices):
            # Contiguous window fast path: only compute rolling features on [start_idx - 72 : end_idx]
            pad_start = max(0, start_idx - 72)
            offset = start_idx - pad_start
            t_feats_sub = compute_thermal_features(temp_series[pad_start:end_idx])
            t_feats_target = {k: v[offset:] for k, v in t_feats_sub.items()}
            df_target = df_full.iloc[start_idx:end_idx]
            y_pred = np.zeros(len(df_target), dtype=float)
            hours_target = df_target["hour"].to_numpy()
            for h in range(1, 25):
                local_pos = np.flatnonzero(hours_target == h)
                if len(local_pos) == 0:
                    continue
                df_h = df_target.iloc[local_pos]
                t_feats_h = {k: v[local_pos] for k, v in t_feats_target.items()}
                X_h = self._build_design_matrix(df_h, t_feats_h, hour=h, fit_splines=False)
                y_pred[local_pos] = self.models[h].predict(X_h) + self.recent_offset.get(h, 0.0)
            return y_pred

        t_feats_all = compute_thermal_features(temp_series)
        df_target = df_full.loc[target_mask]
        y_pred = np.zeros(len(df_target), dtype=float)
        hours_target = df_target["hour"].to_numpy()

        for h in range(1, 25):
            local_pos = np.flatnonzero(hours_target == h)
            if len(local_pos) == 0:
                continue
            global_idx = target_indices[local_pos]
            df_h = df_full.iloc[global_idx]
            t_feats_h = {k: v[global_idx] for k, v in t_feats_all.items()}
            X_h = self._build_design_matrix(df_h, t_feats_h, hour=h, fit_splines=False)
            y_pred[local_pos] = self.models[h].predict(X_h) + self.recent_offset.get(h, 0.0)

        return y_pred

    def generate_temperature_scenarios(
        self,
        df_full: pd.DataFrame,
        temp_series: np.ndarray,
        target_mask: np.ndarray,
        scenario_years: Sequence[int],
        day_shifts: Sequence[int] = (-6, -4, -2, 0, 2, 4, 6),
    ) -> List[np.ndarray]:
        """
        Construct physically coherent temperature scenarios for the target window by extracting
        historical temperature paths from `scenario_years` around the same day-of-year (+/- `day_shifts` days),
        stitching them seamlessly after the actual historical temperatures prior to `target_mask`
        so that building thermal inertia features at the month boundary remain physically continuous.
        """
        target_indices = np.flatnonzero(target_mask)
        start_idx = int(target_indices[0])
        horizon = len(target_indices)

        df_target = df_full.iloc[target_indices]
        start_doy = int(df_target["day_of_year"].iloc[0])
        start_hour = int(df_target["hour"].iloc[0])

        scenarios: List[np.ndarray] = []
        years_arr = df_full["year"].to_numpy()
        doy_arr = df_full["day_of_year"].to_numpy()
        hour_arr = df_full["hour"].to_numpy()

        for yr in scenario_years:
            for shift in day_shifts:
                target_doy = start_doy + shift
                # Wrap day-of-year within 1..365
                if target_doy < 1:
                    cand_yr = yr - 1
                    cand_doy = 365 + target_doy
                elif target_doy > 365:
                    cand_yr = yr + 1
                    cand_doy = target_doy - 365
                else:
                    cand_yr = yr
                    cand_doy = target_doy

                match = np.flatnonzero(
                    (years_arr == cand_yr) & (doy_arr == cand_doy) & (hour_arr == start_hour)
                )
                if len(match) == 0:
                    continue
                hist_start = int(match[0])
                if hist_start + horizon > int(target_indices[0]):
                    # Do not peek into target period or beyond available history
                    continue

                scen_temp = temp_series.copy()
                scen_temp[start_idx : start_idx + horizon] = temp_series[
                    hist_start : hist_start + horizon
                ]
                scenarios.append(scen_temp)

        return scenarios

    def compute_pre_origin_diagnostics(
        self,
        df_full: pd.DataFrame,
        temp_series: np.ndarray,
        target_mask: np.ndarray,
        scenario_years: Sequence[int],
        day_shifts: Sequence[int] = (-6, -4, -2, 0, 2, 4, 6),
    ) -> Dict[str, float]:
        """
        Compute strictly causal (pre-origin, t < start_idx) meteorological & load-residual
        diagnostics to evaluate whether the boundary regime is stationary/persistent or undergoing
        a rapid frontal transition / mean reversion toward climatology.
        """
        target_indices = np.flatnonzero(target_mask)
        start_idx = int(target_indices[0])
        if start_idx < 48:
            return {
                "dt_day_abs": 0.0,
                "dt_day_signed": 0.0,
                "anom_mag": 0.0,
                "anom_growth": 0.0,
                "err_day1_mean": 0.0,
                "err_day2_mean": 0.0,
                "temp_gate": 1.0,
                "err_gate": 0.6,
            }

        # Extract Day-1 climatological profile across historical scenarios
        df_target = df_full.iloc[start_idx : start_idx + 24]
        start_doy = int(df_target["day_of_year"].iloc[0])
        start_hour = int(df_target["hour"].iloc[0])
        years_arr = df_full["year"].to_numpy()
        doy_arr = df_full["day_of_year"].to_numpy()
        hour_arr = df_full["hour"].to_numpy()

        day1_scens = []
        for yr in scenario_years:
            for shift in day_shifts:
                target_doy = start_doy + shift
                if target_doy < 1:
                    cand_yr, cand_doy = yr - 1, 365 + target_doy
                elif target_doy > 365:
                    cand_yr, cand_doy = yr + 1, target_doy - 365
                else:
                    cand_yr, cand_doy = yr, target_doy
                match = np.flatnonzero(
                    (years_arr == cand_yr) & (doy_arr == cand_doy) & (hour_arr == start_hour)
                )
                if len(match) == 0:
                    continue
                hist_start = int(match[0])
                if hist_start + 24 > start_idx:
                    continue
                day1_scens.append(temp_series[hist_start : hist_start + 24])

        scen_day1_mean = np.mean(day1_scens, axis=0) if day1_scens else temp_series[start_idx - 24 : start_idx]

        t_m2 = temp_series[start_idx - 48 : start_idx - 24]
        t_m1 = temp_series[start_idx - 24 : start_idx]
        dt_signed = float(np.mean(t_m1) - np.mean(t_m2))
        dt_abs = abs(dt_signed)

        anom_m1 = t_m1 - scen_day1_mean
        anom_mag = float(np.mean(np.abs(anom_m1)))
        # Intraday anomaly evolution: is departure from seasonal norm strengthening (>=0) or decaying (<0)?
        anom_growth = float(np.mean(np.abs(anom_m1[12:])) - np.mean(np.abs(anom_m1[:12])))

        # Causal gating weight gamma_temp in [0, 1]:
        # High day-over-day temperature jumps or decaying intraday anomalies indicate active frontal transition / mean reversion
        effective_jump = dt_abs if anom_growth < 0.0 else 0.45 * dt_abs
        temp_gate = float(np.exp(-((effective_jump / 3.2) ** 2)))
        if anom_growth < -1.5:
            temp_gate *= float(np.exp(-(((abs(anom_growth) - 1.5) / 2.0) ** 2)))

        pre_mask_48 = np.zeros(len(df_full), dtype=bool)
        pre_mask_48[start_idx - 48 : start_idx] = True
        err_48 = df_full.loc[pre_mask_48, "LOAD"].to_numpy(dtype=float) - self.predict_point(
            df_full, temp_series, pre_mask_48
        )
        err_day2 = float(np.mean(err_48[:24]))
        err_day1 = float(np.mean(err_48[24:48]))
        err_gate = temp_gate * (0.60 if (err_day1 * err_day2 > 0) else 0.25)

        return {
            "dt_day_abs": dt_abs,
            "dt_day_signed": dt_signed,
            "anom_mag": anom_mag,
            "anom_growth": anom_growth,
            "err_day1_mean": err_day1,
            "err_day2_mean": err_day2,
            "temp_gate": temp_gate,
            "err_gate": err_gate,
        }

    def predict_quantiles_via_scenarios(
        self,
        df_full: pd.DataFrame,
        temp_series: np.ndarray,
        target_mask: np.ndarray,
        scenario_years: Sequence[int],
        day_shifts: Sequence[int] = (-6, -4, -2, 0, 2, 4, 6),
        include_residual_uncertainty: bool = True,
        residual_scale: float = 0.85,
        ar_half_life_hours: Optional[float] = None,
        ar_cutoff_hours: float = 48.0,
        adaptive_ar_gating: bool = False,
        freeze_thermal_inertia_after_cutoff: bool = False,
        similarity_weight_lambda: float = 0.0,
        similarity_weight_tau: float = 5.0,
        similarity_decay_hours: Optional[float] = None,
        correction_geometry: str = "legacy",
        include_load_residual_correction: bool = True,
        shrink_residual_uncertainty: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Two-stage probabilistic load forecasting:
        1. Extract N_scen historical temperature trajectories over `target_mask` (with 72h pre-boundary lookback).
        2. Optionally apply short-term boundary correction over `1..ar_cutoff_hours`:
           - If `adaptive_ar_gating=False`: unconditional persistence blend (weight=1.0).
           - If `adaptive_ar_gating=True`: pre-origin regime-stability gated blend (weight=gamma(t_0) in [0, 1]),
             shifting the scenario ensemble center while preserving (1 - gamma) of the inter-scenario spread.
           - Note on thermal inertia propagation: even when direct temperature injection ends at `ar_cutoff_hours` (48h),
             the 24h and 72h rolling temperature means (`t_mean24`, `t_mean72`) naturally propagate the modified
             1..48h temperatures into hours 49..120 unless `freeze_thermal_inertia_after_cutoff=True`.
        3. Optionally weight historical scenarios by pre-origin temperature similarity (`similarity_weight_lambda` in [0, 1]).
        4. Convolve scenario load distribution with zero-centered conditional residual quantiles.

        correction_geometry="legacy" preserves the original combined gate/spread behavior.
        "individual" pulls each trajectory towards the recent profile; "center" moves
        the ensemble center by the same hourly increment for every trajectory.
        Disable load-residual correction and residual shrinkage for a temperature-only
        gate x geometry factorial comparison. Preserving temperature spread does not
        guarantee unchanged load intervals after nonlinear regression.

        Returns:
            (y_quantiles [H, 99], y_mean [H])
        """
        if correction_geometry not in ("legacy", "individual", "center"):
            raise ValueError("correction_geometry must be legacy, individual or center")
        if not 0 <= similarity_weight_lambda <= 1:
            raise ValueError("similarity_weight_lambda must lie in [0, 1]")
        if ar_cutoff_hours <= 0 or similarity_weight_tau <= 0:
            raise ValueError("Correction cutoff and similarity bandwidth must be positive")
        if similarity_decay_hours is not None and similarity_decay_hours <= 0:
            raise ValueError("Similarity half-life must be positive")
        target_indices = np.flatnonzero(target_mask)
        if len(target_indices) == 0:
            raise RuntimeError("Empty target_mask provided.")
        if np.any(np.diff(target_indices) != 1):
            raise ValueError("Forecast target must be a contiguous hourly window")
        start_idx = int(target_indices[0])
        end_idx = int(target_indices[-1]) + 1
        horizon = len(target_indices)
        pad_start = max(0, start_idx - 72)
        offset = start_idx - pad_start

        df_target = df_full.iloc[start_idx:end_idx]
        start_doy = int(df_target["day_of_year"].iloc[0])
        start_hour = int(df_target["hour"].iloc[0])

        years_arr = df_full["year"].to_numpy()
        doy_arr = df_full["day_of_year"].to_numpy()
        hour_arr = df_full["hour"].to_numpy()

        base_sub = temp_series[pad_start:end_idx]
        scen_subs: List[np.ndarray] = []
        for yr in scenario_years:
            for shift in day_shifts:
                target_doy = start_doy + shift
                if target_doy < 1:
                    cand_yr, cand_doy = yr - 1, 365 + target_doy
                elif target_doy > 365:
                    cand_yr, cand_doy = yr + 1, target_doy - 365
                else:
                    cand_yr, cand_doy = yr, target_doy

                match = np.flatnonzero(
                    (years_arr == cand_yr) & (doy_arr == cand_doy) & (hour_arr == start_hour)
                )
                if len(match) == 0:
                    continue
                hist_start = int(match[0])
                if hist_start + horizon > start_idx:
                    continue

                sub = base_sub.copy()
                sub[offset:] = temp_series[hist_start : hist_start + horizon]
                scen_subs.append(sub)

        if not scen_subs:
            raise RuntimeError("No valid historical temperature scenarios generated.")

        n_scen = len(scen_subs)
        last_24h_temp = temp_series[start_idx - 24 : start_idx].copy() if start_idx >= 24 else None

        # Compute pre-origin scenario similarity weights if requested
        if similarity_weight_lambda > 0.0 and last_24h_temp is not None:
            dists = np.array(
                [np.sqrt(np.mean((s[offset : offset + 24] - last_24h_temp) ** 2)) for s in scen_subs],
                dtype=float,
            )
            sim_w = np.exp(-0.5 * ((dists - np.min(dists)) / max(0.5, similarity_weight_tau)) ** 2)
            sim_w = sim_w / np.sum(sim_w)
        else:
            sim_w = np.full(n_scen, 1.0 / n_scen, dtype=float)

        # Short-term boundary persistence setup
        temp_gate = 1.0
        err_coeff = 0.60
        if ar_half_life_hours is not None and ar_half_life_hours > 0 and start_idx >= 24 and last_24h_temp is not None:
            if adaptive_ar_gating:
                diag = self.compute_pre_origin_diagnostics(
                    df_full, temp_series, target_mask, scenario_years, day_shifts
                )
                temp_gate = diag["temp_gate"]
                err_coeff = diag["err_gate"]

            pre_mask = np.zeros(len(df_full), dtype=bool)
            pre_mask[start_idx - 24 : start_idx] = True
            y_pre_true = df_full.loc[pre_mask, "LOAD"].to_numpy(dtype=float)
            y_pre_hat = self.predict_point(df_full, temp_series, pre_mask)
            recent_errors = y_pre_true - y_pre_hat

            h_steps = np.arange(1, horizon + 1, dtype=float)
            cutoff_taper = np.clip(1.0 - (h_steps - 1.0) / ar_cutoff_hours, 0.0, 1.0)
            temp_decay = temp_gate * np.power(0.5, h_steps / ar_half_life_hours) * cutoff_taper
            if not include_load_residual_correction:
                err_coeff = 0.0
            err_decay = err_coeff * np.power(0.5, h_steps / (0.5 * ar_half_life_hours)) * cutoff_taper
            diurnal_err = np.resize(recent_errors, horizon)
        else:
            temp_decay = None
            err_decay = None
            diurnal_err = None

        # Compute thermal inertia features per scenario (and optionally freeze propagation after cutoff)
        t_feats_list = []
        ens_day1_mean = np.mean([s[offset : offset + 24] for s in scen_subs], axis=0)
        for sub in scen_subs:
            if temp_decay is not None and last_24h_temp is not None:
                sub_mod = sub.copy()
                scen_day1_temp = sub[offset : offset + 24]
                if correction_geometry == "center":
                    anchor = ens_day1_mean
                elif correction_geometry == "individual":
                    anchor = scen_day1_temp
                elif adaptive_ar_gating:
                    # Retain the teammate's original behavior as an explicit reference.
                    anchor = temp_gate * scen_day1_temp + (1.0 - temp_gate) * ens_day1_mean
                else:
                    anchor = scen_day1_temp
                diurnal_anom = np.resize(last_24h_temp - anchor, horizon)
                sub_mod[offset:] += diurnal_anom * temp_decay
                tf_mod = compute_thermal_features(sub_mod)
                if freeze_thermal_inertia_after_cutoff:
                    tf_raw = compute_thermal_features(sub)
                    cut_idx = offset + int(ar_cutoff_hours)
                    for k in tf_mod:
                        arr_k = tf_mod[k].copy()
                        arr_k[cut_idx:] = tf_raw[k][cut_idx:]
                        tf_mod[k] = arr_k
                t_feats_list.append({k: v[offset:] for k, v in tf_mod.items()})
            else:
                tf = compute_thermal_features(sub)
                t_feats_list.append({k: v[offset:] for k, v in tf.items()})

        # Batched evaluation across all N_scen scenarios for each hour h=1..24
        hours_target = df_target["hour"].to_numpy()
        scen_mat = np.zeros((n_scen, horizon), dtype=float)

        for h in range(1, 25):
            local_pos = np.flatnonzero(hours_target == h)
            if len(local_pos) == 0:
                continue
            df_h = df_target.iloc[local_pos]
            cal_X = np.tile(self._build_calendar_matrix(df_h), (n_scen, 1))
            t_feats_batch = {
                k: np.concatenate([tf[k][local_pos] for tf in t_feats_list])
                for k in t_feats_list[0].keys()
            }
            t_raw = t_feats_batch["t_raw"].reshape(-1, 1)
            B_raw = self.spline_raw[h].transform(t_raw)

            dow = df_h["dayofweek"].to_numpy().copy()
            if self.use_holiday_correction:
                dow[df_h["is_major_holiday"].to_numpy() == 1] = 6
                dow[df_h["is_holiday_bridge"].to_numpy() == 1] = 5
            is_wknd = np.tile((dow >= 5).astype(float).reshape(-1, 1), (n_scen, 1))

            doy = df_h["day_of_year"].to_numpy(dtype=float)
            omega = (2.0 * np.pi * doy / 365.25).reshape(-1, 1)
            sin1 = np.tile(np.sin(omega), (n_scen, 1))
            cos1 = np.tile(np.cos(omega), (n_scen, 1))

            parts = [cal_X, B_raw, B_raw * is_wknd, B_raw * sin1, B_raw * cos1]
            if self.use_thermal_inertia:
                B_m24 = self.spline_m24[h].transform(t_feats_batch["t_mean24"].reshape(-1, 1))
                B_m72 = self.spline_m72[h].transform(t_feats_batch["t_mean72"].reshape(-1, 1))
                lag_mat = np.column_stack(
                    [
                        t_feats_batch["t_lag1"],
                        t_feats_batch["t_lag2"],
                        t_feats_batch["t_lag3"],
                        t_feats_batch["t_lag6"],
                        t_feats_batch["t_lag24"],
                        t_feats_batch["t_max24"],
                        t_feats_batch["t_min24"],
                        t_feats_batch["t_diff24"],
                    ]
                )
                parts.extend([B_m24, B_m72, lag_mat])

            X_batch = np.hstack(parts)
            preds_h = (self.models[h].predict(X_batch) + self.recent_offset.get(h, 0.0)).reshape(
                n_scen, len(local_pos)
            )
            scen_mat[:, local_pos] = preds_h

        if diurnal_err is not None and err_decay is not None:
            scen_mat = scen_mat + diurnal_err * err_decay

        # Compute scenario weights per horizon step t_idx
        unif_w = np.full(n_scen, 1.0 / n_scen, dtype=float)
        if similarity_weight_lambda <= 0.0:
            y_mean = np.mean(scen_mat, axis=0)
        elif similarity_decay_hours is None:
            w_static = (1.0 - similarity_weight_lambda) * unif_w + similarity_weight_lambda * sim_w
            y_mean = np.sum(scen_mat * w_static[:, None], axis=0)
        else:
            h_steps = np.arange(1, horizon + 1, dtype=float)
            lam_t = similarity_weight_lambda * np.power(0.5, h_steps / similarity_decay_hours)
            w_mat = (1.0 - lam_t[None, :]) * unif_w[:, None] + lam_t[None, :] * sim_w[:, None]
            y_mean = np.sum(scen_mat * w_mat, axis=0)

        if not include_residual_uncertainty:
            y_quantiles = np.quantile(scen_mat, QUANTILES_99, axis=0).T
            return np.sort(y_quantiles, axis=1), y_mean

        # Convolve scenario predictions with conditional residual quantiles
        hours_t = df_target["hour"].to_numpy()
        is_wknd_t = (df_target["dayofweek"].to_numpy() >= 5).astype(int)

        res_grid_idx = np.linspace(2, 96, 19, dtype=int)
        n_res = len(res_grid_idx)
        y_quantiles = np.zeros((horizon, len(QUANTILES_99)), dtype=float)

        for t_idx in range(horizon):
            h = int(hours_t[t_idx])
            wk = int(is_wknd_t[t_idx])
            res_q = self.residual_quantiles[(h, wk)][res_grid_idx] * residual_scale
            if shrink_residual_uncertainty and ar_half_life_hours is not None and ar_half_life_hours > 0 and t_idx < ar_cutoff_hours:
                cutoff_taper = max(0.0, 1.0 - t_idx / ar_cutoff_hours)
                shrink = 1.0 - 0.25 * temp_gate * (0.5 ** ((t_idx + 1) / ar_half_life_hours)) * cutoff_taper
                res_q = res_q * shrink

            combined_samples = (scen_mat[:, t_idx : t_idx + 1] + res_q.reshape(1, -1)).ravel()
            if similarity_weight_lambda <= 0.0:
                y_quantiles[t_idx, :] = np.quantile(combined_samples, QUANTILES_99)
            else:
                if similarity_decay_hours is None:
                    lam_cur = similarity_weight_lambda
                else:
                    lam_cur = similarity_weight_lambda * (0.5 ** ((t_idx + 1) / similarity_decay_hours))
                w_t = (1.0 - lam_cur) * unif_w + lam_cur * sim_w
                joint_w = np.repeat(w_t, n_res) / float(n_res)
                sorter = np.argsort(combined_samples)
                s_sorted = combined_samples[sorter]
                w_sorted = joint_w[sorter]
                cdf = np.cumsum(w_sorted) - 0.5 * w_sorted
                cdf = cdf / cdf[-1]
                y_quantiles[t_idx, :] = np.interp(QUANTILES_99, cdf, s_sorted)

        return np.sort(y_quantiles, axis=1), y_mean


class DirectCalendarQuantileBaseline:
    """
    Baseline without future temperature scenario generation:
    Directly fits quantile regression (or empirical hourly/calendar quantile profiles)
    using only calendar, seasonal Fourier harmonics, and holiday features (since future T_t is unknown).
    """

    def __init__(self, use_holiday_correction: bool = True):
        self.use_holiday_correction = use_holiday_correction
        self.base_gam = HourlySplineGAM(use_holiday_correction=use_holiday_correction, use_thermal_inertia=False)
        self.models: Dict[int, RidgeCV] = {}
        self.res_quantiles: Dict[Tuple[int, int], np.ndarray] = {}

    def fit(self, df_full: pd.DataFrame, train_mask: np.ndarray) -> "DirectCalendarQuantileBaseline":
        valid_mask = train_mask & df_full["LOAD"].notna().to_numpy()
        max_year = float(df_full.loc[valid_mask, "year"].max())

        for h in range(1, 25):
            h_mask = valid_mask & (df_full["hour"].to_numpy() == h)
            df_h = df_full.loc[h_mask]
            y_h = df_h["LOAD"].to_numpy(dtype=float)
            X_h = self.base_gam._build_calendar_matrix(df_h)

            age_years = max_year - (
                df_h["year"].to_numpy(dtype=float) + df_h["day_of_year"].to_numpy(dtype=float) / 365.25
            )
            weights = np.power(0.5, np.maximum(0.0, age_years) / 3.5)

            model = RidgeCV(alphas=(0.1, 1.0, 10.0, 50.0))
            model.fit(X_h, y_h, sample_weight=weights)
            self.models[h] = model

            res = y_h - model.predict(X_h)
            recent_mask = age_years <= 4.0
            month_arr = df_h["month"].to_numpy()
            for m in range(1, 13):
                sub = res[recent_mask & (month_arr == m)]
                if len(sub) < 30:
                    sub = res[recent_mask]
                self.res_quantiles[(h, m)] = np.quantile(sub, QUANTILES_99)

        return self

    def predict_quantiles(self, df_full: pd.DataFrame, target_mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        df_target = df_full.loc[target_mask]
        horizon = len(df_target)
        y_mean = np.zeros(horizon, dtype=float)
        y_quantiles = np.zeros((horizon, len(QUANTILES_99)), dtype=float)

        hours_t = df_target["hour"].to_numpy()
        months_t = df_target["month"].to_numpy()
        target_indices = np.flatnonzero(target_mask)

        for h in range(1, 25):
            local_pos = np.flatnonzero(hours_t == h)
            if len(local_pos) == 0:
                continue
            global_idx = target_indices[local_pos]
            df_h = df_full.iloc[global_idx]
            X_h = self.base_gam._build_calendar_matrix(df_h)
            mu_h = self.models[h].predict(X_h)
            y_mean[local_pos] = mu_h
            for idx_in_h, pos in enumerate(local_pos):
                m = int(months_t[pos])
                y_quantiles[pos, :] = mu_h[idx_in_h] + self.res_quantiles[(h, m)]

        return np.sort(y_quantiles, axis=1), y_mean
