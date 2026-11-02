"""Frozen cross-period gate x temperature-geometry validation.

python -m src.boundary_validation --output results/boundary_merged --save-predictions
No threshold search, within-month update, or target truth is used for predictions.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
from threadpoolctl import threadpool_limits
from src.data_loader import load_full_gefcom2014_l
from src.metrics import QUANTILES_99, evaluate_all_metrics
from src.models import HourlySplineGAM

WINDOWS={"first_48h":slice(0,48),"hours_49_120":slice(48,120),
         "hours_121_end":slice(120,None),"full_month":slice(None)}
MONTHS=[str(x) for x in pd.period_range("2008-01","2009-12",freq="M")]+[
    str(x) for x in pd.period_range("2010-10","2011-12",freq="M")]


def configurations():
    legacy={"correction_geometry":"legacy"}
    pure={"include_load_residual_correction":False,"shrink_residual_uncertainty":False}
    return {
        "no_correction":dict(legacy),
        "legacy_unconditional":dict(legacy,ar_half_life_hours=36.),
        "legacy_gated":dict(legacy,ar_half_life_hours=36.,adaptive_ar_gating=True),
        "legacy_hybrid":dict(legacy,ar_half_life_hours=36.,adaptive_ar_gating=True,
                             similarity_weight_lambda=.35,similarity_decay_hours=48.),
        **{f"{gate}_{geometry}":dict(pure,ar_half_life_hours=36.,
                adaptive_ar_gating=(gate=="gated"),correction_geometry=geometry)
           for gate in ("fixed","gated") for geometry in ("individual","center")},
    }


def git(*args):
    try:
        return subprocess.check_output(["git",*args],text=True,encoding="utf-8",stderr=subprocess.DEVNULL).strip()
    except (OSError,subprocess.CalledProcessError):
        return "unavailable"


def fingerprint(paths):
    return {p.as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def summarize(root,rows):
    metrics=pd.DataFrame(rows)
    metrics.to_csv(root/"monthly_metrics.csv",index=False)
    summary=[]
    for (period,method,window),block in metrics.groupby(["period","method","window"],sort=True):
        row=dict(period=period,method=method,window=window,months=len(block),hours=int(block.hours.sum()))
        for key in ["Pinball","Pinball_Tail","Pinball_Central","MAE","MAPE(%)","PICP_80(%)","PICP_90(%)","MIW_90(MW)"]:
            row[key]=float(np.average(block[key],weights=block.hours))
        row["RMSE"]=float(np.sqrt(np.average(block.RMSE**2,weights=block.hours)))
        row["macro_Pinball"]=float(block.Pinball.mean())
        row["worst_month_Pinball"]=float(block.Pinball.max())
        base=metrics[(metrics.period==period)&(metrics.method=="no_correction")&(metrics.window==window)]
        paired=block.merge(base,on="month",suffixes=("_method","_base"))
        row["wins_vs_no_correction"]=int((paired.Pinball_method<paired.Pinball_base-1e-10).sum())
        row["macro_delta_vs_no_correction"]=float((paired.Pinball_method-paired.Pinball_base).mean())
        summary.append(row)
    pd.DataFrame(summary).to_csv(root/"summary.csv",index=False)
    pairs=[]
    for base,changed,effect in [
        ("fixed_individual","gated_individual","gate_individual"),
        ("fixed_center","gated_center","gate_center"),
        ("fixed_individual","fixed_center","geometry_fixed"),
        ("gated_individual","gated_center","geometry_gated"),
        ("legacy_gated","legacy_hybrid","extra_similarity")]:
        a=metrics[metrics.method==base]
        b=metrics[metrics.method==changed]
        joined=a.merge(b,on=["month","period","window"],suffixes=("_base","_changed"))
        for row in joined.to_dict("records"):
            pairs.append({"month":row["month"],"period":row["period"],"window":row["window"],
                          "effect":effect,"base":base,"changed":changed,
                          "delta_Pinball":row["Pinball_changed"]-row["Pinball_base"],
                          "base_Pinball":row["Pinball_base"],"changed_Pinball":row["Pinball_changed"]})
    pd.DataFrame(pairs).to_csv(root/"paired_effects.csv",index=False)


def run(output,months,save_predictions=False):
    root=Path(output)
    root.mkdir(parents=True,exist_ok=False)
    if save_predictions: (root/"predictions").mkdir()
    protocol={"started_utc":datetime.now(timezone.utc).isoformat(),"months":months,
        "periods":{"transfer_2008_2009":"Frozen teammate rule transferred to earlier years; retrospective temporal transfer, not a blind future test",
                   "explored_2010_2011":"Previously explored competition months; not an independent holdout"},
        "configs":configurations(),"thresholds":{"jump_scale_F":3.2,"growing_jump_factor":.45,
                   "anomaly_decay_threshold_F":-1.5,"anomaly_decay_scale_F":2.,
                   "residual_error_same_sign_weight":.60,"opposite_sign_weight":.25},
        "common":{"stations":"25-station equal mean","day_shifts":[-6,-4,-2,0,2,4,6],
                  "ar_cutoff_hours":48.,"ar_half_life_hours":36.,"residual_scale":.85,
                  "point_metrics":"predictive median","trend":"inherited capped 2008 basis, empirical assumption, not causal evidence"},
        "controls":"Factorial methods change ONLY temperature gating and geometry. Load-residual correction and residual shrinkage are disabled; no similarity weighting.",
        "no_tuning":"All thresholds/configurations fixed before execution; report every method and month. No winner is chosen to replace defaults.",
        "git_revision":git("rev-parse","HEAD"),"git_status":git("status","--porcelain"),
        "working_diff_sha256":hashlib.sha256(git("diff","HEAD").encode()).hexdigest(),
        "source_sha256":fingerprint(sorted(Path("src").glob("*.py"))),
        "data_sha256":fingerprint(sorted(Path("Data/Load").rglob("*.csv"))),
        "versions":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__,
                    "scipy":scipy.__version__,"scikit_learn":sklearn.__version__},
        "save_predictions":save_predictions,
        "limitations":["Rule was designed using later competition months; earlier transfer is not prospective validation",
                       "In-sample residuals with empirical 0.85 scale and 19 representative samples",
                       "Only one load zone; overlapping historical scenarios are dependent",
                       "Temperature spread preservation does not guarantee load interval calibration"]}
    (root/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf-8")
    df=load_full_gefcom2014_l()
    rows,diagnostics=[],[]
    for month in months:
        t0=time.monotonic()
        origin=pd.Timestamp(month)
        target=((df.year==origin.year)&(df.month==origin.month)).to_numpy()
        idx=np.flatnonzero(target)
        if not len(idx): raise ValueError(f"No observations for {month}")
        stop=int(idx[-1])+1
        known=df.iloc[:stop].copy()
        target=target[:stop]
        train=np.arange(stop)<idx[0]
        truth=known.loc[target,"LOAD"].to_numpy(float).copy()
        stamps=known.loc[target,"timestamp"].to_numpy(dtype="datetime64[ns]")
        known.loc[target,"LOAD"]=np.nan
        weather=[f"w{i}" for i in range(1,26)]+["temp_mean25","temp_std25","temp_min25","temp_max25"]
        known.loc[target,weather]=np.nan
        temp=known.temp_mean25.to_numpy(float)
        gam=HourlySplineGAM().fit(known,temp,train)
        # Eligibility checks exclude current-year target paths. Retain the teammate's
        # year-list convention for the diagnostic climatology as an inherited rule.
        years=list(range(2001,origin.year+1))
        diag=gam.compute_pre_origin_diagnostics(known,temp,target,years)
        diagnostics.append(dict(month=month,**diag))
        period="transfer_2008_2009" if origin.year<2010 else "explored_2010_2011"
        if save_predictions:
            np.savez_compressed(root/"predictions"/f"{month}_truth.npz",timestamp=stamps,y_true=truth)
        for name,config in configurations().items():
            q,_=gam.predict_quantiles_via_scenarios(known,temp,target,years,**config)
            if save_predictions:
                np.savez_compressed(root/"predictions"/f"{month}_{name}.npz",timestamp=stamps,
                    quantiles=QUANTILES_99,y_quantiles=q,origin=str(origin),method=name)
            for window,selection in WINDOWS.items():
                score=evaluate_all_metrics(truth[selection],q[selection])
                rows.append(dict(month=month,period=period,method=name,window=window,
                                 hours=len(truth[selection]),**score))
        summarize(root,rows)
        pd.DataFrame(diagnostics).to_csv(root/"origin_diagnostics.csv",index=False)
        print(f"{month}: 8 configurations; {time.monotonic()-t0:.1f}s",flush=True)
    protocol.update(completed_utc=datetime.now(timezone.utc).isoformat(),completed_months=len(months))
    (root/"protocol.json").write_text(json.dumps(protocol,ensure_ascii=False,indent=2),encoding="utf-8")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",default="results/boundary_merged")
    parser.add_argument("--months",nargs="+",default=MONTHS)
    parser.add_argument("--save-predictions",action="store_true")
    args=parser.parse_args()
    with threadpool_limits(limits=1): run(args.output,args.months,args.save_predictions)


if __name__=="__main__": main()
