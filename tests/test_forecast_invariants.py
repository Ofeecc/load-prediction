"""Chronology, distribution and mechanism invariants for the merged pipeline."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from src.data_loader import load_full_gefcom2014_l, load_task_benchmark, validate_raw_timestamps
from src.metrics import evaluate_all_metrics, pinball_loss
from src.models import HourlySplineGAM, weighted_quantile, historical_scenario_windows, compute_thermal_features


class ForecastInvariants(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.limits=threadpool_limits(limits=1)
        cls.df=load_full_gefcom2014_l()
        cls.temp=cls.df.temp_mean25.to_numpy(float)
        cls.target=((cls.df.year==2011)&(cls.df.month==3)).to_numpy()
        cls.train=np.arange(len(cls.df))<np.flatnonzero(cls.target)[0]
        cls.gam=HourlySplineGAM().fit(cls.df,cls.temp,cls.train)

    @classmethod
    def tearDownClass(cls): cls.limits.restore_original_limits()

    def test_source_and_benchmark_alignment(self):
        self.assertEqual(len(self.df),96408)
        self.assertEqual(int(self.df.LOAD.notna().sum()),61344)
        for task in range(1,16): self.assertEqual(load_task_benchmark(task).shape[1],99)
        corrupt=self.df.iloc[:4].copy()
        corrupt.loc[0,"raw_ts"]=corrupt.loc[1,"raw_ts"]
        with self.assertRaisesRegex(ValueError,"mismatch"): validate_raw_timestamps(corrupt)
        with self.assertRaises(ValueError): load_task_benchmark(16)

    def test_calendar_alignment_including_february29(self):
        starts,meta=historical_scenario_windows(self.df,self.target,[2008,2010],(0,))
        self.assertEqual(len(starts),2)
        self.assertTrue(all("-03-01 01:00:00" in m["source_start"] for m in meta))
        feb=((self.df.year==2008)&(self.df.month==2)&(self.df.day==29)).to_numpy()
        _,meta=historical_scenario_windows(self.df,feb,[2004,2007],(0,))
        self.assertEqual(meta[0]["source_start"],"2004-02-29 01:00:00")
        self.assertEqual(meta[1]["source_start"],"2007-02-28 01:00:00")
        self.assertTrue(all(pd.Timestamp(m["source_end"])<pd.Timestamp("2008-02-29 01:00") for m in meta))

    def test_invalid_target_and_training_overlap(self):
        bad=self.target.copy(); bad[np.flatnonzero(bad)[1]]=False
        with self.assertRaises(ValueError): self.gam.predict_quantiles_via_scenarios(self.df,self.temp,bad,[2010])
        earlier=((self.df.year==2011)&(self.df.month==2)).to_numpy()
        with self.assertRaisesRegex(ValueError,"training history"):
            self.gam.predict_quantiles_via_scenarios(self.df,self.temp,earlier,[2010])

    def test_metric_contracts_and_known_loss(self):
        self.assertAlmostEqual(pinball_loss(np.array([2.]),np.array([[1.,3.]]),np.array([.25,.75])),.25)
        with self.assertRaises(ValueError): evaluate_all_metrics(np.array([1.,2.]),np.ones((1,99)))
        q=np.tile(np.arange(99,dtype=float),(2,1)); q[:,[1,2]]=q[:,[2,1]]
        with self.assertRaisesRegex(ValueError,"Crossing"): evaluate_all_metrics(np.array([1.,2.]),q)
        with self.assertRaises(ValueError): evaluate_all_metrics(np.array([np.nan]),np.ones((1,99)))

    def test_weighted_quantile_uniform_limit(self):
        samples=np.array([5.,2.,2.,13.,-1.]); levels=np.arange(1,100)/100
        expected=np.quantile(samples,levels)
        np.testing.assert_allclose(weighted_quantile(samples,np.ones(5),levels),expected,atol=1e-12)
        weights=np.ones(5)+1e-12*np.arange(5)
        np.testing.assert_allclose(weighted_quantile(samples,weights,levels),expected,atol=1e-10)
        with self.assertRaises(ValueError): weighted_quantile(samples,np.zeros(5),levels)
        with self.assertRaises(ValueError): weighted_quantile(samples,-np.ones(5),levels)

    def test_static_forecast_does_not_read_future_truth(self):
        options=dict(scenario_years=[2008,2009,2010],day_shifts=(-2,0,2),ar_half_life_hours=36.,
                     adaptive_ar_gating=True,similarity_weight_lambda=.35,similarity_decay_hours=48.)
        normal,_=self.gam.predict_quantiles_via_scenarios(self.df,self.temp,self.target,**options)
        altered=self.df.copy(); future=np.arange(len(altered))>=np.flatnonzero(self.target)[0]
        altered.loc[future,"LOAD"]=np.nan
        altered.loc[future,[f"w{i}" for i in range(1,26)]+["temp_mean25","temp_std25","temp_min25","temp_max25"]]=np.nan
        temp=altered.temp_mean25.to_numpy(float)
        refitted=HourlySplineGAM().fit(altered,temp,self.train)
        hidden,_=refitted.predict_quantiles_via_scenarios(altered,temp,self.target,**options)
        np.testing.assert_allclose(normal,hidden,atol=1e-10,rtol=0)
        self.assertTrue(np.isfinite(hidden).all())
        self.assertTrue((np.diff(hidden,axis=1)>=0).all())

    def test_forecast_weighting_limit_and_no_residual_branch(self):
        args=dict(scenario_years=[2008,2009,2010],day_shifts=(-2,0,2))
        q0,_=self.gam.predict_quantiles_via_scenarios(self.df,self.temp,self.target,**args)
        qt,_=self.gam.predict_quantiles_via_scenarios(self.df,self.temp,self.target,similarity_weight_lambda=1e-12,**args)
        np.testing.assert_allclose(q0,qt,atol=1e-7,rtol=0)
        q0,_=self.gam.predict_quantiles_via_scenarios(self.df,self.temp,self.target,include_residual_uncertainty=False,**args)
        q1,_=self.gam.predict_quantiles_via_scenarios(self.df,self.temp,self.target,include_residual_uncertainty=False,similarity_weight_lambda=.8,**args)
        self.assertGreater(float(np.max(np.abs(q0-q1))),.1)

    def capture_temperature_paths(self,**kwargs):
        paths=[]
        expected_length=int(self.target.sum())+72
        def collect(values):
            if len(values)==expected_length: paths.append(np.asarray(values).copy())
            return compute_thermal_features(values)
        with patch("src.models.compute_thermal_features",side_effect=collect):
            self.gam.predict_quantiles_via_scenarios(self.df,self.temp,self.target,[2008,2009,2010],day_shifts=(0,),
                include_load_residual_correction=False,shrink_residual_uncertainty=False,**kwargs)
        return np.vstack(paths)

    def test_center_geometry_preserves_temperature_differences(self):
        raw=self.capture_temperature_paths()
        for gating in (False,True):
            center=self.capture_temperature_paths(ar_half_life_hours=36.,adaptive_ar_gating=gating,correction_geometry="center")
            np.testing.assert_allclose(center[1]-center[0],raw[1]-raw[0],atol=1e-12)
        individual=self.capture_temperature_paths(ar_half_life_hours=36.,correction_geometry="individual")
        std_raw=np.std(raw[:,72:96],axis=0)
        std_mod=np.std(individual[:,72:96],axis=0)
        self.assertTrue(np.all(std_mod<=std_raw+1e-12))

    def test_batched_prediction_matches_individual_paths_on_holidays(self):
        target=((self.df.year==2011)&(self.df.month==12)).to_numpy()
        scenarios=self.gam.generate_temperature_scenarios(self.df,self.temp,target,[2008,2009,2010],day_shifts=(0,))
        individual=np.mean([self.gam.predict_point(self.df,t,target) for t in scenarios],axis=0)
        _,batched=self.gam.predict_quantiles_via_scenarios(self.df,self.temp,target,[2008,2009,2010],day_shifts=(0,),include_residual_uncertainty=False)
        np.testing.assert_allclose(individual,batched,atol=1e-10,rtol=0)

    def test_recent_offset_moves_whole_distribution(self):
        args=dict(scenario_years=[2009,2010],day_shifts=(0,))
        q0,_=self.gam.predict_quantiles_via_scenarios(self.df,self.temp,self.target,**args)
        saved=self.gam.recent_offset.copy()
        try:
            self.gam.recent_offset={h:x+5 for h,x in saved.items()}
            q1,_=self.gam.predict_quantiles_via_scenarios(self.df,self.temp,self.target,**args)
            np.testing.assert_allclose(q1-q0,5.,atol=1e-10)
        finally: self.gam.recent_offset=saved

    def test_shared_holiday_daytype(self):
        rows=pd.DataFrame({"dayofweek":[0,0,6,0],"is_major_holiday":[1,0,1,0],
                           "is_holiday":[1,1,1,0],"is_holiday_bridge":[1,0,1,1]})
        np.testing.assert_equal(self.gam.effective_dayofweek(rows),[6,5,6,5])


if __name__=="__main__": unittest.main()
