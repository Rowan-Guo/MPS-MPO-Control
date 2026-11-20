"""IEEE33 physics, measured-profile mapping and high-dimensional TT tests."""
import tempfile
import unittest
from pathlib import Path
from datetime import datetime,timedelta
import csv
import numpy as np

from feeder33 import Feeder33,action_injections,monitored_loss,unsafe_voltage
from measured_profiles import (DATA_PATH,SITES,OBSERVED_SITES,construct_profiles,
    fit_categories,apply_categories,split_indices,read_ausgrid,state_power)
from realistic_benchmark import (calibrate_multi_terminal,operators_from_affine,
    choose_actions,independent_probability_tt,block_bootstrap_ci,run_benchmark)
from probability_tt import probability_scores


class TestRealisticPhysics(unittest.TestCase):
    def test_case33bw_topology_and_ac(self):
        f=Feeder33()
        self.assertEqual(f.parent.shape,(33,))
        self.assertTrue(all(f.parent[k]<k for k in range(1,33)))
        v=f.ac(f.nominal_pq)
        self.assertEqual(v.shape,(33,))
        self.assertAlmostEqual(abs(v[0]),1.02,places=13)
        self.assertAlmostEqual(abs(v[17]),.9662941190392644,places=10)
        self.assertGreater(abs(f.ac(f.nominal_pq,actions=action_injections((.05,.05)))[-1]),abs(v[-1]))
        # Verify complex power balance at every non-slack bus independently.
        z=f.impedance; parent=f.parent
        i_branch=np.zeros(33,dtype=complex)
        for k in range(1,33): i_branch[k]=(v[parent[k]]-v[k])/z[k]
        local=i_branch.copy()
        for k in range(1,33):local[parent[k]]-=i_branch[k]
        observed=v[1:]*np.conj(local[1:])
        target=f.nominal_pq[1:,0]+1j*f.nominal_pq[1:,1]
        np.testing.assert_allclose(observed,target,atol=1e-9,rtol=0)

    def test_physically_meaningful_power_and_voltage(self):
        f=Feeder33()
        v=f.ac(f.nominal_pq)
        self.assertGreater(monitored_loss(v,deadband=.02),0)
        self.assertFalse(unsafe_voltage(v))
        self.assertTrue(np.isfinite(v).all())
        with self.assertRaises(ValueError):f.ac(np.zeros((32,2)))

    def test_leakage_free_split_and_categories(self):
        f=Feeder33()
        t=np.arange(366*48)/48
        gross=.8+.2*np.sin(t*2*np.pi)
        pv=.7*np.maximum(0,np.sin(t*2*np.pi))
        powers=construct_profiles(gross,pv,f)
        train,test=split_indices(len(powers),day_lags=tuple(range(0,36,3)))
        self.assertGreater(test[0],train[-1]+33*48)
        threshold,center,cats=fit_categories(powers[train])
        self.assertEqual(cats.shape,(7680,len(SITES)))
        self.assertEqual(apply_categories(powers[test[:20]],threshold).shape,(20,len(SITES)))
        np.testing.assert_allclose(state_power(f.nominal_pq,center,SITES,cats[0])[np.array(SITES)-1],
                                   center[np.arange(len(SITES)),cats[0]])
        with self.assertRaises(ValueError):split_indices(len(powers),test_start_day=170,day_lags=tuple(range(0,36,3)))

    def test_calibrated_fourier_benchmark(self):
        sites=(6,18,33)
        f=Feeder33()
        centers=np.stack([f.nominal_pq[np.asarray(sites)-1]*k for k in (.7,1.0,1.3)],axis=1)
        const,contrib,calls=calibrate_multi_terminal(f,centers,sites)
        self.assertEqual(calls,6*(2*3+1))
        ref=np.ones(3,dtype=int)
        ref_v=f.ac(state_power(f.nominal_pq,centers,sites,ref),actions=action_injections((0,-.05)))
        np.testing.assert_allclose(const[0]+1,abs(ref_v)[[17,32]],atol=1e-12)
        ops,domain=operators_from_affine(const,contrib,harmonics=8)
        self.assertEqual(len(ops),6*2)
        self.assertEqual(max(x.shape[-1] for x in ops[0]),17)
        np.testing.assert_equal([x.shape[1] for x in ops[0]],[3,3,3])
        p=independent_probability_tt(np.array([[0,1,2],[2,1,0],[1,1,1]]))
        a,b,gap,mass=choose_actions(p,ops,const,contrib,{0:1},num_monitored=2)
        self.assertIn(a,range(6))
        self.assertIn(b,range(6))
        self.assertGreater(mass,0)

    def test_block_bootstrap_ci(self):
        x=np.array([1.,2.,3.,4.,4.,4.])
        days=np.array([1,1,2,2,3,3])
        lo,hi=block_bootstrap_ci(x,days,seed=3)
        self.assertLess(lo,hi)
        self.assertLessEqual(lo,x.mean())
        self.assertGreaterEqual(hi,x.mean())

    def test_data_reader_and_complete_pipeline(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'measured_like.csv'
            rng=np.random.default_rng(222)
            with path.open('w',newline='') as f:
                writer=csv.writer(f);writer.writerow(['','GC','GG'])
                for i in range(366*48):
                    hour=(i%48)/2
                    g=0.45+0.35*np.sin(hour/24*2*np.pi)**2+rng.random()*.3
                    s=np.maximum(0,np.sin(np.pi*(hour-6)/12))*(.4+.2*np.sin(i/(48*30))**2)
                    writer.writerow([(datetime(2011,7,1)+timedelta(minutes=30*i)).isoformat(' '),g,s])
            times,g,p=read_ausgrid(path)
            self.assertEqual(len(times),366*48)
            r=run_benchmark(path,sites=(6,8,18,30),observed_sites=(1,3),components=2,
                            harmonics=4,max_test=24)
            self.assertEqual(r['possible_states'],81)
            self.assertEqual(r['offline_ac_calibration_calls'],6*(2*4+1))
            self.assertEqual(r['evaluated_intervals'],24)
            self.assertEqual(r['total_heldout_ac_calls'],24*7)
            self.assertTrue(np.isfinite(r['improvement_vs_mean']))


if __name__=='__main__':unittest.main()
