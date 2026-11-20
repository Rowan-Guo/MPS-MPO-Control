import unittest
from tempfile import TemporaryDirectory
from pathlib import Path
import contextlib
from io import StringIO
import json
import numpy as np

from tensor import amplitude_state, diagonal_mpo, conditional_scores, posterior, evidence, contract, identity_mpo
from run import run, decision, Controller, generate_joint, load_config

class TestDemo(unittest.TestCase):
    def test_unsafe_action_is_rejected_and_masked_by_evidence(self):
        joint=np.ones((3,3,3))/27
        psi=amplitude_state(joint)
        energy=diagonal_mpo(np.zeros_like(joint))
        flag=np.zeros_like(joint)
        flag[2,0,1]=1
        unsafe=diagonal_mpo(flag)
        noflag=diagonal_mpo(np.zeros_like(joint))
        a,den=conditional_scores(psi,(energy,energy),{1:0,2:1},(unsafe,noflag))
        self.assertTrue(np.isinf(a[0]))
        self.assertAlmostEqual(a[1],0)
        self.assertAlmostEqual(den,1/9)
        a,_=conditional_scores(psi,(energy,),{1:1,2:1},(unsafe,))
        self.assertAlmostEqual(a[0],0)
    
    def test_empty_evidence_and_padding(self):
        joint=np.zeros((3,3,3))
        joint[0,0,0]=1.
        psi=amplitude_state(joint)
        self.assertAlmostEqual(contract(psi,identity_mpo(psi),evidence({})),1)
        self.assertAlmostEqual(contract(psi,identity_mpo(psi),evidence({0:0,1:0,2:0})),1)
        with self.assertRaisesRegex(ValueError,'zero probability'):
            conditional_scores(psi,[identity_mpo(psi)],{1:2})
        with self.assertRaisesRegex(ValueError,'invalid'):
            evidence({1:3})

    def test_full_pipeline_and_saved_online_match(self):
        cfg=load_config()
        with TemporaryDirectory() as temp:
            output=Path(temp)
            with contextlib.redirect_stdout(StringIO()):
                run(output,cfg)
            with (output/'all_action_scores.csv').open(encoding='utf-8') as f:
                import csv
                rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),540)
            self.assertLess(max(float(r['max_enum_error']) for r in rows),1e-12)
            f=StringIO()
            with contextlib.redirect_stdout(f):
                decision(output/'model.npz','B',20261120,.05,.10)
            replay=json.loads(f.getvalue())
            selected=[r for r in rows if r['regime']=='B' and r['seed']=='20261120'
                      and r['P6']=='0.05' and r['P9']=='0.1']
            np.testing.assert_allclose([float(r['mpo_expected_loss']) for r in selected],
                                       [r['loss'] for r in replay['action_scores']],atol=1e-12)
            self.assertAlmostEqual(sum(replay['conditional_probability']),1.)

if __name__=='__main__':
    unittest.main()
