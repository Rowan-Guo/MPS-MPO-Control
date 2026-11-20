"""Offline synthetic training and online conditional MPS–MPO control."""
import argparse
import csv
import json
from itertools import product
from pathlib import Path
import math
import numpy as np
from powerflow import LEVELS, ACTIONS, ac, loss
from tensor import (STATES, amplitude_state, diagonal_mpo, conditional_scores,
                    posterior, tt_scalar_count, tt_bond_ranks)

HERE=Path(__file__).resolve().parent
SITES=tuple(product(range(3),repeat=2))
METHODS=('mpo','conditional_mean','regime_mean','full_information')
V=np.array(LEVELS)
ACT=np.array(ACTIONS)

def load_config():
    return json.loads((HERE/'config.json').read_text(encoding='utf-8'))

def select(scores):
    eligible=np.flatnonzero(np.isfinite(scores))
    if not len(eligible):
        raise ValueError('no action satisfies the hard voltage and device limits on the conditional support')
    best=min(scores[eligible])
    return min((i for i in eligible if scores[i]<=best+1e-12),
               key=lambda i:(math.hypot(*ACTIONS[i]),ACTIONS[i]))

def generate_joint(cfg,name):
    target=tuple(cfg['target_visible_indices'])
    other=(1-cfg['target_visible_mass'])/8
    true=np.zeros((3,3,3))
    base=np.array(cfg['target_hidden_laws'][name])
    assert abs(sum(base)-1)<1e-13
    for b,c in SITES:
        w=base*np.exp(.25*(b+c-sum(target))*np.arange(3))
        true[:,b,c]=(cfg['target_visible_mass'] if (b,c)==target else other)*w/sum(w)
    return true

def sample(joint,n,rng):
    ids=rng.choice(27,n,p=joint.ravel())
    return np.array(np.unravel_index(ids,(3,3,3))).T

def estimate(train,alpha):
    counts=np.full((3,3,3),alpha)
    np.add.at(counts,tuple(train.T),1.)
    return counts/counts.sum()

class Controller:
    def __init__(self,cfg, *, max_bond=None, tt_tol=0.0):
        self.cfg=cfg
        self.max_bond=max_bond
        self.tt_tol=tt_tol
        self.loss=np.empty((6,3,3,3))
        self.voltage=np.empty_like(self.loss)
        self.unsafe=np.empty_like(self.loss,dtype=bool)
        self.minmax=np.empty((6,3,3,3,2))
        for j,a in enumerate(ACTIONS):
            for h,b,c in STATES:
                key=(j,h,b,c)
                volt=ac(V[h],V[b],V[c],a,cfg['slack'],return_all=True)
                self.voltage[key]=volt[9]
                self.loss[key]=loss(volt[9],cfg['deadband'])
                self.minmax[key]=min(volt),max(volt)
                lo,hi=cfg['hard_voltage_limits']
                self.unsafe[key]=(min(volt)<lo or max(volt)>hi or
                                  math.hypot(*a)>cfg['device_rating'])
        self.ops=tuple(diagonal_mpo(e, max_bond=max_bond, relative_tolerance=tt_tol)
                       for e in self.loss)
        # NEVER truncate a hard safety indicator: false negatives are unacceptable.
        self.flags=tuple(map(diagonal_mpo,self.unsafe.astype(float)))

    def policy(self,psi,trained,b,c,check=True):
        seen={1:b,2:c}
        score,den=conditional_scores(psi,self.ops,seen,self.flags)
        post=posterior(psi,seen,0,den)
        safe=~np.any(self.unsafe[:,trained[:,b,c]>0,b,c],axis=1)
        scores_ref=np.where(safe,self.loss[:,:,b,c]@ (trained[:,b,c]/trained[:,b,c].sum()),np.inf)
        error=float(np.max(np.abs(score[safe]-scores_ref[safe]),initial=0.))
        if check:
            np.testing.assert_allclose(den,trained[:,b,c].sum(),atol=1e-12,rtol=0)
            np.testing.assert_allclose(post,trained[:,b,c]/trained[:,b,c].sum(),atol=1e-12,rtol=0)
            np.testing.assert_array_equal(np.isfinite(score),safe)
            np.testing.assert_allclose(score,scores_ref,atol=1e-12,rtol=0)

        def point(m):
            val=np.array([loss(ac(m,V[b],V[c],a,self.cfg['slack']),self.cfg['deadband'])
                          for a in ACTIONS])
            return select(np.where(safe,val,np.inf))
        j=select(score)
        if check:
            assert j==select(scores_ref)
        return (j,point(post@V),point(trained.sum(axis=(1,2))@V)),score,post,error

    def full(self,h,b,c):
        return select(np.where(self.unsafe[:,h,b,c],np.inf,self.loss[:,h,b,c]))

def write_csv(dest,name,header,rows):
    with (dest/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)

def run(dest,cfg, *, max_bond=None, tt_tol=0.0):
    dest.mkdir(parents=True,exist_ok=True)
    ctrl=Controller(cfg, max_bond=max_bond, tt_tol=tt_tol)
    approximate=max_bond is not None or tt_tol > 0
    b0,c0=cfg['target_visible_indices']
    assert {select(np.array([loss(ac(V[h],V[b],V[c],a,1.),0.) for a in ACTIONS]))
            for h,b,c in STATES}=={ACTIONS.index((.05,.05))}
    control=[]; histories=[]; scores=[]; paired=[]; summaries=[]; models={}; worst=0.
    diagnostics=[]
    models.update(n_sites=np.array(3), format_version=np.array(2))
    for j,(op,flag) in enumerate(zip(ctrl.ops,ctrl.flags)):
        models.update({f'op_{j}_{k}':g for k,g in enumerate(op)})
        models.update({f'unsafe_{j}_{k}':g for k,g in enumerate(flag)})
    policy_rows=[]
    for name in cfg['target_hidden_laws']:
        true=generate_joint(cfg,name)
        js,energy,post,err=ctrl.policy(amplitude_state(true),true,b0,c0,check=not approximate)
        np.testing.assert_allclose(post,cfg['target_hidden_laws'][name],rtol=0,atol=1e-12)
        direct=ctrl.loss[:,:,b0,c0]@post
        oracle=sum(post[h]*ctrl.loss[ctrl.full(h,b0,c0),h,b0,c0] for h in range(3))
        control.append([name,post@V,*ACT[js[1]],*ACT[js[0]],direct[js[1]],energy[js[0]],oracle])
        for seed in cfg['seeds']:
            # Stable explicit streams; no test outcomes reused in estimation or dispatch.
            trng,erng=[np.random.default_rng(s) for s in np.random.SeedSequence(
                        [seed,ord(name)]).spawn(2)]
            train=sample(true,cfg['train_samples']//2,trng)
            test=sample(true,cfg['test_samples']//2,erng)
            trained=estimate(train,cfg['training_pseudocount'])
            psi=amplitude_state(trained, max_bond=max_bond, relative_tolerance=tt_tol)
            models.update({f'psi_{name}_{seed}_{k}':g for k,g in enumerate(psi)})
            policy={}
            for b,c in SITES:
                choices,energy,post,err=ctrl.policy(psi,trained,b,c,check=not approximate)
                worst=max(worst,err)
                policy[b,c]=choices
                support=trained[:,b,c]>0
                allowed=~np.any(ctrl.unsafe[:,support,b,c],axis=1)
                weights=trained[:,b,c]/trained[:,b,c].sum()
                exact_scores=np.where(allowed,ctrl.loss[:,:,b,c]@weights,np.inf)
                selected=choices[0]
                regret=exact_scores[selected]-min(exact_scores)
                gap=float(np.sort(exact_scores[np.isfinite(exact_scores)])[1]-min(exact_scores))
                diagnostics.append([name,seed,V[b],V[c],max_bond or 'exact',tt_tol,
                                    err,regret,gap,int(selected==select(exact_scores))])
                policy_rows.append([name,seed,V[b],V[c],*ACT[choices[0]],*ACT[choices[1]],*ACT[choices[2]]])
                for j,(p,q) in enumerate(ACTIONS):
                    scores.append([name,seed,V[b],V[c],p,q,*post,energy[j],err])
            for split,data in [('train',train),('test',test)]:
                for k,(h,b,c) in enumerate(data):
                    histories.append([name,seed,split,k,V[h],V[b],V[c]])
            realized=np.zeros(4); truth=np.zeros(4); target=np.zeros(4)
            for h,b,c in STATES:
                js=(*policy[b,c],ctrl.full(h,b,c))
                y=ctrl.loss[list(js),h,b,c]
                assert y[-1]<=min(y)+1e-12
                truth+=true[h,b,c]*y
                if (b,c)==(b0,c0): target+=true[h,b,c]*y/true[:,b0,c0].sum()
            for k,(h,b,c) in enumerate(test):
                js=(*policy[b,c],ctrl.full(h,b,c))
                y=ctrl.loss[list(js),h,b,c]
                realized+=y/len(test)
                for method,j,ell in zip(METHODS,js,y):
                    v9=ctrl.voltage[j,h,b,c]
                    vlo,vhi=ctrl.minmax[j,h,b,c]
                    paired.append([name,seed,k,V[h],V[b],V[c],method,*ACT[j],v9,ell,vlo,vhi,int(ctrl.unsafe[j,h,b,c])])
            for m,x,y,z in zip(METHODS,realized,truth,target):
                summaries.append([name,seed,m,x,y,z])
    if not approximate:
        assert [row[4:6] for row in control]==[[0.,.05],[.05,.05]]
    write_csv(dest,'positive_control.csv',['regime','conditional_mean','mean_P','mean_Q','mpo_P','mpo_Q','mean_loss','mpo_loss','full_loss'],control)
    write_csv(dest,'samples.csv',['regime','seed','split','index','P3','P6','P9'],histories)
    write_csv(dest,'all_action_scores.csv',['regime','seed','P6','P9','action_P','action_Q','post_low','post_mid','post_high','mpo_expected_loss','max_enum_error'],scores)
    write_csv(dest,'heldout_actions.csv',['regime','seed','index','realized_P3','P6','P9','method','action_P','action_Q','V9','loss','min_V','max_V','unsafe'],paired)
    write_csv(dest,'performance.csv',['regime','seed','method','heldout_loss','true_generator_expected_loss','target_context_expected_loss'],summaries)
    write_csv(dest,'policies.csv',['regime','seed','P6','P9','mpo_P','mpo_Q','conditional_mean_P','conditional_mean_Q','regime_mean_P','regime_mean_Q'],policy_rows)
    write_csv(dest,'operator_layout.csv',['P','Q','bond_ranks','core_scalars','direct_table_scalars'],
              [[*a,';'.join(map(str,tt_bond_ranks(op))),tt_scalar_count(op),27]
               for a,op in zip(ACTIONS,ctrl.ops)])
    write_csv(dest,'decision_diagnostics.csv',
              ['regime','seed','P6','P9','max_bond','tt_tol','max_abs_score_error',
               'decision_regret','exact_action_gap','action_matches_exact'],diagnostics)
    np.savez_compressed(dest/'model.npz',**models)
    print('Distinct conditional MPO/mean actions on learned policies:',sum(r[4:6]!=r[6:8] for r in policy_rows))

    for j in control: print('known-law control', j)
    for m in METHODS:
        vals=np.array([row[3:] for row in summaries if row[2]==m])
        print(m,'mean held-out / true-generator / target-only:',*vals.mean(axis=0))
    print(f'Max MPO/direct discrepancy: {worst:.3g}; {len(scores)} action scores, {len(paired)} held-out method rows')
    print('Max rank-approximation regret:',max(r[7] for r in diagnostics))
    return ctrl

def decision(path,name,seed,p6,p9):
    with np.load(path,allow_pickle=False) as data:
        n_sites=int(data['n_sites'])
        psi=tuple(data[f'psi_{name}_{seed}_{k}'] for k in range(n_sites))
        ops=tuple(tuple(data[f'op_{j}_{k}'] for k in range(n_sites)) for j in range(len(ACTIONS)))
        flags=tuple(tuple(data[f'unsafe_{j}_{k}'] for k in range(n_sites)) for j in range(len(ACTIONS)))
    axis={v:i for i,v in enumerate(LEVELS)}
    try:
        b,c=axis[p6],axis[p9]
    except KeyError as exc:
        raise ValueError('P6/P9 must be one of 0.05, 0.10, 0.15 p.u.') from exc
    # Online: load cores and contract, no load P3 input and no AC calls.
    score,den=conditional_scores(psi,ops,{1:b,2:c},flags)
    i=select(score)
    print(json.dumps({'selected_action':ACTIONS[i],
                      'conditional_probability':posterior(psi,{1:b,2:c},0,den).tolist(),
                      'action_scores':[{'P':p,'Q':q,'loss':float(w) if np.isfinite(w) else None}
                                       for (p,q),w in zip(ACTIONS,score)]},indent=2))

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--out',type=Path,default=HERE/'results')
    p.add_argument('--decide',nargs=4,metavar=('REGIME','SEED','P6','P9'),
                   help='decide from an offline model in --out; omit remote P3')
    p.add_argument('--max-bond',type=int,help='cap MPS and loss MPO bond ranks (not safety operator)')
    p.add_argument('--tt-tol',type=float,default=0.,help='per-tensor relative TT-SVD tolerance')
    args=p.parse_args()
    if args.decide:
        name,seed,p6,p9=args.decide
        decision(args.out/'model.npz',name,int(seed),float(p6),float(p9))
    else:
        run(args.out,load_config(),max_bond=args.max_bond,tt_tol=args.tt_tol)

if __name__=='__main__':
    main()
