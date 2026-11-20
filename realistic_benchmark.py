"""IEEE33 + measured-Ausgrid load/PV, leakage-controlled MPS--MPO benchmark.

1. IEEE 33-bus real impedance/base-demand *benchmark*; measured household
   gross demand and PV, mapped into synthetic node-level trajectories.
2. Chronological train/test split has zero overlap even after spatial day-lags.
3. Fit per-time-regime nonnegative probability TTs from training categories.
4. Calibrate multi-terminal affine voltage operators with AC power flow.
5. Build Fourier MPOs without enumerating the joint categorical state space.
6. Evaluate all policies on held-out *continuous* source/load and nonlinear AC.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
import numpy as np

from feeder33 import Feeder33, action_injections, monitored_loss, unsafe_voltage
from measured_profiles import (DATA_PATH,SITES,OBSERVED_SITES,read_ausgrid,construct_profiles,
    split_indices,fit_categories,apply_categories,construct_category_base,state_power,regime)
from probability_tt import fit_mixture_em,mixture_probability_mps,probability_scores,probability_posterior
from scalable import fourier_deadband_mpo, fourier_uniform_error_bound
from powerflow import ACTIONS
from fast_fourier import FactorizedFourier, fourier_domain

MONITORED=(18,33)
DEADBAND=0.020


def calibrate_multi_terminal(feeder,centers,site_buses=SITES,monitored=MONITORED,actions=ACTIONS,
                             *, dispatch_map=action_injections):
    """One full AC solve at reference + two per categorical site per action.

    Returns c[a,monitor], f[a,monitor,site,category] for additive voltage
    surrogate; same AC solve calibrates ALL monitored buses simultaneously.
    """
    m=len(site_buses)
    base=construct_category_base(feeder,site_buses)
    ref=np.ones(m,dtype=int)
    const=np.zeros((len(actions),len(monitored)))
    f=np.zeros((len(actions),len(monitored),m,3))
    buses=np.asarray(monitored)-1
    for a_idx,a in enumerate(actions):
        v0=np.abs(feeder.ac(state_power(base,centers,site_buses,ref),actions=dispatch_map(a)))[buses]
        const[a_idx]=v0-1
        for j in range(m):
            for k in (0,2):
                changed=ref.copy();changed[j]=k
                vk=np.abs(feeder.ac(state_power(base,centers,site_buses,changed),
                     actions=dispatch_map(a)))[buses]
                f[a_idx,:,j,k]=vk-v0
    return const,f,len(actions)*(1+2*m)


def operators_from_affine(constants,contributions,deadband=DEADBAND,harmonics=8):
    domain=fourier_domain(constants,contributions,deadband)
    ops=tuple(fourier_deadband_mpo(c,g,deadband,domain,harmonics)
              for row_c,row_f in zip(constants,contributions,strict=True)
              for c,g in zip(row_c,row_f,strict=True))
    return ops,domain


def independent_mixture(states):
    # Independent marginals estimated on same training data in each clock regime.
    counts=np.stack([(np.bincount(states[:,i],minlength=3)+.5)/(len(states)+1.5)
                     for i in range(states.shape[1])],axis=0)
    return np.ones(1),counts[None,:,:]


def independent_probability_tt(states):
    return mixture_probability_mps(*independent_mixture(states))


def choose_actions(model,ops,constants,contributions,observed,*,num_monitored=len(MONITORED),deadband=DEADBAND):
    """Compute correlated-TT decision and plug-in mean decision on same model."""
    if isinstance(ops,FactorizedFourier):
        scores,margins,den=ops.scores_and_marginals(*model,observed)
    else:
        scores,den=probability_scores(model,ops,observed)
        margins=np.zeros((len(contributions[0,0]),3))
        for site in range(len(margins)):
            if site in observed:
                margins[site,observed[site]]=1.
            else:
                margins[site]=probability_posterior(model,observed,site,den=den)
    objective=np.mean(scores.reshape(len(ACTIONS),num_monitored),axis=1)
    ranks=np.argsort(objective)
    voltage=constants+np.einsum('amij,ij->am',contributions,margins,optimize=True)
    mean_loss=np.maximum(0,np.abs(voltage)-deadband).mean(axis=1)
    return int(ranks[0]),int(np.argmin(mean_loss)),float(objective[ranks[1]]-objective[ranks[0]]),float(den)


def block_bootstrap_ci(values,days,seed=20261120,replications=500):
    """Paired daily block bootstrap, retaining within-day dependence."""
    values=np.asarray(values,dtype=float)
    days=np.asarray(days)
    groups=np.unique(days)
    rng=np.random.default_rng(seed)
    # Weighted resampling of whole days; each day may have different sample count.
    daily_sum=np.array([values[days==d].sum() for d in groups])
    daily_n=np.array([np.sum(days==d) for d in groups])
    idx=rng.integers(0,len(groups),size=(replications,len(groups)))
    samples=daily_sum[idx].sum(axis=1)/daily_n[idx].sum(axis=1)
    return (float(np.quantile(samples,.025)),float(np.quantile(samples,.975)))


def run_benchmark(path=DATA_PATH,*,sites=SITES,observed_sites=OBSERVED_SITES,
                  harmonics=8,components=3,max_test=320,seed=20261120,
                  train_days=160,test_start_day=215,test_days=115,solar_scale=1.5,
                  slack=1.02,load_scale=.65,deadband=DEADBAND,backend='factorized'):
    if harmonics<1 or components<1 or max_test<20:
        raise ValueError('harmonics and components must be positive; max_test >= 20')
    if backend not in ('factorized','dense'):
        raise ValueError('expected factorized or dense backend')
    times,gc,pv=read_ausgrid(path)
    feeder=Feeder33(slack=slack,base_load_scale=load_scale)
    powers=construct_profiles(gc,pv,feeder,sites=sites,day_lags=tuple(range(0,len(sites)*3,3)),solar_scale=solar_scale)
    lags=tuple(range(0,len(sites)*3,3))
    train,test=split_indices(len(powers),train_days=train_days,test_start_day=test_start_day,
                             test_days=test_days,day_lags=lags)
    thresholds,centers,cats_train=fit_categories(powers[train],sites)
    cats_test=apply_categories(powers[test],thresholds,sites)
    models={}; independent={}; size_info={}
    train_regimes=regime(train)
    for bucket in range(4):
        subset=cats_train[train_regimes==bucket]
        fit=fit_mixture_em(subset,components=components,seed=seed+bucket,restarts=2,iterations=85)
        mixture=fit['weights'],fit['local_probabilities']
        independent_mix=independent_mixture(subset)
        if backend=='factorized':
            models[bucket],independent[bucket]=mixture,independent_mix
        else:
            models[bucket]=mixture_probability_mps(*mixture)
            independent[bucket]=mixture_probability_mps(*independent_mix)
        count,site_count,levels=mixture[1].shape
        size_info[bucket]={'samples':len(subset),'em_iter':fit['em_iterations'],
                           'probability_scalars':levels*(2*count+(site_count-2)*count**2),
                           'probability_array_bytes':sum(x.nbytes for x in models[bucket])}
    constants,contributions,offline_calls=calibrate_multi_terminal(feeder,centers,sites)
    if backend=='factorized':
        ops=FactorizedFourier(constants,contributions,deadband,harmonics)
        domain=ops.domain
        operator_bytes=ops.nbytes
    else:
        ops,domain=operators_from_affine(constants,contributions,deadband,harmonics)
        operator_bytes=sum(core.nbytes for op in ops for core in op)
    n=len(test)
    selected=np.unique(np.linspace(0,n-1,min(max_test,n),dtype=int))
    test_idx=test[selected]
    true_states=cats_test[selected]
    # Cache policies across clock bucket and observed category combination.
    policies={}
    choices=np.empty((len(selected),3),dtype=int) # TT, conditional-mean, independent TT
    gaps=[];masses=[]
    for row,(t,cat) in enumerate(zip(test_idx,true_states,strict=True)):
        bucket=int(regime([t])[0])
        evidence={int(i):int(cat[i]) for i in observed_sites}
        key=(bucket,tuple((i,evidence[i]) for i in observed_sites))
        if key not in policies:
            a,b,gap,mass=choose_actions(models[bucket],ops,constants,contributions,evidence,
                                     num_monitored=len(MONITORED),deadband=deadband)
            ind,_,_,_=choose_actions(independent[bucket],ops,constants,contributions,evidence,
                                  num_monitored=len(MONITORED),deadband=deadband)
            policies[key]=(a,b,ind,gap,mass)
        a,b,ind,gap,mass=policies[key]
        choices[row]=[a,b,ind]
        gaps.append(gap);masses.append(mass)
    # Continuous-measurement power flow is the held-out ground truth.
    measured_losses=np.empty((len(test_idx),len(ACTIONS)))
    measured_unsafe=np.empty_like(measured_losses,dtype=bool)
    mae=[]
    for i,t in enumerate(test_idx):
        for j,action in enumerate(ACTIONS):
            v=feeder.ac(powers[t],actions=action_injections(action))
            measured_losses[i,j]=monitored_loss(v,deadband=deadband)
            measured_unsafe[i,j]=unsafe_voltage(v)
        chosen=choices[i,0]
        # Compare AC voltage against calibrated categorical approximation.
        v=feeder.ac(powers[t],actions=action_injections(ACTIONS[chosen]))
        approx=1+constants[chosen]+contributions[chosen,np.arange(len(MONITORED))[:,None],
                         np.arange(len(sites))[None,:],true_states[i][None,:]].sum(axis=-1)
        mae.append(float(np.mean(np.abs(np.abs(v)[np.asarray(MONITORED)-1]-approx))))
    selected_losses=np.stack([measured_losses[np.arange(len(test_idx)),choices[:,j]] for j in range(3)],axis=1)
    baseline_mean=selected_losses[:,1]
    gains=baseline_mean-selected_losses[:,0]
    indep_gains=selected_losses[:,2]-selected_losses[:,0]
    days=test_idx//48
    lo,hi=block_bootstrap_ci(gains,days,seed)
    illo,ilhi=block_bootstrap_ci(indep_gains,days,seed+1)
    source_label=('Ausgrid Customer 12 measured GC/GG July 2011-June 2012'
                  if Path(path).resolve()==DATA_PATH.resolve() else f'custom measurements: {Path(path).name}')
    return dict(dataset=source_label,
        benchmark='Baran-Wu IEEE 33-bus, MATPOWER case33bw (radial normally-closed lines)',
        mapping='synthetic node-level scaling and disjoint same-hour whole-day lags of one real household',
        train_intervals=len(train),test_intervals=len(test),evaluated_intervals=len(test_idx),
        random_sites=len(sites),observed_sites=len(observed_sites),
        possible_states=int(3**len(sites)),
        probability_scalars=max(x['probability_scalars'] for x in size_info.values()),
        probability_max_bond=components,
        # Preserve reference dense TT layout metrics for cross-run comparisons.
        mpo_scalars_per_action_monitor=3*(2*(2*harmonics+1)+(len(sites)-2)*(2*harmonics+1)**2),
        mpo_max_bond=2*harmonics+1,
        operator_backend=backend,
        operator_array_bytes=operator_bytes,
        offline_ac_calibration_calls=offline_calls,
        total_heldout_ac_calls=int(len(test_idx)*len(ACTIONS)+len(test_idx)),
        fourier_uniform_bound=float(fourier_uniform_error_bound(domain,harmonics)),
        voltage_surrogate_mae=float(np.mean(mae)),
        tt_mean_loss=float(selected_losses[:,0].mean()),
        mean_plug_in_loss=float(selected_losses[:,1].mean()),
        independent_tt_mean_loss=float(selected_losses[:,2].mean()),
        oracle_mean_loss=float(np.min(measured_losses,axis=1).mean()),
        tt_oracle_action_rate=float(np.mean(np.isclose(selected_losses[:,0],
                    np.min(measured_losses,axis=1),atol=1e-12,rtol=0.0))),
        action_histograms={name:[int(np.sum(choices[:,j]==a)) for a in range(len(ACTIONS))]
                    for j,name in enumerate(('tt','mean','independent'))},
        improvement_vs_mean=float(gains.mean()),
        improvement_vs_mean_percent=float(100*gains.mean()/max(baseline_mean.mean(),1e-12)),
        improvement_vs_mean_ci95=[lo,hi],
        improvement_vs_independent=float(indep_gains.mean()),
        improvement_vs_independent_ci95=[illo,ilhi],
        different_actions_vs_mean=float(np.mean(choices[:,0]!=choices[:,1])),
        different_actions_vs_independent=float(np.mean(choices[:,0]!=choices[:,2])),
        tt_unsafe_fraction=float(measured_unsafe[np.arange(len(test_idx)),choices[:,0]].mean()),
        mean_unsafe_fraction=float(measured_unsafe[np.arange(len(test_idx)),choices[:,1]].mean()),
        tt_regret_vs_oracle=float(np.mean(selected_losses[:,0]-np.min(measured_losses,axis=1))),
        train_regime_sizes={str(k):v for k,v in size_info.items()},
        empirical_conditions_used=len(policies),
        source_first_timestamp=str(times[0]),source_last_timestamp=str(times[-1]),
        train_last_measured_day=int((train[-1]//48)+max(lags)),
        test_first_measured_day=int(test[0]//48),
        configuration=dict(harmonics=harmonics,components=components,seed=seed,
            deadband=deadband,slack=slack,base_load_scale=load_scale,
            monitor_buses=list(MONITORED),control_buses=[18,33],
            variable_buses=list(sites),observed_site_positions=list(observed_sites),
            train_days=train_days,test_start_day=test_start_day,test_days=test_days,
            solar_scale=solar_scale))


def compare_fourier_backends(path=DATA_PATH, *, harmonics=16, components=3, repeats=5, seed=20261120):
    """Time equivalent uncached scoring across all 4 time regimes and 9 evidence pairs."""
    if repeats < 1:
        raise ValueError('repeats must be positive')
    from platform import machine, python_version, system
    start = perf_counter()
    _, gross, pv = read_ausgrid(path)
    feeder = Feeder33()
    lags = tuple(range(0, len(SITES)*3, 3))
    powers = construct_profiles(gross, pv, feeder, day_lags=lags)
    train, _ = split_indices(len(powers), day_lags=lags)
    _, centers, states = fit_categories(powers[train])
    data_seconds = perf_counter() - start
    start = perf_counter()
    mixtures = []
    buckets = regime(train)
    for index in range(4):
        fit = fit_mixture_em(states[buckets == index], components=components, seed=seed+index,
                             restarts=2, iterations=85)
        mixtures.append((fit['weights'], fit['local_probabilities']))
    em_seconds = perf_counter() - start
    start = perf_counter()
    constants, contributions, ac_calls = calibrate_multi_terminal(feeder, centers)
    calibration_seconds = perf_counter() - start
    observed_cases = [dict(zip(OBSERVED_SITES, levels, strict=True))
                      for levels in np.ndindex(3, 3)]
    results = {}
    score_arrays = {}
    for backend in ('dense', 'factorized'):
        construction = []
        scoring = []
        for _ in range(repeats):
            start = perf_counter()
            if backend == 'dense':
                ops, _ = operators_from_affine(constants, contributions, DEADBAND, harmonics)
                probabilities = [mixture_probability_mps(*mix) for mix in mixtures]
            else:
                ops = FactorizedFourier(constants, contributions, DEADBAND, harmonics)
                probabilities = mixtures
            construction.append(perf_counter() - start)
            start = perf_counter()
            if backend == 'dense':
                rows = [probability_scores(prob, ops, obs)[0].reshape(constants.shape)
                        for prob in probabilities for obs in observed_cases]
            else:
                rows = [ops.scores(*mixture, obs)
                        for mixture in probabilities for obs in observed_cases]
            scoring.append(perf_counter() - start)
        score_arrays[backend] = np.array(rows)
        if backend == 'dense':
            array_bytes = sum(c.nbytes for operator in ops for c in operator)
        else:
            array_bytes = ops.nbytes
        model_bytes = sum(x.nbytes for model in probabilities for x in model)
        results[backend] = dict(operator_bytes=array_bytes, probability_bytes=model_bytes,
                                construction_seconds_median=float(np.median(construction)),
                                seconds_per_evidence_median=float(np.median(scoring))/len(rows))
    dense, fast = score_arrays['dense'], score_arrays['factorized']
    result = dict(harmonics=harmonics, evidence_cases=len(dense), mixture_components=components,
                  repeats=repeats, seed=seed, platform=system(), architecture=machine(),
                  python=python_version(), numpy=np.__version__,
                  timing_scope='Uncached scoring of six actions at two monitors per context; 4 regimes x 9 observed pairs.',
                  storage_scope='Array payload of all 12 loss operators and all 4 probability models. Excludes workspace and Python object overhead.',
                  data_preparation_seconds=data_seconds, mixture_training_seconds=em_seconds,
                  ac_calibration_seconds=calibration_seconds, common_ac_solves=ac_calls,
                  max_abs_score_difference=float(np.max(np.abs(dense-fast))),
                  action_agreement=float(np.mean(np.argmin(dense.mean(axis=-1), axis=-1) ==
                                                 np.argmin(fast.mean(axis=-1), axis=-1))), backends=results)
    result['online_speedup'] = results['dense']['seconds_per_evidence_median']/results['factorized']['seconds_per_evidence_median']
    result['operator_storage_ratio'] = results['dense']['operator_bytes']/results['factorized']['operator_bytes']
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=DATA_PATH)
    parser.add_argument('--harmonics',type=int,default=16)
    parser.add_argument('--components',type=int,default=3)
    parser.add_argument('--max-test',type=int,default=320)
    parser.add_argument('--seed',type=int,default=20261120)
    parser.add_argument('--solar-scale',type=float,default=1.5)
    parser.add_argument('--backend',choices=('factorized','dense'),default='factorized')
    parser.add_argument('--compare-backends',action='store_true',
                        help='time equivalent dense and phase-factorized conditional scoring')
    parser.add_argument('--out',type=Path)
    args=parser.parse_args()
    if args.compare_backends:
        result=compare_fourier_backends(args.data,harmonics=args.harmonics,components=args.components,seed=args.seed)
    else:
        result=run_benchmark(args.data,harmonics=args.harmonics,components=args.components,
                   max_test=args.max_test,seed=args.seed,solar_scale=args.solar_scale,backend=args.backend)
        if args.out is None:
            args.out=Path('examples/ieee33_ausgrid_results.json')
    if args.out is not None:
        args.out.parent.mkdir(parents=True,exist_ok=True)
        args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
