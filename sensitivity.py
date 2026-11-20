"""Ausgrid+IEEE33 robustness sweep: different PV penetrations and EM seeds.

Each row uses the same chronological train/test split and the same held-out
continuous AC evaluation. The seed changes EM initialization and bootstrap.
"""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from realistic_benchmark import run_benchmark
from measured_profiles import DATA_PATH

COLUMNS=('solar_scale','seed','evaluated_intervals','tt_mean_loss','mean_plug_in_loss',
         'independent_tt_mean_loss','improvement_vs_mean',
         'improvement_vs_mean_percent','ci95_mean_low','ci95_mean_high',
         'improvement_vs_independent','ci95_ind_low','ci95_ind_high',
         'tt_oracle_action_rate','tt_unsafe_fraction','mean_unsafe_fraction',
         'different_actions_vs_mean','voltage_surrogate_mae',
         'offline_ac_calibration_calls')


def sweep(scales=(.8,1.5,2.2),seeds=(20261120,20261121,20261122),max_test=240,harmonics=16,data_path=DATA_PATH):
    output=[]
    for scale in scales:
        for seed in seeds:
            row=run_benchmark(data_path,solar_scale=scale,seed=seed,
                              max_test=max_test,harmonics=harmonics)
            low,high=row['improvement_vs_mean_ci95']
            ilo,ihi=row['improvement_vs_independent_ci95']
            sample={k:row[k] for k in COLUMNS if k in row}
            sample.update(solar_scale=scale,seed=seed,ci95_mean_low=low,ci95_mean_high=high,
                          ci95_ind_low=ilo,ci95_ind_high=ihi)
            output.append(sample)
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=DATA_PATH)
    p.add_argument('--scales',nargs='+',type=float,default=[.8,1.5,2.2])
    p.add_argument('--seeds',nargs='+',type=int,default=[20261120,20261121,20261122])
    p.add_argument('--max-test',type=int,default=240)
    p.add_argument('--harmonics',type=int,default=16)
    p.add_argument('--out',type=Path,default=Path('examples/ieee33_ausgrid_sensitivity.csv'))
    args=p.parse_args()
    results=sweep(args.scales,args.seeds,args.max_test,args.harmonics,args.data)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=COLUMNS)
        w.writeheader();w.writerows(results)
    print(f'Wrote {args.out} ({len(results)} runs)')
    for r in results:
        print(f"PV={r['solar_scale']:4.1f}, seed={r['seed']}: "
              f"loss TT/mean/ind={r['tt_mean_loss']:.6g}/"
              f"{r['mean_plug_in_loss']:.6g}/{r['independent_tt_mean_loss']:.6g}, "
              f"gain={r['improvement_vs_mean_percent']:.1f}%, "
              f"CI=[{r['ci95_mean_low']:.2g},{r['ci95_mean_high']:.2g}]")


if __name__=='__main__':main()
