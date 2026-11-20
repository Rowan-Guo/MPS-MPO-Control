"""Ausgrid Customer 12 measured gross consumption / rooftop PV + IEEE33 mapping.

Source: Ausgrid Solar Home Electricity Data (CC BY 4.0), 2011--2012,
curated by Pierre Haessig in the solarhome-control-bench project.
The *measured* data contains only one household: the mapping across feeder
nodes uses different-day lags and explicit scaling, hence site-level loads are
scenario constructions, not metered IEEE33 node injections.
"""
from __future__ import annotations
import csv
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np

from feeder33 import Feeder33

DATA_PATH=Path(__file__).resolve().parent/'data'/'ausgrid_customer12_2011_2012.csv'
SITES=(5,6,8,11,14,17,18,23,24,25,30,33)
DAY_LAGS=(0,2,4,7,10,13,16,19,22,25,28,31)
OBSERVED_SITES=(2,10) # categorical site indices, buses 8 and 30


def read_ausgrid(path=DATA_PATH):
    """Load half-hourly measured average kW, preserve exact timestamps."""
    dates=[];gross=[];pv=[]
    with open(path,newline='',encoding='utf-8-sig') as f:
        rows=csv.DictReader(f)
        if not {'GC','GG'}.issubset(rows.fieldnames):
            raise ValueError('expected Ausgrid GC and GG columns')
        # Source column is unnamed by design.
        dt_col=rows.fieldnames[0]
        for row in rows:
            dt=datetime.fromisoformat(row[dt_col].strip())
            a,b=float(row['GC']),float(row['GG'])
            if not np.isfinite(a+b) or min(a,b)<0: raise ValueError('nonphysical power in measurements')
            dates.append(dt);gross.append(a);pv.append(b)
    if len(dates)<96:raise ValueError('at least two days of measurements required')
    diffs=np.diff(np.array(dates,dtype='datetime64[m]')).astype('timedelta64[m]').astype(int)
    if not np.all(diffs==30):
        raise ValueError('the source must be a continuous 30-minute series')
    return np.array(dates,dtype='datetime64[m]'),np.asarray(gross),np.asarray(pv)


def construct_profiles(gross,pv,feeder=Feeder33(),*,sites=SITES,day_lags=DAY_LAGS,
                       solar_scale=1.5,load_clip=3.5):
    """Translate real home GC/GG measurements to scenario nodal PQ trajectories.

    All site profiles use real paired GC/GG intervals, with whole-day offsets
    to preserve within-day diurnal shape; node scaling is synthetic and explicit.
    Each site has local load and optional rooftop PV generation (unity PF PV).
    """
    gross=np.asarray(gross,dtype=float);pv=np.asarray(pv,dtype=float)
    if gross.shape!=pv.shape or gross.ndim!=1 or len(gross)<200 or np.any(gross<0) or np.any(pv<0):
        raise ValueError('invalid measured profiles')
    if len(sites)!=len(day_lags) or len(set(sites))!=len(sites):
        raise ValueError('site mapping must have unique sites and a lag per site')
    if not all(2<=i<=33 for i in sites) or min(day_lags)<0 or not np.isfinite(solar_scale):
        raise ValueError('invalid sites, lag or solar scale')
    total=len(gross)-48*max(day_lags)
    if total<=0:raise ValueError('insufficient measured time horizon')
    # All normalization constants estimated from the *early* 160 measured days.
    # The caller uses a strict train/test time split, avoiding future leakage.
    ref_end=min(len(gross),160*48)
    gross_ref=max(float(np.median(gross[:ref_end])),0.1)
    pv_ref=max(float(np.quantile(pv[:ref_end],0.95)),0.1)
    base=feeder.nominal_pq
    loads=np.broadcast_to(base,(total,33,2)).copy()
    for j,(bus,lag) in enumerate(zip(sites,day_lags,strict=True)):
        src=slice(48*lag,48*lag+total)
        demand_factor=0.45+0.55*np.clip(gross[src]/gross_ref,0,load_clip)
        # Alternating solar adoption; per-bus capacities proportional to P_nom.
        adoption=solar_scale*(0.6+0.15*(j%4)) if j%3!=0 else 0.0
        loads[:,bus-1,0]=base[bus-1,0]*(demand_factor-adoption*pv[src]/pv_ref)
        loads[:,bus-1,1]=base[bus-1,1]*demand_factor
    return loads


def split_indices(n, *,train_days=160,test_start_day=215,test_days=115,day_lags=DAY_LAGS):
    """Disjoint raw-measurement support across train and held-out test periods."""
    train_end=train_days*48
    test_start=test_start_day*48
    test_end=(test_start_day+test_days)*48
    if (train_end<=0 or train_end+48*max(day_lags)>test_start or test_end>n):
        raise ValueError('invalid chronological split: lagged raw data may overlap')
    return np.arange(0,train_end,dtype=int),np.arange(test_start,test_end,dtype=int)


def fit_categories(train_powers,site_buses=SITES):
    """Train-only 1/3 and 2/3 net-P quantile thresholds and P/Q means."""
    x=train_powers[:,np.asarray(site_buses)-1,:]
    net=x[:,:,0]
    thresholds=np.quantile(net,[1/3,2/3],axis=0).T
    states=np.sum(net[:,:,None]>thresholds[None,:,:],axis=2).astype(int)
    centers=np.empty((len(site_buses),3,2))
    for site in range(len(site_buses)):
        for cat in range(3):
            selected=states[:,site]==cat
            if not np.any(selected):raise ValueError('empty categorical bin')
            centers[site,cat,:]=x[selected,site,:].mean(axis=0)
    return thresholds,centers,states


def apply_categories(powers,thresholds,site_buses=SITES):
    net=powers[:,np.asarray(site_buses)-1,0]
    return np.sum(net[:,:,None]>thresholds[None,:,:],axis=2).astype(int)


def construct_category_base(feeder,site_buses=SITES):
    """Non-uncertain node loads remain at their MATPOWER demand base."""
    return feeder.nominal_pq.copy()


def state_power(base,centers,site_buses,cat):
    power=base.copy()
    power[np.asarray(site_buses)-1]=centers[np.arange(len(site_buses)),cat]
    return power


def regime(indices):
    """Measured clock-hour proxy: four six-hour operating regimes."""
    return ((np.asarray(indices)%48)//12).astype(int)
