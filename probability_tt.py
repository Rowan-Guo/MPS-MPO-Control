"""Sample-learned, nonnegative classical probability MPS (tensor train).

A finite mixture of K categorical product distributions is learned by EM from
complete NORMAL-COMMUNICATION training samples. Its *probability* tensor has
an exact TT/MPS representation of bond rank <= K, without a 3**sites table.

This differs from the original Born MPS: here p(x) itself is factored, rather
than sqrt(p(x)). Expectations use ONE probability layer, not <psi|O|psi>.
The contrast is deliberate and documented; no quantum advantage is claimed.
"""
from __future__ import annotations

import numpy as np
from tensor import evidence, validate_cores


def fit_mixture_em(samples, *, components=3, pseudocount=0.5,
                   iterations=150, seed=20261120, tolerance=1e-7, restarts=3):
    """Learn K-product mixture from N x sites integer samples via EM.

    Training O(restarts * iterations * N * sites * K * categories), memory
    O(N*sites*categories + sites*K). No dense 3**sites states created.
    Pseudocount is a symmetric additive Dirichlet regularizer per category.
    This is an approximate restricted model, not a guaranteed global MLE.
    """
    x=np.asarray(samples)
    if (x.ndim!=2 or len(x)<2 or x.shape[1]<2 or not np.issubdtype(x.dtype,np.integer)
        or np.any(x<0) or components<1 or pseudocount<=0 or iterations<1 or restarts<1):
        raise ValueError('invalid samples, components, pseudocount or iteration count')
    n,sites=x.shape
    d=int(np.max(x))+1
    if d<2:
        raise ValueError('need at least two observed categorical levels')
    one_hot=np.eye(d)[x]
    rng=np.random.default_rng(seed)
    best=None
    for restart in range(restarts):
        weights=rng.dirichlet(np.ones(components))
        local=rng.dirichlet(np.ones(d)*2.,size=(components,sites))
        prev=-np.inf
        for step in range(iterations):
            log_joint=np.broadcast_to(np.log(weights),(n,components)).copy()
            for site in range(sites):
                log_joint+=np.log(local[:,site,x[:,site]].T)
            offset=np.max(log_joint,axis=1,keepdims=True)
            exp=np.exp(log_joint-offset)
            denominator=exp.sum(axis=1,keepdims=True)
            responsibility=exp/denominator
            log_likelihood=float(np.sum(offset+np.log(denominator)))
            if step and abs(log_likelihood-prev)<=tolerance*max(1,abs(prev)):
                break
            prev=log_likelihood
            weights=(responsibility.sum(axis=0)+pseudocount)
            weights/=weights.sum()
            counts=np.einsum('nk,nid->kid',responsibility,one_hot,optimize=True)
            local=(counts+pseudocount)
            local/=local.sum(axis=-1,keepdims=True)
        # Calculate final likelihood after the last M step, not the penultimate.
        score=mixture_log_likelihood(x,weights,local)
        if best is None or score>best[2]:
            best=(weights.copy(),local.copy(),score,step+1)
    return dict(weights=best[0],local_probabilities=best[1],
                train_log_likelihood=best[2],em_iterations=best[3])


def mixture_log_likelihood(samples,weights,local):
    x=np.asarray(samples)
    n,sites=x.shape
    log_joint=np.broadcast_to(np.log(weights),(n,len(weights))).copy()
    for site in range(sites):
        log_joint+=np.log(local[:,site,x[:,site]].T)
    mx=np.max(log_joint,axis=1)
    return float(np.sum(mx+np.log(np.exp(log_joint-mx[:,None]).sum(axis=1))))


def mixture_probability_mps(weights,local_probabilities):
    """Exact nonnegative TT for p(x)=sum_k w_k prod_i p_i(x_i|k)."""
    w=np.asarray(weights,dtype=float)
    p=np.asarray(local_probabilities,dtype=float)
    if (p.ndim!=3 or p.shape[1]<2 or p.shape[2]<2 or w.shape!=(p.shape[0],)
        or np.any(w<0) or not np.isclose(w.sum(),1,atol=1e-9)
        or not np.allclose(p.sum(axis=-1),1,atol=1e-9) or np.any(p<0)):
        raise ValueError('invalid mixture probability parameters')
    k,sites,d=p.shape
    first=np.einsum('k,kd->dk',w,p[:,0]).reshape(1,d,k)
    out=[first]
    for site in range(1,sites-1):
        core=np.zeros((k,d,k))
        ids=np.arange(k)
        core[ids,:,ids]=p[:,site]
        out.append(core)
    out.append(p[:,-1].reshape(k,d,1))
    return tuple(out)


def probability_contract(prob,diagonal_operator,mask=None):
    """Single-layer TT contraction sum_x p(x) O(x) mask(x)."""
    dims=validate_cores(prob)
    validate_cores(diagonal_operator,dims=dims)
    if len(prob)!=len(diagonal_operator):
        raise ValueError('incompatible TT site counts')
    if mask is None:
        mask=tuple(np.ones(d) for d in dims)
    if len(mask)!=len(dims) or any(np.shape(m)!=(d,) for m,d in zip(mask,dims)):
        raise ValueError('invalid evidence mask')
    env=np.ones((1,1))
    for a,w,m in zip(prob,diagonal_operator,mask,strict=True):
        env=np.einsum('ij,isa,jsb,s->ab',env,a,w,m,optimize=True)
    return float(np.real_if_close(env[0,0]))


def probability_scores(prob,operators,seen,unsafe=None,*,risk_tolerance=0.0,
                       return_risks=False):
    if not 0<=risk_tolerance<=1:
        raise ValueError('risk_tolerance must be in [0,1]')
    dims=validate_cores(prob)
    mask=evidence(seen,dims)
    identity=tuple(np.ones((1,d,1)) for d in dims)
    den=probability_contract(prob,identity,mask)
    if den<=1e-14:
        raise ValueError('zero probability evidence')
    if unsafe is not None and len(unsafe)!=len(operators):
        raise ValueError('expected one safety indicator per action')
    risk=np.zeros(len(operators)) if unsafe is None else np.array([
        np.clip(probability_contract(prob,flag,mask)/den,0,1) for flag in unsafe])
    scores=np.array([probability_contract(prob,op,mask)/den if r<=risk_tolerance+1e-10
                     else np.inf for op,r in zip(operators,risk,strict=True)])
    return (scores,den,risk) if return_risks else (scores,den)


def probability_posterior(prob,seen,site,den=None):
    dims=validate_cores(prob)
    if site in seen or site not in range(len(dims)):
        raise ValueError('target site must be missing')
    identity=tuple(np.ones((1,d,1)) for d in dims)
    if den is None:
        den=probability_contract(prob,identity,evidence(seen,dims))
    if den<=1e-14:
        raise ValueError('zero probability evidence')
    return np.array([probability_contract(prob,identity,evidence({**seen,site:v},dims))/den
                     for v in range(dims[site])])
