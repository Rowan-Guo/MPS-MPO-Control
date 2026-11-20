import math
from itertools import product
LEVELS = (0.05, 0.10, 0.15)
ACTIONS = tuple(product((0.0, 0.05), (-0.05, 0.0, 0.05)))
REGIMES = {'A': (0.01, 0.58, 0.41), 'B': (0.29, 0.02, 0.69)}
OBS = (0.05, 0.10)
SLACK = 1.0104
DEADBAND = 0.002
Z = .01 + .02j

def power(h, b, c, a):
    p, q = a
    return (0j, .02, .02, h, .02, .02, b, .02, .02, c - p - q*1j)

def ac(h, b, c, a, slack, return_all=False):
    """Complex-current backward/forward sweep, radial constant-PQ loads."""
    load = power(h, b, c, a)
    v = [complex(slack)] * 10
    for _ in range(200):
        i = [0j] * 11
        for k in range(9, 0, -1):
            i[k] = (load[k] / v[k]).conjugate() + i[k+1]
        new = [complex(slack)]
        for k in range(1, 10):
            new.append(new[-1] - Z * i[k])
        if max(abs(x-y) for x,y in zip(v,new)) < 1e-13:
            mismatch = max(abs(new[k]*((new[k-1]-new[k])/Z -
                             ((new[k]-new[k+1])/Z if k<9 else 0)).conjugate()-load[k])
                           for k in range(1,10))
            assert mismatch < 1e-10, mismatch
            mags = tuple(map(abs,new))
            return mags if return_all else mags[9]
        v = new
    raise RuntimeError("AC sweep did not converge")

def linear(h,b,c,a,slack):
    return math.sqrt(slack**2 - 2*sum(k*(Z.real*s.real + Z.imag*s.imag)
                                       for k,s in enumerate(power(h,b,c,a))))

def loss(v,db):
    return max(0., abs(v-1.)-db)

def best(d):
    return min(d, key=lambda a:(d[a], math.hypot(*a),a))

