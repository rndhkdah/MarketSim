"""Impulse-response harness for the reference prototype (see dynamic_core_prototype.py)."""
import numpy as np
from dynamic_core_prototype import *

def ar(persist_q): return np.exp(-1.0/(3.0*persist_q))

def run_shock(kind, mag, T=240, P=None):
    e=Economy(P); z=mag
    pers=dict(demand=6,cost_push=8,monetary=4,supply=12,fiscal=8,row=6)[kind]; rho=ar(pers)
    for t in range(T):
        if kind=="demand": e.sh['dem']=z
        elif kind=="cost_push":
            c=np.zeros(S); c[IDX['ENERGY']]=z; c[IDX['AGRIFOOD']]=0.3*z; e.sh['cost']=c
        elif kind=="monetary": e.sh['mon']=z
        elif kind=="supply": e.sh['sup']=np.full(S,z)
        elif kind=="fiscal": e.sh['fisc']=z
        elif kind=="row": e.sh['row']=z
        e.step(); z*=rho
    return e

def summary(e,label):
    L=e.log; g=np.array([o['gap'] for o in L]); pi=np.array([o['infl'] for o in L]); r=np.array([o['r'] for o in L])
    I=np.array([o['I'] for o in L]); C=np.array([o['C'] for o in L]); U=np.array([o['U'] for o in L])
    pk=int(np.abs(g).argmax())
    print(f"\n== {label}")
    print(f"   GDP gap   : peak {g[pk]*100:+.2f}% at month {pk+1};  @3m {g[2]*100:+.2f}  @12m {g[11]*100:+.2f}  @24m {g[23]*100:+.2f}  @48m {g[47]*100:+.2f}  @120m {g[119]*100:+.2f}  @240m {g[-1]*100:+.3f}")
    k=int(np.abs(pi).argmax()); print(f"   inflation : peak {pi[k]*100:+.2f}pp at month {k+1};  @12m {pi[11]*100:+.2f}  @24m {pi[23]*100:+.2f}  @48m {pi[47]*100:+.2f}")
    print(f"   policy r  : max {r.max()*100:.2f}%  min {r.min()*100:.2f}%   |  I: min {I.min():.2f} max {I.max():.2f} (base 19)  C: min {C.min():.2f} max {C.max():.2f} (base 62)  U: {U.min()*100:.2f}-{U.max()*100:.2f}%")
    X=np.array([o['x'] for o in L])/e.x0-1
    return g,pi,X

if __name__=="__main__":
    e=run_shock("demand",0.02); g,pi,X=summary(e,"DEMAND +2% consumption propensity, persistence 6q")
    print("   sign restriction (output +, inflation +):", g[:24].mean()>0 and pi[6:36].mean()>0)
    amp={c: np.abs(X[:,IDX[c]]).max() for c in CODES}
    print("   peak |output dev| by sector (bullwhip ordering):", ", ".join(f"{c} {amp[c]*100:.2f}" for c in sorted(amp,key=lambda c:-amp[c])[:8]))
    print("   upstream amplification: SEMIS/DISCRET =%.2f  MATERIALS/STAPLES=%.2f"%(amp['SEMIS']/amp['DISCRET'], amp['MATERIALS']/amp['STAPLES']))

    e=run_shock("monetary",0.01); g,pi,X=summary(e,"MONETARY +100bp, persistence 4q")
    print("   sign restriction (output -, inflation -):", g[:36].mean()<0 and pi[12:48].mean()<0)
    tm={c:int(np.argmin(X[:60,IDX[c]]))+1 for c in ("CONSTRUCT","AUTOS","REALESTATE","CAPGOODS","SEMIS","MATERIALS","STAPLES")}
    print("   trough month:", tm, "| trough depth %:", {c: round(float(X[:60,IDX[c]].min()*100),2) for c in tm})
    print("   hump-shaped (trough later than month 6 and then recovers):", np.argmin(g[:120])>6 and g[119]>g[:120].min()*0.5)

    e=run_shock("cost_push",0.30); g,pi,X=summary(e,"COST-PUSH ENERGY +30% (AGRIFOOD +9%), persistence 8q")
    print("   sign restriction (output -, inflation +):", g[6:48].mean()<0 and pi[3:36].mean()>0)
    P=np.array([o['p'] for o in e.log]); k=12
    print("   relative price rise at 12m %:", {c: round(float((P[k-1,IDX[c]]-1)*100),2) for c in ("ENERGY","UTILITIES","MATERIALS","TRANSPORT","AUTOS","STAPLES","SOFTWARE")})

    e=run_shock("supply",-0.03); g,pi,X=summary(e,"SUPPLY -3% effective capacity everywhere, persistence 12q")
    print("   sign restriction (output -, inflation +):", g[:48].mean()<0 and pi[3:48].mean()>0)
