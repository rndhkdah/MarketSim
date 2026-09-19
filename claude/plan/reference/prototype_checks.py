"""Checks behind master plan §13: exact steady state, level-window sign tests, timing, stochastic runs."""
import sys
import numpy as np
from dynamic_core_prototype import *
from prototype_irf import run_shock
def lvl_of(e):
    cpi = np.array([o['cpi'] for o in e.log]); t = np.arange(1, len(cpi)+1)
    return np.log(cpi) - e.P['pi_star']*t/12
for mode in ("nominal_drift", "real"):
    for pis in (0.0, 0.02):
        P = dict(pi_star=pis, smooth_mode=mode)
        e, out = run(360, P=P); dev = max(np.abs(o["x"]/e.x0-1).max() for o in out); o = out[-1]
        line = f"[{mode:<13} pi*={pis}] steady dev {dev:.1e} infl {o['infl']:.5f} W/(B+D) {o['W']/(o['B']+o['debt']):.8f}"
        res = {}
        for kind, mag in [("demand",0.02),("monetary",0.01),("cost_push",0.30),("supply",-0.03),("fiscal",0.05),("row",-0.10)]:
            e = run_shock(kind, mag, T=240, P=P); g = np.array([o['gap'] for o in e.log]); res[kind] = (g, lvl_of(e), e)
        g, lvl, e = res["monetary"]; k = int(np.argmin(g[:120])); X = np.array([o['x'] for o in e.log])/e.x0-1
        tm = {c: int(np.argmin(X[:60, IDX[c]]))+1 for c in ("AUTOS","CONSTRUCT","CAPGOODS")}
        line += f" | MON trough {g[k]*100:+.2f}% @m{k+1} rebound {g[k:120].max()/-g[k]:.2f} lvl18-36 {lvl[17:36].mean()*100:+.2f} timing {tm}"
        g, lvl, e = res["cost_push"]; k = int(np.argmin(g[:120]))
        line += f" | COST trough {g[k]*100:+.2f}% @m{k+1} rebound {g[k:160].max()/-g[k]:.2f} lvl@12 {lvl[11]*100:+.2f}"
        g, lvl, e = res["demand"]; k = int(np.argmax(g[:120]))
        line += f" | DEM peak {g[k]*100:+.2f}% @m{k+1} undershoot {(-g[k:160].min())/g[k]:.2f}"
        ok = (res["demand"][0][:24].sum()>0 and res["demand"][1][17:36].mean()>0 and res["monetary"][0][:36].sum()<0 and res["monetary"][1][17:36].mean()<0
              and res["cost_push"][0][5:48].sum()<0 and res["cost_push"][1][5:24].mean()>0 and res["supply"][0][:48].sum()<0 and res["supply"][1][5:36].mean()>0
              and res["fiscal"][0][:24].sum()>0 and res["fiscal"][1][17:36].mean()>0 and res["row"][0][:24].sum()<0)
        print(line, "| all sign tests:", bool(ok))
print()
def stochastic(P, seed, years=100):
    e = Economy(P); rng = np.random.default_rng(seed); zd = zs = 0.0; zc = np.zeros(S)
    for t in range(12*years):
        zd = 0.90*zd + 0.004*rng.standard_normal(); zs = 0.95*zs + 0.0015*rng.standard_normal()
        zc[IDX["ENERGY"]] = 0.93*zc[IDX["ENERGY"]] + 0.02*rng.standard_normal()
        e.sh['dem'] = zd; e.sh['sup'] = np.full(S, zs); e.sh['cost'] = zc.copy(); e.step()
    L = e.log; g = np.array([o['gap'] for o in L]); U = np.array([o['U'] for o in L]); r = np.array([o['r'] for o in L]); infl = np.array([o['infl'] for o in L])
    I = np.array([o['I'] for o in L]); C = np.array([o['C'] for o in L]); gd_ = np.array([o['gdp'] for o in L]); yoy = lambda a: np.log(a[12:]/a[:-12])
    X = np.array([o['x'] for o in L]); sdx = {c: yoy(X[:, IDX[c]]).std()*100 for c in CODES}; top = sorted(sdx, key=lambda c: -sdx[c])
    print(f"[{P['smooth_mode']:<13} seed {seed}] gap [{g.min()*100:+.1f},{g.max()*100:+.1f}] sd(yoy GDP) {yoy(gd_).std()*100:.2f}% U [{U.min()*100:.1f},{U.max()*100:.1f}] infl [{infl.min()*100:+.1f},{infl.max()*100:+.1f}] r [{r.min()*100:.1f},{r.max()*100:.1f}] ELB {(r<=1e-9).sum()}m sdI/sdY {yoy(I).std()/yoy(gd_).std():.1f} sdC/sdY {yoy(C).std()/yoy(gd_).std():.2f} | most vol: {', '.join(f'{c} {sdx[c]:.1f}' for c in top[:4])} | least: {', '.join(f'{c} {sdx[c]:.1f}' for c in top[-3:])}")
for mode in (("nominal_drift", "real") if "--slow" in sys.argv else ()):
    for s_ in (1, 2): stochastic(dict(pi_star=0.02, smooth_mode=mode), s_)
