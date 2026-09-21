from pathlib import Path
import numpy as np
from marketsim.core.config import load_config
from marketsim.layer1.io import load_io, resolve_io_path
from marketsim.real.economy import RealEconomy
for kap in (0.5, 0.95, 1.5):
    cfg = load_config(Path("config")); io = load_io(resolve_io_path(cfg))
    eco = RealEconomy(cfg, io, pi_star=0.0, check_sfc=False)
    eco.step_month()
    i = eco.codes.index("CONSTRUCT")
    eco.bus.inject("catastrophe", kap, targets=["CONSTRUCT"], economy=eco, tick=1)
    try:
        for _ in range(3): eco.step_month()
        print(f"kappa={kap}: survived. k={eco.k[i]:.4g} p[i]={eco.p[i]:.4g} nan_p={int(np.isnan(eco.p).sum())} u_s={eco.u_s[i]:.4g}")
    except Exception as e:
        print(f"kappa={kap}: RAISED {type(e).__name__}: {e}  k={eco.k[i]}")
