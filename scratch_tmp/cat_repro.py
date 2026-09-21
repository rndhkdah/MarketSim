from pathlib import Path
import numpy as np
from marketsim.core.config import load_config
from marketsim.layer1.io import load_io, resolve_io_path
from marketsim.real.economy import RealEconomy

cfg = load_config(Path("config"))
io = load_io(resolve_io_path(cfg))
eco = RealEconomy(cfg, io, pi_star=0.0, check_sfc=False)
eco.step_month()
i = eco.codes.index("CONSTRUCT")
print("k before:", float(eco.k[i]))
claims = eco.bus.inject("catastrophe", 1.0, targets=["CONSTRUCT"], economy=eco, tick=1)
print("claims:", claims, "k after cat:", float(eco.k[i]))
try:
    eco.step_month()
    print("step ok; u_s:", eco.u_s[i], "p:", eco.p[i])
except Exception as e:
    print("RAISED:", type(e).__name__, e)
