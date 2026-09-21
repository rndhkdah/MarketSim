from pathlib import Path
import traceback
import numpy as np
from marketsim.core.config import load_config
from marketsim.layer1.io import load_io, resolve_io_path
from marketsim.real.economy import RealEconomy

cfg = load_config(Path("config"))
io = load_io(resolve_io_path(cfg))
eco = RealEconomy(cfg, io, pi_star=0.0, check_sfc=False)
eco.step_month()
i = eco.codes.index("CONSTRUCT")
eco.bus.inject("catastrophe", 1.0, targets=["CONSTRUCT"], economy=eco, tick=1)
try:
    eco.step_month()
except Exception:
    traceback.print_exc()
print("u_s nan count:", int(np.isnan(eco.u_s).sum()), "p nan:", int(np.isnan(eco.p).sum()))
