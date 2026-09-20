# Proposed Korea flavour — keys that already exist, plus ones we cannot add

`src/marketsim/core/config.py` uses `extra="forbid"`. This file is **not** loaded
by the engine. Copy a row into `World.create(overrides=…)` only after a human
review. ADR-015: optional flavour, not an applied retune of US seed elasticities.

## Constructor / enum — cannot go in YAML without editing config.py

| proposed | why it is not a YAML key | how to set it |
|---|---|---|
| `π* = 0.02` | `pi_star` is `RealEconomy` / `make_real_world(..., pi_star=)` and the CB state, not `PolicyFile` or `DynamicsConfig` | `make_real_world(config_dir, pi_star=0.02)` |
| `world.io_source: bok` | `WorldSettings.io_source` is `Literal["seed", "bea"]` | keep `seed`; review a local aggregated table. Do not add an enum. |

BOK medium-term inflation target is 2 % (Monetary Policy Board, target adopted
2016 and renewed). That is the same number the seed already uses in stationarity
tests; it is not a new elasticity.

## Dotted overrides that already exist (DynamicsConfig / PolicyFile)

Do not invent keys. These are the documented Korean-flavoured knobs whose names
already live on the pydantic models:

```python
{
    # Value-Added Tax Act (부가가치세법) Art. 30: standard rate 10 %. Seed vat is 0.
    "dynamics.fiscal.vat": 0.10,

    # Corporate Tax Act top national bracket 24 % (2023 revision). Local income
    # tax (10 % of CIT) is not a separate config key — do not invent one.
    "dynamics.fiscal.corp_tax": 0.24,

    # General-government D2 near 50 % (MOSF / BOK, 2023–24). Seed is 0.60.
    "dynamics.fiscal.debt_to_gdp": 0.50,

    # Structural unemployment near 3–3.5 % (BOK / OECD-style NAIRU). Seed 0.05.
    "dynamics.labour.u_star": 0.035,
}
```

## Already-default policy knobs (thin `policy.yaml` snippet)

Bank of Korea Monetary Policy Board: 8 scheduled rate meetings per year (since
2017) and a conventional 25 bp grid. Those keys already exist and already match
the D15 defaults — `config/kr/policy.yaml` only restates them.

```python
{
    "policy.monetary.calendar.meetings_per_year": 8,
    "policy.monetary.calendar.rate_step": 0.0025,
}
```

## Do not propose (ADR-015)

- `config/sectors.yaml` / `config/edges.yaml` elasticities
- a second 18-sector list
- `io_source: bok` or a `notes:` / `flavour:` field on `WorldSettings`
- replacing `config/io_table.json` in the repo
