# Tier-3 counterfactual validation (T8.10)

Date: 2026-09-20. This is a research card. Nothing here loosens an earlier gate
or retunes `edges.yaml` / `sectors.yaml`.

## Question

Can we claim that an agent's order *caused* a price (or an unrelated path)
inside MarketSim, and what would that claim be worth outside the sim?

## What we can identify (in-sim, actions of others held fixed)

The engine is a deterministic dynamical system. The do-operator is well-defined:
replay the same seed and the same actions for every other agent, change only
one input, and compare.

`tests/validation/test_intervention_consistency.py` records three facts:

1. **Treated-name impact.** On the §6.4 kernel, a signed volume `q` on name `j`
   moves `ξ_impact,j` and leaves `ξ_impact,k` unchanged when `q_k = 0`.
   Unrelated *kernel* paths are invariant under a single-name intervention.
2. **Dose-response.** `|ΔI|` is strictly increasing in `|q|` at fixed ADV and
   `δ = 0.5` (concave: 4× volume is not 4× impact). Sign(`ΔI`) = sign(`q`).
3. **Determinism.** Same seed + same actions → same `World.state_hash()`.

These are mechanism identities, not estimated treatment effects.

## What we cannot claim

1. **Real-world policy / trading counterfactuals.** The model is not a
   structural causal model of the listed economy. Holding other agents fixed
   is an in-sim construct; outside, other traders would re-optimise (Lucas,
   general-equilibrium feedback, T6.19 edges not yet on `World`).
2. **Empty-world "no impact".** `World.create` with `modules=[]` drops the
   inbox on `PUBLISH` and has no venue. Submit-then-step hashes equal to
   idle-step. That is a missing book, not a null effect. Price impact is
   identified only on a venue (`ImpactKernel`, engine MM, CLOB).
3. **Invariance of *economic* unrelated paths.** A demand shock to HOUSEHOLD
   is supposed to move many sectors through `A` (rule 2–4). Those comovements
   are the model. "Unrelated" must be defined as *not on the typed-edge /
   IO closure of the intervention*. We did not prove that a goods-market
   order leaves, say, `GB_BILL` fair yield unchanged once T6.19 is wired.
4. **Attribution of a live month to one agent.** NPC background flow, the
   engine MM, and other agents share the same book. The residual after
   subtracting a no-order replay is the *partial* effect of that order given
   everyone else's realised actions — not a ceteris-paribus field experiment.
5. **Tier-3 "why did X happen" as a unique decomposition.** T8.06 identities
   (`qty` product, `Δln P` log-sum) are accounting, not causal. They say how
   a realised number adds up, not which primitive we should have blocked.

## Design implication

Keep counterfactual tools on the **do-operator + venue** (replay log + one
edited `(agent, seq)`). Do not advertise a "what if I hadn't traded" button
as a historical fact. T9.02/T7.10 already store the inputs needed to rerun.

## Status

Unsolved as a scientific claim; solved as an engineering check on the kernel
and on determinism. Human gate not requested.
