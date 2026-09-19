# claude/plan — how to use this plan with Cursor

1. Copy the bundle into the repo root, keeping paths: `AGENTS.md`, `.cursor/rules/marketsim.mdc`, `claude/plan/**`.
   Cursor reads `AGENTS.md` automatically; the `.mdc` rule is an always-on pointer to it.
2. Read `00-MASTER-PLAN.md` once yourself — §12 lists the defaults the agents will assume until you change them, §14 the
   places where this plan deliberately differs from the design docs.
3. Start an agent session with:

   > Read `AGENTS.md` and `claude/plan/00-MASTER-PLAN.md` §6–§8. Then do task **T0.01** from
   > `claude/plan/01-phase0-foundation.md`. One task only. Follow the card's Tests. Update `claude/plan/PROGRESS.md`
   > and stop.

   For later tasks replace the ID and file. Cards name the spec sections to read first.
4. Review at the gates (master plan §9) and at every **HUMAN GATE** card. Questions from agents accumulate in
   `QUESTIONS.md`; decisions you take go in `DECISIONS.md` as short ADRs.
5. Parallel work: Track B (market microstructure, T6.07–T6.13) can run alongside Phases 3–5 once T2.03 is done;
   Phase 4 can run alongside Phase 3 after T2.18 (cards that need region masks or want shifters say so).

v1.2 adds the monetary policy framework (`02-…` §2.14): committee, calendar, dual mandate, published-data information
set, makeup / risk-management / financial-conditions options — decision D15.

v1.1 adds the bond market (`06-…` §6.11), government and central bank as policy authorities (`02-…` §2.13) and one common
borrowing rate for all firms (`02-…` §2.7) — decisions D12–D14 in the master plan.

`reference/` holds the numerical prototype behind Phase 2. Run it from the repo root with
`python claude/plan/reference/prototype_checks.py` (needs numpy, scipy, pyyaml and the legacy `claude/marketsim`
tree, or set `MARKETSIM_ROOT`). It is a reference for equations and expected numbers — **not** production code: no
ledger, no banks, one region, and it scales the whole cost-of-capital gap by leverage (see `02-…` §2.6).
