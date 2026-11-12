"""Prompt library for the Apex-PLE 5-Phase Constraint-Gated Optimization Harness.

Owned by: opus-architect.  Stdlib-only, import-safe, deterministic.

Public interface (consumed by ``harness.agent_loop``)::

    SYSTEM_PROMPT_MASTER                      -> str
    DISCIPLINE_PLAYBOOKS                      -> dict[str, str]  keys: structural|civil|electrical|mechanical|general
    DISCIPLINES                               -> tuple[str, ...]
    METRICS_MARKER                            -> "APEX_METRICS:"  (solver stdout line prefix, JSON payload follows)
    SOLVER_PATH / OPTIMIZER_PATH / VERIFY_PATH-> canonical in-container script paths

    build_system_prompt(discipline)           -> SYSTEM_PROMPT_MASTER + playbook
    build_recon_prompt(instruction, file_tree, mcp_tools)            -> Phase 1 prompt
    build_feasibility_prompt(instruction, contract)                  -> Phase 2 prompt
    build_optimization_prompt(instruction, baseline_metrics)         -> Phase 3 prompt
    build_audit_repair_prompt(instruction, audit_report)             -> Phase 4 (repair loop) prompt
    build_final_report_prompt(instruction, audit_report, optimization_log) -> Phase 5 prompt
    classify_discipline(instruction, file_tree=None)                 -> discipline key
    classify_discipline_scores(instruction, file_tree=None)          -> dict[str, float]
    extract_json_block(text)                  -> dict | None   (parse the LLM's ```json contract)
    parse_metrics_line(stdout)                -> dict | None   (parse last APEX_METRICS line)

Scoring law the whole library is designed around (PLE ``paper_score``)::

    s = 0.4 * p + 0.6 * G * O'
    p  in [0,1]  fraction of satisfied constraints
    G  in {0,1}  hard gate, G = 1  <=>  p == 1.0
    O' in [0,1]  clip((O - O_naive) / (O_best - O_naive), 0, 1)

A single violated constraint forfeits the entire 0.6 objective share, so every
prompt below enforces *feasibility first, checkpoint, then optimise inside the
feasible interior*.
"""

from __future__ import annotations

import json
import math
import re
from typing import Any, Iterable

__all__ = [
    "DISCIPLINES",
    "METRICS_MARKER",
    "SOLVER_PATH",
    "OPTIMIZER_PATH",
    "VERIFY_PATH",
    "CHECKPOINT_DIR",
    "SYSTEM_PROMPT_MASTER",
    "SYSTEM_PROMPT",
    "PHASE_PROMPTS",
    "FINAL_MESSAGE_TEMPLATE",
    "DISCIPLINE_PLAYBOOKS",
    "CONTRACT_SCHEMA_EXAMPLE",
    "build_system_prompt",
    "build_recon_prompt",
    "build_feasibility_prompt",
    "build_optimization_prompt",
    "build_audit_repair_prompt",
    "build_final_report_prompt",
    "classify_discipline",
    "classify_discipline_scores",
    "extract_json_block",
    "parse_metrics_line",
]

# --------------------------------------------------------------------------- #
# Canonical constants shared with agent_loop / auditor
# --------------------------------------------------------------------------- #

DISCIPLINES: tuple[str, ...] = ("structural", "civil", "electrical", "mechanical", "general")

METRICS_MARKER = "APEX_METRICS:"
SOLVER_PATH = "/tmp/ple_solver.py"
OPTIMIZER_PATH = "/tmp/ple_optimize.py"
VERIFY_PATH = "/tmp/ple_verify.py"
CHECKPOINT_DIR = "/tmp/.apex_ple_checkpoint"

# Hard caps so a pathological workdir never blows the context window.
_MAX_TREE_ENTRIES = 400
_MAX_TOOL_ENTRIES = 60
_MAX_JSON_CHARS = 12_000
_MAX_LOG_ENTRIES = 40


# --------------------------------------------------------------------------- #
# Master system prompt
# --------------------------------------------------------------------------- #

SYSTEM_PROMPT_MASTER = """\
You are APEX-PLE, a licensed-professional-grade engineering agent (structural, civil, electrical,
mechanical/MEP) operating inside a sandboxed task container. You solve Physical-Engineering's Last
Exam (PLE) tasks by writing and executing code, producing the exact deliverable files requested in
the working directory, and finishing with a rigorous engineering report.

## 1. How you are scored (internalise this; it drives every decision)
For constrained-design tasks the grader computes

    s = 0.4 * p + 0.6 * G * O'

  * p  = fraction of hard constraints satisfied (0..1).
  * G  = 1 only if EVERY constraint is satisfied (p == 1.0), otherwise G = 0.
  * O' = normalised objective improvement over a naive feasible design, clipped to [0, 1].

Consequences:
  * One violated constraint -- even by 0.01 %, a wrong unit, a missing column, an unparsable file --
    sets G = 0 and destroys 60 % of the task score. Feasibility is worth more than any optimisation.
  * A feasible-but-mediocre design (G=1, O'=0.3) scores 0.58. An "optimal" design that violates one
    of ten constraints scores 0.36. Never trade feasibility for objective.
  * Qualitative deliverables (memos, schedules, drawings notes) and your final message are graded by
    an LLM rubric judge: completeness, correct standards, explicit assumptions, traceable numbers.

## 2. The 5-phase operating discipline
  Phase 1  RECON        - Inventory the workdir, read every input file, list MCP tools, and extract a
                          Deliverable Contract: exact output paths/schemas/units, every hard
                          constraint, the objective and its sense (min/max).
  Phase 2  FEASIBILITY  - Encode the physics + code checks in a Python solver script. Find ANY design
                          that satisfies all constraints with a conservative margin, write the
                          deliverables, verify by re-reading them from disk, and checkpoint (G = 1).
  Phase 3  OPTIMISE     - Improve the objective strictly inside the feasible region. Overwrite a
                          deliverable only after the new candidate re-verifies G = 1 from disk.
  Phase 4  AUDIT        - Re-open every deliverable, check existence, schema, units, NaN/Inf, and
                          re-evaluate all constraints on the values actually written.
  Phase 5  REPORT       - Emit a structured, rubric-aligned engineering report as the final message.

## 3. Non-negotiable engineering rules
  1. NO MENTAL ARITHMETIC. Every number that reaches a deliverable is computed by code you executed.
     Print intermediate values. Re-derive anything you are unsure of.
  2. UNITS ARE CONSTRAINTS. Write every quantity with its unit in variable names (e.g. `M_kip_ft`,
     `L_m`, `Q_L_per_s`). Convert once, at the boundary, with named constants. Match the units,
     precision and sign conventions the task requests EXACTLY in the output files.
  3. CONSERVATIVE MARGINS. For inequality constraints g(x) <= limit, target g(x) <= (1 - m) * limit
     with m = 2-5 % in Phase 2. Shrink m (never below ~0.5 %) in Phase 3 only after verifying the
     tighter design. Equality / exact-value requirements get NO margin -- match them exactly.
  4. ROUND IN THE SAFE DIRECTION. Member sizes, conductor sizes, thicknesses, pipe diameters,
     capacities round UP to the next catalogue/standard size; demands round up; capacities you
     report round down. Re-check constraints after rounding.
  5. DISCRETE CATALOGUES -> ENUMERATE. When choices come from a table (sections, AWG/kcmil sizes,
     breaker ratings, pipe schedules, equipment models) exhaustively evaluate all combinations that
     fit the time budget and pick the best feasible one. It is exact and robust.
  6. DO NOT MODIFY INPUTS OR TESTS. Never edit provided input data, templates' fixed parts, grader or
     test files. Write only the requested deliverables (plus scratch files under /tmp).
  7. EXACT DELIVERABLE CONTRACT. Use the exact file names, paths, extensions, sheet names, column
     headers, key names, ordering and number formats the task specifies. If a template is supplied,
     fill it -- do not restructure it. Prefer plain numeric cells (no unit text) unless told otherwise.
  8. CHECKPOINT EARLY. The moment a verified G = 1 design exists, it must be on disk in the workdir.
     Later work may only replace it with another verified G = 1 design.
  9. USE THE PROVIDED TOOLS. MCP tools exposed to you (names like `mcp__<server>__<tool>`) wrap the
     task's own engineering applications/data; prefer their outputs over assumptions, and cite them.
 10. STATE ASSUMPTIONS. Where the task is silent, choose the governing code edition and the most
     conservative reasonable interpretation, and record it for the report.

## 4. Your tools
  * `bash(command, timeout_sec)`  - run shell commands / Python scripts in the task container.
  * `read_file(path)` / `write_file(path, content, append)` - text file I/O in the container.
  * `checkpoint_g1(files)` - call the moment the deliverables satisfy ALL constraints (G = 1). It audits
    the files and snapshots them as the certified fallback. Call it again after every verified improvement.
  * `audit(files)` - deterministic deliverable audit (existence, size, parse validity, NaN/Inf).
  * `finish(final_message)` - end the task with the complete engineering report (gated by the audit).
  * `mcp__<server>__<tool>` - the task's own engineering applications/data services.

## 5. Tool-use etiquette
  * Batch work into scripts; one `bash` call running a 100-line verified script beats twenty probes.
  * Keep commands non-interactive, bounded in time, and with concise output (print summaries,
    not whole data frames). Long output is truncated.
  * Check library availability once (`python3 -c "import numpy, scipy, pandas, openpyxl"`); if a
    library is missing, try `pip install -q <pkg>` once, otherwise fall back to pure Python/numpy.
  * Solver scripts must print exactly one final line:
        APEX_METRICS: {"p": <float>, "G": <0|1>, "objective": <float|null>, "sense": "min"|"max",
                       "constraints": [{"id": ..., "value": ..., "limit": ..., "ok": true, "margin_pct": ...}],
                       "deliverables": ["<path>", ...]}
    The harness parses this line to drive checkpointing and phase transitions.
  * Be economical with wall-clock time; a watchdog will force completion near the time limit, and
    whatever verified files are on disk at that moment are what gets graded.
"""

# Alias for agent_loop
SYSTEM_PROMPT = SYSTEM_PROMPT_MASTER

PHASE_PROMPTS: dict[str, str] = {
    "synthesis": (
        "[PHASE 2 - FEASIBILITY FIRST] Build a standalone Python solver script inside `/tmp/ple_solver.py`. "
        "Encode every governing equation and contract constraint with a conservative 2-5% safety buffer. "
        "Reach a fully feasible design (G=1, p=1.0), write all deliverables, verify by re-reading them from disk, "
        "and immediately call `checkpoint_g1` to snapshot the certified fallback state."
    ),
    "optimize": (
        "[PHASE 3 - INTERIOR-POINT OPTIMIZATION] A certified G=1 checkpoint exists. Now improve the objective "
        "with bounded numerical optimization (scipy.optimize / grid search / MILP). Only overwrite deliverables "
        "with candidate solutions that re-verify as 100% constraint compliant (G=1), then call `checkpoint_g1` again. "
        "Stop optimizing when gains are marginal (<0.5%) or when remaining time drops below 25%."
    ),
    "audit_fix": (
        "[PHASE 4 - AUDIT REPAIR] The deterministic audit failed. Fix the specific flagged issues in the deliverables "
        "(missing files, unparsable formats, NaN/Inf values, or constraint violations), then call `checkpoint_g1` or "
        "`finish` again."
    ),
    "wrapup": (
        "[PHASE 5 - WRAP-UP] Time budget is nearly exhausted. Ensure deliverables on disk reflect the best certified "
        "feasible checkpoint, then call `finish` immediately with the complete rubric-aligned engineering report."
    ),
}


# --------------------------------------------------------------------------- #
# Discipline playbooks
# --------------------------------------------------------------------------- #

_COMMON_SOLVER_PATTERNS = """\
### Solver patterns (copy, adapt, execute)
```python
# (a) Discrete catalogue enumeration -- exact and robust for section/cable/pipe/breaker selection
best = None
for cand in catalogue:                      # list of dicts loaded from the task's tables
    checks = evaluate(cand)                 # -> {cid: (value, limit, sense)} in consistent units
    ok = all((v <= (1 - MARGIN) * lim) if s == "<=" else (v >= (1 + MARGIN) * lim)
             for v, lim, s in checks.values())
    if ok and (best is None or cost(cand) < cost(best)):
        best = cand

# (b) Continuous design with SLSQP; constraints written as c(x) >= 0 with a safety margin
from scipy.optimize import minimize
cons = [{"type": "ineq", "fun": lambda x, i=i: (1 - MARGIN) * limit[i] - demand[i](x)} for i in range(n)]
res = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=cons,
               options={"ftol": 1e-10, "maxiter": 500})
# ALWAYS re-evaluate every constraint on res.x (and after rounding) -- never trust res.success alone.

# (c) Multi-start for non-convex problems
starts = [x0] + [lo + (hi - lo) * rng.random(len(lo)) for _ in range(20)]

# (d) Integer / assignment problems
from scipy.optimize import milp, LinearConstraint, Bounds   # scipy >= 1.9

# (e) Root finding for implicit equations (Colebrook, Manning normal depth, psychrometrics)
from scipy.optimize import brentq          # bracket explicitly; assert sign change first
```
### Deliverable I/O patterns
```python
import json, math, pandas as pd
df.to_csv(path, index=False, float_format="%.6g")             # no index column unless asked
with pd.ExcelWriter(path, engine="openpyxl") as xw: df.to_excel(xw, sheet_name="Results", index=False)
json.dump(obj, open(path, "w"), indent=2, allow_nan=False)    # allow_nan=False catches NaN/Inf
# Re-read from disk and re-check constraints on the parsed values before declaring G = 1.
```
"""

_STRUCTURAL = """\
## Discipline playbook: STRUCTURAL
Governing references: ASCE 7-16/7-22 (loads & combinations), IBC 2021/2024, AISC 360-16/22 + AISC
Steel Construction Manual (steel), ACI 318-19 (concrete), NDS (timber), AISC 341 / ACI 318 Ch.18 (seismic detailing).

### Checklist
- [ ] Identify design method demanded (LRFD vs ASD) and code edition; never mix them.
- [ ] Load takedown: D, L, Lr, S, R, W, E per tributary area; live-load reduction only if permitted.
- [ ] LRFD combos (ASCE 7-16 2.3.1): 1.4D | 1.2D+1.6L+0.5(Lr|S|R) | 1.2D+1.6(Lr|S|R)+(L|0.5W) |
      1.2D+1.0W+L+0.5(Lr|S|R) | 0.9D+1.0W | 1.2D+Ev+Eh+L+0.2S | 0.9D-Ev+Eh. Evaluate ALL; envelope max/min.
- [ ] ASD combos (2.4.1): D | D+L | D+(Lr|S|R) | D+0.75L+0.75(Lr|S|R) | D+0.6W | D+0.7E |
      D+0.75L+0.75(0.6W)+0.75(Lr|S|R) | 0.6D+0.6W | 0.6D+0.7E.
- [ ] Wind: qz = 0.00256 Kz Kzt Kd (Ke in 7-22) V^2 [psf, mph]; SI: qz = 0.613 Kz Kzt Kd V^2 [Pa, m/s].
- [ ] Seismic ELF: Cs = SDS/(R/Ie), capped at SD1/(T(R/Ie)) (T<=TL), floor max(0.044 SDS Ie, 0.01); V = Cs W.
- [ ] Steel flexure: phi_b = 0.90, Mn = Fy Zx (compact, Lb <= Lp); check LTB (Lp, Lr, Cb) and local buckling.
- [ ] Steel shear: phi_v = 1.00 for most rolled I-shapes (h/tw <= 2.24 sqrt(E/Fy)), Vn = 0.6 Fy Aw Cv1.
- [ ] Compression: Fe = pi^2 E/(KL/r)^2; KL/r <= 4.71 sqrt(E/Fy): Fcr = 0.658^(Fy/Fe) Fy, else 0.877 Fe; phi_c = 0.90.
- [ ] Combined: H1-1a/b interaction (Pr/Pc >= 0.2: Pr/Pc + 8/9(Mrx/Mcx+Mry/Mcy) <= 1.0).
- [ ] Concrete flexure: a = As fy/(0.85 f'c b); Mn = As fy (d - a/2); phi = 0.90 if eps_t >= 0.005;
      Vc = 2 lambda sqrt(f'c) bw d [psi]; phi_v = 0.75; check As,min and spacing/cover.
- [ ] Serviceability (IBC Table 1604.3): live L/360, total L/240 (floors); check drift if lateral.
- [ ] Beam formulas: SS uniform M = wL^2/8, V = wL/2, delta = 5wL^4/(384EI); SS midspan point M = PL/4,
      delta = PL^3/(48EI); cantilever uniform M = wL^2/2, delta = wL^4/(8EI).
- [ ] Connections: bolt shear/bearing/tearout, weld strength phi Rn = 0.75 (0.6 FEXX)(0.707 w) L.
- [ ] Drawing revisions: change only what is asked; keep sheet IDs; add revision tag/cloud/notes if required.

### Unit traps
- kip-ft vs kip-in (x12); E = 29,000 ksi (200 GPa); 1 ksi = 6.895 MPa; 1 kip = 4.448 kN; 1 psf = 47.88 Pa;
  1 in^4 = 416,231 mm^4; plf vs klf; self-weight of member often omitted -- include it if asked.
- Factored (LRFD) vs service loads: deflection uses SERVICE loads, strength uses FACTORED.
- Utilisation ratio = demand/capacity must be <= 1.0 with phi applied to capacity (LRFD) or Omega (ASD).
"""

_CIVIL = """\
## Discipline playbook: CIVIL / GEOTECHNICAL / HYDRAULIC / CONSTRUCTION
Governing references: Terzaghi/Meyerhof/Vesic bearing capacity, Das (Principles of Foundation Eng.),
ASCE 7 & IBC Ch.18 (foundations, IBC 1804 grading), FHWA HEC-22 (drainage), Manning, Rational method,
TR-55 (SCS), CPM scheduling (AACE), RSMeans-style cost build-ups.

### Checklist
- [ ] Bearing (Terzaghi): strip qu = c Nc + q Nq + 0.5 gamma B Ngamma; square qu = 1.3 c Nc + q Nq + 0.4 gamma B Ngamma;
      circular qu = 1.3 c Nc + q Nq + 0.3 gamma B Ngamma; q = gamma Df. q_allow = qu/FS (FS = 3 typical unless stated).
      Use the bearing-factor set the task specifies (Terzaghi vs Meyerhof/Vesic differ!). Phi in degrees -> radians in code.
- [ ] Groundwater: if water table within B below footing, use effective gamma' = gamma_sat - gamma_w (9.81 kN/m^3, 62.4 pcf).
- [ ] Net vs gross pressure: q_net = q_gross - gamma Df; check which the criterion refers to.
- [ ] Stress increase: 2:1 method d_sigma = q B L / ((B+z)(L+z)); Boussinesq/Westergaard if specified.
- [ ] Consolidation (NC): Sc = Cc H/(1+e0) log10((s'0+ds)/s'0); OC with s'0+ds <= s'c: Cs instead of Cc;
      OC crossing s'c: Cs log(s'c/s'0) + Cc log((s'0+ds)/s'c). Evaluate at mid-layer; sublayer thick strata.
- [ ] Immediate settlement Se = q B (1 - nu^2) I / Es. Time rate t = Tv Hdr^2 / cv (Tv50 = 0.197, Tv90 = 0.848;
      Hdr = H/2 for double drainage). Check differential settlement / angular distortion limits (e.g. L/500).
- [ ] Manning: V = (k/n) R^(2/3) S^(1/2), k = 1.49 (US, ft/s) or 1.0 (SI, m/s); R = A/P. Solve normal depth with brentq.
- [ ] Rational: Q = C i A [cfs, in/hr, acres] or Q = C i A / 360 [m^3/s, mm/h, ha]; area-weighted C; Tc via Kirpich/NRCS.
- [ ] Pipe/channel: self-cleansing V >= 2 ft/s (0.6 m/s), max ~10 ft/s; check full vs partial flow; HGL below rim.
- [ ] Grading: IBC 1804.4 -- 5 % fall for 10 ft away from foundations (impervious: 2 %); ADA running slope <= 5 %, cross <= 2 %.
- [ ] Earthwork: average-end-area V = L (A1+A2)/2; 1 yd^3 = 27 ft^3; apply swell/shrink factors consistently (bank vs loose vs compacted).
- [ ] Scheduling (CPM): forward/backward pass, total float = LS - ES, critical path float = 0; respect FS/SS/FF lags and calendars.
- [ ] Cost/fee tables: unit cost x quantity, then subtotals, then markups in the stated order (overhead -> profit -> contingency
      -> fees/taxes); verify row sums and grand totals in code; never hard-code a total.

### Unit traps
- kPa vs kN/m^2 (same) vs psf (1 kPa = 20.885 psf) vs ksf; tsf (1 tsf = 95.76 kPa); unit weight kN/m^3 vs pcf (1 kN/m^3 = 6.366 pcf).
- log10 vs ln in consolidation; percent vs decimal slopes; ft vs m in Manning constant; acres vs ha (1 ha = 2.471 ac).
- Rainfall intensity per hour vs per duration; settlement mm vs in.
"""

_ELECTRICAL = """\
## Discipline playbook: ELECTRICAL (power distribution)
Governing references: NFPA 70 / NEC 2020-2023 (Art. 110, 210, 215, 220, 230, 240, 250, 310, 430, 450, 700),
IEEE 141 (Red), IEEE 242 (Buff, protection & coordination), IEEE 399/551 (short circuit), IEEE 1584 (arc flash),
IEC 60364 / IEC 60909 when the task is IEC-based.

### Checklist
- [ ] System: voltage, phases, wires, frequency, PF; 3-phase kVA = sqrt(3) V_LL I / 1000; 1-phase kVA = V I / 1000.
- [ ] Load schedule: connected vs demand load (NEC 220 demand factors, e.g. 220.44 receptacles: first 10 kVA @100 %,
      remainder @50 %); continuous loads at 125 %; largest motor at 125 % (430.24); balance phases (report max imbalance %).
- [ ] Conductor ampacity: Table 310.16; termination temperature (110.14(C): <=100 A or 14-1 AWG -> 60 C column unless
      listed 75 C); ambient correction (310.15(B)); bundling adjustment (310.15(C)(1): 4-6 CCC 80 %, 7-9 70 %, 10-20 50 %).
      Required: I_design <= Ampacity_table x F_ambient x F_adjust.
- [ ] OCPD: >= 125 % continuous + 100 % non-continuous; next standard size (240.6(A)); conductor protected per 240.4
      (small-conductor rule 240.4(D): 14 AWG 15 A, 12 AWG 20 A, 10 AWG 30 A Cu).
- [ ] Motors: use Table 430.250/430.248 FLC (not nameplate) for conductors (125 %) and 430.52 OCPD maxima.
- [ ] Voltage drop: 3-ph VD = sqrt(3) I L (R cos(th) + X sin(th)) (per-length R, X); simplified VD = sqrt(3) K I L / CM
      (K = 12.9 Cu, 21.2 Al ohm-cmil/ft); 1-ph uses factor 2 instead of sqrt(3). NEC info notes: <= 3 % branch, <= 5 % total
      (unless task sets other limits). L is ONE-WAY length.
- [ ] Short circuit: transformer I_sc = kVA x 1000 / (sqrt(3) V_LL Z_pu) (infinite bus); point-to-point
      f = sqrt(3) L I_sc / (C n E_LL), M = 1/(1+f), I_sc,downstream = M I_sc; include motor contribution (~4-6 x FLA) if asked.
      Equipment interrupting rating (AIC/SCCR) >= available fault current (110.9, 110.10).
- [ ] Protection coordination: TCC selectivity (downstream clears first, no overlap of curves across the fault range);
      selective coordination mandatory for 700.32/701.32/708.54 systems; transformer protection per 450.3(B).
- [ ] Grounding/bonding: EGC Table 250.122 (size by OCPD, upsized proportionally when phase conductors are upsized for VD);
      GEC Table 250.66; bonding jumpers 250.102.
- [ ] Power factor correction: kvar = P (tan(acos PF1) - tan(acos PF2)).
- [ ] Energization/commissioning sequence: verify LOTO removal, insulation resistance (megger), torque, phase rotation,
      energize upstream -> downstream, one feeder at a time, verify voltages/phase sequence before loads; document hold points.

### Unit traps
- AWG/kcmil vs mm^2 (1 kcmil = 0.5067 mm^2); circular mils (1 AWG ladder); line-to-line vs line-to-neutral voltage;
  one-way vs round-trip length; ohm/1000 ft vs ohm/km; kVA vs kW (PF!); %Z on its own base; per-unit conversions.
"""

_MECHANICAL = """\
## Discipline playbook: MECHANICAL / MEP / HVAC / PIPING / TANKS / FIRE
Governing references: ASHRAE Fundamentals (psychrometrics, load calc RTS/CLTD), ASHRAE 62.1 (ventilation), ASHRAE 90.1,
Darcy-Weisbach + Colebrook/Swamee-Jain, Hazen-Williams, Hydraulic Institute (pumps, NPSH), AMCA (fans),
API 650 (new welded tanks), API 653 (inspection, repair, service life), NFPA 13 (sprinklers), IPC/UPC (plumbing slopes).

### Checklist
- [ ] Psychrometrics: W = 0.622 pw/(P - pw); h = 1.006 t + W (2501 + 1.86 t) [kJ/kg dry air, t in C];
      saturation pressure from ASHRAE Hyland-Wexler or Magnus; use site barometric pressure if altitude is given.
- [ ] Air-side loads (US, std air): Qs = 1.08 CFM dT [Btu/h]; Ql = 0.68 CFM d(grains/lb); Qt = 4.5 CFM dh [Btu/lb].
      SI: Qs = 1.2 V[L/s] dT [W]; Qt = 1.2 V dh [W, kJ/kg]. 1 ton = 12,000 Btu/h = 3.517 kW. Sensible heat ratio = Qs/Qt.
- [ ] Ventilation (62.1 VRP): Vbz = Rp Pz + Ra Az; Voz = Vbz/Ez; check outdoor-air minimums are met.
- [ ] Pipe/duct friction: Darcy-Weisbach hf = f (L/D) V^2/(2g); Re = rho V D / mu; laminar f = 64/Re (Re < 2300);
      turbulent Colebrook (solve) or Swamee-Jain f = 0.25 / [log10(eps/(3.7 D) + 5.74/Re^0.9)]^2; minor losses K V^2/(2g).
- [ ] Hazen-Williams (water, fire): p[psi/ft] = 4.52 Q^1.85 / (C^1.85 d^4.87) [gpm, in]; sprinkler Q = K sqrt(P).
- [ ] Pumps/fans: affinity Q ~ N D, H ~ N^2 D^2, P ~ N^3 D^5; hydraulic power = rho g Q H; shaft = hydraulic/eta;
      operating point = system curve n pump curve (solve numerically); NPSHa >= NPSHr + margin (HI 9.6.1).
- [ ] API 650 one-foot method: td = 4.9 D (H - 0.3) G / Sd + CA [mm; D, H in m; Sd MPa]
      (US: td = 2.6 D (H - 1) G / Sd + CA [in; ft; psi]); tt (hydrotest) without G and CA; observe minimum nominal thickness by D.
- [ ] API 653: shell t_min = 2.6 (H - 1) D G / (S E) [in], >= 0.1 in; remaining life = (t_actual - t_min)/corrosion_rate;
      corrosion rate = (t_previous - t_actual)/years; inspection interval rules per 653 Sec. 6 (internal: per RCA / rate,
      max 20 yr without RBI); bottom: MRT >= 0.10 in (0.05 in with release prevention barrier).
- [ ] 3-D MEP routing / clash detection: model elements as axis-aligned boxes inflated by required clearances; two boxes
      clash iff they overlap on ALL three axes; route with A*/Dijkstra on a voxel grid, penalise bends, honour gravity
      drain slopes (IPC 704.1: <=2.5 in pipe 1/4 in/ft, 3-6 in 1/8 in/ft, >=8 in 1/16 in/ft), and re-run the clash check
      on the final geometry. Report clash count = 0 explicitly.
- [ ] Fire tests / sprinklers: NFPA 13 density/area curves, remote-area hydraulic calc, hose allowance, supply curve
      check (demand point below supply curve with safety margin, commonly >= 10 %).

### Unit traps
- Btu/h vs W (1 W = 3.412 Btu/h); in.wc vs Pa (1 in.wc = 249.1 Pa); psi vs ft head (2.31 ft/psi for water, divide by SG);
  gpm vs L/s (1 gpm = 0.06309 L/s); CFM vs L/s (1 CFM = 0.4719 L/s); C vs F vs K in property lookups;
  grains vs lb (7000 gr/lb); gauge vs absolute pressure (NPSH uses absolute!); tank H in ft vs m in the 1-ft method.
"""

_GENERAL = """\
## Discipline playbook: GENERAL / MULTI-DISCIPLINE
### Checklist
- [ ] Decompose the task into sub-problems per discipline; apply each relevant section of the standards below.
- [ ] Build a single constraint table (id, expression, limit, sense, unit, source clause) before solving.
- [ ] Dimensional analysis on every equation: both sides must reduce to identical SI base units.
- [ ] Sanity bounds: order-of-magnitude check every result against typical engineering ranges.
- [ ] Interfaces between disciplines (e.g. structural loads from mechanical equipment, electrical loads from HVAC motors,
      drainage under foundations) must use the SAME numbers in every deliverable -- compute once, reuse everywhere.
- [ ] Spreadsheets: formulas are fine only if the task wants them; otherwise write computed values (graders read cached
      values -- openpyxl-written formulas have NO cached value until Excel recalculates).
- [ ] Reports/memos: purpose, basis of design, governing codes + editions, inputs, method, results table, compliance
      statement, assumptions, limitations, recommendations.

### Key cross-discipline references
Structural: ASCE 7, AISC 360, ACI 318 | Civil: Terzaghi/Meyerhof, Manning, Rational, CPM |
Electrical: NEC, IEEE 141/242/399/1584 | Mechanical: ASHRAE, Darcy-Weisbach, API 650/653, NFPA 13.

### Unit traps
- Mixed US/SI inputs in the same task; percent vs fraction; gauge vs absolute; per-unit bases; one-way vs round-trip.
"""

DISCIPLINE_PLAYBOOKS: dict[str, str] = {
    "structural": _STRUCTURAL + "\n" + _COMMON_SOLVER_PATTERNS,
    "civil": _CIVIL + "\n" + _COMMON_SOLVER_PATTERNS,
    "electrical": _ELECTRICAL + "\n" + _COMMON_SOLVER_PATTERNS,
    "mechanical": _MECHANICAL + "\n" + _COMMON_SOLVER_PATTERNS,
    "general": _GENERAL + "\n" + _COMMON_SOLVER_PATTERNS,
}


def build_system_prompt(discipline: str | None = None) -> str:
    """Master system prompt with the matching discipline playbook appended."""
    key = discipline if discipline in DISCIPLINE_PLAYBOOKS else "general"
    return SYSTEM_PROMPT_MASTER + "\n\n" + DISCIPLINE_PLAYBOOKS[key]


# --------------------------------------------------------------------------- #
# Small formatting helpers (private)
# --------------------------------------------------------------------------- #

def _safe_json(obj: Any, limit: int = _MAX_JSON_CHARS) -> str:
    """JSON-dump anything (falls back to str) and cap the length head+tail."""
    try:
        text = json.dumps(obj, indent=2, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        text = str(obj)
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return text[:head] + f"\n... [{len(text) - limit} chars truncated] ...\n" + text[-tail:]


def _format_file_tree(file_tree: Iterable[str] | None) -> str:
    entries = [str(p) for p in (file_tree or []) if str(p).strip()]
    if not entries:
        return "(empty or not yet listed -- run `find <workdir> -maxdepth 4 -type f | head -400` first)"
    shown = entries[:_MAX_TREE_ENTRIES]
    text = "\n".join(f"  {p}" for p in shown)
    if len(entries) > len(shown):
        text += f"\n  ... ({len(entries) - len(shown)} more entries omitted)"
    return text


def _tool_name_desc(tool: dict) -> tuple[str, str, Any]:
    """Accept MCP-style ({name, description, inputSchema}) or OpenAI-style tool dicts."""
    if not isinstance(tool, dict):
        return str(tool), "", None
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else None
    if fn is not None:
        return str(fn.get("name", "?")), str(fn.get("description", "") or ""), fn.get("parameters")
    name = tool.get("name") or tool.get("qualified_name") or "?"
    server = tool.get("server")
    if server and not str(name).startswith("mcp__"):
        name = f"mcp__{server}__{name}"
    schema = tool.get("inputSchema") or tool.get("input_schema") or tool.get("parameters")
    return str(name), str(tool.get("description", "") or ""), schema


def _format_mcp_tools(mcp_tools: list[dict] | None) -> str:
    tools = list(mcp_tools or [])
    if not tools:
        return "(no MCP tools discovered -- rely on files in the workdir)"
    lines = []
    for tool in tools[:_MAX_TOOL_ENTRIES]:
        name, desc, schema = _tool_name_desc(tool)
        desc = " ".join(desc.split())
        if len(desc) > 240:
            desc = desc[:237] + "..."
        args = ""
        if isinstance(schema, dict) and isinstance(schema.get("properties"), dict):
            req = set(schema.get("required") or [])
            args = ", ".join(f"{k}{'*' if k in req else ''}" for k in list(schema["properties"])[:12])
        lines.append(f"  - {name}({args}): {desc}" if args else f"  - {name}: {desc}")
    if len(tools) > _MAX_TOOL_ENTRIES:
        lines.append(f"  ... ({len(tools) - _MAX_TOOL_ENTRIES} more tools omitted)")
    return "\n".join(lines)


def _instruction_block(instruction: str) -> str:
    return "<task_instruction>\n" + (instruction or "").strip() + "\n</task_instruction>"


# --------------------------------------------------------------------------- #
# Phase 1 -- Recon & Deliverable Contract
# --------------------------------------------------------------------------- #

CONTRACT_SCHEMA_EXAMPLE: dict[str, Any] = {
    "discipline": "electrical",
    "task_type": "constrained_optimization | calculation | document | drawing_revision | mixed",
    "governing_standards": ["NEC 2023 Art. 310", "IEEE 141"],
    "inputs": [{"path": "/app/data/loads.csv", "purpose": "panel load list", "key_fields": ["load_kva"]}],
    "deliverables": [
        {
            "path": "/app/output/cable_schedule.csv",
            "format": "csv",
            "required_fields": ["feeder_id", "size_awg_kcmil", "ampacity_a", "vd_pct"],
            "units": {"ampacity_a": "A", "vd_pct": "%"},
            "notes": "one row per feeder, same order as input",
        }
    ],
    "constraints": [
        {
            "id": "C1",
            "description": "Feeder voltage drop <= 3 %",
            "expression": "vd_pct <= 3.0",
            "sense": "<=",
            "limit": 3.0,
            "units": "%",
            "source": "task text / NEC 215.2 IN2",
            "hard": True,
            "margin_pct": 3,
        }
    ],
    "objective": {
        "description": "minimise total copper cost",
        "sense": "min",
        "metric": "sum(length_ft * cost_per_ft)",
        "units": "USD",
        "naive_baseline": "largest conductor everywhere",
    },
    "mcp_tools_to_use": ["mcp__apps__get_cable_catalog"],
    "final_message_requirements": ["state chosen sizes", "show VD per feeder"],
    "assumptions": ["75 C terminations (task silent)"],
    "open_questions": [],
}


def build_recon_prompt(instruction: str, file_tree: list[str], mcp_tools: list[dict]) -> str:
    """Phase 1: inspect environment and emit a machine-readable Deliverable Contract."""
    return "\n".join([
        "# PHASE 1 -- RECONNAISSANCE & DELIVERABLE CONTRACT",
        "",
        _instruction_block(instruction),
        "",
        "## Workdir inventory (pre-listed by the harness)",
        _format_file_tree(file_tree),
        "",
        "## MCP tools available to you",
        _format_mcp_tools(mcp_tools),
        "",
        "## What to do now",
        "1. Read EVERY input file that could carry data, constraints, templates or catalogues (CSV/XLSX: print",
        "   shape, headers, first rows and sheet names; JSON/YAML: keys; PDF/DXF/IFC/MD: extract text/entities with",
        "   a short script). Look for README/spec/template/example-output files and any provided checker script.",
        "2. Call read-only MCP tools that expose task data (catalogues, project parameters, model queries).",
        "3. Re-read the instruction sentence by sentence. Every 'must', 'shall', 'not exceed', 'at least',",
        "   'within', 'no more than', 'limit', 'minimum', 'maximum', numeric threshold, or code reference is a",
        "   candidate HARD constraint. Implicit code requirements (e.g. NEC ampacity, AISC capacity) count too.",
        "4. Identify every deliverable: exact path, format, required columns/keys/sheets, units, ordering, and",
        "   number formatting. If the instruction names a file, that exact path is mandatory.",
        "5. Identify the objective (what is minimised or maximised) and what a naive feasible design would be.",
        "",
        "## Output",
        "After exploring, reply with a short summary followed by ONE fenced ```json block containing the",
        "Deliverable Contract with exactly these top-level keys (example values shown):",
        "```json",
        _safe_json(CONTRACT_SCHEMA_EXAMPLE),
        "```",
        "Rules for the contract: use absolute paths; list constraints atomically (one inequality each);",
        "`sense` is one of \"<=\", \">=\", \"==\", \"in\", \"exists\"; set `objective` to null when the task has no",
        "optimisation objective; put any ambiguity in `assumptions` with the conservative choice you will make.",
        "Do NOT write deliverables yet.",
    ])


# --------------------------------------------------------------------------- #
# Phase 2 -- Feasibility-first solver (G = 1)
# --------------------------------------------------------------------------- #

def build_feasibility_prompt(instruction: str, contract: dict) -> str:
    """Phase 2: produce a verified G=1 design with 2-5 % margins and checkpoint it."""
    contract = contract if isinstance(contract, dict) else {"raw_contract": contract}
    n_cons = len(contract.get("constraints") or [])
    deliverables = [d.get("path") if isinstance(d, dict) else str(d) for d in (contract.get("deliverables") or [])]
    deliverable_list = "\n".join(f"  - {p}" for p in deliverables if p) or "  - (derive from the instruction)"
    return "\n".join([
        "# PHASE 2 -- FEASIBILITY-FIRST SYNTHESIS (target G = 1)",
        "",
        _instruction_block(instruction),
        "",
        f"## Deliverable Contract ({n_cons} constraints)",
        "```json",
        _safe_json(contract),
        "```",
        "",
        "## Required deliverables",
        deliverable_list,
        "",
        "## What to do now",
        f"1. Write `{SOLVER_PATH}`: a standalone, deterministic Python script that",
        "   a. loads all inputs from disk (never re-type numbers that exist in files);",
        "   b. encodes the governing equations and EVERY contract constraint as a function returning",
        "      (value, limit, sense) in consistent units;",
        "   c. searches for a FEASIBLE design first -- enumerate catalogue options or solve with margin",
        "      m = 0.03 (use 0.02-0.05; equality constraints get no margin); if nothing is feasible with",
        "      margin, relax toward m = 0 before declaring infeasibility;",
        "   d. prefers a simple, robust, clearly-feasible design over a clever one at this stage;",
        "   e. writes every deliverable in the exact contract format;",
        "   f. RE-READS the written files, recomputes all constraints from the parsed values, and prints",
        f"      the single final line `{METRICS_MARKER} {{...}}` (keys: p, G, objective, sense, constraints",
        "      [id, value, limit, ok, margin_pct], deliverables).",
        "2. Run it. If any constraint fails, diagnose (units? wrong combination? wrong table column?), fix,",
        "   and rerun. Do not proceed while G = 0 unless the problem is provably infeasible -- in that",
        "   case maximise p (number of satisfied constraints) and document why.",
        "3. Once the metrics line shows G = 1, stop and report: the harness will checkpoint these files.",
        "   Keep the solver script; Phase 3 will reuse its constraint functions.",
        "",
        "## Pitfalls that silently zero the gate",
        "- deliverable at a slightly different path/filename/extension, or wrong sheet/column name;",
        "- values written in different units/precision than requested, or as text with unit suffixes;",
        "- NaN/Inf/empty cells; index column written by pandas; trailing summary rows inside data tables;",
        "- rounding a value across a limit after the check; checking factored vs service loads wrongly;",
        "- one-way vs round-trip length, LL vs LN voltage, gauge vs absolute pressure, log10 vs ln.",
    ])


# --------------------------------------------------------------------------- #
# Phase 3 -- Interior-point objective optimisation (maximise O')
# --------------------------------------------------------------------------- #

def build_optimization_prompt(instruction: str, baseline_metrics: dict) -> str:
    """Phase 3: improve the objective while strictly preserving G=1."""
    baseline_metrics = baseline_metrics if isinstance(baseline_metrics, dict) else {"raw": baseline_metrics}
    sense = str(baseline_metrics.get("sense") or "min").lower()
    obj = baseline_metrics.get("objective")
    obj_txt = f"{obj}" if obj is not None else "unknown (define it from the contract)"
    return "\n".join([
        "# PHASE 3 -- CONSTRAINED OBJECTIVE OPTIMISATION (maximise O' while keeping G = 1)",
        "",
        _instruction_block(instruction),
        "",
        "## Certified checkpoint (G = 1) -- baseline metrics",
        "```json",
        _safe_json(baseline_metrics),
        "```",
        f"Baseline objective: {obj_txt} (sense: {sense}). This design is saved; you may only replace it",
        "with a design that re-verifies G = 1 from disk.",
        "",
        "## What to do now",
        f"1. Write `{OPTIMIZER_PATH}` reusing the constraint functions of `{SOLVER_PATH}` (import or copy).",
        "2. Search the feasible interior for a better objective:",
        "   - discrete choices: exhaustive or branch-and-bound enumeration over the catalogue;",
        "   - continuous: SLSQP/trust-constr with multi-start, constraints as (1 - m) * limit - g(x) >= 0;",
        "   - mixed: enumerate the discrete part, optimise the continuous part inside each branch.",
        "3. Margin annealing: start at m = 0.02, then try 0.01 and 0.005 ONLY if the objective gain is",
        "   material. Never go below 0.005 on inequality constraints evaluated with tabulated/rounded data.",
        "   Expected-score rule: a tighter margin is worth it only if the O' gain clearly outweighs the risk",
        "   of losing 0.6 * O' through a grader-side rounding or interpretation difference.",
        "4. Accept a candidate only if: (a) every constraint passes after rounding to the output precision,",
        f"   (b) the objective strictly improves, (c) files re-read from disk reproduce both. Print `{METRICS_MARKER}`",
        "   for the accepted design.",
        "5. Write candidate files to a /tmp staging directory first; copy over the deliverables only after",
        "   verification passes. If anything fails, leave the checkpointed deliverables untouched.",
        "6. Stop when improvements fall below ~0.5 % of the objective, the design is at a proven optimum",
        "   (exhaustive search), or ~60 % of the remaining time budget is used -- reserve time for audit/report.",
    ])


# --------------------------------------------------------------------------- #
# Phase 4 -- Audit repair loop
# --------------------------------------------------------------------------- #

def build_audit_repair_prompt(instruction: str, audit_report: dict) -> str:
    """Phase 4 (optional loop): fix only what the deterministic auditor flagged."""
    return "\n".join([
        "# PHASE 4 -- DELIVERABLE AUDIT FAILED: TARGETED REPAIR",
        "",
        _instruction_block(instruction),
        "",
        "## Auditor findings",
        "```json",
        _safe_json(audit_report),
        "```",
        "",
        "## What to do now",
        "1. Fix ONLY the flagged problems (missing/empty file, parse error, missing column/key, NaN/Inf,",
        "   failed constraint). Do not re-architect a design that already passes.",
        "2. If a fix would break a constraint, restore the last checkpointed file instead and note it.",
        f"3. Re-run `{SOLVER_PATH}` (or `{OPTIMIZER_PATH}`) verification so the final `{METRICS_MARKER}` line",
        "   reflects the files on disk, then reply with a one-paragraph summary of the repairs.",
    ])


# --------------------------------------------------------------------------- #
# Phase 5 -- Rubric-aligned final report
# --------------------------------------------------------------------------- #

_REPORT_TEMPLATE = """\
# <Project / task title> -- Engineering Design Report

## 1. Executive Summary
<3-6 sentences: what was designed/checked, the governing outcome, compliance status (all N constraints
satisfied), and the key objective result (e.g. "total cost reduced 18.4 % vs. baseline").>

## 2. Basis of Design & Governing Standards
<code editions and clauses actually applied; design method (LRFD/ASD, NEC edition, ASHRAE method); key inputs with
units and their source file / MCP tool.>

## 3. Deliverables Manifest
| File | Format | Contents | Verified |
|---|---|---|---|
| /app/... | CSV | ... | parsed OK, no NaN, required columns present |

## 4. Method
<numbered steps: equations used (written out), solver approach, catalogues searched, how constraints were checked.>

## 5. Constraint Compliance Matrix
| ID | Requirement (source) | Design value | Limit | Utilisation / margin | Status |
|---|---|---|---|---|---|
| C1 | ... | ... | ... | ... | PASS |

## 6. Optimisation Results
<baseline -> optimised objective, % improvement, why further improvement is limited (which constraint governs).>

## 7. Assumptions, Limitations & Recommendations
<explicit assumptions where the task was silent, conservative choices made, items for field verification / next steps.>
"""

FINAL_MESSAGE_TEMPLATE = _REPORT_TEMPLATE


def build_final_report_prompt(instruction: str, audit_report: dict, optimization_log: list) -> str:
    """Phase 5: produce the `final_message` markdown that the rubric judge reads."""
    log = list(optimization_log or [])
    if len(log) > _MAX_LOG_ENTRIES:
        log = log[:5] + [{"note": f"... {len(log) - _MAX_LOG_ENTRIES} entries omitted ..."}] + log[-(_MAX_LOG_ENTRIES - 6):]
    return "\n".join([
        "# PHASE 5 -- FINAL ENGINEERING REPORT (this text becomes `final_message`)",
        "",
        _instruction_block(instruction),
        "",
        "## Verified audit report (ground truth -- only these numbers may be reported as verified)",
        "```json",
        _safe_json(audit_report),
        "```",
        "",
        "## Optimisation log (baseline -> accepted candidates)",
        "```json",
        _safe_json(log),
        "```",
        "",
        "## Instructions",
        "Write the final report in GitHub markdown following the template below. Requirements:",
        "- Every number must come from the audit report, the optimisation log, or files you verified;",
        "  never invent values. If something was not verified, say so plainly.",
        "- Answer every explicit question in the task instruction directly (quote the asked quantity and give",
        "  the value with units), and include anything the task says the final message must contain.",
        "- Name governing standards with edition and clause; show the key equations once.",
        "- Be honest about any constraint that is not satisfied or any deliverable that could not be produced;",
        "  a truthful limitation scores better with the rubric judge than an unsupported claim.",
        "- Concise and skimmable: tables over prose, roughly 400-1200 words. No tool-call chatter.",
        "- Output ONLY the report markdown (no preamble, no code fences around the whole report).",
        "",
        "## Template",
        _REPORT_TEMPLATE,
    ])


# --------------------------------------------------------------------------- #
# Deterministic discipline classifier
# --------------------------------------------------------------------------- #

# (pattern, weight). Patterns are matched case-insensitively with word boundaries where sensible.
_KEYWORDS: dict[str, list[tuple[str, float]]] = {
    "structural": [
        (r"\bstructural\b", 3), (r"\basce\s*7\b", 3), (r"\baisc\b", 3), (r"\baci\s*318\b", 3), (r"\bibc\b", 1),
        (r"\bbeams?\b", 2), (r"\bgirders?\b", 2), (r"\bcolumns?\b", 1), (r"\btruss(es)?\b", 2), (r"\bjoists?\b", 2),
        (r"\bmoment\b", 1.5), (r"\bshear\b", 1), (r"\bdeflection\b", 2), (r"\bbuckling\b", 2),
        (r"\bload combinations?\b", 3), (r"\blrfd\b", 2.5), (r"\basd\b", 1.5), (r"\bseismic\b", 2),
        (r"\bwind loads?\b", 2), (r"\bsnow loads?\b", 2), (r"\bdead loads?\b", 1.5), (r"\blive loads?\b", 1.5),
        (r"\bdrift\b", 1), (r"\brebar\b", 2), (r"\breinforc", 1.5), (r"\bconnections?\b", 1), (r"\bwelds?\b", 1.5),
        (r"\bbolts?\b", 1), (r"\bsteel sections?\b", 2), (r"\bw\d{1,2}x\d{1,3}\b", 3), (r"\bslabs?\b", 1.5),
        (r"\bsection modulus\b", 2.5), (r"\bmoment of inertia\b", 1.5), (r"\bdemand.capacity\b", 1.5),
        (r"\bnds\b", 1.5), (r"\btimber\b", 1.5), (r"\bframe\b", 1),
    ],
    "civil": [
        (r"\bcivil\b", 3), (r"\bgeotech", 3), (r"\bsettlement\b", 3), (r"\bbearing capacity\b", 3.5),
        (r"\bterzaghi\b", 3.5), (r"\bmeyerhof\b", 3.5), (r"\bconsolidation\b", 3), (r"\bfootings?\b", 2),
        (r"\bfoundations?\b", 2), (r"\bpiles?\b", 2), (r"\bsoil\b", 2), (r"\bclay\b", 1.5), (r"\bsand\b", 1),
        (r"\bborings?\b", 2), (r"\bspt\b", 2), (r"\bgrading\b", 2.5), (r"\bdrainage\b", 2.5), (r"\bstorm", 2),
        (r"\bmanning\b", 3.5), (r"\brational method\b", 3.5), (r"\brunoff\b", 3), (r"\bculverts?\b", 3),
        (r"\bswales?\b", 2.5), (r"\bcatch basins?\b", 2.5), (r"\bdetention\b", 2), (r"\bhydrolog", 2.5),
        (r"\bearthwork\b", 3), (r"\bcut and fill\b", 3), (r"\bexcavation\b", 2), (r"\bretaining wall\b", 2.5),
        (r"\bpavement\b", 2.5), (r"\broadway\b", 2), (r"\bconstruction sequenc", 3), (r"\bsequencing\b", 1.5),
        (r"\bcritical path\b", 2.5), (r"\bcpm\b", 2), (r"\bgantt\b", 2), (r"\bschedule of values\b", 2),
        (r"\bcost estimate\b", 2), (r"\bfee (table|schedule)s?\b", 2.5), (r"\bbid\b", 1), (r"\bsite plan\b", 1.5),
        (r"\bgroundwater\b", 2), (r"\bwater table\b", 2),
    ],
    "electrical": [
        (r"\belectrical\b", 3), (r"\bnec\b", 3), (r"\bnfpa\s*70\b", 3.5), (r"\bieee\b", 2), (r"\bvoltage drop\b", 3.5),
        (r"\bshort[- ]circuit\b", 3.5), (r"\bfault current\b", 3.5), (r"\bampacity\b", 3.5), (r"\bderat", 2),
        (r"\bbreakers?\b", 2.5), (r"\bpanel ?boards?\b", 3), (r"\bpanels?\b", 1), (r"\bswitchgear\b", 3),
        (r"\btransformers?\b", 2.5), (r"\bkva\b", 2.5), (r"\bkw\b", 1), (r"\bvolts?\b", 1.5), (r"\bamps?\b", 1.5),
        (r"\bamperes?\b", 1.5), (r"\bfeeders?\b", 2.5), (r"\bbranch circuits?\b", 3), (r"\bconductors?\b", 2),
        (r"\bawg\b", 3), (r"\bkcmil\b", 3), (r"\bconduit\b", 2), (r"\bcables?\b", 1.5), (r"\bload schedule\b", 3),
        (r"\bprotection coordination\b", 3.5), (r"\bselective coordination\b", 3.5), (r"\btcc\b", 2.5),
        (r"\brelays?\b", 2), (r"\bfuses?\b", 2), (r"\barc flash\b", 3), (r"\bgrounding\b", 2), (r"\benergiz", 3),
        (r"\bpower factor\b", 2.5), (r"\bgenerators?\b", 2), (r"\bups\b", 1.5), (r"\bphotovoltaic\b", 2.5),
        (r"\bsingle[- ]line\b", 3), (r"\bone[- ]line\b", 3), (r"\bmotor control\b", 2.5), (r"\bmcc\b", 2.5),
    ],
    "mechanical": [
        (r"\bmechanical\b", 3), (r"\bhvac\b", 3.5), (r"\bpsychrometric", 3.5), (r"\bcooling loads?\b", 3.5),
        (r"\bheating loads?\b", 3), (r"\bashrae\b", 3), (r"\bair handl", 3), (r"\bahu\b", 3), (r"\bchillers?\b", 3),
        (r"\bboilers?\b", 2.5), (r"\bducts?\b", 2.5), (r"\bductwork\b", 3), (r"\bcfm\b", 3), (r"\bdampers?\b", 2),
        (r"\bpumps?\b", 2.5), (r"\bfans?\b", 1.5), (r"\baffinity laws?\b", 3.5), (r"\bpressure drop\b", 3),
        (r"\bhead loss\b", 3), (r"\bdarcy\b", 3.5), (r"\bweisbach\b", 3.5), (r"\bcolebrook\b", 3.5),
        (r"\bhazen[- ]williams\b", 3), (r"\breynolds\b", 2), (r"\bnpsh\b", 3.5), (r"\bpiping\b", 2.5),
        (r"\bpipes?\b", 1.5), (r"\bmep\b", 3), (r"\bclash", 3), (r"\brouting\b", 1.5), (r"\bapi\s*65[03]\b", 4),
        (r"\bstorage tanks?\b", 3), (r"\btanks?\b", 1.5), (r"\bcorrosion\b", 2.5), (r"\bshell thickness\b", 3),
        (r"\bremaining life\b", 3), (r"\bsprinklers?\b", 3), (r"\bnfpa\s*13\b", 3.5), (r"\bfire test", 3),
        (r"\bfire protection\b", 2.5), (r"\brefrigerant\b", 2.5), (r"\bheat exchangers?\b", 3),
        (r"\bthermal\b", 1), (r"\benthalpy\b", 2.5), (r"\bhumidity\b", 2.5), (r"\bwet[- ]bulb\b", 3),
        (r"\bdew ?point\b", 2.5), (r"\bbtu", 2), (r"\btons? of refrigeration\b", 3), (r"\bplumbing\b", 2.5),
        (r"\bventilation\b", 2.5), (r"\bvav\b", 3),
    ],
}

_EXTENSION_HINTS: dict[str, dict[str, float]] = {
    ".ifc": {"mechanical": 1.0, "structural": 1.0},
    ".rvt": {"mechanical": 0.5, "structural": 0.5},
    ".dxf": {"civil": 0.5, "structural": 0.5},
    ".dwg": {"civil": 0.5, "structural": 0.5},
    ".s2k": {"structural": 2.0},
    ".e2k": {"structural": 2.0},
    ".std": {"structural": 1.5},
    ".hec": {"civil": 1.5},
    ".swmm": {"civil": 2.0},
    ".inp": {"civil": 0.5, "mechanical": 0.5},
    ".tcc": {"electrical": 2.0},
    ".etap": {"electrical": 2.0},
    ".glm": {"electrical": 2.0},
    ".idf": {"mechanical": 2.0},
    ".epw": {"mechanical": 1.5},
}

_PATH_TOKEN_HINTS: dict[str, dict[str, float]] = {
    "structural": {"structural": 1.0}, "beam": {"structural": 0.5}, "column": {"structural": 0.3},
    "seismic": {"structural": 0.5}, "geotech": {"civil": 1.0}, "boring": {"civil": 0.5}, "soil": {"civil": 0.5},
    "drainage": {"civil": 0.5}, "grading": {"civil": 0.5}, "schedule": {"civil": 0.2}, "cost": {"civil": 0.2},
    "panel": {"electrical": 0.5}, "feeder": {"electrical": 0.5}, "cable": {"electrical": 0.5},
    "breaker": {"electrical": 0.5}, "one_line": {"electrical": 0.8}, "single_line": {"electrical": 0.8},
    "hvac": {"mechanical": 1.0}, "duct": {"mechanical": 0.5}, "pipe": {"mechanical": 0.4},
    "pump": {"mechanical": 0.5}, "tank": {"mechanical": 0.5}, "psychro": {"mechanical": 0.8},
    "sprinkler": {"mechanical": 0.8}, "mep": {"mechanical": 0.8},
}

_COMPILED_KEYWORDS: dict[str, list[tuple[re.Pattern[str], float]]] = {
    disc: [(re.compile(pat, re.IGNORECASE), w) for pat, w in pats] for disc, pats in _KEYWORDS.items()
}

# Deterministic tie-break order (by PLE task frequency: ME 30, EE 27, CE 18, SE 10).
_TIE_ORDER = ("mechanical", "electrical", "civil", "structural")
_MIN_CONFIDENT_SCORE = 3.0


def classify_discipline_scores(instruction: str, file_tree: list[str] | None = None) -> dict[str, float]:
    """Return raw per-discipline evidence scores (for logging / debugging)."""
    text = instruction or ""
    scores = {d: 0.0 for d in _TIE_ORDER}
    for disc, pats in _COMPILED_KEYWORDS.items():
        for rx, w in pats:
            hits = len(rx.findall(text))
            if hits:
                # Diminishing returns: first hit full weight, repeats add log-scaled evidence.
                scores[disc] += w * (1.0 + math.log(hits))
    for raw in file_tree or []:
        path = str(raw).lower()
        dot = path.rfind(".")
        if dot != -1:
            for disc, w in _EXTENSION_HINTS.get(path[dot:], {}).items():
                scores[disc] += w
        for token, hint in _PATH_TOKEN_HINTS.items():
            if token in path:
                for disc, w in hint.items():
                    scores[disc] += w
    return {d: round(s, 4) for d, s in scores.items()}


def classify_discipline(instruction: str, file_tree: list[str] | None = None) -> str:
    """Deterministic keyword + file-extension classifier.

    Returns one of ``"structural" | "civil" | "electrical" | "mechanical" | "general"``.
    ``"general"`` is returned when evidence is weak, or when two disciplines are both strong and
    nearly tied (genuinely multi-discipline task).
    """
    scores = classify_discipline_scores(instruction, file_tree)
    ranked = sorted(_TIE_ORDER, key=lambda d: (-scores[d], _TIE_ORDER.index(d)))
    top, runner = ranked[0], ranked[1]
    if scores[top] < _MIN_CONFIDENT_SCORE:
        return "general"
    if scores[runner] >= _MIN_CONFIDENT_SCORE * 2 and scores[runner] >= 0.85 * scores[top]:
        return "general"
    return top


# --------------------------------------------------------------------------- #
# Parsing helpers for LLM output and solver stdout
# --------------------------------------------------------------------------- #

_JSON_FENCE_RX = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)


def extract_json_block(text: str) -> dict | None:
    """Return the last fenced JSON object in ``text`` (falls back to the outermost {...})."""
    if not text:
        return None
    for candidate in reversed(_JSON_FENCE_RX.findall(text)):
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            return obj if isinstance(obj, dict) else None
        except json.JSONDecodeError:
            return None
    return None


def parse_metrics_line(stdout: str) -> dict | None:
    """Parse the LAST ``APEX_METRICS: {...}`` line from solver stdout.

    Normalises ``G`` to int and derives it from ``p``/constraint flags when missing.
    """
    if not stdout:
        return None
    for line in reversed(stdout.splitlines()):
        idx = line.find(METRICS_MARKER)
        if idx == -1:
            continue
        payload = line[idx + len(METRICS_MARKER):].strip()
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        cons = data.get("constraints") or []
        if "p" not in data and cons:
            oks = [bool(c.get("ok")) for c in cons if isinstance(c, dict)]
            data["p"] = (sum(oks) / len(oks)) if oks else 0.0
        try:
            p = float(data.get("p", 0.0))
        except (TypeError, ValueError):
            p = 0.0
        data["p"] = p
        gate = data.get("G")
        data["G"] = 1 if (gate in (1, True, "1") and p >= 1.0 - 1e-12) else (1 if gate is None and p >= 1.0 - 1e-12 else 0)
        return data
    return None


if __name__ == "__main__":  # pragma: no cover - quick self-check
    demo = "Size feeders per NEC 310.16 so voltage drop stays below 3% and minimise copper cost (kcmil)."
    print(classify_discipline(demo), classify_discipline_scores(demo))
    print(len(build_system_prompt("electrical")), "chars system prompt")
