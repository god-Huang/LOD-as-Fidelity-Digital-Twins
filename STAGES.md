# Stages

26 stages run in this order under `experiments/run_all.sh`.
Each writes its own tables into `experiments/results/`; the columns are
documented in the module docstring of the stage that produces them.

- **01** `01_index.py` — Build the complete LOD index of the 3D Tiles dataset
- **02** `02_census.py` — Full-dataset mesh census
- **03** `03_cut_ladder.py` — LOD-native multi-fidelity ladder via 3D Tiles level cuts
- **04** `04_fidelity_audit.py` — Measured geometric error of every LOD level, per block
- **05** `05_network.py` — Roadway graph from the delivered centrelines, scan-validated junction completion, station sampling, and per-fidelity cross-section extraction
- **06** `06_vent_verify.py` — Verification and validation of the ventilation network solver
- **08** `08_fidelity_propagation.py` — How scan LOD fidelity propagates into the ventilation solution
- **09** `09_lod_rom.py` — Innovation 1 in its calibrated form: the fidelity-anchored reduced-order reconstruction of the cross-section field, evaluated against native LOD…
- **12** `12_di_operator.py` — Conservation-preserving, discretisation-invariant operator that transfers the *physics* across LOD fidelity levels
  - run configuration: `EPOCHS=260 N_SCEN=500 SEEDS=0,1,2 TRAIN_LEVELS=14,16`
- **11** `11_di_no.py` — Fidelity-anchored, discretisation-invariant operator on the PARAMETRIC SECTION field: cross-LOD zero-shot transfer of the cross-section geometry…
  - run configuration: `EPOCHS=260`
- **07** `07_semantic.py` — Texture-and-geometry semantics for a scan product that carries no labels
- **14** `14_eal.py` — Error-aware active refinement: which roadway branches to re-survey next
  - run configuration: `N_SEED=8 BUDGET=40 BASE_LEVELS=14,16,18`
- **18** `18_effort_rescore.py` — Re-score the stored refinement trajectories against a spatially non-uniform survey effort, without re-running any network solve
- **22** `22_amplification.py` — Signed geometry-to-resistance transfer and the empirical amplification exponent
- **23** `23_alpha_field.py` — Spatially structured friction uncertainty
- **24** `24_penalty_baseline.py` — A tuned soft-constraint baseline for the conservation claim
- **25** `25_crossdiscretisation.py` — A genuine cross-discretisation test
- **26** `26_filling.py` — How much of the section field is interpolated, and what the filling rule is worth
- **27** `27_holm.py` — Statistical strength of the acquisition-policy comparison
- **28** `28_bytes.py` — see the module docstring
- **30** `30_floor_ci.py` — Does the estimator floor's own uncertainty propagate into the excess?
- **31** `31_provenance.py` — Machine, environment and measured wall-clock cost of the pipeline
- **15** `15_index.py` — DT-Q: a computable, calibrated composite index for a multi-fidelity digital twin, together with the trust dimension it needs
- **19** `19_pareto.py` — Weight-free evaluation of the fidelity ladder and of the learned operators, over (streamed cost, physical quality)
- **16** `16_repetition.py` — Reproduction study: how many repetitions, how stable, does the ordering hold
- **20** `20_timing.py` — Measured compute cost of the three ways to answer a ventilation question: native solve at a fidelity level, fidelity-codec decode plus solve, and the…
