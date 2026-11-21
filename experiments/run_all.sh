#!/usr/bin/env bash
# run_all.sh — reproduce the complete experimental programme from raw data to figures.
# Stages are strictly ordered because later stages read the derived data of earlier ones, and
# the heavy CPU stages are serialised so that no two of them contend for memory.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
run () { echo "=== $* ==="; "$@" 2>&1 | tail -n "$((TAIL:-40))" | tee -a "logs/${*// /_}.log"; }

# 1. raw 3D Tiles -> index, exact mesh census, fidelity ladder, measured geometric audit
run python src/01_index.py
run python src/02_census.py --workers 6
run python src/03_cut_ladder.py
run python src/04_fidelity_audit.py
# 2. centrelines -> roadway graph -> per-fidelity parametric section field
run python src/05_network.py
# 3. solver verification (analytic, conservation, homogeneity, subdivision, energy, loop)
run python src/06_vent_verify.py
# 4. propagation of geometric fidelity into the ventilation solution (the core experiment)
run python src/08_fidelity_propagation.py
# 5. methods: LOD-ROM reconstruction, cross-LOD operators, semantic physics, active refinement
run python src/09_lod_rom.py
run env EPOCHS=260 N_SCEN=500 SEEDS=0,1,2 TRAIN_LEVELS=14,16 python src/12_di_operator.py
run env EPOCHS=260 python src/11_di_no.py
run python src/07_semantic.py
run env N_SEED=8 BUDGET=40 BASE_LEVELS=14,16,18 python src/14_eal.py
run python src/18_effort_rescore.py
# 6a. revision experiments (each answers a specific referee item)
run python src/22_amplification.py          # signed deltas, empirical elasticity, sign-randomisation control
run python src/23_alpha_field.py            # spatially structured alpha fields, error vs width norm
run python src/24_penalty_baseline.py       # tuned soft-constraint opponent for the conservation claim
run python src/25_crossdiscretisation.py    # true cross-grid transfer at fixed weights
run python src/26_filling.py                # section-field coverage and filling-rule sensitivity
run python src/27_holm.py                   # paired t-tests with Holm correction, per-family best policy
run python src/28_bytes.py                  # one byte accounting: archive / stop / cumulative / sum-of-cuts
run python src/30_floor_ci.py               # floor propagated, paired vs unpaired, rank monotonicity
run python src/31_provenance.py             # hardware, libraries, stage times, artefact inventory
# 6. synthesis: composite index, weight-free Pareto evaluation, repetition study
run python src/15_index.py
run python src/19_pareto.py
run python src/16_repetition.py
# 7. stage timing
run python src/20_timing.py
echo "ALL STAGES COMPLETE"
