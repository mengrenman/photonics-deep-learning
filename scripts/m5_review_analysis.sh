#!/bin/bash
# Analysis commands of the milestone-5 review follow-up (docs/DESIGN.md 12.6), as run after the batch of scripts/make_m5_review_jobs.py had finished.
# Environment: conda activate photonics-dl; run from the repository root. Single processes; the finalize steps need the chains of the new runs.
# Everything new goes to results/prior_comparison/m5/review/; the m5/ files that the review corrected (edge analysis, convergence audit with the
# location of the slow pixels, prior energies, cross-run pools with the continuation flag, independent features) are regenerated in place.
set -e
D=results/prior_comparison
M5=$D/m5
R=$M5/review
mkdir -p $R
PROBLEM="--n 48 --side 2 --eps-r 2 --snr 30 --seed 0 --chains 4"
tv() { echo "--prior tv --lam $1 --delta 0.01 --sigma-0 1 --map-maxiter 4000 --adapt"; }
SEGS=${SEGS:-1}   # recorded segments of 100 draws per chain of the new lambda runs actually written: the stop file scripts/jobs/stop/m5v was created at 04:01 (58 min
                  # into the 90-minute budget; the machine ran at load 11), so the loops ended after segment 0 (50 warmup + 100 recorded draws per chain)

# ---- per-run summaries of the new runs (stage finalize) ----
python scripts/run_prior_comparison.py $PROBLEM $(tv 1.25) --out pc_tv_lam1.25_adapt --start-theta $D/pc_tv_lam1.25_adapt/starts.npy --stage finalize --max-segments $SEGS
python scripts/run_prior_comparison.py $PROBLEM $(tv 8.5) --out pc_tv_lam8.5_adapt --start-theta $D/pc_tv_lam8.5_adapt/starts.npy --stage finalize --max-segments $SEGS
python scripts/run_prior_comparison.py ${PROBLEM/--chains 4/--chains 2} $(tv 3.75) --out pc_tv_lam3.75_indep --start-theta $D/pc_tv_lam3.75_indep/starts.npy --stage finalize

OLD="pc_matern pc_qggmrf_sg0.05 pc_qggmrf_sg0.1 pc_gmrf pc_gmrf_adapt pc_gmrf_adapt_burn2 pc_gmrf_adapt2 pc_tv_lam3.75_adapt pc_tv_lam3.75_adapt2 pc_tv_lam11.25_adapt pc_tv_lam11.25_adapt2"
NEW="pc_tv_lam1.25_adapt pc_tv_lam8.5_adapt pc_tv_lam3.75_indep"
ALL="$OLD $NEW"

# ---- corrected / extended versions of the milestone-5 analyses ----
python scripts/edge_analysis.py $ALL --out $M5
python scripts/prior_energy_audit.py $ALL --out $M5/prior_energy.json
python scripts/convergence_audit.py $ALL --null-reps 60 --out $M5/convergence_audit.json
python scripts/cross_run_check.py pc_tv_lam3.75_adapt pc_tv_lam3.75_adapt2 --out $M5/cross_run_tv_lam3.75.json
python scripts/cross_run_check.py pc_tv_lam11.25_adapt pc_tv_lam11.25_adapt2 --out $M5/cross_run_tv_lam11.25.json
python scripts/cross_run_check.py pc_gmrf_adapt pc_gmrf_adapt2 --last 200 --out $M5/cross_run_gmrf.json
python scripts/check_features_independent.py $OLD $NEW --out $M5/independent_features.json

# ---- new analyses ----
python scripts/resolution_analysis.py pc_matern pc_qggmrf_sg0.05 pc_qggmrf_sg0.1 pc_gmrf pc_gmrf_adapt pc_tv_lam3.75_adapt pc_tv_lam11.25_adapt $NEW --out $R/resolution_analysis.json
python scripts/feature_scales.py pc_matern pc_qggmrf_sg0.05 pc_qggmrf_sg0.1 pc_gmrf pc_gmrf_adapt pc_gmrf_adapt2 pc_tv_lam3.75_adapt pc_tv_lam11.25_adapt $NEW \
    --radii 0 0.04 0.06 0.08 0.10 0.12 0.16 --out $R/feature_scales.json
# the Laplace approximation at the GMRF MAP (exact Gaussian draws, no MCMC; imports JAX, one XLA thread)
python scripts/feature_scales.py pc_gmrf_adapt --radii 0 0.04 0.06 0.08 0.10 0.12 0.16 --laplace pc_gmrf_adapt --laplace-draws 4000 --out $R/feature_scales_laplace.json
python scripts/equal_length_audit.py --out $R/equal_length_audit.json
python scripts/start_drift_audit.py pc_gmrf_adapt pc_tv_lam3.75_adapt pc_tv_lam11.25_adapt --block 100 --burn 50 --out $R/start_drift.json
python scripts/start_drift_audit.py $NEW --block 25 --burn 50 --out $R/start_drift_new_runs.json
python scripts/lambda_band.py --runs pc_tv_lam1.25_adapt pc_tv_lam3.75_adapt pc_tv_lam3.75_indep pc_tv_lam8.5_adapt pc_tv_lam11.25_adapt --out $R/lambda_band.json --figure $R/lambda_band.png
python scripts/cross_run_check.py pc_tv_lam3.75_adapt pc_tv_lam3.75_indep --last 200 --out $R/cross_run_tv_lam3.75_indep.json

# ---- tables (every number is read from the JSON files above); the verdict file m5/comparison.json of m5_analysis.sh is unchanged ----
MAIN="pc_matern pc_qggmrf_sg0.05 pc_qggmrf_sg0.1 pc_gmrf_adapt pc_tv_lam3.75_adapt pc_tv_lam11.25_adapt"
python scripts/prior_comparison_tables.py $MAIN $NEW --audit-dir $M5 > $M5/tables_main.md
python scripts/milestone5_tables.py --audit-dir $M5 --runs pc_gmrf_adapt pc_gmrf_adapt2 pc_tv_lam3.75_adapt pc_tv_lam3.75_adapt2 \
    pc_tv_lam11.25_adapt pc_tv_lam11.25_adapt2 $NEW --compare pc_matern pc_qggmrf_sg0.05 pc_qggmrf_sg0.1 pc_gmrf > $M5/tables_m5.md
