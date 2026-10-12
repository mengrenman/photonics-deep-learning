#!/bin/bash
# Commands behind results/prior_comparison/m5/ and the milestone-5 report (docs/DESIGN.md 12.5), as run, after the chains had finished.
# Environment: conda activate photonics-dl; run from the repository root. Every step is a single process with at most 2 BLAS threads and
# takes seconds to a few minutes; the finalize steps (physical Phi of every stored draw) take 1 to 7 minutes each.
#
# Chains (scripts/batch_run.py with the job files written by scripts/make_m5_resume.py; one segment_loop.py job per chain):
#   python scripts/make_m5_resume.py --out scripts/jobs/m5_batch2.json            # GMRF (first map) + TV lambda* + TV 3 lambda*
#   python scripts/make_m5_resume.py --batch 3 --out scripts/jobs/m5_batch3.json  # the same runs on re-adapted maps (scripts/readapt_map.py)
#   python scripts/batch_run.py scripts/jobs/m5_batch4.json --cpu-slots 4 --threads 1 --log-dir results/prior_comparison/logs
#       (batch 4 = the four gmrf chain jobs of batch 2 under new names, after the re-adapted run was stopped; see the run notes in
#        results/prior_comparison/logs/m5_resume_notes.txt)
set -e
D=results/prior_comparison
M5=$D/m5
mkdir -p $M5
PROBLEM="--n 48 --side 2 --eps-r 2 --snr 30 --seed 0 --chains 4"
GMRF="--prior gmrf --sigma 0.5 --ell 0.4 --adapt"
TVA="--prior tv --lam 3.75 --delta 0.01 --sigma-0 1 --map-maxiter 4000 --adapt"
TVB="--prior tv --lam 11.25 --delta 0.01 --sigma-0 1 --map-maxiter 4000 --adapt"

# ---- second-stage maps from the recorded draws (never overwrite the source run) ----
# python scripts/readapt_map.py --from pc_gmrf_adapt --to pc_gmrf_adapt2 -- $PROBLEM $GMRF
# python scripts/readapt_map.py --from pc_tv_lam3.75_adapt --to pc_tv_lam3.75_adapt2 -- $PROBLEM $TVA
# python scripts/readapt_map.py --from pc_tv_lam11.25_adapt --to pc_tv_lam11.25_adapt2 -- $PROBLEM $TVB

# ---- per-run summaries (stage finalize; --max-segments K uses the first K segments of every chain) ----
# python scripts/run_prior_comparison.py $PROBLEM $GMRF --out pc_gmrf_adapt --start-theta $D/pc_gmrf_adapt/m4_ends.npy --stage finalize --max-segments 6
# pc_gmrf_adapt_burn2 = the same segments with the first two of every chain discarded as burn-in (symlinks to the files of pc_gmrf_adapt):
# python scripts/run_prior_comparison.py $PROBLEM $GMRF --out pc_gmrf_adapt_burn2 --start-theta $D/pc_gmrf_adapt/m4_ends.npy --stage finalize --max-segments 6 --skip-segments 2
# python scripts/run_prior_comparison.py $PROBLEM $GMRF --out pc_gmrf_adapt2 --start-theta $D/pc_gmrf_adapt2/starts.npy --stage finalize
# python scripts/run_prior_comparison.py $PROBLEM $TVA --out pc_tv_lam3.75_adapt --stage finalize
# python scripts/run_prior_comparison.py $PROBLEM $TVA --out pc_tv_lam3.75_adapt2 --start-theta $D/pc_tv_lam3.75_adapt2/starts.npy --stage finalize --max-segments 5
# python scripts/run_prior_comparison.py $PROBLEM $TVB --out pc_tv_lam11.25_adapt --stage finalize
# python scripts/run_prior_comparison.py $PROBLEM $TVB --out pc_tv_lam11.25_adapt2 --start-theta $D/pc_tv_lam11.25_adapt2/starts.npy --stage finalize

ALL="pc_matern pc_qggmrf_sg0.05 pc_qggmrf_sg0.1 pc_gmrf pc_gmrf_adapt pc_gmrf_adapt_burn2 pc_gmrf_adapt2 pc_tv_lam3.75_adapt pc_tv_lam3.75_adapt2 pc_tv_lam11.25_adapt pc_tv_lam11.25_adapt2"
MAIN="pc_matern pc_qggmrf_sg0.05 pc_qggmrf_sg0.1 pc_gmrf_adapt pc_tv_lam3.75_adapt pc_tv_lam11.25_adapt"

# ---- comparison across priors (milestone-4 runs, the adapted GMRF, TV at lambda* and 3 lambda*): data check, disagreement maps,
#      feature probabilities and verdicts (the milestone-4 GMRF pc_gmrf is superseded and does not vote) ----
python scripts/compare_priors.py $MAIN --out $M5
# ---- edge statistics, prior-energy audit, rank-normalized convergence audit, slow directions, cross-run pooled chains ----
python scripts/edge_analysis.py $ALL --out $M5
python scripts/prior_energy_audit.py $ALL --out $M5/prior_energy.json
python scripts/convergence_audit.py $ALL --null-reps 60 --out $M5/convergence_audit.json
python scripts/slow_modes.py pc_gmrf_adapt pc_gmrf_adapt2 pc_tv_lam3.75_adapt pc_tv_lam3.75_adapt2 pc_tv_lam11.25_adapt pc_tv_lam11.25_adapt2 \
    --out $M5/slow_modes.json --figure $M5/slow_modes_theta_pca.png
python scripts/cross_run_check.py pc_tv_lam3.75_adapt pc_tv_lam3.75_adapt2 --out $M5/cross_run_tv_lam3.75.json
python scripts/cross_run_check.py pc_tv_lam11.25_adapt pc_tv_lam11.25_adapt2 --out $M5/cross_run_tv_lam11.25.json
python scripts/cross_run_check.py pc_gmrf_adapt pc_gmrf_adapt2 --last 200 --out $M5/cross_run_gmrf.json
python scripts/check_features_independent.py $ALL --out $M5/independent_features.json
# ---- tables (every number is read from the JSON files above) ----
python scripts/prior_comparison_tables.py $MAIN --audit-dir $M5 > $M5/tables_main.md
python scripts/milestone5_tables.py --audit-dir $M5 --runs pc_gmrf_adapt pc_gmrf_adapt2 pc_tv_lam3.75_adapt pc_tv_lam3.75_adapt2 \
    pc_tv_lam11.25_adapt pc_tv_lam11.25_adapt2 --compare pc_matern pc_qggmrf_sg0.05 pc_qggmrf_sg0.1 pc_gmrf > $M5/tables_m5.md
