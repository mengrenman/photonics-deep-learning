#!/bin/bash
# Commands behind the experiment tables in results/tables (as run; read it as documentation, most lines take minutes to an hour).
# Environment: conda activate photonics-dl; run from the repository root. The runs of one group can go in parallel,
# but keep the number of simultaneous jobs below the number of cores divided by two: each JAX process spawns a thread pool,
# and 12 jobs on 16 cores gave load averages of 200 and 3 to 5 times slower runs. Wall times in the tables are
# therefore indicative only; use evaluations. The Laplace cache of a problem is written by the first run that uses
# it, so start the "mala_map" run of a problem and wait until results/laplace_cache/<name>.npz exists.
#
# Common arguments:
#   weak      = --n 32 --side 2 --phantom cylinder --eps-r 1.5 --snr 30 --sigma 0.5 --ell 0.4 --seed 0
#   moderate  = --n 32 --side 2 --phantom austria_fixed --eps-r 2 --snr 30 --sigma 0.5 --ell 0.4 --seed 0
set -e
WEAK="--n 32 --side 2 --phantom cylinder --eps-r 1.5 --snr 30 --sigma 0.5 --ell 0.4 --seed 0 --chains 2"
MOD="--n 32 --side 2 --phantom austria_fixed --eps-r 2 --snr 30 --sigma 0.5 --ell 0.4 --seed 0 --chains 2"
LAP=scripts/run_posterior_laplace.py

# ---- identity-mass samplers (groups a: weak, b: moderate, c: book scale n = 48; none converged) ----
for s in pcn pcn_langevin mala; do
  python scripts/run_posterior.py $WEAK --phantom cylinder --sampler $s --warmup 600 --samples 2400 --thin 4 --init map --map-maxiter 200 --out a_${s}_map
  python scripts/run_posterior.py $MOD  --sampler $s --warmup 600 --samples 2400 --thin 4 --init map --map-maxiter 200 --out b_${s}_map
done
python scripts/run_posterior.py $WEAK --sampler pcn --warmup 1000 --samples 1000 --thin 2 --init prior --map-maxiter 200 --out a_pcn_prior
for s in pcn_langevin mala; do   # prior starts need a tiny first step
  python scripts/run_posterior.py $WEAK --sampler $s --warmup 1000 --samples 1000 --thin 2 --init prior --step0 1e-8 --map-maxiter 200 --out a_${s}_prior
  python scripts/run_posterior.py $MOD  --sampler $s --warmup 1000 --samples 1000 --thin 2 --init prior --step0 1e-8 --map-maxiter 200 --out b_${s}_prior
done
python scripts/run_posterior.py $MOD --sampler pcn --warmup 1000 --samples 1000 --thin 2 --init prior --map-maxiter 200 --out b_pcn_prior
python scripts/run_posterior.py --n 48 --side 5 --phantom austria_fixed --eps-r 2 --snr 30 --sigma 0.5 --ell 0.4 --sampler mala \
    --chains 2 --warmup 40 --samples 160 --thin 2 --init map --map-maxiter 300 --seed 0 --out c_mala_map
# book scale on a finer inference grid at a lower SNR (grid error 0.26 of the noise instead of 2.5); MAP does not converge
python scripts/run_posterior.py --n 96 --side 5 --phantom austria_fixed --eps-r 2 --snr 20 --sigma 0.5 --ell 0.4 --sampler mala \
    --chains 2 --warmup 20 --samples 60 --thin 2 --init map --map-maxiter 400 --seed 0 --out c_book96_mala_map
python scripts/operating_point.py a_pcn_map a_pcn_prior a_pcn_langevin_map a_pcn_langevin_prior a_mala_map a_mala_prior \
    b_pcn_map b_pcn_prior b_pcn_langevin_map b_pcn_langevin_prior b_mala_map b_mala_prior c_mala_map   # added after the fact

# ---- Laplace-whitened samplers, equal budget of 3001 evaluations per chain (group L) ----
for prob in weak mod; do
  if [ $prob = weak ]; then P="$WEAK"; C=results/laplace_cache/weak.npz; else P="$MOD"; C=results/laplace_cache/mod.npz; fi
  python $LAP $P --map-maxiter 400 --sampler mala --warmup 300 --samples 2700 --thin 3 --init map --laplace-cache $C --out L_${prob}_mala_map
  python $LAP $P --map-maxiter 400 --sampler mala --warmup 300 --samples 2700 --thin 3 --init map --laplace-init-scale 2 --laplace-cache $C --out L_${prob}_mala_over
  python $LAP $P --map-maxiter 400 --sampler pcn_langevin --warmup 300 --samples 2700 --thin 3 --init map --laplace-cache $C --out L_${prob}_pcnl_map
  python $LAP $P --map-maxiter 400 --sampler hmc --warmup 60 --samples 240 --thin 1 --init map --laplace-cache $C --out L_${prob}_hmc_map
  python $LAP $P --map-maxiter 400 --sampler hmc --warmup 60 --samples 240 --thin 1 --init map --laplace-init-scale 2 --laplace-cache $C --out L_${prob}_hmc_over
  # four times the budget (12001 evaluations per chain; group Ll)
  python $LAP $P --map-maxiter 400 --sampler mala --warmup 300 --samples 11700 --thin 12 --init map --laplace-cache $C --out Ll_${prob}_mala_map
  python $LAP $P --map-maxiter 400 --sampler hmc --warmup 150 --samples 1050 --thin 1 --init map --laplace-cache $C --out Ll_${prob}_hmc_map
done
# chain started at a prior draw (mapped through M^-1): does not reach the posterior in this budget
python $LAP $WEAK --map-maxiter 400 --sampler hmc --warmup 400 --samples 240 --thin 1 --init prior --step0 0.01 \
    --laplace-cache results/laplace_cache/weak.npz --out L_weak_hmc_prior
# MAP from six prior-draw starts (two-stage: plain L-BFGS, then L-BFGS in Laplace-whitened coordinates)
for prob in weak mod; do
  if [ $prob = weak ]; then P="--n 32 --side 2 --phantom cylinder --eps-r 1.5"; else P="--n 32 --side 2 --phantom austria_fixed --eps-r 2"; fi
  python scripts/map_multistart.py $P --snr 30 --sigma 0.5 --ell 0.4 --seed 0 --map-maxiter 800 --starts 6 --refine-maxiter 400 \
      --precondition --laplace-cache results/laplace_cache/$prob.npz --out MS_$prob
done

# ---- controls for the edge miscoverage of the moderate object (group M) ----
python $LAP $MOD --data-factor 1 --map-maxiter 400 --sampler mala --warmup 300 --samples 2700 --thin 3 --init map --laplace-cache results/laplace_cache/mod_df1.npz --out Mdf1_mod_mala_map
python $LAP $MOD --data-factor 1 --map-maxiter 400 --sampler hmc --warmup 60 --samples 240 --thin 1 --init map --laplace-cache results/laplace_cache/mod_df1.npz --out Mdf1_mod_hmc_map
M48="--n 48 --side 2 --phantom austria_fixed --eps-r 2 --snr 30 --sigma 0.5 --ell 0.4 --seed 0 --chains 2"
python $LAP $M48 --map-maxiter 400 --sampler mala --warmup 300 --samples 2700 --thin 3 --init map --laplace-cache results/laplace_cache/mod_n48.npz --out Mn48_mod_mala_map
python $LAP $M48 --map-maxiter 400 --sampler hmc --warmup 60 --samples 240 --thin 1 --init map --laplace-cache results/laplace_cache/mod_n48.npz --out Mn48_mod_hmc_map
# GMRF prior: the log-normal re-centering shift would be 830 Laplace sd and the chain does not recover (Mgmrf_mod_mala_map);
# --laplace-recenter none is used instead
python $LAP $MOD --prior gmrf --map-maxiter 400 --sampler mala --warmup 300 --samples 2700 --thin 3 --init map --laplace-recenter lognormal --laplace-recenter-max-sd 1e9 --laplace-cache results/laplace_cache/mod_gmrf.npz --out Mgmrf_mod_mala_map
python $LAP $MOD --prior gmrf --map-maxiter 400 --sampler mala --warmup 300 --samples 2700 --thin 3 --init map --laplace-recenter none --laplace-cache results/laplace_cache/mod_gmrf.npz --out Mgmrf_mod_mala_none
python $LAP $MOD --prior gmrf --map-maxiter 400 --sampler hmc --warmup 60 --samples 240 --thin 1 --init map --laplace-recenter none --laplace-cache results/laplace_cache/mod_gmrf.npz --out Mgmrf_mod_hmc_none
python $LAP $MOD --prior gmrf --map-maxiter 400 --sampler hmc --warmup 150 --samples 1050 --thin 1 --init map --laplace-recenter none --laplace-cache results/laplace_cache/mod_gmrf.npz --out Mgmrf_mod_hmc_long

# ---- analysis, tables and figures ----
python scripts/analyze_run.py a_pcn_map a_pcn_prior --pool a_pcn_pool
python scripts/analyze_run.py a_pcn_langevin_map a_pcn_langevin_prior --pool a_pcn_langevin_pool
python scripts/analyze_run.py a_mala_map a_mala_prior --pool a_mala_pool
python scripts/analyze_run.py b_pcn_map b_pcn_prior --pool b_pcn_pool
python scripts/analyze_run.py b_pcn_langevin_map b_pcn_langevin_prior --pool b_pcn_langevin_pool
python scripts/analyze_run.py b_mala_map b_mala_prior --pool b_mala_pool
python scripts/analyze_run.py c_mala_map c_book96_mala_map
for prob in weak mod; do
  python scripts/analyze_run.py L_${prob}_mala_map L_${prob}_mala_over --pool L_${prob}_mala_pool
  python scripts/analyze_run.py L_${prob}_hmc_map L_${prob}_hmc_over --pool L_${prob}_hmc_pool
  python scripts/analyze_run.py L_${prob}_pcnl_map Ll_${prob}_mala_map Ll_${prob}_hmc_map
done
python scripts/analyze_run.py L_weak_hmc_prior Mdf1_mod_mala_map Mdf1_mod_hmc_map Mn48_mod_mala_map Mn48_mod_hmc_map \
    Mgmrf_mod_mala_map Mgmrf_mod_mala_none Mgmrf_mod_hmc_none Mgmrf_mod_hmc_long
python scripts/make_tables.py a_weak a_pcn_map a_pcn_langevin_map a_mala_map a_pcn_prior a_pcn_langevin_prior a_mala_prior --pool a_pcn_pool a_pcn_langevin_pool a_mala_pool
python scripts/make_tables.py b_moderate b_pcn_map b_pcn_langevin_map b_mala_map b_pcn_prior b_pcn_langevin_prior b_mala_prior --pool b_pcn_pool b_pcn_langevin_pool b_mala_pool
python scripts/make_tables.py c_book c_mala_map c_book96_mala_map
python scripts/make_tables.py L_weak L_weak_mala_map L_weak_mala_over L_weak_pcnl_map L_weak_hmc_map L_weak_hmc_over --pool L_weak_mala_pool L_weak_hmc_pool
python scripts/make_tables.py L_mod L_mod_mala_map L_mod_mala_over L_mod_pcnl_map L_mod_hmc_map L_mod_hmc_over --pool L_mod_mala_pool L_mod_hmc_pool
python scripts/make_tables.py Ll_weak Ll_weak_mala_map Ll_weak_hmc_map
python scripts/make_tables.py Ll_mod Ll_mod_mala_map Ll_mod_hmc_map
python scripts/make_tables.py M_edges L_mod_mala_map L_mod_hmc_map Mdf1_mod_mala_map Mdf1_mod_hmc_map Mn48_mod_mala_map Mn48_mod_hmc_map \
    Mgmrf_mod_mala_map Mgmrf_mod_mala_none Mgmrf_mod_hmc_none Mgmrf_mod_hmc_long
python scripts/make_tables.py P_weak L_weak_hmc_map L_weak_hmc_prior
python scripts/plot_gn_spectrum.py
