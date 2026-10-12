# Milestone 5: the commands that were actually run

Reconstructed from the job files in `scripts/jobs/`, the headers of the logs in `results/prior_comparison/logs/` and the run notes
`logs/m5_resume_notes.txt`. Where a command differs from what a job file says, the difference is listed (review of milestone 5: the job file
and the report said `--init-scale 1.0` for the TV chains, and that is not what produced the draws). All commands were run from the repository root in the
conda environment `photonics-dl`; PROBLEM = `--n 48 --side 2 --eps-r 2 --snr 30 --seed 0 --chains 4`.

## 1. lambda sweep

`scripts/tv_lambda_sweep.py` through `scripts/jobs/tv_lambda_sweep*.json` (delta 0.01 and 0.003, lambda 0.3 to 11.25, `--update` to add lambdas); output
`results/prior_comparison/tv_lambda_sweep*.json`. lambda* = 3.75 (Phi_MAP closest to N = 2400); 3 lambda* = 11.25.

## 2. First attempt (killed by a reboot, 2026-10-11 about 20:55) and its surviving pieces

`scripts/make_m5_batch.py` wrote `scripts/jobs/m5_batch1.json`: prepare, adapt, chain segments and extensions, gated by `scripts/wait_then.py` (dependency, go and skip files).
What survived and is used: the prepared and adapted maps of the three runs and the first TV segments.

* `pc_gmrf_adapt`: adaptation chain started AT the end state of chain 0 of the stuck milestone-4 run (`m4_end0.npy`, `--mass-theta`, `--adapt-start-scale 0`), windows 50 + 100 + 200, trajectory length 1.5,
  kappa = 0.25 N: `... --prior gmrf --sigma 0.5 --ell 0.4 --adapt --out pc_gmrf_adapt --adapt-init 50 --adapt-first-window 50 --adapt-windows 3 --adapt-growth 2 --adapt-final 100 --adapt-kappa-per-dim 0.25 --traj-length 1.5 --mass-theta results/prior_comparison/pc_gmrf_adapt/m4_end0.npy --adapt-start-scale 0 --stage adapt`.
  The recording chains start at `m4_ends.npy` (`--start-theta`, the four milestone-4 end states).
* `pc_tv_lam3.75_adapt` and `pc_tv_lam11.25_adapt`: `--prior tv --lam <3.75 | 11.25> --delta 0.01 --sigma-0 1 --map-maxiter 4000 --adapt` with the TV schedule `--adapt-init 50 --adapt-first-window 50 --adapt-windows 2 --adapt-growth 2 --adapt-final 50 --adapt-kappa-per-dim 0.25 --traj-length 1.0`
  (windows 50 and 100, kappa 576, trajectory length 1.0).
* **TV chains, as run.** The gated jobs of the batch file have `--init-scale 1.0`; their starts have a physical Phi of 4e4 to 8e4, so the gate files `scripts/jobs/skip/tvA_seg0`, `tvA_seg1`, `tvB_*` made those jobs exit without running and
  the segments were launched BY HAND (logs `pc_tv_lam*_adapt_c*_s*_manual.log`, whose first line names the reason): segment 0 of every chain with `--init-scale 0.5 --stage chain --chain-id C --segment 0 --samples 400 --warmup 50`
  (lambda*: 2026-10-11 17:00; 3 lambda*: 17:01) and, for lambda*, segment 1 with `--stage chain --chain-id C --segment 1 --samples 400` (18:44), i.e. 400 + 400 recorded draws per chain (`pc_tv_lam3.75_adapt`, 800 draws).
  The same segment 1 of 3 lambda* was launched at 20:39 and lost in the reboot of about 20:55, so `pc_tv_lam11.25_adapt` holds the 400 draws of segment 0 only.
  The starts are 0.5 N(0, I) in the coordinates of the adapted map around the final state of the adaptation chain (field `starts` of every segment file).
* GMRF chains of the first attempt: segment 0 launched by hand with 300 recorded trajectories (`pc_gmrf_adapt_c*_s0_manual.log`, 2026-10-11 18:09), starts `--start-theta results/prior_comparison/pc_gmrf_adapt/m4_ends.npy`, extension segments through the job file.

## 3. Resumption (batch 2) and re-adaptation (batch 3), batch 4

* `python scripts/make_m5_resume.py --out scripts/jobs/m5_batch2.json && python scripts/batch_run.py scripts/jobs/m5_batch2.json --cpu-slots 12 --threads 1 --log-dir results/prior_comparison/logs`
  (four `segment_loop.py` jobs per run; the TV groups waited for go files; the TV segments beyond the first attempt's were stopped by stop files).
* `python scripts/readapt_map.py --from pc_gmrf_adapt --to pc_gmrf_adapt2 -- PROBLEM --prior gmrf --sigma 0.5 --ell 0.4 --adapt`, and the same for `pc_tv_lam3.75_adapt` -> `pc_tv_lam3.75_adapt2`, `pc_tv_lam11.25_adapt` -> `pc_tv_lam11.25_adapt2`
  (TV arguments as above). The new runs CONTINUE the same chains (`starts.npy` = final states of the source's chains).
* `python scripts/make_m5_resume.py --batch 3 --out scripts/jobs/m5_batch3.json` and `batch_run.py ... --cpu-slots 12`; stop files for `tvA2` (00:04), `gmrf2` (00:51), `tvB2`.
* `python scripts/batch_run.py scripts/jobs/m5_batch4.json --cpu-slots 4 --threads 1 ...`: the first-map GMRF chains continued to 600 recorded draws (killed during segment 6 at 02:14; the segment in flight was lost).

## 4. Finalize and analysis

`scripts/m5_analysis.sh` (finalize commands in its comments; the GMRF `--max-segments 6`; `pc_gmrf_adapt_burn2` = `--skip-segments 2`), then `scripts/m5_review_analysis.sh` (review follow-up).

## 5. Review follow-up (this batch)

`python scripts/make_m5_review_jobs.py --out scripts/jobs/m5_review_batch.json` and
`python scripts/batch_run.py scripts/jobs/m5_review_batch.json --cpu-slots 17 --threads 1 --log-dir results/prior_comparison/logs` (one batch; waiting jobs sleep; at most 10 XLA threads compute at any time):
TV at lambda = 1.25 and 8.5 (prepare, adapt, chains from four different starts), and an independent-start replication of lambda* in `pc_tv_lam3.75_indep`. See DESIGN 12.6.
