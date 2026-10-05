"""Add the operating point (grid model error at the truth against the noise) to finished runs.

For every run directory results/<name>/ (summary.json + chains.npz) it rebuilds the forward model of the inference grid,
evaluates it on the stored coverage-averaged truth and compares with the stored noise-free data, exactly as
scripts/run_posterior.py now does at run time (``bayes_scatter.inference.operating_point``), and stores the result
as summary.json["operating_point"]. Needed for runs that were made before that check existed.

Usage:  python scripts/operating_point.py RUN [RUN ...]   (RUN is a name under results/ or an absolute path)
"""

import json
import sys
from pathlib import Path

import numpy as np

import bayes_scatter  # noqa: F401
from bayes_scatter.forward import Forward
from bayes_scatter.geometry import ch15_setup
from bayes_scatter.inference import operating_point

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    names = sys.argv[1:] if argv is None else argv
    for name in names:
        d = Path(name) if Path(name).is_absolute() else ROOT / "results" / name
        s = json.loads((d / "summary.json").read_text())
        a = s["args"]
        z = np.load(d / "chains.npz")
        forward = Forward(ch15_setup(n=a["n"], side=a["side"], n_tx=a["n_tx"], n_rx=a["n_rx"]), tol=a["tol"])
        model = (lambda chi: (forward.born(chi), None)) if a.get("born") else forward
        op = operating_point(model, z["chi_true"], z["d_clean"], z["d_obs"], float(z["noise_std"]))
        s["operating_point"] = op
        (d / "summary.json").write_text(json.dumps(s, indent=2, default=float))
        print(f"{name}: n={a['n']} side={a['side']} model error {100 * op['rel_model_error']:.2f}% vs noise "
              f"{100 * op['rel_noise']:.2f}% (ratio {op['model_to_noise']:.2f}); Phi(truth) {op['phi_truth']:.0f} "
              f"(N {op['n_data']}, clean fine-grid data {op['phi_clean']:.0f}, +{op['phi_excess_sd']:.1f} sd)")


if __name__ == "__main__":
    main()
