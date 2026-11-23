# %% [markdown]
# # Band edges: high-frequency homogenization, little groups, Dirac cones and the envelope's boundary condition
#
# Classical homogenization describes a periodic medium near the band edge $\omega\to0$, where the Bloch function is a slowly varying envelope times a constant. The same construction works at any other band edge $(\mathbf k_0,\omega_0)$, with the constant replaced by the Bloch function of the edge: the field is $e^{i\mathbf k_0\cdot\mathbf x}p(\mathbf x)F(\varepsilon\mathbf x)$ (with $p$ cell periodic), and the envelope $F$ obeys an effective equation whose coefficients are fixed by the edge wave functions (high-frequency homogenization, Craster, Kaplunov and Pichugin 2010; the Bloch-wave version is Conca and Vanninathan 1997, Allaire and Conca 1998). If the edge is degenerate, $F$ is a vector with one component per degenerate state and the coefficients are matrices. The question here is how much of that effective equation is already determined by the little group of $\mathbf k_0$, and whether exact band computations show it: the number of free constants, the quadratic touching of two bands at $M$, the Dirac cone at $K$ with its Berry phase, and, for a finite crystal, the quantization of the envelope together with the boundary condition that the walls impose on it.
#
# It matters because it is where representation theory enters homogenization: the number of independent constants in the effective tensor at a degenerate edge is the dimension of a Hom space between little-group representations, which a character computation gives and a Bloch solver can confirm. And it matters for the finite-size question of notebook 04 in its high-frequency form: the envelope's boundary condition is not a property of the interior medium alone, it depends on where the wall cuts the cell.
#
# ## What this notebook shows
#
# - **The solver and its symmetry.** With a circular plane-wave cutoff the degenerate levels are degenerate to about $10^{-13}$ (checked to $10^{-10}$) at $M$ and $\Gamma$ of the square lattice and at $K$ of the honeycomb lattice; a parallelogram cutoff of 289 plane waves splits the honeycomb Dirac point by $1.1\times10^{-6}$ in $f$, a circular one of 294 by less than $10^{-13}$. The plane-wave frequencies at $X$ and $M$ agree with the multipole solver to at most $3.15\times10^{-5}$ (relative, bands 1-4), inside three times the change between 600 and 1000 plane waves; most of that difference is plane-wave truncation (at 2000 plane waves the largest difference is $1.07\times10^{-5}$).
# - **Little groups.** Character projection finds a single irrep for every degenerate set at $\Gamma$, $X$, $M$ (square) and $K$ (honeycomb); bands 2-3 at $M$ are the $E$ doublet of $C_{4v}$ and bands 1-2 at $K$ the $E$ doublet of $C_{3v}$, with traces equal to the tabulated characters.
# - **Non-degenerate edge.** Band 1 at $X$ has no linear term and an inverse-mass tensor $\mathrm{diag}(-0.6865,\,+0.1210)$ in $\lambda$ units (a saddle). Perturbation theory matches Richardson finite differences to better than $10^{-7}$ at $|\mathbf q|a=0.01,0.02$. At the finer steps $10^{-3},2\times10^{-3}$ the comparison is limited to about $10^{-5}$ by round-off in the second difference, so the agreement of $10^{-6}$ expected beforehand (PLAN.md) is not reachable there.
# - **Degenerate edge at $M$.** $\dim\operatorname{Hom}_{C_{4v}}(\operatorname{Sym}^2V,\operatorname{End}E)=3$, $\dim\operatorname{Hom}(V,\operatorname{End}E)=0$ and the cubic Hom space is also zero, computed from the characters of the computed doublet. The 2x2 $k\cdot p$ matrix has the three-constant form with a residual of about $10^{-14}$; it predicts $\lambda-\lambda_0$ with a relative error of $1.0\times10^{-3}$ at $|\mathbf q|a=0.04$, the error scales as $q^2$ (ratio 4.01 between $|\mathbf q|a=0.08$ and $0.04$), and the pseudospin of the upper band winds twice.
# - **Breaking $C_4$.** Elliptical rods split the doublet linearly in $e$; the first-order degenerate perturbation slope agrees with the PWE splitting$/e$ to $2.5\times10^{-3}$ at $e=0.005$, and the splitting$/e$ falls from $0.9975$ to $0.9807$ of that slope between $e=0.005$ and $0.04$.
# - **Dirac cone.** $\dim\operatorname{Hom}_{C_{3v}}(V,\operatorname{End}E)=1$; the $k\cdot p$ velocity agrees with the PWE slope to better than $10^{-7}$ when $\pm\mathbf q$ are averaged (a single direction is off by $7.5\times10^{-4}$, trigonal warping). The loop phase of equal rods is exactly $\pi$, which is a test of the code. For unequal rods the closed formula for the massive Dirac loop phase (with the unequal-rod velocity $v=0.58261$) deviates from the 256-point PWE loop by $0.0011\pi$, $0.0055\pi$, $0.0200\pi$ and $0.0345\pi$ at $r=m/2v,\ m/v,\ 2m/v,\ 3m/v$: **the bound of $0.03\pi$ expected beforehand (PLAN.md) fails at $3m/v$**. The deviation does not scale as $r^2$ (local log-log slopes $2.29$, $2.00$, $1.66$, $1.41$, $1.25$ against $2$; deviation$/r^2$ varies by a factor $1.44$). Loop discretization (256 against 512 points: $1.2\times10^{-5}\pi$) and plane-wave truncation (200 against 400: $1.1\times10^{-5}\pi$) are small, but 64-point loops are off by up to $2.5\times10^{-4}\pi$, $22\%$ of the smallest deviation. Projecting the PWE eigenvector onto the doublet at $K$ shows where the deviation comes from: the part of the Bloch vector outside the doublet is $73\%$ of the deviation at $m/2v$, $47\%$ at $m/v$ and $30\%$ at $3m/v$; the remainder is a doublet-only effect that the second-order $k\cdot p$ matrix reproduces to $0.0013\pi$ or better. The second-order $k\cdot p$ loop alone therefore cannot beat $0.0008\pi$ at $m/2v$. Attribution of the doublet-only part to trigonal warping specifically is a conjecture (the second-order matrix contains warping among other terms). The phases at $K$ and $K'$ are opposite to better than $10^{-12}$, which is an exact symmetry of the discrete scheme (a test of the code).
# - **Finite stacks.** With both walls at the same cell phase (at the mirror plane, or at $a/4$) every in-band mode has Bloch wave number $(N-j)\pi/(Na)$ to $10^{-13}$, the two cases have identical in-band spectra, and the envelope error falls as $N^{-4}$ (slopes between $-3.8$ and $-4.0$ for $j=1,2,3$ at both edges); there is no end correction. Only the wall phase decides the edge modes: walls at 0 make exactly one gap edge a finite-stack root (the lower one), walls at $a/4$ make neither one and create a gap mode. For walls at different cell phases (0 and $Na+a/4$) the exact condition is $kNa+\Theta(\omega)=n\pi$ (verified to $3.5\times10^{-12}$); $\Theta_e$ is $0$ at the upper and $\pi/2$ at the lower edge, which fixes integer and half-integer $j$, and a fit-free end correction from $d\Theta/dq$ restores the $N^{-4}$ error. The end correction depends on the length convention. With the convention $L=Na+2\delta$ of PLAN.md (uncorrected length $Na$) a single $\delta$ fitted at $N=20$, $j=1$ gives $\delta=0.201a$ at the upper edge and cuts the error by $15.3\times$ and $24.1\times$ at $N=10$ and $40$: an earlier estimate of $0.20a$ and $15\times$-$24\times$ is reproduced, and the expected reduction of $10\times$ is met. With the geometric length $Na+s$ ($s=a/4$) as uncorrected baseline, $L=Na+s+2\delta$, the same correction is $\delta=0.076a$ and the reductions are $6.2\times$ and $8.4\times$: the same physics under two conventions, $\delta_{\rm geo}=\delta_{\rm bare}-s/2$. The end correction proper is $0.0595a$ (fit-free, from $-c-s$); the fit at $N=20$ also absorbs the band's quartic term.
## Conventions
#
# - Time dependence $e^{-i\omega t}$. TM means the electric field along the rod axis $z$, so the field equation is the generalized eigenproblem $-(\nabla+i\mathbf k)^2u=\lambda\,\varepsilon u$ for the cell-periodic part $u$, with $\lambda=(\omega/c)^2$. Frequencies are reported as $f=\omega a/2\pi c=a/\lambda_{\rm vac}$, and all lengths are in units of the period $a=1$. Band indices are 1-based in the text and tables.
# - Plane-wave method: unknown coefficients $c_{\mathbf G}$ on a set of reciprocal vectors, $A_{\mathbf G\mathbf G'}=|\mathbf k+\mathbf G|^2\delta_{\mathbf G\mathbf G'}$, $B_{\mathbf G\mathbf G'}=\hat\varepsilon(\mathbf G-\mathbf G')$ (positive definite), eigenvectors $B$-orthonormal. The "number of plane waves" is the size of the set, which is a disk $|\mathbf k_0+\mathbf G|\le G_{\max}$ around the expansion point $\mathbf k_0$.
# - Square lattice: one rod of radius $R=0.3$, $\varepsilon=12$, background 1, little groups $C_{4v}$ at $\Gamma$ and $M$, $C_{2v}$ at $X$. Honeycomb lattice: $\mathbf a_1=(1,0)$, $\mathbf a_2=(\tfrac12,\tfrac{\sqrt3}2)$, rods at $(\mathbf a_1+\mathbf a_2)/3$ and $2(\mathbf a_1+\mathbf a_2)/3$ with radius $0.2$, $\varepsilon=12$ (the origin is the hexagon center, group $C_{6v}$), little group $C_{3v}$ at $K=(\mathbf b_1+2\mathbf b_2)/3$ and $K'=(2\mathbf b_1+\mathbf b_2)/3$.
# - Tolerances stated "a priori" were fixed before the computation; those set after a convergence study or after seeing a result are marked where they appear and listed in the caveats.

# %%
SETUP

# %%
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path.cwd()))                                   # bloch.py sits next to this notebook
sys.path.insert(0, str(pathlib.Path.cwd().parent / "spatial_dispersion"))      # rodlattice.py (multipole reference)
import bloch as bl
import rodlattice as rl

T_START = time.perf_counter()
N_CHECKS = 0
_check, _check_in = check, check_in


def check(*a, **k):                      # same as the setup cell's check, but counted
    global N_CHECKS
    N_CHECKS += 1
    return _check(*a, **k)


def check_in(*a, **k):
    global N_CHECKS
    N_CHECKS += 1
    return _check_in(*a, **k)


lam_of = lambda f: (2 * np.pi * f) ** 2          # lambda = (omega/c)^2 from f = a/lambda_vac, a = 1
f_of = lambda lam: np.sqrt(lam) / (2 * np.pi)

sq = bl.Lattice.square()
hc = bl.Lattice.honeycomb()
EPS, R_SQ = 12.0, 0.3
rods_sq = [bl.disk((0, 0), R_SQ, EPS)]
s_h = hc.sites
rods_h = [bl.disk(s_h[0], 0.2, EPS), bl.disk(s_h[1], 0.2, EPS)]
rods_h_un = [bl.disk(s_h[0], 0.21, EPS), bl.disk(s_h[1], 0.19, EPS)]


def gset(lat, n, center):
    """One circular plane-wave set of at least n vectors around `center` (complete shells), reused for everything at that point."""
    k0 = np.asarray(center, float)
    return bl.gvectors(lat, bl.gmax_for_count(lat, n, k0), k0)[0]


# %% [markdown]
# ## 1. A plane-wave solver with exact symmetry
#
# The truncation is the only approximation of the solver, and it can break the symmetry that the physics has. The little group of $\mathbf k_0$ acts on the coefficient vectors by permuting reciprocal vectors: for a point-group element $g$ with $g\mathbf k_0=\mathbf k_0+\mathbf G_g$, $(P_g c)_{\mathbf G}=c_{g^{-1}(\mathbf G+\mathbf k_0)-\mathbf k_0}$, which is a permutation of the index set exactly when the set is invariant under $\mathbf G\mapsto g(\mathbf k_0+\mathbf G)-\mathbf k_0$. A disk $|\mathbf k_0+\mathbf G|\le G_{\max}$ is invariant under every orthogonal $g$ that fixes $\mathbf k_0$ modulo the lattice, so $P_g$ commutes with $A$ and $B$ and the truncated problem has the same little group and the same degeneracies, exactly, not merely to the truncation error. A parallelogram $|i|,|j|\le m$ in the lattice indices is not invariant under the sixfold rotation and breaks it. Section 1 measures both, converges the square-lattice frequencies in the number of plane waves, and checks them against the independent multipole solver of `notebooks/spatial_dispersion/rodlattice.py` (a different method with its own truncation, five multipoles).

# %%
SIZES = (300, 600, 1000)
fsq, npw = {}, {}
for nm in ("G", "X", "M"):
    k0 = sq.points[nm]
    for n in SIZES:
        G = gset(sq, n, k0)
        fsq[(nm, n)] = bl.tm_solve(sq, rods_sq, k0, G, 6)[0]
        npw[(nm, n)] = len(G)

print("Square lattice (R = 0.3, eps = 12), TM, f = a/lambda; 1000 plane waves")
print(f"{'point':>6} {'n_pw':>5} " + " ".join(f"{'band ' + str(b):>9}" for b in range(1, 7)))
for nm in ("G", "X", "M"):
    print(f"{nm:>6} {npw[(nm, 1000)]:>5} " + " ".join(f"{x:9.6f}" for x in fsq[(nm, 1000)]))

print("\nConvergence in the number of plane waves (bands 1-4, relative change of f)")
print(f"{'point':>6} {'n_pw':>5} {'f1':>9} {'f2':>9} {'f3':>9} {'f4':>9} {'max rel. change vs next':>26}")
for nm in ("X", "M"):
    for i, n in enumerate(SIZES):
        chg = "" if i == len(SIZES) - 1 else f"{np.max(np.abs(fsq[(nm, SIZES[i + 1])][:4] - fsq[(nm, n)][:4]) / fsq[(nm, SIZES[i + 1])][:4]):.2e}"
        print(f"{nm:>6} {npw[(nm, n)]:>5} " + " ".join(f"{x:9.6f}" for x in fsq[(nm, n)][:4]) + f" {chg:>26}")

# %%
# PWE against the multipole solver at X and M, four lowest bands.
# Tolerance fixed after the convergence study above: 3 x (relative change between 600 and 1000 plane waves), capped at 2e-4.
print(f"{'point':>6} {'band':>5} {'PWE (1000)':>11} {'multipole':>11} {'rel. diff':>10} {'tolerance':>10}")
REF = {}
for nm in ("X", "M"):
    ref = rl.bloch_frequencies(sq.points[nm], R_SQ, EPS, "TM", M=5, fmax=0.7, nscan=300)[:4]   # M doublet is listed twice, as in the PWE
    REF[nm] = ref
    p = np.array([fsq[(nm, n)][:4] for n in SIZES])
    for b in range(4):
        diff = abs(p[2, b] - ref[b]) / ref[b]
        tol = min(3 * abs(p[2, b] - p[1, b]) / p[2, b], 2e-4)
        print(f"{nm:>6} {b + 1:>5} {p[2, b]:11.6f} {ref[b]:11.6f} {diff:10.2e} {tol:10.2e}")
        check(f"PWE vs multipole at {nm}, band {b + 1}", diff, tol)

# %%
# One more point of the convergence study: 2000 plane waves at X and M, bands 1-4. It separates the plane-wave truncation error of the
# 1000-wave frequencies from the difference to the multipole solver.
f2k = {}
print(f"{'point':>6} {'n_pw':>5} {'max rel. change 1000 -> 2000':>30} {'max rel. diff to multipole, 1000 pw':>37} {'2000 pw':>9}")
for nm in ("X", "M"):
    G2 = gset(sq, 2000, sq.points[nm])
    f2k[nm] = bl.tm_solve(sq, rods_sq, sq.points[nm], G2, 4)[0]
    c1000 = np.max(np.abs(f2k[nm] - fsq[(nm, 1000)][:4]) / f2k[nm])
    c600 = np.max(np.abs(fsq[(nm, 1000)][:4] - fsq[(nm, 600)][:4]) / fsq[(nm, 1000)][:4])
    d1000 = np.max(np.abs(fsq[(nm, 1000)][:4] - REF[nm]) / REF[nm])
    d2000 = np.max(np.abs(f2k[nm] - REF[nm]) / REF[nm])
    print(f"{nm:>6} {len(G2):>5} {c1000:30.2e} {d1000:37.2e} {d2000:9.2e}")
    check(f"{nm}: change from 1000 to 2000 plane waves is below the change from 600 to 1000 (convergence is not yet reversed)", c1000, c600)

# %%
# Exact degeneracies of the circular cutoff, and the parallelogram cutoff for comparison.
print("Exact degeneracies with the circular cutoff")
for n in SIZES:
    fm = fsq[("M", n)]
    check(f"square M, bands 2 and 3, about {n} plane waves: |f2 - f3|", abs(fm[2] - fm[1]), 1e-10)
fg = fsq[("G", 1000)]
check("square Gamma, bands 2 and 3: |f2 - f3|", abs(fg[2] - fg[1]), 1e-10)

fK = {}
for n in SIZES:
    G = gset(hc, n, hc.points["K"])
    fK[n] = bl.tm_solve(hc, rods_h, hc.points["K"], G, 4)[0]
    check(f"honeycomb K, bands 1 and 2, about {n} plane waves: |f1 - f2|", abs(fK[n][1] - fK[n][0]), 1e-10)
print(f"\nHoneycomb (rods 0.2, eps 12) at K, 1000 plane waves: f = " + ", ".join(f"{x:.6f}" for x in fK[1000]))
print("worst degeneracy splitting of the circular cutoffs: square M " + f"{max(abs(fsq[('M', n)][2] - fsq[('M', n)][1]) for n in SIZES):.2e}, square Gamma {abs(fg[2] - fg[1]):.2e}, honeycomb K {max(abs(fK[n][1] - fK[n][0]) for n in SIZES):.2e}")

m_ = np.arange(-8, 9)
Ii, Jj = np.meshgrid(m_, m_, indexing="ij")
Gpar = Ii.ravel()[:, None] * hc.b1 + Jj.ravel()[:, None] * hc.b2          # parallelogram |i|, |j| <= 8, 289 vectors
fp = bl.tm_solve(hc, rods_h, hc.points["K"], Gpar, 2)[0]
Gci = gset(hc, len(Gpar), hc.points["K"])
fc = bl.tm_solve(hc, rods_h, hc.points["K"], Gci, 2)[0]
print(f"Dirac splitting at K, f2 - f1: parallelogram cutoff ({len(Gpar)} plane waves) {fp[1] - fp[0]:.2e}; circular cutoff ({len(Gci)} plane waves) {fc[1] - fc[0]:.2e}")
check_in("parallelogram cutoff splits the Dirac point (resolved above the circular cutoff's round-off; about 1e-6 in exploratory runs)", fp[1] - fp[0], 1e-8, 1.0)
check("circular cutoff: Dirac splitting", fc[1] - fc[0], 1e-10)

# %%
# Figure 1: band structures along the boundary of the irreducible Brillouin zone (240 plane waves around Gamma; the plot needs 1e-3 only).
def path(pts, n=18):
    out, lab = [], [0.0]
    for p, q in zip(pts[:-1], pts[1:]):
        t = np.linspace(0, 1, n, endpoint=False)[:, None]
        out.append(np.asarray(p) + t * (np.asarray(q) - np.asarray(p)))
        lab.append(lab[-1] + np.linalg.norm(np.asarray(q) - np.asarray(p)))
    out.append(np.asarray(pts[-1])[None, :])
    return np.vstack(out), lab


def bands_on_path(lat, rods, pts, nb):
    ks, lab = path(pts)
    G = gset(lat, 240, [0.0, 0.0])
    B = bl.eps_matrix(lat, rods, G)
    f = np.array([bl.tm_solve(lat, rods, k, G, nb, B=B)[0] for k in ks])
    x = np.concatenate([np.linspace(lab[i], lab[i + 1], 18, endpoint=False) for i in range(len(lab) - 1)] + [[lab[-1]]])
    return x, f, lab


Pq = sq.points
xs, fs, labs = bands_on_path(sq, rods_sq, [Pq["G"], Pq["X"], Pq["M"], Pq["G"]], 6)
Ph = hc.points
xh, fh, labh = bands_on_path(hc, rods_h, [Ph["G"], Ph["M"], Ph["K"], Ph["G"]], 5)

fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.0))
for a_, x, f, lab, names, title in ((ax[0], xs, fs, labs, ["$\\Gamma$", "X", "M", "$\\Gamma$"], "square lattice, $R=0.3$, $\\varepsilon=12$"),
                                     (ax[1], xh, fh, labh, ["$\\Gamma$", "M", "K", "$\\Gamma$"], "honeycomb lattice, $R=0.2$, $\\varepsilon=12$")):
    for b in range(f.shape[1]):
        a_.plot(x, f[:, b], color=CAT[0], lw=1.3)
    for l in lab:
        a_.axvline(l, color=AXIS, lw=0.8)
    a_.set_xticks(lab)
    a_.set_xticklabels(names)
    a_.set_ylabel("$f=a/\\lambda$")
    a_.set_title(title)
    a_.set_xlim(0, lab[-1])
ax[0].set_ylim(0, 0.75)
ax[1].set_ylim(0, 0.62)
ax[0].annotate("E doublet at M\n(bands 2-3)", xy=(labs[2], fsq[("M", 1000)][1]), xytext=(labs[2] - 2.3, 0.62), color=INK, fontsize=8.5,
               arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
ax[1].annotate("Dirac point at K\n(bands 1-2)", xy=(labh[2], fK[1000][0]), xytext=(labh[2] - 3.6, 0.43), color=INK, fontsize=8.5,
               arrowprops=dict(arrowstyle="->", color=INK2, lw=0.8))
plt.show()
print(f"[{time.perf_counter() - T_START:.1f} s elapsed]")

# %% [markdown]
# *Figure 1. TM bands along the boundary of the irreducible zone. The doublet at $M$ of the square lattice (bands 2 and 3) and the Dirac point of the honeycomb lattice at $K$ (bands 1 and 2) are the two degenerate edges studied below.*
#
# Table 1 (above) gives the lowest six bands at $\Gamma$, $X$, $M$ with about 1000 plane waves. The convergence table shows relative changes of $1.1\times10^{-4}$ (300 to 600) and $3.6\times10^{-5}$ (600 to 1000) at the worst band, so the 1000-wave frequencies are good to about $3\times10^{-5}$ (the change from 1000 to 2000 plane waves is $2.1\times10^{-5}$ at $X$ and $1.9\times10^{-5}$ at $M$), and the comparison with the multipole solver stays below $3.2\times10^{-5}$ (largest: $3.15\times10^{-5}$) for all eight tested bands, each inside the tolerance fixed from this study (three times the 600-to-1000 change, capped at $2\times10^{-4}$). Against the 2000-wave frequencies the difference to the multipole solver falls to $1.07\times10^{-5}$ ($X$) and $9.9\times10^{-6}$ ($M$), so most of the $3\times10^{-5}$ is plane-wave truncation, not an error of the multipole solver. The two solvers use different expansions, so agreement at this level confirms the Fourier coefficients of the disk and the generalized-problem formulation, not only self-consistency.
#
# The degeneracies of the circular cutoff are exact to round-off (below $2\times10^{-13}$: worst $1.9\times10^{-13}$ at $K$ with 1000 plane waves), independent of the number of plane waves; the parallelogram cutoff splits the Dirac point at $K$ by about $10^{-6}$, as exploratory runs before the notebook had found, and a circular cutoff with the same number of vectors does not. Every later $k\cdot p$ and Berry-phase computation therefore uses one circular set around the expansion point, built once and reused for finite differences, ellipse perturbations and loops.

# %% [markdown]
# ## 2. Little groups and irreps
#
# At $\mathbf k_0$ the little group $G_{\mathbf k_0}$ (the point-group elements fixing $\mathbf k_0$ modulo the reciprocal lattice, here also required to preserve the rods) acts on each degenerate eigenspace $E$ through $D(g)=C^HB\,P_gC$, with $C$ the $B$-orthonormal eigenvectors. The matrices $D(g)$ form a unitary representation; its character $\chi_E(g)=\operatorname{tr}D(g)$ decomposes uniquely into irreducible characters, and $E$ is a single irrep exactly when $\langle\chi_E,\chi_E\rangle=1$. This is gauge free: the trace does not depend on the basis chosen inside a degenerate set, which is why the notebook works with traces and projectors and never with individual eigenvectors of a degenerate set.
#
# The table gives, for every degenerate set, the irrep found by character projection, and the decomposition error (distance of the multiplicities from integers, and from a single irrep). The last degenerate set of each run may be cut by the number of computed bands and is not shown.

# %%
rows, irr = [], {}
for lab, lat, rods, nm, nb in (("square Gamma", sq, rods_sq, "G", 8), ("square X", sq, rods_sq, "X", 8),
                               ("square M", sq, rods_sq, "M", 8), ("honeycomb K", hc, rods_h, "K", 6)):
    k0 = lat.points[nm]
    G = gset(lat, 400, k0)
    ops, name = bl.point_group_ops(lat, k0, rods)
    f, C, B = bl.tm_solve(lat, rods, k0, G, nb)
    res = bl.irreps_of_bands(lat, f, C, G, k0, B, ops)[:-1]
    irr[lab] = (name, res, ops, G, k0, B, C)
    print(f"{lab} ({name}, {len(G)} plane waves)")
    for r in res:
        bnds = f"{r['bands'][0] + 1}" if len(r["bands"]) == 1 else f"{r['bands'][0] + 1}-{r['bands'][-1] + 1}"
        print(f"    bands {bnds:>4}   f = {r['f']:.6f}   irrep {str(r['irrep']):>3}   decomposition error {r['error']:.1e}")
    check_in(f"{lab}: every degenerate set is a single irreducible representation (1 = yes)", float(all(r["irrep"] is not None for r in res)), 1, 1)
    check(f"{lab}: worst character decomposition error", max(r["error"] for r in res), 1e-8)
names = [irr[k][0] for k in irr]
print("\nlittle groups found:", ", ".join(f"{k}: {irr[k][0]}" for k in irr))
assert names == ["C4v", "C2v", "C4v", "C3v"]

# %%
# Independent check of the two doublets against the character tables of C4v and C3v, class by class.
TABLE = {"square M": {"E": 2, "C2": -2, "C4": 0, "sv": 0, "sd": 0},     # E irrep of C4v (rotation by pi has trace -2)
         "honeycomb K": {"E": 2, "C3": -1, "sv": 0}}                    # E irrep of C3v
print(f"{'doublet':>12} {'class':>5} {'size':>4} {'computed chi':>13} {'table chi':>10}")
worst = {}
for lab, bands in (("square M", [1, 2]), ("honeycomb K", [0, 1])):
    name, res, ops, G, k0, B, C = irr[lab]
    Ds = bl.rep_matrices(sq if "square" in lab else hc, C[:, bands], G, k0, ops, B)
    eu, eh = bl.rep_error(Ds, ops)
    check(f"{lab} doublet: D(g) unitary", eu, 1e-10)
    check(f"{lab} doublet: D(g1 g2) = D(g1) D(g2)", eh, 1e-10)
    cls = {}
    for o, D in zip(ops, Ds):
        cls.setdefault(o["cls"], []).append(np.trace(D))
    w = 0.0
    for c, v in cls.items():
        print(f"{lab:>12} {c:>5} {len(v):>4} {np.mean(v).real:13.8f} {TABLE[lab][c]:>10}")
        w = max(w, abs(np.mean(v) - TABLE[lab][c]), np.ptp(np.real(v)))
    worst[lab] = w
    check(f"{lab} doublet: traces equal the irreducible characters, constant on classes", w, 1e-8)

# %% [markdown]
# The table lists a single irrep for every degenerate set: $A_1$ and the $E$ doublet at $\Gamma$ and $M$, one-dimensional irreps of $C_{2v}$ at $X$ (all bands non-degenerate there), and at $K$ the $E$ doublet (bands 1-2, the Dirac pair), then $A_1$ and a second $E$ doublet. The class-by-class traces of the two doublets used below match the character tables to better than $10^{-14}$, and the matrices $D(g)$ are unitary and multiply correctly to better than $10^{-14}$ (they would not for an operation outside the little group; the module raises an error in that case). The agreement is not a tautology: $D(g)$ is built from the eigenvectors and the permutation of the plane-wave set, while the table values come from group theory.

# %% [markdown]
# ## 3. A non-degenerate edge: effective mass by perturbation theory
#
# For a non-degenerate eigenvalue $\lambda_0$ at $\mathbf k_0$, write $\mathbf k=\mathbf k_0+\mathbf q$. In the generalized problem $A(\mathbf k)c=\lambda Bc$ only $A$ depends on $\mathbf k$, $A(\mathbf k_0+\mathbf q)=A_0+\mathbf q\cdot\mathbf w+|\mathbf q|^2$ with $\mathbf w=2(\mathbf k_0+\mathbf G)$ (diagonal), and $B$ does not move. With $B$-orthonormal eigenvectors $c_n$ and eigenvalues $\lambda_n$, Löwdin perturbation theory through second order gives
#
# $$\lambda(\mathbf q)=\lambda_0+\mathbf q\cdot\mathbf H_1+q_iq_j\,H_{2,ij}+O(q^3),\qquad \mathbf H_1=c_0^H\mathbf w\,c_0,\qquad H_{2,ij}=\delta_{ij}+\sum_{n\ne0}\frac{(c_0^Hw_ic_n)(c_n^Hw_jc_0)}{\lambda_0-\lambda_n},$$
#
# the sum running over every other eigenstate of the truncated problem (the module takes all of them, so the second-order coefficient is exact for the truncated problem). Its Hessian is $2H_2$, and $H_2$ is the inverse-mass tensor of the envelope equation $\lambda_0F-H_{2,ij}\,\partial_i\partial_jF=\lambda F$ in $\lambda$ units. At $X$ the little group $C_{2v}$ contains $q_x\to-q_x$ and $q_y\to-q_y$ separately, which kills $\mathbf H_1$ and the off-diagonal entry of $H_2$; band 1 there is a saddle: the Hessian has one negative and one positive eigenvalue, a maximum along $\Gamma X$ and a minimum along $XM$.
#
# The finite-difference comparison uses the same truncated plane-wave set. Central second differences carry an $O(h^2)$ truncation error that a Richardson step removes, and a round-off error of order $\epsilon_{\rm mach}\lambda_{\max}/h^2$, where $\lambda_{\max}$ is the largest eigenvalue of the truncated problem (about $10^4$ for 600 plane waves). Steps $|\mathbf q|a=10^{-3},2\cdot10^{-3}$ were the first choice, but round-off limits the comparison there to about $10^{-5}$, so the steps used for the check are $10^{-2}$ and $2\cdot10^{-2}$ and the finer pair is run as well and reported.

# %%
k0 = sq.points["X"]
G_X = gset(sq, 600, k0)
B_X = bl.eps_matrix(sq, rods_sq, G_X)
sol_X = bl.tm_solve(sq, rods_sq, k0, G_X, None, B=B_X)
kpX = bl.kp_second_order(sq, rods_sq, k0, G_X, [0], solution=sol_X)
H1X = kpX.H1[:, 0, 0].real
HessX = 2 * kpX.H2[:, :, 0, 0].real                    # lambda(q) = lambda0 + q . (Hess / 2) . q
H2X = kpX.H2[:, :, 0, 0].real


def lam_q(q):
    return lam_of(bl.tm_solve(sq, rods_sq, k0 + np.asarray(q, float), G_X, 1, B=B_X)[0][0])


def fd_h2(h):
    l0 = lam_q([0, 0])
    H = np.zeros((2, 2))
    H[0, 0] = (lam_q([h, 0]) + lam_q([-h, 0]) - 2 * l0) / (2 * h * h)
    H[1, 1] = (lam_q([0, h]) + lam_q([0, -h]) - 2 * l0) / (2 * h * h)
    H[0, 1] = H[1, 0] = (lam_q([h, h]) + lam_q([-h, -h]) - lam_q([h, -h]) - lam_q([-h, h])) / (8 * h * h)
    return H


rich = lambda h: (4 * fd_h2(h) - fd_h2(2 * h)) / 3
scale = np.abs(H2X).max()
rel = lambda H: max(abs(H[0, 0] - H2X[0, 0]) / abs(H2X[0, 0]), abs(H[1, 1] - H2X[1, 1]) / abs(H2X[1, 1]), abs(H[0, 1] - H2X[0, 1]) / scale)
e_fine, e_coarse = rel(rich(1e-3)), rel(rich(1e-2))

print(f"Band 1 at X: f = {kpX.f0:.6f}, lambda0 = {kpX.lam0:.5f}")
print(f"  H1 (linear coefficient)      : ({H1X[0]:+.1e}, {H1X[1]:+.1e})")
print(f"  H2 = Hessian/2, k.p          : [[{H2X[0, 0]:+.6f}, {H2X[0, 1]:+.1e}], [{H2X[1, 0]:+.1e}, {H2X[1, 1]:+.6f}]]")
for h in (1e-2, 1e-3):
    Hf = rich(h)
    print(f"  Richardson FD, h a = {h:.0e} {'':>3}: [[{Hf[0, 0]:+.6f}, {Hf[0, 1]:+.1e}], [{Hf[1, 0]:+.1e}, {Hf[1, 1]:+.6f}]]   relative difference {rel(Hf):.1e}")
# in frequency units: df = d(lambda)/(2 (2 pi)^2 f0), so d2f/dq2 = Hess/(2 (2 pi)^2 f0)
print(f"  d2f/dqx2 = {HessX[0, 0] / (2 * (2 * np.pi) ** 2 * kpX.f0):+.5f}, d2f/dqy2 = {HessX[1, 1] / (2 * (2 * np.pi) ** 2 * kpX.f0):+.5f}   (a/lambda per (1/a)^2)")

check("H1 at X (no linear term, C2v)", np.abs(H1X).max(), 1e-10)
check("off-diagonal inverse mass at X (C2v)", abs(H2X[0, 1]), 1e-10)
check_in("saddle: H2_xx < 0 < H2_yy (maximum along Gamma-X, minimum along X-M)", float(H2X[0, 0] < 0 < H2X[1, 1]), 1, 1)
check("k.p tensor vs Richardson finite differences, |q| a = 0.01, 0.02 (relative)", e_coarse, 1e-6)
# The finer steps, 1e-3 and 2e-3: round-off limited. Only the one-sided bound the prose relies on is asserted; the value itself varies with BLAS and thread count.
print(f"k.p tensor vs finite differences at the finer steps |q| a = 1e-3, 2e-3: {e_fine:.1e} (the 1e-6 expected beforehand is {'reached' if e_fine <= 1e-6 else 'not reached'}; round-off limited)")
check("k.p tensor vs finite differences at the finer steps |q| a = 1e-3, 2e-3 (round-off limited; one-sided bound)", e_fine, 1e-4)

# %%
# How far does the quadratic form reach? Relative error of lambda - lambda0 along the two axes.
print(f"{'direction':>10} {'|q| a':>7} {'PWE dlambda':>12} {'k.p dlambda':>12} {'rel. error':>11}")
errX = {}
for name, d in (("Gamma-X", np.array([1.0, 0.0])), ("X-M", np.array([0.0, 1.0]))):
    for qa in (0.04, 0.08):
        q = qa * d
        dl = lam_q(q) - kpX.lam0
        pred = q @ H2X @ q
        errX[(name, qa)] = abs(dl - pred) / abs(dl)
        print(f"{name:>10} {qa:7.2f} {dl:12.6f} {pred:12.6f} {errX[(name, qa)]:11.2e}")
rat = [errX[(n, 0.08)] / errX[(n, 0.04)] for n in ("Gamma-X", "X-M")]
print(f"error ratio |q| a = 0.08 vs 0.04: {rat[0]:.3f} (Gamma-X), {rat[1]:.3f} (X-M)")
for n, r_ in zip(("Gamma-X", "X-M"), rat):
    check_in(f"relative error of lambda - lambda0, ratio between |q| a = 0.08 and 0.04, {n} (q^2 scaling gives 4)", r_, 3, 5)

# %% [markdown]
# The $k\cdot p$ inverse-mass tensor of band 1 at $X$ is diagonal and indefinite, as symmetry predicts, and matches the finite differences to better than $10^{-7}$ with the larger steps (printed: $4\times10^{-8}$ in this execution). The finer pair of steps gives about $10^{-5}$ (printed: $8\times10^{-6}$ in this execution; this number varies from run to run, as round-off does), as the round-off estimate in the introduction predicts (the off-diagonal entry, zero by symmetry, comes out near $10^{-7}$ there), so the deviation from the $10^{-6}$ expected beforehand (PLAN.md) is the finite-difference method, not the $k\cdot p$ tensor; the check asserts only the one-sided bound $10^{-4}$, because the value itself depends on BLAS and thread count. The quadratic form reproduces $\lambda-\lambda_0$ along both axes with relative errors $7.6\times10^{-4}$ ($\Gamma X$) and $5.1\times10^{-5}$ ($XM$) at $|\mathbf q|a=0.04$, and these grow by factors $3.995$ and $4.001$ when $|\mathbf q|$ doubles. That is the scaling of an absent cubic term: the absolute error is $O(q^4)$ and the relative error $O(q^2)$. (The ratio 16 would be the absolute-error scaling; the interval $[3,5]$ fixed in advance is the relative one, and the check is stated for the relative error.)

# %% [markdown]
# ## 4. The degenerate edge at M: three constants and a quadratic touching
#
# Let $E$ be the degenerate set at $\mathbf k_0$ with little group $G$ and let $V=\mathbb R^2$ be the plane of $\mathbf q$. Symmetry of the truncated problem under $G$ gives $H(g\mathbf q)=D(g)H(\mathbf q)D(g)^{-1}$ for the effective $d\times d$ matrix, so the coefficient tensors are intertwiners: $\mathbf H_1\in\operatorname{Hom}_G(V,\operatorname{End}E)$ and $H_2\in\operatorname{Hom}_G(\operatorname{Sym}^2V,\operatorname{End}E)$, and the number of independent real constants is the dimension of these spaces (for the representations here, which are real or real-symmetric, the real and complex dimensions agree). By character orthogonality
#
# $$\dim\operatorname{Hom}_G(W,\operatorname{End}E)=\frac1{|G|}\sum_{g}\overline{\chi_W(g)}\,|\chi_E(g)|^2,\qquad \chi_{\operatorname{Sym}^2V}(g)=\tfrac12\big(\chi_V(g)^2+\chi_V(g^2)\big),$$
#
# with $\chi_V(g)=\operatorname{tr}g$ the trace of the $2\times2$ matrix of $g$. At $M$ in $C_{4v}$ the doublet is the irrep $E$, which also is $V$; the sum gives $3$ for $\operatorname{Sym}^2V$ and $0$ for $V$. So there is no linear term, and in a mirror-adapted basis (index 0 odd under $x\to-x$) the Löwdin matrix is
#
# $$H(\mathbf q)=\lambda_0+A\,q^2+B\,(q_x^2-q_y^2)\,\sigma_z+C\,q_xq_y\,\sigma_x,$$
#
# with three real constants. Its eigenvalues are $\lambda_0+Aq^2\pm q^2\sqrt{B^2\cos^22\varphi+\tfrac14C^2\sin^22\varphi}$, a quadratic touching along $\Gamma M$ and $MX$ directions modulated in $\varphi$, with no fit parameter once $A,B,C$ come from perturbation theory. The pseudospin, the Bloch vector $\mathbf n=(\langle\sigma_x\rangle,\langle\sigma_y\rangle,\langle\sigma_z\rangle)$ of the upper band's two-component vector, lies in the $(z,x)$ plane because the matrix is real, and it winds twice around the origin as $\varphi$ goes once around $M$: the angle of $(B\cos2\varphi,\tfrac12C\sin2\varphi)$ turns by $4\pi$. A cubic term of the matrix would be an element of $\operatorname{Hom}_G(\operatorname{Sym}^3V,\operatorname{End}E)$; the character sum below shows that this space is zero too, so the $k\cdot p$ matrix has no cubic term, the eigenvalue error is $O(q^4)$ and the relative error of $\lambda-\lambda_0$ should scale as $q^2$. The vanishing has a one-line reason that does not use $C_{4v}$ in detail: $C_2$ lies in every little group used here, acts as $-1$ on odd-degree polynomials in $\mathbf q$ and trivially on $\operatorname{End}E$, so every odd-order Hom space is zero and the Löwdin matrix is even in $\mathbf q$. The character sum for $\operatorname{Sym}^3V$ is therefore a consistency test of the computed characters (it only probes $|\chi_E|^2$ on $E$ and $C_2$); the discriminating test is the error ratio $4$ (a cubic term would make the relative error scale as $q$ and the ratio $2$).

# %%
# Hom dimensions from the characters of the computed doublet, with the group elements' own 2x2 matrices (independent of bl.hom_dimension).
name, res, ops, G_M, kM, B_M, C_M = irr["square M"]
sol_M = bl.tm_solve(sq, rods_sq, kM, G_M, None)
kpM = bl.kp_second_order(sq, rods_sq, kM, G_M, [1, 2], solution=sol_M)
DsM = bl.rep_matrices(sq, kpM.C, G_M, kM, ops, sol_M[2])
chiE = np.array([np.trace(D) for D in DsM])
gmats = [o["matrix"] for o in ops]
chiV = np.array([np.trace(g) for g in gmats])
chiS2 = np.array([0.5 * (np.trace(g) ** 2 + np.trace(g @ g)) for g in gmats])
chiS3 = np.array([(np.trace(g) ** 3 + 3 * np.trace(g) * np.trace(g @ g) + 2 * np.trace(g @ g @ g)) / 6 for g in gmats])
hom = lambda chiW: float(np.real(np.sum(np.conj(chiW) * np.abs(chiE) ** 2)) / len(ops))
print("Hom dimensions at M (C4v, doublet E): sum over the 8 group elements of conj(chi_W) |chi_E|^2 / 8")
print(f"  W = V          : {hom(chiV):.12f}   (module: {bl.hom_dimension(ops, chiE, 'linear'):.12f})")
print(f"  W = Sym^2 V    : {hom(chiS2):.12f}   (module: {bl.hom_dimension(ops, chiE, 'quadratic'):.12f})")
print(f"  W = Sym^3 V    : {hom(chiS3):.12f}   (cubic terms of the k.p matrix)")
check("Hom_C4v(Sym^2 V, End E): dimension", hom(chiS2), 1e-12, 3)
check("Hom_C4v(V, End E): dimension", hom(chiV), 1e-12, 0)
check("Hom_C4v(Sym^3 V, End E): dimension (consistency test of the computed characters: C2 acts as -1 on odd powers of q)", hom(chiS3), 1e-12, 0)
check("character formula equals bl.hom_dimension", abs(hom(chiS2) - bl.hom_dimension(ops, chiE, "quadratic")) + abs(hom(chiV) - bl.hom_dimension(ops, chiE, "linear")), 1e-12)

mx = [D for o, D in zip(ops, DsM) if np.allclose(o["matrix"], np.diag([-1.0, 1.0]))][0]
U, H1a, H2a = bl.adapt_doublet(kpM, mx)
A_M, B_M_, C_M_, resid = bl.c4v_constants(H2a)
print(f"\nSymmetry-adapted constants (lambda units, a = 1): A = {A_M:.5f}, B = {B_M_:.5f}, C = {C_M_:.5f}")
print(f"  ||H1|| on the doublet = {np.linalg.norm(kpM.H1):.1e};  residual of H2 after projection onto the three allowed matrices = {resid:.1e}")
check("no linear term: ||H1|| restricted to the doublet", np.linalg.norm(kpM.H1), 1e-10)
check("three-parameter form: residual of H2 after projection onto {1, (qx^2 - qy^2) sz, qx qy sx}", resid, 1e-10)

# %%
# k.p against the PWE on circles around M: 24 angles in [0, pi), both branches.
print(f"{'|q| a':>6} {'max rel. error of lambda - lambda0':>36}")
errs, dl_by_q = {}, {}
for qa in (0.02, 0.04, 0.08):
    e_, rows_ = [], []
    for phi in np.linspace(0, np.pi, 24, endpoint=False):
        q = qa * np.array([np.cos(phi), np.sin(phi)])
        fe = bl.tm_solve(sq, rods_sq, kM + q, G_M, 3, B=B_M)[0][1:3]
        dl = lam_of(fe) - kpM.lam0
        s_ = qa ** 2 * np.sqrt(B_M_ ** 2 * np.cos(2 * phi) ** 2 + C_M_ ** 2 / 4 * np.sin(2 * phi) ** 2)
        pred = A_M * qa ** 2 + np.array([-1, 1]) * s_
        e_.append(np.abs(dl - pred) / np.abs(dl))
        rows_.append((phi, dl))
    errs[qa] = np.max(e_)
    dl_by_q[qa] = rows_
    print(f"{qa:6.2f} {errs[qa]:36.2e}")
print(f"ratio of the errors, |q| a = 0.08 over 0.04: {errs[0.08] / errs[0.04]:.3f}")
check_in("k.p prediction of lambda - lambda0 at |q| a = 0.04 (max relative error, both branches, 24 angles) <= 1e-2", errs[0.04], 0, 1e-2)
check_in("error ratio between |q| a = 0.08 and 0.04 (relative error; q^2 scaling gives 4)", errs[0.08] / errs[0.04], 3, 5)

th, nvec = bl.pseudospin_curve(sq, rods_sq, kM, 0.04, 2, nk=64, G=G_M, doublet=kpM.C, U=U)
wn = bl.winding_number(nvec)
print(f"pseudospin of the upper band on |q| a = 0.04 (64 points): winding number {wn:+.4f}; max |n_y| = {np.abs(nvec[:, 1]).max():.1e}; best-fit-plane winding {bl.winding_number(nvec, axes=None):.4f}")
check("winding number of the pseudospin: | |w| - 2 |", abs(abs(wn) - 2), 0.05)
check("pseudospin stays in the (z, x) plane: max |n_y|", np.abs(nvec[:, 1]).max(), 1e-8)

# %% [markdown]
# The character sums give 3, 0 and 0 for $\operatorname{Sym}^2V$, $V$ and $\operatorname{Sym}^3V$, equal to the module's values; they are computed here from the traces of the computed doublet and the $2\times2$ matrices of the group elements. The projection of the second-order matrix onto the three allowed forms leaves a residual of about $10^{-14}$ and the linear matrix vanishes to a few $10^{-13}$, so the symmetry-allowed form is not an approximation of the numerics: the exact $k\cdot p$ matrix has no component outside it. The constants are $A=0.0233$, $B=0.417$ and $C=0.279$ in $\lambda$ units. $B\neq C/2$ means that the touching is anisotropic: along the axes ($\varphi=0$) the two branches are $A\pm B$, along the diagonals $A\pm C/2$ (Figure 2, left; the dots are the PWE and the lines the prediction without any fit).
#
# The prediction error at $|\mathbf q|a=0.04$ is $1.0\times10^{-3}$ relative to $\lambda-\lambda_0$ and quadruples when $|\mathbf q|$ doubles, as for the nondegenerate edge. The pseudospin of the upper band on the circle $|\mathbf q|a=0.04$ (64 points) winds exactly twice ($+2.0000$) and stays in the $(z,x)$ plane to $10^{-16}$ (Figure 2, middle: the pseudospin angle against the direction of $\mathbf q$, from the PWE and from $B$ and $C$ alone). This is the quadratic band touching of the $E$ doublet of $C_{4v}$. For a real $2\times2$ matrix the band's Berry phase is $\pi$ times the winding, here $2\pi$, trivial modulo $2\pi$; that is not computed here.

# %% [markdown]
# ## 5. Breaking C4: elliptical rods
#
# Replace the disk by an area-preserving ellipse with semi-axes $R(1+e)^{1/2}$ along $x$ and $R(1+e)^{-1/2}$ along $y$. The group at $M$ drops from $C_{4v}$ to $C_{2v}$, under which the doublet splits into two one-dimensional irreps (odd and even under $x\to-x$), and the splitting is linear in $e$ because a perturbation $\delta B=e\,\partial_eB$ that transforms like $q_x^2-q_y^2$ (the $B_1$ component of $\operatorname{Sym}^2V$ at $C_{4v}$) has a nonzero matrix element between the two states. In the generalized problem only $B$ changes, and first-order degenerate perturbation theory (at fixed $A$) gives the $2\times2$ matrix $-\lambda_0\,C^H\,\partial_eB\,C$ in the doublet; $\partial_eB$ is a central difference of the exact Fourier matrix $B(e)$ with step $10^{-4}$. The splitting is measured in $\lambda$ units.

# %%
rodsE = lambda e: [bl.ellipse((0, 0), R_SQ * np.sqrt(1 + e), R_SQ / np.sqrt(1 + e), EPS)]
f0, C0, _ = bl.tm_solve(sq, rodsE(0.0), kM, G_M, 3)
h = 1e-4
dBe = (bl.eps_matrix(sq, rodsE(h), G_M) - bl.eps_matrix(sq, rodsE(-h), G_M)) / (2 * h)
ev = np.linalg.eigvalsh(bl.degenerate_first_order(lam_of(f0[1]), C0[:, 1:3], dBe))
slope_pt = ev[1] - ev[0]
spl = {}
print(f"{'e':>7} {'splitting of lambda':>20} {'splitting / e':>14} {'ratio to first-order slope':>27}")
for e in (0.005, 0.01, 0.02, 0.04):
    ff = bl.tm_solve(sq, rodsE(e), kM, G_M, 3)[0]
    spl[e] = lam_of(ff[2]) - lam_of(ff[1])
    print(f"{e:7.3f} {spl[e]:20.6f} {spl[e] / e:14.5f} {spl[e] / e / slope_pt:27.4f}")
print(f"first-order degenerate perturbation theory: slope = {slope_pt:.5f} (eigenvalues {ev[0]:+.5f}, {ev[1]:+.5f} per unit e)")
check("first-order slope: PT vs PWE splitting/e at e = 0.005 (relative)", abs(spl[0.005] / 0.005 / slope_pt - 1), 1e-2)
check_in("linearity: splitting(0.01)/splitting(0.005)", spl[0.01] / spl[0.005], 1.95, 2.05)
check("the two first-order shifts are opposite: |sum| / slope (the perturbation dB/de is traceless in the doublet)", abs(ev[0] + ev[1]) / slope_pt, 1e-6)

# Figure 2: the doublet at M.
fig, ax = plt.subplots(1, 3, figsize=(11.8, 4.6))
phi_f = np.linspace(0, np.pi, 200)
sroot = lambda phi: np.sqrt(B_M_ ** 2 * np.cos(2 * phi) ** 2 + C_M_ ** 2 / 4 * np.sin(2 * phi) ** 2)
ph = np.array([r[0] for r in dl_by_q[0.04]])
dl = np.array([r[1] for r in dl_by_q[0.04]]) / 0.04 ** 2
ax[0].plot(phi_f / np.pi, A_M + sroot(phi_f), color=CAT[1], label=r"$k\cdot p$: $A\pm\sqrt{B^2\cos^22\varphi+C^2\sin^22\varphi/4}$")
ax[0].plot(phi_f / np.pi, A_M - sroot(phi_f), color=CAT[1])
ax[0].plot(ph / np.pi, dl[:, 0], "o", color=CAT[0], ms=4, label="PWE, $|q|a=0.04$")
ax[0].plot(ph / np.pi, dl[:, 1], "o", color=CAT[0], ms=4)
ax[0].set_xlabel(r"direction $\varphi/\pi$ of $\mathbf{q}$ ($\varphi=0$: $M\to X$)")
ax[0].set_ylabel(r"$(\lambda-\lambda_0)/q^2$")
ax[0].set_title("two bands near the touching")
ax[0].legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.2))
ang = np.unwrap(np.arctan2(nvec[:, 0], nvec[:, 2]))                      # pseudospin angle in the (z, x) plane
phf = np.linspace(0, 2 * np.pi, 200)
ax[1].plot(phf / np.pi, np.unwrap(np.arctan2(0.5 * C_M_ * np.sin(2 * phf), B_M_ * np.cos(2 * phf))) / np.pi, color=CAT[1], label="from $B$, $C$ (no fit)")
ax[1].plot(th / np.pi, ang / np.pi, "o", color=CAT[0], ms=4, label="PWE, upper band")
ax[1].set_xlabel(r"direction $\varphi/\pi$ of $\mathbf{q}$ around $M$")
ax[1].set_ylabel(r"pseudospin angle $/\pi$")
ax[1].set_title(f"pseudospin of the upper band, winding {wn:+.0f}")
ax[1].legend(fontsize=8, loc="upper left")
es = np.array(sorted(spl))
ax[2].plot(es, [spl[e] / e for e in es], "o-", color=CAT[0], label="PWE, $C_{2v}$")
ax[2].axhline(slope_pt, color=CAT[1], ls="--", label="first-order degenerate PT")
ax[2].set_xlabel("ellipticity $e$")
ax[2].set_ylabel(r"splitting of $\lambda$ / $e$")
ax[2].set_ylim(4.4, 4.9)
ax[2].set_title("breaking $C_4$: linear splitting")
ax[2].legend(fontsize=8, loc="lower left")
plt.show()

# %% [markdown]
# The splitting of $\lambda$ is linear in $e$ at the sizes tried: the ratio of the splittings at $e=0.01$ and $0.005$ is $1.995$, and the first-order perturbation theory, which uses only the unperturbed doublet and $\partial_eB$, predicts the slope to $2.5\times10^{-3}$ at $e=0.005$ (Figure 2, right). The ratio of the splitting$/e$ to the first-order slope falls from $0.9975$ to $0.9807$ between $e=0.005$ and $0.04$ (table above), the second-order term of the series. The first-order shifts $\pm2.3786$ per unit $e$ sum to zero because the mode odd and the mode even under $x\to-x$ are shifted in opposite directions, as the $q_x^2-q_y^2$ symmetry of $\partial_eB$ requires.

# %% [markdown]
# ## 6. Dirac cone at K and its Berry phase
#
# At $K$ the little group is $C_{3v}$ and the doublet is again the irrep $E$, now $\cong V$ as a representation of $C_{3v}$. The character sum gives $\dim\operatorname{Hom}_{C_{3v}}(V,\operatorname{End}E)=\tfrac16(1\cdot2\cdot4+2\cdot(-1)\cdot1+3\cdot0)=1$: the first-order matrix is $\mathbf H_1\cdot\mathbf q=v\,(q_x\sigma_x+q_y\sigma_y)$ in a suitable basis, with one real constant $v$. The symmetry fixes the form of the matrix; it does not fix $v\ne0$, which is a computed fact here ($v=0.58129$ in $\lambda$ units for equal rods), not a consequence of the Hom count. The degeneracy itself is protected because the two bands form the irrep $E$ (shown by the character projection of section 2). The statement is the one of Fefferman and Weinstein (2012) for $-\Delta+V$ with honeycomb symmetry, where $v\ne0$ is a nontrivial hypothesis. Here the operator is the generalized TM problem $-\Delta u=\lambda\varepsilon u$, not the Schrödinger operator, so their theorem is not what is used: what carries over is the symmetry argument (the dimension of the Hom space), and the theorem serves as an analogy, not as a proof.
#
# **Berry phase.** The honeycomb lattice with equal rods is invariant under inversion about the hexagon center at the origin, so $\hat\varepsilon(\mathbf G)$ is real, $B$ is real symmetric, $A$ is diagonal, and the eigenvectors can be chosen real for every $\mathbf k$ and every fixed plane-wave set. The Wilson loop of real vectors is a product of real overlaps, hence $\pm1$, and the phase is $0$ or $\pi$ by construction; around a Dirac point it is $\pi$ because the real eigenvector changes sign. This is a test of the code (and of the fixed-set construction), not of the physics: it holds for any real symmetric pencil. The physics check is the unequal-rod lattice, which breaks inversion, opens a gap $2m$ at $K$ and makes the phase non-quantized: for $H=v(q_x\sigma_x+q_y\sigma_y)+m\sigma_z$, the lower-band eigenvector at radius $r$ has polar angle $\theta$ with $\cos\theta=m/\sqrt{m^2+v^2r^2}$ on the Bloch sphere, and the loop encloses the solid angle $\Omega=2\pi(1-\cos\theta)$, so the magnitude of the Berry phase ($\Omega/2$) is
#
# $$|\gamma(r)|=\pi\Big(1-\frac{m}{\sqrt{m^2+v^2r^2}}\Big),$$
#
# with opposite signs at $K$ and $K'$ (time reversal maps $K$ to $K'$ and reverses the curvature; the mass term is the same at both valleys). The formula neglects every $O(q^2)$ term of the matrix and the trigonal warping, so it must fail at large enough $r$; the expectation recorded beforehand (PLAN.md) was agreement within $0.03\pi$ up to $r=3m/v$, with the deviation scaling as $r^2$. The unequal-rod run tests that, and shows the deviation divided by $r^2$, the second-order $k\cdot p$ loop phase (which keeps $H_2$) and a third loop built from the PWE eigenvector projected onto the doublet at $K$ (renormalized; the same plane-wave set) next to it. The projected loop is a doublet-only Berry connection but with the true eigenvector's doublet components, so it separates the part of the deviation that comes from the Bloch vector outside the doublet (remote-band admixture, which no matrix on the doublet can represent) from the doublet-only part. Loops are discrete Wilson products $-\arg\prod_j\langle C_j|B|C_{j+1}\rangle$ on one fixed plane-wave set around the loop center, with 256 points for the table (64 for the $K'$ comparison) and the convergence in the number of points and plane waves measured below; $m$ is half the $\lambda$ splitting at $K$ and $v$ is the first-order velocity in $\lambda$ units of the lattice being tested (for unequal rods $0.58261$, not the equal-rod $0.58129$); the two are converted to frequency units only for the comparison with the PWE slope.

# %%
NPW = 200
K, Kp = hc.points["K"], hc.points["Kp"]
GK, GKp = gset(hc, NPW, K), gset(hc, NPW, Kp)
solK = bl.tm_solve(hc, rods_h, K, GK, None)
kpK = bl.kp_second_order(hc, rods_h, K, GK, [0, 1], solution=solK)
opsK, nameK = bl.point_group_ops(hc, K, rods_h)
DsK = bl.rep_matrices(hc, kpK.C, GK, K, opsK, solK[2])
chiEK = np.array([np.trace(D) for D in DsK])
gK = [o["matrix"] for o in opsK]
homK = lambda chiW: float(np.real(np.sum(np.conj(chiW) * np.abs(chiEK) ** 2)) / len(opsK))
hV = homK(np.array([np.trace(g) for g in gK]))
hS2 = homK(np.array([0.5 * (np.trace(g) ** 2 + np.trace(g @ g)) for g in gK]))
print(f"Little group at K: {nameK}; dim Hom(V, End E) = {hV:.12f}, dim Hom(Sym^2 V, End E) = {hS2:.12f}")
check("Hom_C3v(V, End E): dimension", hV, 1e-12, 1)
check("character formula equals bl.hom_dimension (C3v)", abs(hV - bl.hom_dimension(opsK, chiEK, "linear")), 1e-12)

evx, evy = np.linalg.eigvalsh(kpK.H1[0]), np.linalg.eigvalsh(kpK.H1[1])
v_lam = evx[1]
f0K = kpK.f0
v_f = v_lam / (2 * (2 * np.pi) ** 2 * f0K)                  # df/dq, (a/lambda) per (1/a)
sl = []
for q in (0.002, -0.002):
    ff = bl.tm_solve(hc, rods_h, K + np.array([q, 0.0]), GK, 2)[0]
    sl.append((ff[1] - ff[0]) / abs(q) / 2)
print(f"Dirac velocity: v = {v_lam:.5f} (lambda units), {v_f:.5f} (f units); eigenvalues of H1_x: {evx[0]:+.5f}, {evx[1]:+.5f}; of H1_y: {evy[0]:+.5f}, {evy[1]:+.5f}")
print(f"PWE slope at |q| a = 0.002: +q {sl[0]:.5f}, -q {sl[1]:.5f}, average {np.mean(sl):.5f}; single direction off by {abs(sl[0] / v_f - 1):.1e}, average off by {abs(np.mean(sl) / v_f - 1):.1e}")
check("H1_x and H1_y have the same eigenvalues +-v (isotropic cone)", np.abs(evx - evy).max(), 1e-10)
check("Dirac velocity: k.p vs PWE slope averaged over +q and -q (relative)", abs(np.mean(sl) / v_f - 1), 1e-3)

print("\nEqual rods, Wilson loops of band 1 (64 points): |phase|/pi - 1")
print(f"{'r a':>6} {'K':>12} {'K prime':>12}")
ph_eq = []
for r in (0.01, 0.02, 0.04):
    pk = bl.berry_phase(hc, rods_h, K, r, 0, nk=64, G=GK)
    pkp = bl.berry_phase(hc, rods_h, Kp, r, 0, nk=64, G=GKp)
    ph_eq.append((abs(abs(pk) - np.pi), abs(abs(pkp) - np.pi)))
    print(f"{r:6.2f} {abs(pk) / np.pi - 1:12.2e} {abs(pkp) / np.pi - 1:12.2e}")
check("equal rods (test of the code, exact by construction): max ||phase| - pi|", np.max(ph_eq), 1e-4)
check("both Dirac bands together: |phase| (trivial bundle)", abs(bl.berry_phase(hc, rods_h, K, 0.02, [0, 1], nk=64, G=GK)), 1e-6)

# %%
# Unequal rods (radii 0.21 and 0.19): a massive Dirac point.
solU = bl.tm_solve(hc, rods_h_un, K, GK, 4)
m_lam = 0.5 * (lam_of(solU[0][1]) - lam_of(solU[0][0]))
solUp = bl.tm_solve(hc, rods_h_un, Kp, GKp, 2)
kpU = bl.kp_second_order(hc, rods_h_un, K, GK, [0, 1])
vU = np.linalg.eigvalsh(kpU.H1[0])
v_un = vU[1]                                   # the unequal-rod velocity: used in the closed formula and for the radii below
r0 = m_lam / v_un
print(f"Unequal rods: gap at K, f2 - f1 = {solU[0][1] - solU[0][0]:.5f}; at K' {solUp[0][1] - solUp[0][0]:.5f}; m = {m_lam:.5f}")
print(f"  H1 of the unequal-rod doublet: eigenvalues of H1_x = {vU[0]:+.5f}, {vU[1]:+.5f} (equal-rod v = {v_lam:.5f}); the unequal-rod velocity v = {v_un:.5f} is {v_un / v_lam - 1:.2%} larger and is used below")
print(f"  m/v = {r0:.4f} (1/a) with the unequal-rod v (with the equal-rod v: {m_lam / v_lam:.4f})")
check("unequal rods: gap at K' equals gap at K (time reversal)", abs((solUp[0][1] - solUp[0][0]) - (solU[0][1] - solU[0][0])), 1e-9)

BU = bl.eps_matrix(hc, rods_h_un, GK)


def loop_data(r, nk, G=GK, B=BU, doublet=kpU.C, rods=rods_h_un, center=K):
    """Wilson loop of the lower band on a circle of radius r around the center, from the full PWE vector and from the same vector projected
    onto the doublet at the center (renormalized). Same discrete product as bl.berry_phase. Returns (full phase, projected phase,
    mean weight of the PWE vector outside the doublet)."""
    ths = 2 * np.pi * np.arange(nk) / nk
    Cs = [bl.tm_solve(hc, rods, center + r * np.array([np.cos(t), np.sin(t)]), G, 1, B=B)[1][:, 0] for t in ths]
    ps = [doublet.conj().T @ (B @ C) for C in Cs]
    wt = float(np.mean([1 - np.vdot(p_, p_).real for p_ in ps]))
    ps = [p_ / np.linalg.norm(p_) for p_ in ps]
    pf, pp = 1.0 + 0j, 1.0 + 0j
    for j in range(nk):
        pf *= np.vdot(Cs[j], B @ Cs[(j + 1) % nk])
        pp *= np.vdot(ps[j], ps[(j + 1) % nk])
    return -np.angle(pf), -np.angle(pp), wt


NK_MAIN = 256
MULT = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
tab = []
for mult in MULT:
    r = mult * r0
    pK64 = bl.berry_phase(hc, rods_h_un, K, r, 0, nk=64, G=GK)
    pKp64 = bl.berry_phase(hc, rods_h_un, Kp, r, 0, nk=64, G=GKp)
    pK, pP, wt = loop_data(r, NK_MAIN)
    form = np.pi * (1 - m_lam / np.sqrt(m_lam ** 2 + v_un ** 2 * r ** 2))
    p1 = bl.kp_berry_phase(kpU, r, 0, order=1)
    p2 = bl.kp_berry_phase(kpU, r, 0, order=2)
    tab.append(dict(mult=mult, r=r, pK64=pK64, pKp64=pKp64, pK=pK, pP=pP, wt=wt, form=form, p1=p1, p2=p2))
chk64 = loop_data(r0, 64)[0] - bl.berry_phase(hc, rods_h_un, K, r0, 0, nk=64, G=GK)
check("the loop helper reproduces bl.berry_phase at nk = 64 (interface identity, r = m/v)", abs(chk64), 1e-9)
print(f"\nWilson loops of the lower band, K, {NK_MAIN} points, 200 plane waves; closed formula with m and the unequal-rod v (phases in units of pi)")
print(f"{'r':>7} {'r a':>6} {'PWE':>8} {'projected':>10} {'formula':>8} {'k.p H1':>8} {'k.p H1+H2':>10}")
for t in tab:
    t["dev"] = abs(abs(t["pK"]) - t["form"]) / np.pi
    t["dev2"] = abs(abs(t["p2"]) - abs(t["pK"])) / np.pi
    t["remote"] = abs(abs(t["pK"]) - abs(t["pP"])) / np.pi          # part of the deviation from the Bloch vector outside the doublet
    t["doub"] = abs(abs(t["pP"]) - t["form"]) / np.pi               # doublet-only part
    print(f"{t['mult']:5.1f}m/v {t['r']:6.3f} {t['pK'] / np.pi:+8.4f} {t['pP'] / np.pi:+10.4f} {t['form'] / np.pi:8.4f} {abs(t['p1']) / np.pi:8.4f} {abs(t['p2']) / np.pi:10.4f}")
print(f"\n{'r':>7} {'|PWE-formula|':>14} {'/ r^2':>8} {'|PWE-proj|':>11} {'|proj-formula|':>15} {'remote share':>13} {'|PWE-kp2|':>10} {'|proj-kp2|':>11} {'weight outside':>15}")
for t in tab:
    t["dev3"] = abs(abs(t["pP"]) - abs(t["p2"])) / np.pi
    print(f"{t['mult']:5.1f}m/v {t['dev']:14.4f} {t['dev'] / t['r'] ** 2:8.4f} {t['remote']:11.4f} {t['doub']:15.4f} {t['remote'] / t['dev']:13.2f} {t['dev2']:10.4f} {t['dev3']:11.4f} {t['wt']:15.2e}")

# %%
# Convergence of the loops: number of loop points (64 against 256) and number of plane waves (200 against 400, same radius, 64 points).
print("Wilson-loop discretization and truncation, |change of the PWE phase| in units of pi")
print(f"{'r':>7} {'nk 64 -> 256':>13} {'nk 256 -> 512':>14} {'200 -> 400 pw (nk 64)':>23} {'|PWE-formula| at nk 64':>24}")
G400 = gset(hc, 400, K)
B400 = bl.eps_matrix(hc, rods_h_un, G400)
C400 = bl.tm_solve(hc, rods_h_un, K, G400, 2, B=B400)[1]
PW_MULTS = (0.5, 1.0, 3.0)
NK_MULTS = (0.5, 1.0)
dnk, dnk2, dpw = {}, {}, {}
for t in tab:
    dnk[t["mult"]] = abs(abs(t["pK"]) - abs(t["pK64"])) / np.pi
    dev64 = abs(abs(t["pK64"]) - t["form"]) / np.pi
    c512 = c400 = ""
    if t["mult"] in NK_MULTS:
        dnk2[t["mult"]] = abs(abs(loop_data(t["r"], 512)[0]) - abs(t["pK"])) / np.pi
        c512 = f"{dnk2[t['mult']]:.1e}"
    if t["mult"] in PW_MULTS:
        p400 = loop_data(t["r"], 64, G=G400, B=B400, doublet=C400)[0]
        dpw[t["mult"]] = abs(abs(p400) - abs(t["pK64"])) / np.pi
        c400 = f"{dpw[t['mult']]:.1e}"
    print(f"{t['mult']:5.1f}m/v {dnk[t['mult']]:13.1e} {c512:>14} {c400:>23} {dev64:24.4f}")
dev_min = min(t["dev"] for t in tab)
print(f"smallest deviation to be resolved: {dev_min:.4f}; 64 -> 256 points changes the phase by up to {max(dnk.values()):.1e} ({max(dnk.values()) / dev_min:.0%} of it); "
      f"256 -> 512 by up to {max(dnk2.values()):.1e}; 200 -> 400 plane waves by up to {max(dpw.values()):.1e}")
CONV_TOL = 0.25 * dev_min
check("loop discretization of the loops used (256 points), |phase(256) - phase(512)| at r = m/2v and m/v, below a quarter of the smallest deviation resolved (units of pi)", max(dnk2.values()), CONV_TOL)
check("plane-wave truncation, |phase(200) - phase(400)| at r = m/2v, m/v, 3m/v, below a quarter of the smallest deviation resolved (units of pi)", max(dpw.values()), CONV_TOL)

# %%
D = {t["mult"]: t for t in tab}
for t in tab:
    check(f"valley contrast (exact symmetry of the discrete scheme: test of the code), r = {t['mult']} m/v: |phase(K) + phase(K')| mod 2 pi", abs(np.angle(np.exp(1j * (t["pK64"] + t["pKp64"])))), 1e-6)
    check(f"k.p with H1 only reproduces the closed formula (sanity of the formula), r = {t['mult']} m/v (in units of pi)", abs(abs(t["p1"]) - t["form"]) / np.pi, 2e-3)
    check(f"second-order k.p vs PWE loop phase, r = {t['mult']} m/v (in units of pi)", t["dev2"], 0.015)
for mult in (0.5, 1.0, 2.0):
    check(f"massive Dirac loop phase vs formula, r = {mult} m/v (in units of pi)", D[mult]["dev"], 0.03)
# The expected bound was 0.03 pi at 3 m/v as well. Found: it fails there (one-sided: the bound the prose relies on).
check_in("massive Dirac loop phase vs formula, r = 3 m/v: the expected 0.03 pi is NOT met (deviation in units of pi)", D[3.0]["dev"], 0.03, 1.0)
# Attribution test: how much of the deviation at the smallest radius is the part of the Bloch vector outside the doublet?
check_in("remote-band share of the deviation at r = m/2v: |PWE - projected| / |PWE - formula| (a majority, i.e. >= 0.5)", D[0.5]["remote"] / D[0.5]["dev"], 0.5, 1.0)
check_in("signs at K and K' are opposite for every radius (product of the two phases < 0)", float(all(t["pK64"] * t["pKp64"] < 0 for t in tab)), 1, 1)

# The expectation: deviation / r^2 constant. Local log-log slopes of the deviation (2 would be the r^2 law) and the spread of deviation / r^2.
devs, rs = np.array([t["dev"] for t in tab]), np.array([t["r"] for t in tab])
loc = np.diff(np.log(devs)) / np.diff(np.log(rs))
print("\nlocal log-log slopes of |PWE - formula| between consecutive radii (r^2 law: 2): " + ", ".join(f"{x:.2f}" for x in loc))
q4 = [t["dev"] / t["r"] ** 2 for t in tab if t["mult"] in (0.5, 1.0, 2.0, 3.0)]
q6 = [t["dev"] / t["r"] ** 2 for t in tab]
print(f"deviation / r^2 over the four radii 0.5, 1, 2, 3 m/v: {', '.join(f'{x:.4f}' for x in q4)}; max/min = {max(q4) / min(q4):.3f}; over all six radii: max/min = {max(q6) / min(q6):.3f}")
print(f"loop radius at 3 m/v: r a = {tab[-1]['r']:.3f}, which is {tab[-1]['r'] / np.linalg.norm(K):.1%} of |K| = 4 pi/3")

# Figure 3
rg = np.linspace(0.02, 0.8, 40)
p2g = np.array([abs(bl.kp_berry_phase(kpU, r, 0, order=2)) / np.pi for r in rg])
p1g = np.array([abs(bl.kp_berry_phase(kpU, r, 0, order=1)) / np.pi for r in rg])
fg_ = 1 - m_lam / np.sqrt(m_lam ** 2 + v_un ** 2 * rg ** 2)
fig, ax = plt.subplots(1, 2, figsize=(10.6, 4.0))
ax[0].plot(rg, fg_, color=CAT[1], label=r"leading order $1-m/\sqrt{m^2+v^2r^2}$")
ax[0].plot(rg, p2g, color=CAT[2], ls="--", label=r"second-order $k\cdot p$ loop")
ax[0].plot([t["r"] for t in tab], [abs(t["pK"]) / np.pi for t in tab], "o", color=CAT[0], label="PWE, $K$")
ax[0].plot([t["r"] for t in tab], [abs(t["pP"]) / np.pi for t in tab], "+", color=CAT[3], ms=9, label="PWE vector projected on the doublet")
ax[0].plot([t["r"] for t in tab], [abs(t["pKp64"]) / np.pi for t in tab], "x", color=CAT[6], label="PWE, $K'$ (magnitude, 64 points)")
ax[0].set_xlabel("loop radius $r\\,a$")
ax[0].set_ylabel(r"$|\gamma|/\pi$")
ax[0].legend(loc="lower right", fontsize=8)
ax[0].set_title("unequal rods: loop phase of the lower band")
ax[1].plot([t["r"] for t in tab], [t["dev"] / t["r"] ** 2 for t in tab], "o-", color=CAT[1], label="PWE vs leading order")
ax[1].plot([t["r"] for t in tab], [t["doub"] / t["r"] ** 2 for t in tab], "+-", color=CAT[3], ms=9, label="projected vs leading order (doublet only)")
ax[1].plot([t["r"] for t in tab], [t["dev2"] / t["r"] ** 2 for t in tab], "s-", color=CAT[2], label="PWE vs second-order $k\\cdot p$")
ax[1].set_xlabel("loop radius $r\\,a$")
ax[1].set_ylabel(r"deviation$/\pi\,/\,(r\,a)^2$")
ax[1].set_ylim(bottom=0)
ax[1].legend(loc="lower right", fontsize=8)
ax[1].set_title("deviations scaled by $r^2$")
plt.show()
print(f"[{time.perf_counter() - T_START:.1f} s elapsed]")

# %% [markdown]
# *Figure 3. Left: magnitude of the Berry phase of the lower band around $K$ for the unequal-rod lattice, from the PWE (dots), the closed formula for $H=v\,\mathbf q\cdot\boldsymbol\sigma+m\sigma_z$ (solid) and the $k\cdot p$ matrix with its second-order term (dashed); the $K'$ values (crosses, 64-point loops) coincide in magnitude; all other PWE values use 256 points. Plus signs: the PWE eigenvector projected on the doublet at $K$. Right: the deviations divided by $r^2$, the part that remains after projecting on the doublet, and the deviation from the second-order $k\cdot p$ loop.*
#
# **Velocity.** The $k\cdot p$ velocity of the equal-rod lattice is the same for $q_x$ and $q_y$ (eigenvalues of $H_{1,x}$ and $H_{1,y}$ equal to better than $10^{-13}$), isotropic as $\dim\operatorname{Hom}=1$ requires. Converted to frequency units it is $0.0273$ per $1/a$, and it agrees with the PWE slope to better than $10^{-7}$ once $+\mathbf q$ and $-\mathbf q$ are averaged; one direction alone differs by $7.5\times10^{-4}$, because the trigonal-warping term enters at order $q^2$ with a sign that flips between $+\mathbf q$ and $-\mathbf q$ ($C_{3v}$ has no inversion about $K$), so the average cancels it.
#
# **Equal rods.** The Wilson loops return $\pi$ to machine precision for all three radii and both valleys. As argued above this is a property of the real symmetric pencil, so it checks that the code builds the loop correctly (fixed plane-wave set, $B$-weighted overlaps), and the Wilson loop of both Dirac bands together gives zero.
#
# **Unequal rods.** The gap at $K$ and $K'$ is the same (time reversal) and the 64-point phases at $K$ and $K'$ are opposite to better than $10^{-12}$ at all radii. That agreement is an exact symmetry of the discrete scheme, not a physics test: for real $\varepsilon$ the circular plane-wave sets around $K$ and $K'=-K$ are inversion images, so the loop at $K'$ is the complex conjugate of the loop at $K$. Like the equal-rod case it tests the code and its time-reversal consistency; the physical content is that the sign of the Berry phase flips between valleys while the gap does not, and that the magnitude is not quantized. The $k\cdot p$ matrix with only the first-order term reproduces the closed formula to better than $3.1\times10^{-4}\pi$ at every radius once the formula uses the velocity of the unequal-rod matrix ($0.58261$, $0.23\%$ above the equal-rod value; with the equal-rod value the remainder had been up to $5.6\times10^{-4}\pi$), so the formula is right for the linear model; the rest of that remainder was not traced. Radii are in units of $m/v$ with this velocity, $m/v=0.2259/a$.
#
# Against the PWE, the deviation (table above, column $|\mathrm{PWE}-\mathrm{formula}|$, 256-point loops) grows from $0.0011\pi$ at $r=m/2v$ to $0.0345\pi$ at $3m/v$, and the expected bound of $0.03\pi$ (PLAN.md) holds up to $2m/v$ ($0.0200\pi$) and fails at $3m/v$. The expectation that deviation$/r^2$ is constant fails too: it is $0.089, 0.108, 0.108, 0.098, 0.086, 0.075$ at the six radii, a factor $1.44$ between largest and smallest (the same factor for the four radii $0.5,1,2,3\,m/v$ and for all six), and the local log-log slope of the deviation falls from $2.29$ to $1.25$ across the range; no single $r^2$ coefficient describes it. The loop at $3m/v$ has $ra=0.678$, which is $16.2\%$ of $|K|$.
#
# *Where the deviation comes from.* The projected-vector loop gives a direct test. The part of the Bloch vector outside the doublet has a weight of $5.2\times10^{-4}$ at $m/2v$ and $1.7\times10^{-2}$ at $3m/v$, and the difference $|\mathrm{PWE}-\mathrm{projected}|$ is $0.0008\pi$ of the $0.0011\pi$ deviation at $m/2v$ ($73\%$, the check asserts at least half), $47\%$ at $m/v$, and $30\%$ at $3m/v$. At the smallest radius the deviation is therefore mostly remote-band admixture, not trigonal warping; at the largest radius it is mostly a doublet-only effect ($0.0240\pi$ of $0.0345\pi$). The doublet-only part is reproduced by the second-order $k\cdot p$ matrix: $|\mathrm{projected}-\mathrm{kp2}|\le0.0013\pi$ at every radius against a doublet-only deviation up to $0.0240\pi$. Consequently the second-order $k\cdot p$ loop does not improve on the leading order at $m/2v$ ($0.0008\pi$ against $0.0011\pi$, both limited by the remote part) and improves it only where the doublet-only part dominates ($0.0113\pi$ against $0.0345\pi$ at $3m/v$); its residual against the PWE is essentially the remote part. That the doublet-only part is due to trigonal warping rather than the other terms of $H_2$ was not tested, and stays a conjecture. *Convergence.* Going from 64 to 256 loop points changes the phase by up to $2.5\times10^{-4}\pi$ (22% of the smallest deviation, which is why the table uses 256 points), 256 to 512 points by at most $1.2\times10^{-5}\pi$ (radii $m/2v$ and $m/v$), and 200 to 400 plane waves by at most $1.1\times10^{-5}\pi$ (radii $m/2v$, $m/v$, $3m/v$, 64 points). So the formula for the massive Dirac loop phase is a small-$r$ statement, and the physics check is the existence of this decomposition, not agreement at one radius.

# %% [markdown]
# ## 7. Finite stacks: the envelope equation and its boundary condition (1D, exact)
#
# The 1D counterpart is exactly solvable, which makes the boundary condition visible. Take the symmetric cell $[B\ 0.35a\,|\,A\ 0.3a\,|\,B\ 0.35a]$ with $\varepsilon_A=12$, $\varepsilon_B=1$, so the cell boundary is a mirror plane, and the equation $u''+\omega^2\varepsilon u=0$ ($\omega$ in units of $c/a$, $f=\omega/2\pi$). A finite stack of $N$ whole cells between hard walls ($u=0$) has modes $u=\alpha\psi_k+\beta\psi_{-k}$ with $\psi_{\pm k}$ the Bloch solutions at the mode's frequency.
#
# **Exact quantization, any wall phase.** Let a wall sit at $x_w$. The condition $u(x_w)=0$ fixes $u=\psi_k(x)\psi_{-k}(x_w)-\psi_{-k}(x)\psi_k(x_w)$. If the second wall is at $x_w+Na$ (the same cell phase) Bloch's theorem gives $u(x_w+Na)=2i\sin(kNa)\,\psi_k(x_w)\psi_{-k}(x_w)$, and $\psi_{\pm k}(x_w)\ne0$ in a band: for real $\omega$ and real $\varepsilon$, $\psi_{-k}=\overline{\psi_k}$, so a zero of one is a zero of the other, which the nonzero constant Wronskian $\psi_k\psi_{-k}'-\psi_{-k}\psi_k'$ forbids. So every in-band mode sits at $k=n\pi/(Na)$, for every wall phase and without any mirror symmetry. The envelope prediction of high-frequency homogenization about a gap edge, $\omega_j^2=\omega_e^2+T\,(j\pi/Na)^2$ with $T=\tfrac12\,d^2(\omega^2)/dk^2$ at the edge, is then the exact dispersion with $k$ replaced by its exact value, so its error is the quartic and higher terms of the band: $O(N^{-4})$, with no end correction. What the wall phase decides is which edge standing wave is a mode ($j=0$): with the walls on a mirror plane the edge standing wave that is odd about the wall vanishes there and is a mode of the stack; with walls at $a/4$ neither is, and a surface mode appears in the gap.
#
# **Walls at different cell phases.** If the second wall sits at $x_w+Na+s$ with $0<s<a$, the condition $u(x_w+Na+s)=0$ is still exact but no longer reduces to $\sin kNa=0$. Writing $v_{\pm}$ for the Bloch eigenvectors of the cell transfer matrix $M(\omega)$ from the left wall (eigenvalues $e^{\pm ika}$) and $T_s$ for the transfer matrix of the cell's first $s$, one finds
#
# $$e^{2ikNa}=R(\omega)\equiv\frac{(T_sv_-)_1\,(v_+)_1}{(T_sv_+)_1\,(v_-)_1},\qquad |R|=1,$$
#
# i.e. $kNa+\Theta(\omega)=n\pi$ with $\Theta=-\tfrac12\arg R$. Near the gap edge $\Theta$ is not constant: the two Bloch vectors $v_\pm$ coalesce at the edge (the transfer matrix has a Jordan block there), and $\Theta(q)=\Theta_e+c\,q+O(q^2)$ with $q$ the distance of $k$ from the edge. This gives the envelope's boundary condition from the edge Bloch function alone: $j\to j+\Theta_e/\pi$ (an integer or half-integer shift fixed by the edge standing wave's value at the two walls) and $Na\to Na-c$, i.e. an end correction $\delta$ with $2\delta=-c-s$ beyond the geometric length $Na+s$ (the notebook's convention, $L=Na+s+2\delta_{\rm geo}$; PLAN.md writes $L=Na+2\delta_{\rm bare}$ (uncorrected length $Na$), so $\delta_{\rm geo}=\delta_{\rm bare}-s/2$, and $\delta_{\rm bare}=-c/2$ for the fit-free value). Both conventions are reported below. Without the end correction the error of the envelope prediction is $O(N^{-3})$ at an edge where $\Theta_e\equiv0$ (mod $\pi$), and $O(N^{-2})$ at an edge where $\Theta_e\equiv\pi/2$ if integer $j$ is used (the wrong shift of $j$), $O(N^{-3})$ with the right half-integer $j$.
#
# The three cases below are: **A**, both walls at the cell boundary (a mirror plane); **B**, both walls at $a/4$ inside the cell; **C**, left wall at 0 and right wall at $Na+a/4$. The roots of $u(x_w+L)$ come from a 30000-point scan of $\omega$ plus Brent's method on the transfer matrices, over the window containing the first gap and the first modes of both adjacent bands; $T$ and the edges are exact from the module (`band_edges_1d`, `edge_curvature_1d`).

# %%
LAY = [(1.0, 0.35), (12.0, 0.3), (1.0, 0.35)]
ed, sg = bl.band_edges_1d(LAY, (0.1, 30))
e1, e2 = ed[0], ed[1]
T1, T2 = bl.edge_curvature_1d(LAY, e1, sg[0]), bl.edge_curvature_1d(LAY, e2, sg[1])
print(f"First gap of the three-layer cell: omega = {e1:.6f} to {e2:.6f}  (f = {e1 / (2 * np.pi):.5f} to {e2 / (2 * np.pi):.5f});  T = (1/2) d2(omega^2)/dk2: {T1:+.5f} (lower edge), {T2:+.5f} (upper edge)")
WIN = (0.6 * e1, e2 + 1.5 * (e2 - e1))
NS = (10, 20, 40, 80)


def stack(N, off, extra):
    om, _ = bl.finite_stack_modes(LAY, N, off, WIN, nscan=30000, extra=extra)
    return dict(om=om, below=om[om < e1 - 1e-9][::-1], above=om[om > e2 + 1e-9],
                n_edge=(int(np.sum(np.abs(om - e1) <= 1e-9)), int(np.sum(np.abs(om - e2) <= 1e-9))),
                gap=om[(om > e1 + 1e-9) & (om < e2 - 1e-9)])


CASES = {"A": (0.0, 0.0), "B": (0.25, 0.0), "C": (0.0, 0.25)}
ST = {(c, N): stack(N, *CASES[c]) for c in CASES for N in NS}

print("\nModes of the stack at the edges (counts of finite-stack roots at the edge frequency, and in the gap)")
print(f"{'case':>5} {'N':>4} {'at lower edge':>14} {'at upper edge':>14} {'gap modes':>10}")
for c in CASES:
    for N in (10, 40):
        s_ = ST[(c, N)]
        print(f"{c:>5} {N:>4} {s_['n_edge'][0]:>14} {s_['n_edge'][1]:>14} {len(s_['gap']):>10}" + (f"   at omega = {', '.join(f'{x:.4f}' for x in s_['gap'])}" if len(s_["gap"]) else ""))
check_in("case A: exactly one gap edge is a finite-stack root (N = 10, 20, 40, 80)", float(all(sorted(ST[("A", N)]["n_edge"]) == [0, 1] for N in (10, 20, 40, 80))), 1, 1)
check("case A: the root is at the lower edge (frequency |omega - e1|, N = 40)", float(np.min(np.abs(ST[("A", 40)]["om"] - e1))), 1e-10)
check_in("case B: no edge root, and a surface mode in the gap (N = 10..80)", float(all(ST[("B", N)]["n_edge"] == (0, 0) and len(ST[("B", N)]["gap"]) == 1 for N in NS[1:])), 1, 1)

# exact quantization in cases A and B
qerr = {"A": 0.0, "B": 0.0}
for c in ("A", "B"):
    for N in NS:
        s_ = ST[(c, N)]
        for arr in (s_["below"], s_["above"]):
            for j in (1, 2, 3):
                kj = bl.bloch_k_1d(LAY, arr[j - 1])
                qerr[c] = max(qerr[c], abs(kj.real - (N - j) * np.pi / N), abs(kj.imag))
print(f"\nBloch k of the modes j = 1, 2, 3 at both edges against (N - j) pi/(N a), N = 10..80: worst error, case A {qerr['A']:.1e}, case B {qerr['B']:.1e}")
check("case A (walls at 0): Bloch k of the modes = (N - j) pi/(N a)", qerr["A"], 1e-10)
check("case B (walls at a/4, same cell phase): Bloch k of the modes = (N - j) pi/(N a)", qerr["B"], 1e-10)
dAB = max(np.abs(ST[("A", N)][key][:3] - ST[("B", N)][key][:3]).max() for N in NS for key in ("below", "above"))
print(f"in-band spectra of A and B, modes j = 1, 2, 3 at both edges: max |omega_A - omega_B| = {dAB:.1e}")
check("the in-band spectra of A and B coincide (j = 1, 2, 3, both edges)", dAB, 1e-9)

# %%
# Envelope error: |omega_j^2 - (omega_e^2 + T (j pi / L)^2)| for the modes counted from each edge, N = 10 ... 80.
def env_err(c, edge, j, N, jshift=0.0, Leff=None, signed=False):
    s_ = ST[(c, N)]
    arr, e, T = (s_["below"], e1, T1) if edge == "lower" else (s_["above"], e2, T2)
    L = (N + CASES[c][1]) if Leff is None else Leff(N)
    d = (e ** 2 + T * ((j + jshift) * np.pi / L) ** 2) - arr[j - 1] ** 2          # prediction minus exact
    return d if signed else abs(d)


slope = lambda y, ns: np.polyfit(np.log(ns), np.log(y), 1)[0]
N3 = (10, 20, 40)
print("Slope d log(error) / d log N over N = 10, 20, 40, envelope prediction with the geometric length L = N a + s, no correction")
print(f"{'case':>5} {'edge':>6} {'j shift':>8} " + " ".join(f"{'j=' + str(j):>7}" for j in (1, 2, 3)))
SL = {}
for c, edge, js in (("A", "lower", 0), ("A", "upper", 0), ("B", "lower", 0), ("B", "upper", 0), ("C", "upper", 0), ("C", "lower", 0), ("C", "lower", -0.5)):
    SL[(c, edge, js)] = [slope([env_err(c, edge, j, N, js) for N in N3], N3) for j in (1, 2, 3)]
    print(f"{c:>5} {edge:>6} {js:>8} " + " ".join(f"{x:7.2f}" for x in SL[(c, edge, js)]))
for c in ("A", "B"):
    for edge in ("lower", "upper"):
        check(f"case {c}, {edge} edge: envelope error slope, worst deviation from -4 over j = 1, 2, 3", max(abs(x + 4) for x in SL[(c, edge, 0)]), 0.3)

# %%
# Case C: the exact quantization condition k N a + Theta(omega) = n pi, with Theta from the Bloch eigenvectors of the cell.
def trunc(layers, s):
    out, acc = [], 0.0
    for eps, d in layers:
        if acc + d <= s + 1e-15:
            out.append((eps, d))
            acc += d
        else:
            if s - acc > 1e-15:
                out.append((eps, s - acc))
            break
    return out


def R_of(w, s):
    ev, V = np.linalg.eig(bl.transfer_1d(LAY, np.float64(w)))
    i = int(np.argmax(np.angle(ev)))                      # eigenvalue e^{+ik}, 0 < k a < pi
    vp, vm = V[:, i], V[:, 1 - i]
    Ts = bl.transfer_1d(trunc(LAY, s), np.float64(w))
    return (Ts @ vm)[0] * vp[0] / ((Ts @ vp)[0] * vm[0]), float(np.angle(ev[i]))


S_END = CASES["C"][1]
worst = 0.0
for N in NS:
    for arr in (ST[("C", N)]["below"], ST[("C", N)]["above"]):
        for w in arr[:6]:
            Rv, kk = R_of(w, S_END)
            worst = max(worst, abs(np.exp(2j * kk * N) - Rv), abs(abs(Rv) - 1))
print(f"Case C: max | exp(2 i k N a) - R(omega) | over the first 6 in-band modes at each edge, N = 10..80: {worst:.1e}")
check("case C: exact quantization condition exp(2ikNa) = R(omega) at every mode", worst, 1e-9)

# Theta(q) near each edge, q = pi - k a; extrapolate Theta_e and c = dTheta/dq by a quadratic fit
from scipy.optimize import brentq
QS = np.array([0.005, 0.01, 0.02, 0.04])
EDGE = {}
for edge, e, T in (("lower", e1, T1), ("upper", e2, T2)):
    th_q = []
    for q in QS:
        g = lambda w: 0.5 * np.trace(bl.transfer_1d(LAY, np.float64(w))) - np.cos(np.pi - q)
        w = brentq(g, e - 0.3, e, xtol=1e-15) if T < 0 else brentq(g, e, e + 0.3, xtol=1e-15)
        Rv, kk = R_of(w, S_END)
        th_q.append(-np.angle(Rv) / 2)
    th_q = np.array(th_q)
    pf = np.polyfit(QS, th_q, 2)
    EDGE[edge] = dict(theta_e=pf[2], c=pf[1], d2=pf[0], th=th_q)
print(f"\n{'edge':>6} {'Theta_e / pi':>13} {'c (rad/rad)':>14} {'d2 (rad/rad^2)':>15}   Theta/pi at q = {', '.join(f'{q:g}' for q in QS)}   (Theta in rad, q in rad)")
for edge in ("lower", "upper"):
    d_ = EDGE[edge]
    print(f"{edge:>6} {d_['theta_e'] / np.pi:13.6f} {d_['c']:14.5f} {d_['d2']:15.4f}   " + ", ".join(f"{x / np.pi:.5f}" for x in d_["th"]))
# The edge standing waves at the two walls (k a = pi: M v = -v), amplitude relative to the maximum over the cell
print("\nEdge standing wave u(x) of the infinite crystal, relative to its maximum over the cell: value at the wall x = 0 and at the wall x = a/4")
EDGE_U = {}
for edge, e in (("lower", e1), ("upper", e2)):
    ev, V = np.linalg.eig(bl.transfer_1d(LAY, np.float64(e)))
    v = V[:, int(np.argmin(np.abs(ev + 1)))]
    v = (v / v[int(np.argmax(np.abs(v)))]).real
    xs_ = np.linspace(0, 1, 201)
    u_ = np.array([(bl.transfer_1d(trunc(LAY, x), np.float64(e)) @ v)[0] if x > 0 else v[0] for x in xs_])
    EDGE_U[edge] = (v[0] / np.abs(u_).max(), u_[50] / np.abs(u_).max())
    print(f"  {edge} edge: u(0) = {EDGE_U[edge][0]:+.4f}, u(a/4) = {EDGE_U[edge][1]:+.4f}")
check("lower edge standing wave vanishes at the wall x = 0", EDGE_U["lower"][0], 1e-8)
check_in("lower edge standing wave does not vanish at the wall x = a/4 (|u|/max)", abs(EDGE_U["lower"][1]), 0.1, 1)
check_in("upper edge standing wave does not vanish at x = 0 (|u|/max)", abs(EDGE_U["upper"][0]), 0.1, 1)
check_in("upper edge standing wave does not vanish at x = a/4 (|u|/max)", abs(EDGE_U["upper"][1]), 0.1, 1)

# Theta_e is fixed mod pi by symmetry: 0 where the edge standing wave is nonzero at both walls, pi/2 where it vanishes at one wall
check("Theta_e / pi at the upper edge (edge wave nonzero at both walls: 0 mod 1)", EDGE["upper"]["theta_e"] / np.pi, 1e-5)
check("Theta_e / pi at the lower edge (edge wave vanishes at one wall: 1/2 mod 1)", EDGE["lower"]["theta_e"] / np.pi, 1e-5, 0.5)

# %%
# Corrections tested at the first mode of each edge (j = 1 at the upper edge; j - 1/2 = 1/2 at the lower edge), N = 10 ... 80.
# Length convention: the notebook writes the effective length as L = N a + s + 2 delta_geo (s = a/4 is the geometric extra length of the
# right wall); PLAN.md writes L = N a + 2 delta_bare, so delta_bare = delta_geo + s/2. Two uncorrected baselines are reported:
# the geometric length N a + s (the notebook's) and N a (the convention of PLAN.md).
JS = {"lower": -0.5, "upper": 0.0}


def fitted_delta(edge, j, N0=20):
    s_ = ST[("C", N0)]
    arr, e, T = (s_["below"], e1, T1) if edge == "lower" else (s_["above"], e2, T2)
    Leff = (j + JS[edge]) * np.pi / np.sqrt((arr[j - 1] ** 2 - e ** 2) / T)
    return (Leff - (N0 + S_END)) / 2


DELTA = {edge: fitted_delta(edge, 1) for edge in ("lower", "upper")}
NN = (10, 20, 40, 80)
print("Case C, first mode counted from each edge: |omega^2 - prediction|; reduction = uncorrected error / corrected error at N = 10 and 40,")
print("for the two uncorrected baselines L = N a + s (geometric) and L = N a (bare convention)")
print(f"{'edge':>6} {'prediction':>38} " + " ".join(f"{'N=' + str(N):>9}" for N in NN) + f" {'slope 20-80':>11} {'red. vs Na+s':>16} {'red. vs Na':>14}")
ERR = {}
for edge in ("lower", "upper"):
    c_ = EDGE[edge]["c"]
    cands = [("integer", "integer j, L = N a + s", 0.0, None), ("integerN", "integer j, L = N a", 0.0, lambda N: float(N))]
    if edge == "lower":
        cands += [("half", "half-integer j, L = N a + s", JS[edge], None), ("halfN", "half-integer j, L = N a", JS[edge], lambda N: float(N))]
    dn = DELTA[edge]
    cands += [("fit", f"fitted d_geo = {dn:.3f} (d_bare = {dn + S_END / 2:.3f}), N = 20", JS[edge], lambda N, d=dn: N + S_END + 2 * d),
              ("free", f"fit-free, Theta_e and c = {c_:+.3f}", JS[edge], lambda N, c=c_: N - c)]
    for key, lab, js, Lf in cands:
        ERR[(edge, key)] = np.array([env_err("C", edge, 1, N, js, Lf) for N in NN])
    bS, bN = ("half", "halfN") if edge == "lower" else ("integer", "integerN")
    for key, lab, js, Lf in cands:
        e_ = ERR[(edge, key)]
        sl_ = "" if key == "fit" else f"{slope(e_[1:], NN[1:]):11.2f}"        # the fitted delta is zero-error at N = 20 by construction
        rat = "" if key in ("integer", "integerN", "half", "halfN") else (f"{ERR[(edge, bS)][0] / e_[0]:7.1f} {ERR[(edge, bS)][2] / e_[2]:7.1f}   {ERR[(edge, bN)][0] / e_[0]:7.1f} {ERR[(edge, bN)][2] / e_[2]:7.1f}")
        print(f"{edge:>6} {lab:>38} " + " ".join(f"{x:9.2e}" for x in e_) + f" {sl_:>11} {rat}")
print("\nEnd corrections in the two length conventions (units of a):")
print(f"{'edge':>6} {'fitted d_geo':>12} {'fitted d_bare':>14} {'fit-free d_geo = (-c-s)/2':>26} {'fit-free d_bare = -c/2':>24}")
for edge in ("lower", "upper"):
    print(f"{edge:>6} {DELTA[edge]:12.4f} {DELTA[edge] + S_END / 2:14.4f} {(-EDGE[edge]['c'] - S_END) / 2:26.4f} {-EDGE[edge]['c'] / 2:24.4f}")

bS_of = {"lower": ("half", "halfN"), "upper": ("integer", "integerN")}
RAT = {(edge, key, bl_): (ERR[(edge, bS_of[edge][bi])][0] / ERR[(edge, key)][0], ERR[(edge, bS_of[edge][bi])][2] / ERR[(edge, key)][2])
       for edge in ("lower", "upper") for key in ("fit", "free") for bi, bl_ in enumerate(("s", "N"))}

# Why the single-mode fit gives 0.076 and not the fit-free 0.0595 at the upper edge: the fit at N = 20 also absorbs the band's quartic term.
# With signed errors, d_fit / d_free = E_unc / (E_unc - E_free) at N = 20 (to first order in the correction).
Eu = env_err("C", "upper", 1, 20, 0.0, None, signed=True)
Ef = env_err("C", "upper", 1, 20, 0.0, lambda N, c=EDGE["upper"]["c"]: N - c, signed=True)
d_free = (-EDGE["upper"]["c"] - S_END) / 2
d_pred = d_free * Eu / (Eu - Ef)
print(f"\nUpper edge, N = 20: uncorrected error {Eu:.3e}, fit-free residual {Ef:.3e} (the band's own quartic term); {d_free:.4f} x {Eu:.3e} / ({Eu:.3e} - {Ef:.3e}) = {d_pred:.4f} against the fitted {DELTA['upper']:.4f}")
print("reduction of the fit-free correction relative to L = N a + s, E_unc / E_free, and divided by N: " + ", ".join(f"N = {N}: {ERR[('upper', 'integer')][i] / ERR[('upper', 'free')][i]:.1f} ({ERR[('upper', 'integer')][i] / ERR[('upper', 'free')][i] / N:.2f})" for i, N in enumerate(NN)))
print(f"q^2 coefficient of Theta at the upper edge: {EDGE['upper']['d2']:.4f} rad/rad^2 (small: it does not explain the difference)")
check("upper edge: fitted delta equals fit-free delta x E_unc / (E_unc - E_free) at N = 20 (relative; the band's quartic term is absorbed by the fit)", abs(d_pred / DELTA["upper"] - 1), 2e-2)

check("case C, upper edge, integer j, no correction: error slope over N = 10-40 (j = 1)", SL[("C", "upper", 0)][0], 0.3, -3)
check("case C, lower edge, integer j: error slope over N = 10-40 (j = 1)", SL[("C", "lower", 0)][0], 0.3, -2)
check("case C, lower edge, half-integer j, no end correction: error slope over N = 10-40 (j = 1)", SL[("C", "lower", -0.5)][0], 0.3, -3)
for edge in ("lower", "upper"):
    check(f"case C, {edge} edge, fit-free correction: error slope over N = 20-80 (tends to the band's quartic term)", slope(ERR[(edge, "free")][1:], NN[1:]), 0.3, -4)
    check_in(f"case C, {edge} edge, fit-free correction: error reduction at N = 40 relative to the geometric length N a + s", RAT[(edge, "free", "s")][1], 5, 1e6)
# Expectation (PLAN.md): a fitted delta (N = 20) cuts the error at N = 10 and 40 by >= 10x relative to L = N a. Found (below): yes at both edges.
for edge, lab in (("upper", "integer j"), ("lower", "half-integer j")):
    for i, N in ((0, 10), (1, 40)):
        check_in(f"case C, {edge} edge ({lab}): fitted delta, error reduction at N = {N} relative to L = N a (expected: >= 10x)", RAT[(edge, "fit", "N")][i], 10, 1e6)
        check_in(f"case C, {edge} edge ({lab}): fitted delta, error reduction at N = {N} relative to L = N a + s (stricter baseline; > 1 is an improvement)", RAT[(edge, "fit", "s")][i], 1, 1e6)

# %%
# Figure 4: envelope error against N for the first mode at each edge.
fig, ax = plt.subplots(1, 2, figsize=(10.6, 5.2), sharey=False)
for a_, edge, ttl in ((ax[0], "lower", "lower gap edge (band 1 top)"), (ax[1], "upper", "upper gap edge (band 2 bottom)")):
    Nf = np.array(NN, float)
    eA = np.array([env_err("A", edge, 1, N) for N in NN])
    eB = np.array([env_err("B", edge, 1, N) for N in NN])
    a_.loglog(Nf, eA, "o-", color=CAT[0], label="A: walls at 0")
    a_.loglog(Nf, eB, "x", color=CAT[6], ms=9, label="B: both walls at $a/4$")
    a_.loglog(Nf, ERR[(edge, "integer")], "s-", color=CAT[1], label="C: no correction" if edge == "upper" else "C, integer $j$ (wrong shift)")
    if edge == "lower":
        a_.loglog(Nf, ERR[(edge, "half")], "s-", color=CAT[3], label="C, half-integer $j$, no end correction")
    fit = np.where(ERR[(edge, "fit")] > 1e-15, ERR[(edge, "fit")], np.nan)
    a_.loglog(Nf, fit, "d--", color=CAT[4], label="C, fitted $\\delta$ (exact at $N=20$)")
    a_.loglog(Nf, ERR[(edge, "free")], "^-", color=CAT[2], label="C, fit-free ($\\Theta_e$, $c$)")
    for p_, ls_ in ((-2, ":"), (-3, "-."), (-4, "--")):                     # slope guides through a point a factor 10 below the A curve at N = 10
        ref = 0.1 * eA[0] * (Nf / Nf[0]) ** p_
        a_.loglog(Nf, ref, color=MUTED, ls=ls_, lw=0.9)
        a_.text(Nf[-1] * 1.04, ref[-1], f"$N^{{{p_}}}$", color=MUTED, fontsize=8, va="center")
    a_.set_xlabel("number of cells $N$")
    a_.set_ylabel(r"$|\omega_1^2-\omega^2_{\rm env}|$")
    a_.set_title(ttl)
    a_.set_xlim(8, 110)
    a_.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
plt.show()

total = time.perf_counter() - T_START
print(f"\nTotal run time of the notebook (from the second cell): {total:.1f} s; {N_CHECKS} checks passed")

# %% [markdown]
# *Figure 4. First mode counted from each gap edge: error of the envelope prediction against $N$. Walls at the mirror plane (A) and both at $a/4$ (B) give identical $N^{-4}$ errors (the markers coincide). Walls at different cell phases (C) give $N^{-2}$ or $N^{-3}$ without an end correction, depending on the edge, and approach $N^{-4}$ with the fit-free correction. The gray straight lines are guides with slopes $-2,-3,-4$, drawn through a point a factor 10 below the A curve at $N=10$ (they show slopes only). Here the uncorrected baseline is the geometric length $Na+s$. The fitted-$\delta$ curve is zero by construction at $N=20$ and is left out of the log plot there.*
#
# **Cases A and B.** The Bloch wave numbers of the stack modes equal $(N-j)\pi/(Na)$ to $10^{-13}$ for the walls at the mirror plane and for those at $a/4$, and the in-band frequencies of the two stacks coincide to $5\times10^{-15}$: the walls' common phase changes nothing in the band, as the argument above shows without using mirror symmetry. The expectation recorded beforehand (PLAN.md), that walls at $a/4$ need an end correction of order $a$ and leave an $O(N^{-3})$ error, is therefore wrong as stated; the exact quantization and the slopes of $-3.8$ to $-4.0$ for $j=1,2,3$ show it. What the wall phase does decide is the count at the edge: one finite-stack root at the lower edge for walls at 0 (the standing wave odd about the wall, $j=0$, with Neumann-like envelope), none for walls at $a/4$, where a surface mode appears in the gap instead.
#
# **Case C.** The $\Theta$ table above shows that quantization still holds exactly, in the form $e^{2ikNa}=R(\omega)$ (error $3.5\times10^{-12}$ at every tested mode), but the shift $\Theta$ is not zero. At the upper gap edge $\Theta_e=0$ (mod $\pi$) with $c=d\Theta/dq=-0.369$; at the lower edge $\Theta_e=\pi/2$ and $c=-1.148$ (the $q^2$ coefficient of $\Theta$ is $0.0017$ at the upper edge, negligible). The $\Theta_e$ values go with the printed values of the edge standing waves at the two walls: the upper-edge wave does not vanish at either wall (both ends of the envelope are Dirichlet-like, integer $j$), and the lower-edge wave vanishes at the wall at 0 but not at the wall at $a/4$ (mixed ends, half-integer $j$).
#
# *Two conventions for the uncorrected length.* The end correction is a length added to an uncorrected baseline, and two baselines are in use: the bare length $Na$ (PLAN.md) and the geometric length $Na+s$ (this notebook). With $L=Na$ the fit-free end correction is $\delta_{\rm bare}=-c/2=0.1845a$ (upper) and $0.5740a$ (lower); with the geometric length $Na+s$ it is $\delta_{\rm geo}=(-c-s)/2=0.0595a$ and $0.4490a$. A single $\delta$ fitted at $N=20$, $j=1$ gives $\delta_{\rm bare}=0.2011a$ and $0.5805a$, i.e. $\delta_{\rm geo}=0.0761a$ and $0.4555a$. The error reductions, relative to the uncorrected error at the same $N$:
#
# | fitted $\delta$ | baseline $L=Na$ ($N=10$, $40$) | baseline $L=Na+s$ ($N=10$, $40$) |
# |---|---|---|
# | upper edge, integer $j$ | $15.3\times$, $24.1\times$ | $6.2\times$, $8.4\times$ |
# | lower edge, half-integer $j$ | $182.9\times$, $206.6\times$ | $138.4\times$, $160.4\times$ |
#
# An earlier estimate, $\delta=0.20a$ with a reduction of $15\times$-$24\times$, is therefore reproduced: these are the $L=Na$ numbers, and the expected $10\times$ is met at both edges and both $N$. The values $0.076a$ and $6.2\times$-$8.4\times$ are the same correction measured from $Na+s$ ($0.2011a-s/2=0.0761a$), a stricter baseline, not a contradiction. (The upper-edge error without correction is $8.4\times10^{-3}$ at $N=10$ for $L=Na$ against $3.4\times10^{-3}$ for $L=Na+s$; the geometric extra length already removes part of the error.)
#
# *Slopes and the fit-free correction.* With the geometric length $Na+s$ and integer $j$ the error slopes are $-3.16$ (upper edge, close to the expected $-3$) and $-2.00$ (lower edge, wrong shift of $j$); half-integer $j$ at the lower edge gives $-2.90$. The fit-free prediction $q=(j+\Theta_e/\pi)\pi/(Na-c)$ takes both to slopes of $-3.95$ and $-3.85$ over $N=20$-$80$, i.e. to the band's own quartic term. Relative to $Na+s$ it cuts the error at $N=40$ by $8.0\times$ (upper) and $133\times$ (lower); relative to $Na$ by $23.0\times$ and $171.8\times$.
#
# *Why the fitted $\delta$ differs from the fit-free one.* At the upper edge the fit-free correction leaves a residual at $N=20$ of $8.17\times10^{-5}$, which is the band's own quartic term (the $q^2$ term of $\Theta$ is negligible), against an uncorrected error of $3.77\times10^{-4}$. The single-mode fit at $N=20$ folds that quartic term into $\delta$: $0.0595\times3.77\times10^{-4}/(3.77\times10^{-4}-8.17\times10^{-5})=0.0759$, against the fitted $0.0761$ (agreement $2.4\times10^{-3}$ relative, asserted to $2\times10^{-2}$). That is also why the reduction relative to $Na+s$ stays below $10\times$: a length correction can only remove the $N^{-3}$ part of the error, the band's quartic term stays, and the ratio of the $N^{-3}$ term to the quartic term is about $0.2N$ because the end correction is small, $0.06a$ (printed $E_{\rm unc}/E_{\rm free}$ is $2.9,\,4.6,\,8.0,\,14.9$ at $N=10,20,40,80$, i.e. $0.29,\,0.23,\,0.20,\,0.19$ times $N$; the fit-free reduction at $N=40$ is $8.0\times$ and the fitted one $8.4\times$). Measured from $L=Na$ the end correction is larger ($0.18a$), the $N^{-3}$ term is larger, and the reduction reaches $15\times$-$24\times$. At $N=80$ the fitted-$\delta$ upper-edge error is $9.9\times10^{-7}$, three times that of the fit-free correction ($3.4\times10^{-7}$). The end correction to quote is $0.0595a$ from $-c-s$ (or $0.1845a$ from $L=Na$).
#
# The boundary condition of the envelope is, therefore, a property of the edge Bloch function at the wall, $\Theta(\omega)$ above, and the geometric length $Na$ is not enough: this is the high-frequency counterpart of notebook 04's boundary-layer tails, where the cut of the cell decides the interior error at $O(\varepsilon)$. The exact phase function is computable for any 1D cell; in 2D the corresponding object would be the edge Bloch functions evaluated on the wall, which this notebook does not compute.
#
# ## Caveats and open ends
#
# - **Design record.** PLAN.md records the design, the expectations written before the computations, and the corrections made after an independent review.
# - **Scope.** The envelope boundary condition is tested in 1D only, where it is exact. The 2D $k\cdot p$ constants, Hom dimensions, the Dirac cone and its Berry phase are tested for one TM rod lattice each (square $R=0.3$, honeycomb $R=0.2$ and the $0.21/0.19$ pair, $\varepsilon=12$). No finite 2D crystal, no TE polarization and no non-circular unit cells other than the ellipse were run.
# - **Pre-asymptotic.** The $N^{-4}$ slopes are $-3.8$ to $-4.0$ over $N=10$-$40$ ($j=1,2,3$), the larger $j$ being further from the asymptotic slope, and the $k\cdot p$ error ratios 4.0 hold for $|\mathbf q|a\le0.08$. The massive Dirac formula is asymptotic only for small $r$ (deviation up to $0.0345\pi$ at $3m/v$), and the deviation does not follow an $r^2$ law over $m/2v$ to $3m/v$.
# - **Tolerances set after seeing results.** The PWE-versus-multipole tolerance (three times the 600-to-1000 change, capped at $2\times10^{-4}$) comes from the convergence study printed in section 1. The one-sided checks that record where the expectation recorded beforehand (PLAN.md) fails (finite differences at the steps $10^{-3},2\times10^{-3}$ below $10^{-4}$; Dirac-point splitting of the parallelogram cutoff above $10^{-8}$; $3m/v$ deviation above $0.03\pi$) assert only the bound the prose relies on, with the bound chosen after seeing the values. The $10^{-5}$ tolerance on $\Theta_e$ was set from the quality of the quadratic extrapolation (observed $7\times10^{-7}$ at most). The bounds of the sanity checks on the $k\cdot p$ loop phase ($2\times10^{-3}\pi$ for the first-order matrix against the closed formula, $0.015\pi$ for the second-order matrix against the PWE), the factor 5 in the fit-free end-correction reduction, the tolerance $2\times10^{-2}$ on the quartic-term identity for the fitted $\delta$ (observed $2.4\times10^{-3}$) and the amplitude threshold $0.1$ of the edge standing waves were set with the observed values in view (observed: below $3.1\times10^{-4}\pi$, at most $0.0113\pi$). The loop-convergence checks use a quarter of the smallest deviation as the criterion (chosen a priori, but the 64-point change of $2.5\times10^{-4}\pi$ would have been close to it; the checked change is that of the 256-point loops, $1.2\times10^{-5}\pi$), and the remote-band check asserts a majority ($\ge0.5$; observed $0.73$). The expected $10\times$ end-correction reduction is asserted relative to $L=Na$, the baseline of PLAN.md; the $L=Na+s$ checks assert only an improvement ($>1$). All other tolerances were fixed in advance.
# - **Checks that cannot fail in an informative way.** The Hom-dimension check for $\operatorname{Sym}^3V$ is a consistency test of the computed characters (every odd order vanishes because $C_2\in G$); the equal-rod loop phase and the valley contrast of the unequal-rod loops are exact consequences of the real symmetric pencil and of the inversion image of the plane-wave sets; the check that the loop helper reproduces `bl.berry_phase` is an interface identity. They test the code, not the physics.
# - **Fit-free correction.** $\Theta_e$ and $c$ come from a quadratic fit through four values of $\Theta(q)$, $q=0.005,\dots,0.04$; $c$ is therefore good to a few $10^{-3}$, which is enough for the slopes shown but not a derivation. The exact quantization condition used to define $\Theta$ is checked against the module's independent root finder, not derived in the notebook beyond the algebra in the text. The identity between the fitted and fit-free $\delta$ is first order in the correction and holds here to $2.4\times10^{-3}$.
# - **Wilson loops.** Convergence was measured: 64 to 256 points changes the phase by up to $2.5\times10^{-4}\pi$, 256 to 512 points by up to $1.2\times10^{-5}\pi$ (two radii), 200 to 400 plane waves by up to $1.1\times10^{-5}\pi$ (three radii, 64 points). The $K'$ loops and the valley contrast use 64 points; the 400-plane-wave test used the same radius in $a^{-1}$ as the 200-wave set, not a radius re-derived from the 400-wave $m/v$. The attribution of the small-$r$ deviation to the Bloch vector outside the doublet rests on the projected-vector loop at one plane-wave set; the attribution of the large-$r$ doublet-only part to trigonal warping specifically (rather than other terms of $H_2$) is a conjecture, not tested. The remaining $3\times10^{-4}\pi$ between the first-order $k\cdot p$ loop and the closed formula with the unequal-rod velocity was not traced. The equal-rod phase is exactly $\pi$ by construction and says nothing about convergence.
# - **Hom dimensions.** The real constants equal the complex Hom dimensions because the representations here are real or real-symmetric; this was not tested independently. The Fefferman-Weinstein theorem is for $-\Delta+V$; for the generalized TM problem it is an analogy, and what is computed here is the symmetry count and the cone, not their theorem.
# - **An earlier estimate, resolved.** An earlier estimate of the fitted $\delta$ ($0.20a$, $15\times$-$24\times$) is reproduced when the uncorrected length is $Na$ (section 7), so it is not an open discrepancy. What remains untested is the dependence on the second wall's phase: the other values of $s$ change $\Theta_e$, $c$ and the fitted $\delta$, and only $s=a/4$ was run. The half-integer $j$ at the other edge was confirmed only in this configuration.
#
# ## References
#
# - C. L. Fefferman and M. I. Weinstein, Honeycomb lattice potentials and Dirac points, J. Amer. Math. Soc. 25 (2012), 1169-1220. (checked)
# - R. V. Craster, J. Kaplunov and A. V. Pichugin, High-frequency homogenization for periodic media, Proc. R. Soc. A 466 (2010), 2341-2362. (checked)
# - G. Allaire and C. Conca, Bloch wave homogenization and spectral asymptotic analysis, J. Math. Pures Appl. 77 (1998), 153-208. (checked)
# - C. Conca and M. Vanninathan, Homogenization of periodic structures via Bloch decomposition, SIAM J. Appl. Math. 57 (1997), 1639-1659. (checked)
# - G. Allaire and M. Amar, Boundary layer tails in periodic homogenization, ESAIM: COCV 4 (1999), 209-243. (checked; compare notebook 04)
# - Within the repository: `bloch.py` and `notebooks/spatial_dispersion/rodlattice.py` (multipole reference), `PLAN.md` of this folder.
# - From memory (details not verified): M. V. Berry, Quantal phase factors accompanying adiabatic changes, Proc. R. Soc. Lond. A 392 (1984), 45-57; P.-O. Löwdin, A note on the quantum-mechanical perturbation theory, J. Chem. Phys. 19 (1951), 1396-1401; the character-orthogonality formula for Hom spaces is standard finite-group representation theory (e.g. J.-P. Serre, Linear Representations of Finite Groups).
#