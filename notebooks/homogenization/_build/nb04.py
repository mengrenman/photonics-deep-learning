# %% [markdown]
# # Two scales on a finite domain: rates, the cut of the cell, boundary-layer tails, and the div-curl lemma
#
# For $-\nabla\!\cdot(a(x/\varepsilon)\nabla u_\varepsilon)=1$ on the unit square with $u_\varepsilon=0$ on the boundary, how fast does $u_\varepsilon$ approach the homogenized solution $u_0$, where is the first-order two-scale expansion $w_\varepsilon=u_0+\varepsilon\chi_j(x/\varepsilon)\partial_ju_0$ wrong, and what fixes the size of its interior error? The textbook statements (an $L^2$ rate $\varepsilon$, an $H^1$ rate $\varepsilon^{1/2}$ for $u_\varepsilon-w_\varepsilon$ because of the boundary layer, and weak convergence of energy densities) are stated for general domains. On a periodic medium cut by a flat edge they hide a dependence on where the edge cuts the cell, and this notebook measures it.
#
# The point is homogenization as analysis, measured rather than quoted: H-convergence, correctors, two-scale convergence and the div-curl lemma, on a fixed discrete problem where every number can be recomputed. Three results carry over to the notebooks that follow: the first-order boundary-layer error is a tail $\bar\beta_j$ of a half-strip problem, it is predicted quantitatively and vanishes for edges on mirror lines of the cell; the layer decays as an evanescent static Bloch wave whose exponent is an eigenvalue of a quadratic pencil on one cell; and the effective medium is the weak limit of the flux energy but not of the plain gradient energy. The termination of a periodic structure by a boundary (notebooks 05 and 06) is the same phenomenon with a wave equation in place of the Laplacian.
#
# ## What this notebook shows
#
# - **The textbook $H^1$ rate needs a generic cut.** The $L^2$ error of $u_\varepsilon-u_0$ falls like $\varepsilon$ for all three cuts (slopes 1.01, 0.97, 1.02). The $H^1$ error of the first-order expansion falls like $\varepsilon$ when the edge lies on a mirror line of the cell (overall slopes 0.957 and 0.989) and with a smaller, still falling rate for the generic cut: overall slope 0.646, local rates 0.709, 0.644, 0.587, 0.548 up to $N=64$, and 0.653, 0.594, 0.552 for the second cell ($m=8$, the square inclusion of side $1/2$). The data are **consistent with an asymptotic rate $1/2$, as the standard theory asserts, but do not observe it**: a model $H^1{}^2=B\varepsilon^q+A\varepsilon^2$ with free $q$ fits the five points to 0.22% with $q=1.019\pm0.007$ (rough, two degrees of freedom; $q$ between 0.96 and 1.07 keeps the misfit below 1%), and the two-term model with $q=1$ fitted to $N\le32$ predicts $N=64$ to 0.50% (0.15% at $m=8$). The interior $H^1$ error falls like $\varepsilon$ for all cuts.
# - **The interior error of $w_\varepsilon$ is $O(\varepsilon^2)$ for mirror-line cuts and $O(\varepsilon)$ for the generic cut**: interior maximum errors fall with slopes 1.96 and 1.95 against 1.02 over $N=8$-$32$; the same holds at $m=8$ up to $N=64$.
# - **The $O(\varepsilon)$ error is a boundary-layer tail.** The half-strip tails are 0.13554 and 0.16978 on the two $x_1$ edges for the generic cut and zero to $10^{-15}$ for mirror-line cuts. The interior correction $\bar u_1$ built from them (harmonic, with trace $\sum_j\bar\beta_j\partial_ju_0$) predicts $(u_\varepsilon-w_\varepsilon)/\varepsilon$ in the interior to 7.1% of $\max|\bar u_1|$ at $N=32$, with the residual falling at rate 0.92 over $N=8$-$32$ (0.95 at $m=8$).
# - **The layer is an evanescent static Bloch wave** and its decay on the strips equals the exponent of the quadratic eigenproblem of the matching parity (relative difference $6\times10^{-8}$ and $4.5\times10^{-6}$). At equal pixel resolution the layer is thinner than in a homogeneous medium for the contrasts $a_\text{in}=0.1,10,100$ (smallest exponents 6.74744, 6.83350, 7.15686 against 6.36553 at $m=16$; the margins are 0.19 to 0.79 over $m=8,16,24$ and are not converged in $m$), but **not for all contrasts**. Located by root finding, the smallest exponent of either parity lies below the homogeneous one for $0.77493<a_\text{in}<1.23907$ ($m=16$, $a_\text{in}\ne1$), with dips of $-2.378\times10^{-3}$ (odd parity, at $a_\text{in}=0.88149$) and $-2.011\times10^{-3}$ (even parity, at $1.11223$); the 121-point log grid of the scan misses 6.9% of the dip (the grid value is $-2.214\times10^{-3}$) and the edges by 2.4% and 3.0%. For the parity a generic edge excites (even, data $-\chi_1$) the layer is thicker only for $1<a_\text{in}<1.23907$ and thinner for every $a_\text{in}<1$; the odd window $0.77493<a_\text{in}<1$ concerns data with a vanishing tail. The mean first-order shift of the degenerate pair is zero, so only the splitting survives at first order.
# - **The div-curl lemma holds and the plain gradient energy misses the naive limit by the corrector term.** With the same quadrature on both sides $I_\text{flux}$ and $I_\text{grad}$ converge to $\int\varphi\sigma^*|\nabla u_0|^2$ and $\int\varphi\nabla u_0\!\cdot\!M\nabla u_0$ ($M_{11}=1.504546$) at rate $\approx2$ for the mirror-line cut $s=0$ with $\varphi$ vanishing on $\partial\Omega$ (last pairs 1.984 for $\varphi_1$, 2.07 and 2.10 for $\varphi_3$), and $I_\text{grad}$ differs from $\int\varphi|\nabla u_0|^2$ by 0.9932 of the predicted constant gap at $N=32$ (this restates the convergence to the limit). For the generic cut the flux error has an $O(\varepsilon)$ term $2\sigma^*\varepsilon\int\varphi\nabla u_0\!\cdot\!\nabla\bar u_1$, predicted from the boundary-layer tails; it is negligible for $\varphi_1$ and dominates $\varphi_3$ at $N\le64$, and after subtracting it the error falls at rate 1.995. The mixed version (midpoint integrals against exact cell tensors) has an $\varepsilon$-independent floor of 0.8% of the tensors: overall slopes 1.116 and 1.217 but last-pair rates 0.605 and 0.724, each error within a factor 1.2 of its own floor at $N=32$.
# - **A 301-term torsion series is not converged.** With $p,q\le301$ terms the nodal error is 3.0e-7 (the expectation recorded before the computation, in PLAN.md, was 1e-10) and the gradient error 6.5e-4 on edge nodes; $p,q\le2001$ is used, and the interior numbers change by at most 1.8e-4 (maximum error) and 4.3e-3 (tail residual), relative, with the shorter series.
#
# ## Conventions
#
# Lengths in units of the period of the cell; the cell is $Y=[0,1)^2$ and $\varepsilon=1/N$ with $N$ cells per side of the unit square $\Omega$. The cell is a disk of radius $0.3$ with $a_\text{in}=10$ in $a_\text{out}=1$, center-sampled on $m=16$ pixels per cell: a *staircase* disk. Everything below, the cell problem and the fine problems, uses the same discrete cell and Q1 elements with the cell's pixel grid as the element grid, so the comparisons test homogenization, not geometry (the smooth disk would give a different $\sigma^*$, and section 1 splits the difference into area, staircase shape and discretization). The cell is symmetric under the dihedral group $D_4$, in particular mirror symmetric about the node lines $y_1\in\{0,\tfrac12\}$ and $y_2\in\{0,\tfrac12\}$ and centrosymmetric; $\sigma^*=\sigma^*_h I$ is isotropic. The pattern is shifted by $s$ cells, $s\in\{0,\tfrac14,\tfrac12\}$ (0, 4, 8 pixels at $m=16$); then the left edge $x_1=0$ cuts the cell at the cell coordinate $-s\bmod1$, that is $0$, $\tfrac34$ and $\tfrac12$. The cases $s=0,\tfrac12$ are *mirror-line cuts* and $s=\tfrac14$ is a *generic cut*. Time dependence plays no role here (static problem; the TM reading of $a$ as $\varepsilon_r$ with $E$ along the cylinders is for notebook 06).
#
# Norms: $L^\infty$ over nodes; $L^2$ and $H^1$ seminorm of the Q1 interpolant, exact on the elements; "interior" means nodes, or elements with centers, in $[\tfrac14,\tfrac34]^2$. Slopes are the negative slope of $\log$ error against $\log N$ (least squares over the stated $N$), "last pair" is $\log_2$ of the error ratio for the last doubling of $N$.

# %%
SETUP

# %%
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path.cwd()))
import matplotlib.ticker
import cellsolver as cs
import twoscale as ts
import rayleigh as ry          # only for the smooth-disk reference values in section 1
from scipy.optimize import brentq, minimize_scalar

T0 = time.perf_counter()
BOX = ts.interior_box()


def slope(x, y):
    """Negative log-log slope of y against x (least squares)."""
    return -np.polyfit(np.log(np.asarray(x, float)), np.log(np.asarray(y, float)), 1)[0]


def plain_ticks(ax, ticks):
    """Log axis with the given tick values written as plain numbers."""
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def pairs(y):
    """log2 ratios of consecutive entries: local rates for a doubling sequence."""
    y = np.asarray(y, float)
    return np.log2(y[:-1] / y[1:])


def table(title, rowlab, rows, cols, data, fmt="{:10.3e}", slopes=None, x=None):
    """Aligned table: rows labeled by rowlab values, columns by names; adds overall slope and last-pair rate rows."""
    print(title)
    print(f"{rowlab:>6s} " + " ".join(f"{c:>10s}" for c in cols))
    for i, r in enumerate(rows):
        print(f"{r:>6} " + " ".join(fmt.format(data[c][i]) for c in cols))
    if x is not None:
        print(f"{'slope':>6s} " + " ".join(f"{slope(x, data[c]):10.3f}" for c in cols))
        print(f"{'last':>6s} " + " ".join(f"{pairs(data[c])[-1]:10.3f}" for c in cols))


# %% [markdown]
# ## 1. Correctors and the homogenized problem
#
# With $\chi_j$ the $Y$-periodic zero-mean solution of $-\nabla\!\cdot a(e_j+\nabla\chi_j)=0$, the effective tensor is $\sigma^*_{ij}=\langle a(e_i+\nabla\chi_i)\cdot e_j\rangle$ and $u_0$ solves $-\sigma^*_h\Delta u_0=1$ in $\Omega$, $u_0=0$ on $\partial\Omega$: $u_0=\tau/\sigma^*_h$ with the torsion function $\tau=\sum_{p,q\text{ odd}}\frac{16}{\pi^4pq(p^2+q^2)}\sin p\pi x\sin q\pi y$. This series is the only place where an analytic $u_0$ enters, so its truncation matters: the coefficients decay only like $1/(pq(p^2+q^2))$, the truncation error of $\tau$ behaves like $3/n_\text{terms}^3$, and that of $\nabla\tau$ like $n_\text{terms}^{-2}$, worst at the edges. The convergence is therefore measured here before $u_0$ is used.

# %%
M = 16
CELL = cs.disk_pixels(M, 0.3, 10.0, 1.0)
CHI, SIG = ts.cell_correctors(CELL)
SS = SIG[0, 0]
f_stair = (CELL > 1.5).mean()
RAY_SMOOTH = float(ry.sigma_eff(np.pi * 0.09, 10.0, 40))
RAY_PIX = float(ry.sigma_eff(f_stair, 10.0, 40))
print(f"cell: staircase disk, m = {M}, area fraction {f_stair:.5f} (smooth disk pi 0.3^2 = {np.pi * 0.09:.5f}), a_in = 10, a_out = 1")
print(f"sigma*_h = [[{SIG[0, 0]:.6f}, {SIG[0, 1]:.2e}], [{SIG[1, 0]:.2e}, {SIG[1, 1]:.6f}]]   (Loewner upper bound of the true value on this cell)")


def q1_energy_tensor(cell, chi, weighted=True):
    """Own recomputation of <a (e_i + grad chi_i) . (e_j + grad chi_j)> (a = 1 if not weighted) from the nodal correctors:
    2 x 2 Gauss points per Q1 element (exact for Q1), no call into twoscale or cellsolver."""
    m = cell.shape[0]
    g = (1 - 1 / np.sqrt(3)) / 2, (1 + 1 / np.sqrt(3)) / 2
    aw = cell if weighted else np.ones_like(cell)
    out = np.zeros((2, 2))
    for xi in g:
        for eta in g:
            G = []
            for d in range(2):
                c00, c10 = chi[d], np.roll(chi[d], -1, 0)
                c01, c11 = np.roll(chi[d], -1, 1), np.roll(chi[d], (-1, -1), (0, 1))
                gx = ((c10 - c00) * (1 - eta) + (c11 - c01) * eta) * m + (d == 0)
                gy = ((c01 - c00) * (1 - xi) + (c11 - c10) * xi) * m + (d == 1)
                G.append((gx, gy))
            for i in range(2):
                for j in range(2):
                    out[i, j] += (aw * (G[i][0] * G[j][0] + G[i][1] * G[j][1])).mean() / 4
    return out


SIG_OWN = q1_energy_tensor(CELL, CHI)
print(f"own recomputation of sigma* from the energy of the correctors (exact element integrals): {SIG_OWN[0, 0]:.6f}")
check("isotropy: |s_xx - s_yy|", abs(SIG[0, 0] - SIG[1, 1]), 1e-12)
check("isotropy: |s_xy|", abs(SIG[0, 1]), 1e-12)
check("sigma*_h from the flux formula (cell_correctors) against an own energy recomputation from the correctors (Galerkin orthogonality), max abs difference", np.abs(SIG - SIG_OWN).max(), 1e-10)
# (cell_correctors calls cellsolver.q1_effective, so comparing the two modules would be an interface identity; it is not checked.)

# decomposition of the distance between sigma*_h and the smooth-disk value
LB16, UB16 = cs.q1_bracket(CELL)
LB_R, UB_R = cs.q1_bracket(np.kron(CELL, np.ones((8, 8))))      # the same pixel medium, each pixel cut into 8 x 8 elements
MID_R = 0.5 * (LB_R[0, 0] + UB_R[0, 0])
print(f"\nwhere does sigma*_h = {SS:.6f} lie above the smooth disk (Rayleigh 1892 multipole sums) {RAY_SMOOTH:.6f} (+{100 * (SS / RAY_SMOOTH - 1):.2f}%)? (a_in = 10, 40 terms)")
print(f"   smooth disk at the staircase area fraction {f_stair:.5f}:            {RAY_PIX:.6f}   area            +{100 * (RAY_PIX / RAY_SMOOTH - 1):.2f}%")
print(f"   pixel medium, refined 8 x 8, certified bracket:   [{LB_R[0, 0]:.6f}, {UB_R[0, 0]:.6f}] (midpoint {MID_R:.6f})   staircase shape +{100 * (MID_R / RAY_PIX - 1):.2f}%")
print(f"   pixel medium, Q1 at m = {M}, certified bracket:     [{LB16[0, 0]:.6f}, {UB16[0, 0]:.6f}]   Q1 error at m = {M}: +{100 * (SS / MID_R - 1):.2f}% (upper bound against the refined midpoint)")
check_in("nested trial spaces: the refined bracket lies inside the m = 16 bracket (smallest of L_ref - L_16 and U_16 - U_ref)", min(LB_R[0, 0] - LB16[0, 0], UB16[0, 0] - UB_R[0, 0]), 0.0, 1.0)
check_in("area alone does not explain the distance: the area step is less than half of the 5.3% (ratio of log steps)", np.log(RAY_PIX / RAY_SMOOTH) / np.log(SS / RAY_SMOOTH), 0.0, 0.5)

# torsion series: truncation study
xs = np.linspace(0, 1, 513)
TS = {nt: ts.torsion(xs, xs, nt) for nt in (301, 601, 2001, 4001)}
ib = slice(128, 385)                           # interior nodes [1/4, 3/4]
print("\ntorsion series, truncation p, q <= nterms, compared with the 4001-term sum on the 513 x 513 grid:")
print(f"{'nterms':>7s} {'max|du|':>10s} {'3/nt^3':>11s} {'max|dux| edge nodes':>20s} {'max|dux| interior':>18s}")
for nt in (301, 601, 2001):
    du = np.abs(TS[nt][0] - TS[4001][0]).max()
    dg_edge = max(np.abs(TS[nt][1] - TS[4001][1])[0, :].max(), np.abs(TS[nt][1] - TS[4001][1])[-1, :].max(), np.abs(TS[nt][1] - TS[4001][1])[:, 0].max())
    dg_int = np.abs(TS[nt][1] - TS[4001][1])[ib, ib].max()
    print(f"{nt:7d} {du:10.2e} {3 / nt ** 3:11.2e} {dg_edge:20.2e} {dg_int:18.2e}")
du301_601 = np.abs(TS[301][0] - TS[601][0]).max()
du2001 = np.abs(TS[2001][0] - TS[4001][0]).max()
print(f"max nodal |u(301) - u(601)| = {du301_601:.2e}  (1e-10 was expected beforehand, PLAN.md: unreachable with this series, since the error is 3/nterms^3)")
du301 = np.abs(TS[301][0] - TS[4001][0]).max()
check_in("2001-term series: max nodal |u(2001) - u(4001)| below 5e-9 (measured 1.9e-9, a few times 3/2001^3; the reference's own tail is 5e-11)", du2001, 0, 5e-9)
check_in("the series error falls at least like nterms^-2.5 from 301 to 2001 (ratio of the errors against the 4001-term sum, at least 6.65^2.5 = 114)", du301 / du2001, 114, 1e4)
check("torsion function at the center (published value 0.0736713533)", TS[2001][0][256, 256], 2e-9, 0.0736713533)
NT = 2001                                      # used for u0 and grad u0 everywhere below

errs_h = {}
for n in (128, 256):
    u_h = ts.fine_dirichlet(np.ones((n, n)))
    xg = np.arange(n + 1) / n
    errs_h[n] = np.abs(u_h - ts.torsion(xg, xg, NT)[0]).max()
print(f"\nhomogeneous medium a = 1: max|u_h - tau|  n = 128: {errs_h[128]:.3e}   n = 256: {errs_h[256]:.3e}")
check_in("homogeneous medium: only the Q1 error remains; ratio of max nodal errors n = 128 over 256 (O(h^2) gives 4)", errs_h[128] / errs_h[256], 3.0, 5.0)

# %% [markdown]
# The table shows that the tolerance $10^{-10}$ between 301 and 601 terms, recorded before the computation (PLAN.md), is unreachable: that difference is 3.3e-7, and against the 4001-term sum the 301-term series is off by 3.0e-7 in $\tau$ and by 6.5e-4 in $\partial_x\tau$ on the nodes of the edges (3.2e-6 in the interior). The error of $\tau$ is a few times the $3/n_\text{terms}^3$ estimate in the table, which was fitted on a 129-point grid; the maximum over a 513-point grid sits next to the edges, where the sine series converges slowest. With $p,q\le2001$ the nodal error is 1.9e-9, and the check above sets its tolerance (5e-9) from this table. The gradient is still only accurate to 5.3e-5 on the edge nodes, which matters for $\bar u_1$ near the edges and for $w_\varepsilon$ there; section 4 measures how much this contaminates the interior numbers. The center value of $\tau$ reproduces the published value to 2.4e-12. For the homogeneous medium the error of the Q1 solution falls by a factor of 4.0003 between $n=128$ and $256$: only the $O(h^2)$ discretization error is left in the comparison with $\tau$.
#
# The staircase disk has area fraction 0.29688, not $\pi\,0.3^2=0.28274$, but the area explains less than half of the distance between $\sigma^*_h=1.688269$ and the smooth-disk value 1.602942 (5.32% in all). The printed decomposition is: area, +2.50% (the Rayleigh value 1.643002 at the staircase fraction); staircase shape, +1.94% (the same pixel medium refined $8\times8$ has the certified bracket [1.674091, 1.675669]; a staircase effect of the kind notebook 01 reports); the Q1 error at $m=16$, +0.80% (the Q1 bracket is [1.654121, 1.688269], and $\sigma^*_h$ is its upper end). The refined bracket lies inside the coarse one, as nested trial spaces require. Everything below refers to the Q1 cell at $m=16$, whose $\sigma^*_h$ is the number used throughout; the hierarchy is self-consistent, so the two-scale conclusions do not depend on the gap to the smooth disk. The check that $\sigma^*_h$ from `cell_correctors` equals `cellsolver.q1_effective` was dropped, because the first calls the second (an interface identity); the check against the energy of the correctors, recomputed here with Gauss points, replaces it (Galerkin orthogonality makes the flux and energy formulas agree for the discrete minimizer).
#
# ## 2. Global rates, and why the textbook $H^1$ rate needs a generic cut
#
# The standard results are $\|u_\varepsilon-u_0\|_{L^2}=O(\varepsilon)$ and, for the first-order expansion, $\|\nabla(u_\varepsilon-w_\varepsilon)\|_{L^2}=O(\varepsilon^{1/2})$. The second is sharp in general because $w_\varepsilon$ does not vanish on $\partial\Omega$: its trace is $\varepsilon\chi_j(x/\varepsilon)\partial_ju_0$, and the boundary-layer function that repairs it has gradient of size one over a layer of width $\varepsilon$. On an edge, say $x_1=0$, $\partial_2u_0=0$ (tangential derivative of a function that vanishes there), so the trace is $\varepsilon\chi_1\partial_1u_0$. Since the cell is mirror symmetric, $\chi_1$ is odd about the lines $y_1\in\{0,\tfrac12\}$ and vanishes on them: for an edge on such a line there is no first-order mismatch, and the $H^1$ rate should improve to $\varepsilon$. For a generic cut the trace is a nonzero oscillation of amplitude $O(\varepsilon)$ and the standard estimate $\varepsilon^{1/2}$ is expected to be sharp (the general estimate is in the books of Bensoussan, Lions and Papanicolaou and of Jikov, Kozlov and Oleinik, from memory, not checked here for this boundary condition; Allaire and Amar 1999 describe the tail mechanism that section 4 uses; Gerard-Varet and Masmoudi 2011, 2012 treat the layers for general cuts, not used quantitatively here). The conditions of the improvement are those of the symmetry argument: mirror symmetric discrete cell about the cut node lines (m even, centered staircase), the same cell phase at opposite edges (integer $N$), and a $\partial_2u_0$ that vanishes on the edge; the corners of the torsion solution add at most logarithmic factors.
#
# Fine problems: $N\in\{4,8,16,32\}$ cells per side ($n=64$ to $512$ pixels), $s\in\{0,\tfrac14,\tfrac12\}$.

# %%
Ns = [4, 8, 16, 32]
SHIFTS = {"0": 0.0, "1/4": 0.25, "1/2": 0.5}
FINE = {}                                      # cache of fine solutions: (id of cell, N, shift in pixels) -> (a, u)
U0 = {}


def fine(cell, m, N, spx):
    key = (id(cell), N, spx)
    if key not in FINE:
        a = ts.periodic_medium(cell, N, spx)
        FINE[key] = (a, ts.fine_dirichlet(a))
    return FINE[key]


def u0_grid(n, nt):
    if (n, nt) not in U0:
        xg = np.arange(n + 1) / n
        U0[(n, nt)] = ts.torsion(xg, xg, nt)
    return U0[(n, nt)]


def run_case(cell, chi, sig, m, N, s, nt=NT, keep=False):
    """Fine solve, first-order expansion, boundary-layer tails and u1bar for one (N, s); returns a dict of scalars."""
    spx = int(round(s * m))
    n = N * m
    a, u = fine(cell, m, N, spx)
    t, tx, ty = (q / sig for q in u0_grid(n, nt))
    cn = np.stack([ts.periodic_nodal(chi[d], N, spx) for d in range(2)])
    w = ts.first_order_expansion(t, tx, ty, cn, N)
    tails = ts.edge_tails(a, cn, m, 6, 0)
    u1 = ts.harmonic_dirichlet(ts.edge_data(tails, tx, ty))
    e0, e1, e1i = ts.norms(u - t), ts.norms(u - w), ts.norms(u - w, BOX)
    out = dict(Linf0=e0[0], L20=e0[1], H1=e1[2], H1i=e1i[2], Linfi=e1i[0], tails=tails,
               resid=ts.norms((u - w) * N - u1, BOX)[0], u1max=ts.norms(u1, BOX)[0], cn=cn)
    if keep:
        out.update(err=(u - w) * N, u1=u1)
    return out


RUNS = {}
for sname, s in SHIFTS.items():
    for N in Ns:
        RUNS[(sname, N)] = run_case(CELL, CHI, SS, M, N, s, keep=(sname == "1/4" and N == 32))
print(f"{time.perf_counter() - T0:6.1f} s elapsed (12 fine solves, m = 16)\n")


def col(key, sn, Nl=Ns, runs=None):
    runs = RUNS if runs is None else runs
    return np.array([runs[(sn, N)][key] for N in Nl])


cols = list(SHIFTS)
table("||u_eps - u0||_L2 (rows N, columns: shift s)", "N", Ns, cols, {c: col("L20", c) for c in cols}, x=Ns)
print()
table("||grad(u_eps - w_eps)||_L2, whole square (H1 seminorm)", "N", Ns, cols, {c: col("H1", c) for c in cols}, x=Ns)
print()
table("same over the interior elements, centers in [1/4, 3/4]^2", "N", Ns, cols, {c: col("H1i", c) for c in cols}, x=Ns)

for c in cols:
    check(f"L2 rate of u_eps - u0 over N = 4-32, s = {c} (within 0.15 of 1)", slope(Ns, col("L20", c)), 0.15, 1.0)
H1_GEN_LAST = pairs(col("H1", "1/4"))[-1]
check_in("H1, generic cut s = 1/4: last-pair rate (pre-asymptotic window around an exploratory value, 0.586)", H1_GEN_LAST, 0.45, 0.70)
for c in ("0", "1/2"):
    check(f"H1, mirror-line cut s = {c}: overall rate within 0.1 of 1", slope(Ns, col("H1", c)), 0.1, 1.0)
for c in cols:
    check(f"interior H1 rate, s = {c}: within 0.1 of 1", slope(Ns, col("H1i", c)), 0.1, 1.0)

# %%
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
mk = {"0": "o", "1/4": "s", "1/2": "^"}
for i, c in enumerate(cols):
    ax[0].loglog(Ns, col("H1", c), marker=mk[c], color=CAT[i], label=f"s = {c}")
    ax[1].loglog(Ns, col("H1i", c), marker=mk[c], color=CAT[i], label=f"s = {c}")
for a_, p_ in ((ax[0], 0.5), (ax[0], 1.0), (ax[1], 1.0)):
    y0 = col("H1", "1/4")[0] if a_ is ax[0] else col("H1i", "1/4")[0]
    a_.loglog(Ns, y0 * (np.array(Ns) / Ns[0]) ** (-p_) * (0.45 if p_ == 1.0 else 1.0), "--", color=MUTED, lw=1)
    a_.text(Ns[-1], y0 * (Ns[-1] / Ns[0]) ** (-p_) * (0.45 if p_ == 1.0 else 1.0), f" N^-{p_:g}", color=INK2, va="center", fontsize=9)
ax[0].set(xlabel="N (cells per side)", ylabel=r"$\|\nabla(u_\varepsilon-w_\varepsilon)\|_{L^2(\Omega)}$", title="whole square")
ax[1].set(xlabel="N (cells per side)", ylabel=r"$\|\nabla(u_\varepsilon-w_\varepsilon)\|_{L^2}$, interior", title=r"interior $[1/4,3/4]^2$")
ax[0].legend()
for a_ in ax:
    plain_ticks(a_, Ns)
plt.show()
print("Figure 1. H1 error of the first-order expansion against N. Left: over the whole square, where the boundary layer enters; right: over the interior elements. Dashed: N^-1/2 and N^-1.")

# %% [markdown]
# **$L^2$.** The error of $u_\varepsilon-u_0$ falls like $\varepsilon$ for all three cuts (slopes 1.01, 0.97 and 1.02; last-pair rates 1.00 for all three): the $L^2$ rate does not see where the edge cuts the cell.
#
# **$H^1$.** The three columns of the second table separate. For the mirror-line cuts $s=0,\tfrac12$ the global $H^1$ error falls with slopes 0.957 and 0.989 (last pairs 0.988 and 0.993), that is like $\varepsilon$, as the symmetry argument predicts. For the generic cut it falls with slope 0.646 overall and 0.587 over the last doubling: the boundary layer dominates, and the rate between the two. The third table shows that the interior elements see none of this (slopes 1.007, 1.000, 0.979): the $H^1$ error is $O(\varepsilon)$ there for every cut, the order of the neglected second-order term, and it is the layer of width one cell around the boundary that costs the half power.
#
# The rate for the generic cut is not yet $1/2$ at these $N$. The squared error is a sum of a smooth part $A\varepsilon^2$ and a layer part $B\varepsilon^q$; with $q=1$ the layer part wins only when $\varepsilon<B/A$, and the fit in section 3 puts that crossover at $N\approx4.4$, so the local rate approaches $1/2$ slowly from above. A two-term model with the exponents 1 and 2 built in cannot fail to give an asymptotic rate between $1/2$ and 1, so the informative test is the fit with free $q$ in section 3: $q=1.019\pm0.007$ and an out-of-sample prediction of $N=64$ from $N\le32$ good to 0.50%. The local rates fall monotonically and stay above $1/2$ in both sequences. This is consistent with $\varepsilon^{1/2}$ as the asymptotic rate; it is not an observation of the asymptote.
#
# ## 3. The interior error depends on where the boundary cuts the cell
#
# The first-order expansion would be second-order accurate in the interior, $u_\varepsilon-w_\varepsilon=O(\varepsilon^2)$, were it not for the boundary layer. The tail of the layer does not decay with the distance from the edge on the macroscopic scale: it is a constant offset on the edge, harmonic in the interior, and it enters at order $\varepsilon$. For the mirror-line cuts the offset vanishes by symmetry and the interior error is $O(\varepsilon^2)$. For the generic cut it is $\varepsilon\bar u_1$ with $\bar u_1$ determined in section 4.
#
# The symmetry that makes the traces vanish is checked on the discrete correctors first. Then the interior maximum of $|u_\varepsilon-w_\varepsilon|$ over nodes in $[\tfrac14,\tfrac34]^2$ is tabulated for all cuts. A second table repeats it with a coarser cell, $m=8$, up to $N=64$. At $m=8$ the center-sampled disk of radius 0.3 is a $4\times4$ pixel block, that is, the square inclusion of side $1/2$ ($f=1/4$) with corner singularities, a different shape and not only a different resolution; its effective conductivity is known exactly, $\sqrt{(1+3h)/(3+h)}$ at $h=10$, that is $\sqrt{31/13}=1.544220$. The square array at area fraction $1/4$ is a four-phase checkerboard with three equal tiles, and this value is that case of the Mortola-Steffe formula, conjectured by Mortola and Steffe (1985) and proved in 2001 by Craster and Obnosov and, independently, by Milton (the specialization is from memory and is confirmed below by the certified brackets). The table tests whether the rates of the first sequence are pre-asymptotic.

# %%
check("symmetry zeros: max|chi_1| on the node column y1 = 0 and y1 = 1/2", max(np.abs(CHI[0][0, :]).max(), np.abs(CHI[0][M // 2, :]).max()), 1e-12)
check("symmetry zeros: max|chi_2| on the node rows y2 = 0 and y2 = 1/2", max(np.abs(CHI[1][:, 0]).max(), np.abs(CHI[1][:, M // 2]).max()), 1e-12)
chi1_gen = CHI[0][(-4) % M, :]                 # the left edge of the s = 1/4 domain sits at node column (-s m) mod m = 12
print(f"for comparison, max|chi_1| on the generic cut y1 = 3/4: {np.abs(chi1_gen).max():.4f}  (chi_1 ranges over +-{np.abs(CHI[0]).max():.4f})")

table("interior max |u_eps - w_eps| over nodes in [1/4, 3/4]^2, m = 16", "N", Ns, cols, {c: col("Linfi", c) for c in cols}, x=Ns)
print(f"{'N 8-32':>6s} " + " ".join(f"{slope(Ns[1:], col('Linfi', c)[1:]):10.3f}" for c in cols) + "   (slope over N = 8-32 only)")
for c in cols:
    print(f"   {c:>4s}: " + ", ".join(f"{p:.3f}" for p in pairs(col("Linfi", c))))
for c in ("0", "1/2"):
    check(f"mirror-line cut s = {c}: interior L_inf slope over N = 8-32 within 0.2 of 2", slope(Ns[1:], col("Linfi", c)[1:]), 0.2, 2.0)
check("generic cut s = 1/4: interior L_inf slope over N = 8-32 within 0.1 of 1", slope(Ns[1:], col("Linfi", "1/4")[1:]), 0.1, 1.0)

# second sequence: m = 8, N up to 64 (n = 512)
M8 = 8
CELL8 = cs.disk_pixels(M8, 0.3, 10.0, 1.0)
CHI8, SIG8 = ts.cell_correctors(CELL8)
SS8 = SIG8[0, 0]
Ns8 = [8, 16, 32, 64]
RUNS8 = {(sn, N): run_case(CELL8, CHI8, SS8, M8, N, s) for sn, s in SHIFTS.items() for N in Ns8}
print(f"\nsecond sequence, m = 8: area fraction {(CELL8 > 1.5).mean():.4f}, sigma*_h = {SS8:.6f}   ({time.perf_counter() - T0:.1f} s elapsed)")
IS_SQUARE = bool(np.array_equal(CELL8, cs.square_pixels(M8, 4, 10.0, 1.0)))
H_SQ = 10.0
EXACT_SQ = float(np.sqrt((1 + 3 * H_SQ) / (3 + H_SQ)))
L8, U8 = cs.q1_bracket(CELL8)
L8r, U8r = cs.q1_bracket(np.kron(CELL8, np.ones((8, 8))))
print(f"  the center-sampled disk of radius 0.3 at m = 8 is a 4 x 4 pixel block: the square inclusion of side 1/2 (f = 1/4) of notebook 02: {IS_SQUARE}")
print(f"  exact value sqrt((1 + 3h)/(3 + h)) at h = 10 is sqrt(31/13) = {EXACT_SQ:.6f} (sqrt(31/13) = {np.sqrt(31 / 13):.6f}); Q1 bracket at m = 8 [{L8[0, 0]:.6f}, {U8[0, 0]:.6f}]; refined 8 x 8 bracket [{L8r[0, 0]:.6f}, {U8r[0, 0]:.6f}]")
check_in("m = 8 cell is exactly the pixel square of side 4 (number of differing pixels)", float(np.sum(CELL8 != cs.square_pixels(M8, 4, 10.0, 1.0))), 0.0, 0.0)
check_in("exact sqrt(31/13) lies inside the certified Q1 bracket at m = 8 (smaller of the two margins)", min(EXACT_SQ - L8[0, 0], U8[0, 0] - EXACT_SQ), 0.0, 1.0)
check_in("exact sqrt(31/13) lies inside the refined certified bracket (smaller of the two margins)", min(EXACT_SQ - L8r[0, 0], U8r[0, 0] - EXACT_SQ), 0.0, 1.0)
print(f"  mirror symmetry on nodes (max|chi_1| on node columns 0 and 4): {max(np.abs(CHI8[0][0, :]).max(), np.abs(CHI8[0][4, :]).max()):.1e}")
table("interior max |u_eps - w_eps|, m = 8", "N", Ns8, cols, {c: col("Linfi", c, Ns8, RUNS8) for c in cols}, x=Ns8)
table("global H1 seminorm of u_eps - w_eps, m = 8", "N", Ns8, cols, {c: col("H1", c, Ns8, RUNS8) for c in cols}, x=Ns8)
table("interior H1, m = 8", "N", Ns8, cols, {c: col("H1i", c, Ns8, RUNS8) for c in cols}, x=Ns8)
for c in cols:
    print(f"   local interior-Linf rates, s = {c:>3s}: " + ", ".join(f"{p:.3f}" for p in pairs(col("Linfi", c, Ns8, RUNS8))) + "     global H1: " + ", ".join(f"{p:.3f}" for p in pairs(col("H1", c, Ns8, RUNS8))))
for c in ("0", "1/2"):
    check(f"m = 8, mirror-line cut s = {c}: interior L_inf slope over N = 16-64 within 0.2 of 2", slope(Ns8[1:], col("Linfi", c, Ns8, RUNS8)[1:]), 0.2, 2.0)
check("m = 8, generic cut s = 1/4: interior L_inf slope over N = 16-64 within 0.1 of 1", slope(Ns8[1:], col("Linfi", "1/4", Ns8, RUNS8)[1:]), 0.1, 1.0)
h8 = pairs(col("H1", "1/4", Ns8, RUNS8))
check_in("m = 8, generic cut: the local H1 rates fall monotonically with N (largest increase between consecutive pairs <= 0)", float(np.max(np.diff(h8))), -1.0, 0.0)
check_in("m = 8, generic cut: the last-pair H1 rate (N = 32 to 64) stays above 1/2", h8[-1], 0.5, 0.65)
for c in ("0", "1/2"):
    check(f"m = 8, mirror-line cut s = {c}: global H1 slope over N = 8-64 within 0.1 of 1", slope(Ns8, col("H1", c, Ns8, RUNS8)), 0.1, 1.0)

# extension of the m = 16 generic-cut sequence to N = 64 (n = 1024), the longest single solve of the notebook
R64 = run_case(CELL, CHI, SS, M, 64, 0.25)
H1g = np.array(list(col("H1", "1/4")) + [R64["H1"]])
Ng5 = Ns + [64]
print(f"\nm = 16, generic cut extended to N = 64 (n = 1024): H1 = {R64['H1']:.4e}, interior Linf = {R64['Linfi']:.4e}, residual of the tail prediction = {R64['resid']:.4e}   ({time.perf_counter() - T0:.1f} s elapsed)")
print("   local H1 rates s = 1/4, N = 4-8, ..., 32-64: " + ", ".join(f"{p:.3f}" for p in pairs(H1g)))
# models for the squared error: e^2 = B eps^q + A eps^2 (q = 1 is the two-term model: boundary layer ~ eps^(1/2) in H1 plus a smooth part ~ eps),
# fitted by least squares on the relative error of e^2 (rows weighted by 1/e^2)
def fit_two_term(Ng, e):
    eps = 1.0 / np.asarray(Ng, float)
    X = np.stack([eps, eps ** 2], 1) / (e ** 2)[:, None]
    coef = np.linalg.lstsq(X, np.ones(len(e)), rcond=None)[0]
    res = X @ coef - 1.0
    cov = (res @ res / (len(e) - 2)) * np.linalg.inv(X.T @ X)
    return coef, np.sqrt(np.diag(cov))


def model(coef, Ng, q=1.0):
    eps = 1.0 / np.asarray(Ng, float)
    return np.sqrt(coef[0] * eps ** q + coef[1] * eps ** 2)


def fit_q(Ng, e, q):
    """B, A (linear) at a fixed exponent q; returns the coefficients and the max relative misfit of e."""
    eps = 1.0 / np.asarray(Ng, float)
    X = np.stack([eps ** q, eps ** 2], 1) / (e ** 2)[:, None]
    coef = np.linalg.lstsq(X, np.ones(len(e)), rcond=None)[0]
    return coef, np.abs(model(coef, Ng, q) / e - 1).max()


(Bc, Ac), (sB, sA) = fit_two_term(Ng5, H1g)
fit5 = model((Bc, Ac), Ng5)
MISFIT5 = np.abs(fit5 / H1g - 1).max()
print(f"   two-term model H1^2 = B eps + A eps^2 (relative least squares): B = {Bc:.4e} +- {sB:.1e}, A = {Ac:.4e} +- {sA:.1e}; max relative misfit of H1 over the five points {MISFIT5:.2e}; "
      f"the terms are equal at N = {Ac / Bc:.1f}; model local rate between N = 64 and 128: {np.log2(fit5[-1] / model((Bc, Ac), [128])[0]):.3f}")
# out-of-sample: fit on N <= 32, predict N = 64
(B4, A4), _ = fit_two_term(Ng5[:4], H1g[:4])
PRED64 = float(model((B4, A4), [64])[0])
print(f"   out of sample: two-term fit to N = 4-32 predicts H1(N = 64) = {PRED64:.4e} against the computed {H1g[-1]:.4e} (relative difference {PRED64 / H1g[-1] - 1:+.2e})")
Bm, Am = fit_two_term(Ns8, h1_8 := col("H1", "1/4", Ns8, RUNS8))[0]
(B8_, A8_), _ = fit_two_term(Ns8[:3], h1_8[:3])
PRED64_8 = float(model((B8_, A8_), [64])[0])
print(f"   m = 8: two-term fit to N = 8-32 predicts H1(N = 64) = {PRED64_8:.4e} against {h1_8[-1]:.4e} (relative difference {PRED64_8 / h1_8[-1] - 1:+.2e}); full fit misfit {np.abs(model((Bm, Am), Ns8) / h1_8 - 1).max():.2e}")
# free exponent: B eps^q + A eps^2, the informative test (the exponents 1 and 2 are not built in)
from scipy.optimize import least_squares
def resid_q(par):
    return model((par[0], par[2]), Ng5, par[1]) / H1g - 1


sol = least_squares(resid_q, [Bc, 1.0, Ac], x_scale=[Bc, 1.0, Ac])
QFIT = sol.x[1]
cov_q = (sol.fun @ sol.fun / (5 - 3)) * np.linalg.inv(sol.jac.T @ sol.jac)
qgrid = np.linspace(0.4, 1.6, 241)
mis_q = np.array([fit_q(Ng5, H1g, q)[1] for q in qgrid])
ok_q = qgrid[mis_q <= 0.01]
print(f"   free exponent H1^2 = B eps^q + A eps^2: q = {QFIT:.3f} +- {np.sqrt(cov_q[1, 1]):.3f} (rough, 2 degrees of freedom); max misfit {np.abs(sol.fun).max():.2e}; "
      f"exponents q in [{ok_q.min():.2f}, {ok_q.max():.2f}] keep the max misfit of H1 below 1%")
print(f"   (q = 1 means H1 ~ eps^(1/2) for the layer part; the fixed exponent 2 of the smooth part is kept)")
check_in(f"two-term model fits the five m = 16 generic-cut H1 errors (max relative misfit of H1; 5 points, 2 parameters)", MISFIT5, 0.0, 0.02)
check_in("layer coefficient B is at least 5 standard errors from zero (fit uncertainty from the residuals, 3 degrees of freedom)", Bc / sB, 5.0, 1e6)
check_in("out of sample: the two-term fit to N = 4-32 predicts H1(N = 64) to within 1% (relative difference)", abs(PRED64 / H1g[-1] - 1), 0.0, 0.01)
check_in("free exponent fit: the fitted layer exponent q of the squared error is within 0.25 of 1 (H1 rate 1/2)", abs(QFIT - 1.0), 0.0, 0.25)
check_in("m = 16, generic cut: the last-pair H1 rate (N = 32 to 64) is lower than the previous pair and above 1/2", pairs(H1g)[-1], 0.5, pairs(H1g)[-2])
check("m = 16, generic cut: interior L_inf rate over N = 32 to 64 within 0.1 of 1", pairs([RUNS[("1/4", 32)]["Linfi"], R64["Linfi"]])[0], 0.1, 1.0)

# %%
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
for i, c in enumerate(cols):
    ax[0].loglog(Ns, col("Linfi", c), marker=mk[c], color=CAT[i], label=f"s = {c}")
    ax[1].loglog(Ns8, col("Linfi", c, Ns8, RUNS8), marker=mk[c], color=CAT[i], label=f"s = {c}")
ax[0].loglog(Ns, col("Linfi", "1/4")[0] * (np.array(Ns) / 4.0) ** -1.0, "--", color=MUTED, lw=1)
ax[0].loglog(Ns, col("Linfi", "0")[0] * (np.array(Ns) / 4.0) ** -2.0, "--", color=MUTED, lw=1)
ax[1].loglog(Ns8, col("Linfi", "1/4", Ns8, RUNS8)[0] * (np.array(Ns8) / 8.0) ** -1.0, "--", color=MUTED, lw=1)
ax[1].loglog(Ns8, col("Linfi", "0", Ns8, RUNS8)[0] * (np.array(Ns8) / 8.0) ** -2.0, "--", color=MUTED, lw=1)
for a_, Nl, base1, base0 in ((ax[0], Ns, col("Linfi", "1/4")[0], col("Linfi", "0")[0]), (ax[1], Ns8, col("Linfi", "1/4", Ns8, RUNS8)[0], col("Linfi", "0", Ns8, RUNS8)[0])):
    a_.text(Nl[-1], base1 * (Nl[-1] / Nl[0]) ** -1.0, r" $N^{-1}$", color=INK2, va="center", fontsize=9)
    a_.text(Nl[-1], base0 * (Nl[-1] / Nl[0]) ** -2.0, r" $N^{-2}$", color=INK2, va="center", fontsize=9)
ax[0].set(xlabel="N (cells per side)", ylabel=r"$\max_{\rm int}|u_\varepsilon-w_\varepsilon|$", title="m = 16 pixels per cell")
ax[1].set(xlabel="N (cells per side)", ylabel=r"$\max_{\rm int}|u_\varepsilon-w_\varepsilon|$", title="m = 8 pixels per cell")
ax[0].legend()
plain_ticks(ax[0], Ns)
plain_ticks(ax[1], Ns8)
plt.show()
print("Figure 2. Interior maximum error of the first-order expansion against N for the three cuts, for two pixel resolutions of the cell. Dashed: N^-1 and N^-2.")

# %% [markdown]
# **Symmetry.** $\chi_1$ vanishes on the node columns $y_1=0,\tfrac12$ to $4\times10^{-15}$ and $\chi_2$ on the node rows to $7\times10^{-16}$; on the generic cut $y_1=\tfrac34$ it reaches 0.1865, out of a range of $\pm0.2312$. The trace mismatch is therefore zero on mirror-line cuts and of order 80% of the corrector's amplitude on the generic cut.
#
# **Rates.** For $s=0$ and $\tfrac12$ the interior maximum error falls with slopes 1.96 and 1.95 over $N=8$-$32$ and the local rates (1.848, 1.942, 1.977 for $s=0$; 1.859, 1.930, 1.966 for $s=\tfrac12$) approach 2 from below. For the generic cut the rate is 1.02, with local rates 1.029, 1.028, 1.017, and the error is larger by a factor of 43 at $N=32$. The mirror-line cut at $s=\tfrac12$ (the edge passes through the centers of the inclusions) has a larger error than $s=0$ by a constant factor of about 1.4. The second sequence ($m=8$, $N$ up to 64, the square inclusion of side $1/2$) gives the same picture: local interior rates 1.951, 1.981, 1.992 for $s=0$, 1.894, 1.950, 1.976 for $s=\tfrac12$ and 1.032, 1.020, 1.011 for $s=\tfrac14$; the global $H^1$ rates for the generic cut fall, 0.653, 0.594, 0.552, as in the $m=16$ sequence. The rates are the same at $m=8$ and $m=16$, for two different inclusion shapes; two resolutions do not exclude a dependence on the resolution beyond these.
#
# **Extension to $N=64$.** One more solve for the generic cut at $m=16$ ($n=1024$, the longest solve of the notebook) gives the local $H^1$ rates 0.709, 0.644, 0.587, 0.548. The two-term fit reproduces the five global errors to 0.43% (the printed misfit) and predicts 0.524 for the next doubling. Its layer coefficient is $B=2.5968\times10^{-3}\pm1.6\times10^{-5}$ (162 standard errors, from three degrees of freedom), and fitted on $N\le32$ alone it predicts $H^1(N=64)$ to 0.50% (0.15% for the $m=8$ sequence, fitted on $N=8$-$32$). With a free exponent the five points give $q=1.019\pm0.007$ and a misfit of 0.22%; the exponents $q\in[0.96,1.07]$ keep the misfit below 1%, so $q=1$ ($H^1\sim\varepsilon^{1/2}$ for the layer part) is inside the range the data allow and a layer exponent much different from it is not. The uncertainty is rough: five points, three parameters.
#
# **The cell at $m=8$.** The $4\times4$ block is the square of notebook 02: the pixels differ from `square_pixels(8, 4, ...)` in none, $\sigma^*_h=1.559345$ is the Q1 upper bound, the Q1 bracket is [1.529242, 1.559345] and the $8\times8$ refined bracket [1.543482, 1.544958], both containing the exact $\sqrt{31/13}=1.544220$. The two-scale conclusions do not depend on the gap between $\sigma^*_h$ and the exact value, because the hierarchy is discrete and self-consistent.
#
# ## 4. Boundary-layer tails predict the $O(\varepsilon)$ interior error
#
# Near an edge, $u_\varepsilon-w_\varepsilon$ is $\varepsilon$ times a function of the stretched coordinates that solves the cell problem on a half-strip (one cell wide along the edge, periodic, semi-infinite in the normal direction) with Dirichlet data $-\chi_j$ on the cut, multiplying $\partial_ju_0$ at the point of the edge. Its limit far from the edge, the *tail* $\bar\beta_j$, is a constant; this is the Allaire-Amar tail. On the macroscopic scale the tails act as boundary data for the interior correction: $(u_\varepsilon-w_\varepsilon)/\varepsilon\to\bar u_1$, with $\bar u_1$ harmonic (here $\sigma^*$ is isotropic) and $\bar u_1=\sum_j\bar\beta_j(\text{edge})\,\partial_ju_0$ on each edge. Only the normal-derivative terms survive, because $\bar\beta_j$ vanishes for the tangential corrector: the data $-\chi_2$ on an edge $x_1=\text{const}$ are odd about the mirror $y_2\to1-y_2$ of the cell, so the half-strip solution is odd and the constant tail, which is even, must be zero (a zero mean of the data alone would not imply it). The natural alternative, adding second-order correctors, cannot improve the interior error without the second-order tails.
#
# Strips are cut from the fine medium next to each edge, six cells deep and one cell wide, with natural boundary condition at the far end. The tail is the mean of the last node column; the left edge uses the leading six cells, the right edge the same strip reversed in $x_1$, and the bottom and top edges the transposed strips.

# %%
TL = RUNS[("1/4", 32)]["tails"]
EDG = ts.EDGES
print("tails beta_j(edge) for s = 1/4 (rows: corrector chi_j; columns: edge), N = 32:")
print(f"{'':>8s} " + " ".join(f"{e:>10s}" for e in EDG))
for j in range(2):
    print(f"{'chi_' + str(j + 1):>8s} " + " ".join(f"{TL[j, k]:10.5f}" for k in range(4)))
for sname in ("0", "1/2"):
    print(f"max |tail| over all four edges, both correctors, s = {sname}, N = 4..32: {max(np.abs(RUNS[(sname, N)]['tails']).max() for N in Ns):.2e}")
check("mirror-line cuts have zero tails (s = 0 and 1/2, all N)", max(np.abs(RUNS[(sn, N)]['tails']).max() for sn in ("0", "1/2") for N in Ns), 1e-12)
check("tangential-corrector tails vanish: chi_2 on the x1 edges and chi_1 on the x2 edges (s = 1/4, all N)",
      max(np.abs(RUNS[("1/4", N)]["tails"][[1, 1, 0, 0], [0, 1, 2, 3]]).max() for N in Ns), 1e-12)
check("by the diagonal symmetry of the cell the chi_2 tails on the x2 edges equal the chi_1 tails on the x1 edges", abs(TL[1, 2] - TL[0, 0]) + abs(TL[1, 3] - TL[0, 1]), 1e-12)
a10, _ = fine(CELL, M, 10, 4)
cn10 = np.stack([ts.periodic_nodal(CHI[k], 10, 4) for k in range(2)])
t6, t8 = ts.edge_tails(a10, cn10, M, 6, 0), ts.edge_tails(a10, cn10, M, 8, 0)
check("tail independent of the strip depth: |tail(6 cells) - tail(8 cells)|", np.abs(t6 - t8).max(), 1e-12)
check("tail independent of the offset along the edge", np.abs(t6 - ts.edge_tails(a10, cn10, M, 6, 5)).max(), 1e-12)

print("\ninterior error divided by eps, s = 1/4: tails predict it (m = 16)")
print(f"{'N':>4s} {'max|err/eps|':>13s} {'max|u1bar|':>11s} {'max|err/eps - u1bar|':>21s} {'ratio to max|u1bar|':>20s}")
for N in Ns:
    r = RUNS[("1/4", N)]
    print(f"{N:4d} {r['Linfi'] * N:13.4e} {r['u1max']:11.4e} {r['resid']:21.4e} {r['resid'] / r['u1max']:20.4f}")
res = col("resid", "1/4")
SL_RES = slope(Ns[1:], res[1:])
print(f"slope of the residual over N = 8-32: {SL_RES:.3f}; local rates {', '.join(f'{p:.3f}' for p in pairs(res))}")
check_in("tails predict the interior error: residual slope over N = 8-32 at least 0.8", SL_RES, 0.8, 2.5)
check_in("the residual rate rises with N (local rates increasing), towards the O(eps) rate of the next term", float(np.min(np.diff(pairs(res)))), 0.0, 1.0)
check_in("tails predict the interior error: residual / max|u1bar| at N = 32 at most 0.1", res[-1] / col("u1max", "1/4")[-1], 0.0, 0.1)
check_in("m = 8: residual slope over N = 16-64 at least 0.8", slope(Ns8[1:], col("resid", "1/4", Ns8, RUNS8)[1:]), 0.8, 2.5)
print("m = 8 sequence, s = 1/4: " + ", ".join(f"N={N}: resid {RUNS8[('1/4', N)]['resid']:.3e} (u1max {RUNS8[('1/4', N)]['u1max']:.3e})" for N in Ns8))
res8 = col("resid", "1/4", Ns8, RUNS8)
print(f"  slope over N = 16-64: {slope(Ns8[1:], res8[1:]):.3f}; tails: left {RUNS8[('1/4', 64)]['tails'][0, 0]:.5f}, right {RUNS8[('1/4', 64)]['tails'][0, 1]:.5f}")
check("m = 8: mirror-line cuts also have zero tails", max(np.abs(RUNS8[(sn, N)]['tails']).max() for sn in ("0", "1/2") for N in Ns8), 1e-12)

# sensitivity of the interior numbers to the truncation of the torsion series (other reading: torsion contamination)
print("\nsensitivity to the torsion truncation: interior L_inf and residual with p, q <= 301 instead of 2001")
print(f"{'s':>4s} {'N':>3s} {'Linfi(2001)':>12s} {'Linfi(301)':>12s} {'rel change':>11s} {'resid(2001)':>12s} {'resid(301)':>12s} {'rel change':>11s}")
SENS = 0.0
for sname in ("0", "1/4"):
    for N in (8, 16, 32):
        r2 = RUNS[(sname, N)]
        r3 = run_case(CELL, CHI, SS, M, N, SHIFTS[sname], nt=301)
        rc = abs(r3["Linfi"] / r2["Linfi"] - 1)
        rr = abs(r3["resid"] / r2["resid"] - 1)
        SENS = max(SENS, rc)
        print(f"{sname:>4s} {N:3d} {r2['Linfi']:12.4e} {r3['Linfi']:12.4e} {rc:11.2e} {r2['resid']:12.4e} {r3['resid']:12.4e} {rr:11.2e}")
check_in("the 301-term series changes the interior L_inf errors by less than 2e-3 relative (contamination is not what sets the rates)", SENS, 0.0, 2e-3)

# %%
r = RUNS[("1/4", 32)]
x = np.arange(513) / 512
lim = 1.5 * np.abs(r["u1"]).max()
fig, ax = plt.subplots(1, 3, figsize=(11.2, 3.7))
for a_, f_, t_ in ((ax[0], r["err"], r"$(u_\varepsilon-w_\varepsilon)/\varepsilon$"), (ax[1], r["u1"], r"$\bar u_1$ from the tails"), (ax[2], r["err"] - r["u1"], "difference")):
    im = a_.imshow(f_.T, extent=(0, 1, 0, 1), cmap="div_blue_red", vmin=-lim, vmax=lim)
    a_.plot([0.25, 0.75, 0.75, 0.25, 0.25], [0.25, 0.25, 0.75, 0.75, 0.25], color=INK, lw=0.8, ls="--")
    a_.set(title=t_, xlabel="$x_1$", ylabel="$x_2$", aspect="equal")
    a_.grid(False)
fig.colorbar(im, ax=ax, shrink=0.85, label="same color scale in all panels (+-1.5 max|u1bar|)")
plt.show()
print("Figure 3. Generic cut s = 1/4, N = 32: the scaled error of the first-order expansion (left), the interior correction predicted from the edge tails (center) and their difference (right), on one color scale. The dashed square is the interior region.")

# %% [markdown]
# **Tails.** For the generic cut the $\chi_1$ tails are 0.13554 on the left and 0.16978 on the right edge, and by the diagonal symmetry of the cell the $\chi_2$ tails on the bottom and top edges are the same; the tangential-corrector tails vanish to $3\times10^{-16}$. The two $x_1$ edges are different cuts: seen from the interior, the left edge cuts the cell at coordinate $\tfrac34$ and the right edge, after reflection, at $\tfrac14$, and the tail is not even in the cut position. For the mirror-line cuts all tails vanish to $5\times10^{-16}$. The tail does not depend on the strip depth: 6 cells against 8 differ by $4\times10^{-14}$, rounding, since the decay of section 5 gives $e^{-6\kappa}\approx10^{-18}$.
#
# **Prediction.** The interior error divided by $\varepsilon$ is within 1.4% of the prediction's amplitude already at $N=32$: $\max|\,(u_\varepsilon-w_\varepsilon)/\varepsilon|=2.16\times10^{-2}$ against $\max|\bar u_1|=2.13\times10^{-2}$, and $\max|\bar u_1|$ is independent of $N$ to within $2\times10^{-4}$ relative, as an $O(1)$ field must be. The residual $\max|(u_\varepsilon-w_\varepsilon)/\varepsilon-\bar u_1|$ falls from 43.8% of $\max|\bar u_1|$ at $N=4$ to 7.1% at $N=32$ (an exploratory run gave 7%), with local rates 0.797, 0.891, 0.941 that rise towards the rate 1 of the next order: the residual is the $O(\varepsilon^2)$ part of the error, divided by $\varepsilon$. At $m=8$ the same slope is 0.954 over $N=16$-$64$ and the tails are 0.14071 and 0.17750, 4% from the $m=16$ values: tails belong to the discrete cell, and values are comparable only at equal $m$. Figure 3 shows what the numbers say: away from the edges the scaled error is the harmonic field generated by the four tails, and the difference is confined to the layer of the edges.
#
# **Torsion contamination (other reading).** With the 301-term series, which has a gradient error of 6.5e-4 on edge nodes, the interior maximum errors change by at most 1.8e-4 relative and the tail residual by at most 4.3e-3 (at $N=32$). The truncation of the 301-term series would have been visible at the level of the residual at $N=32$, but it does not change any rate or any conclusion.
#
# ## 5. How thick is the boundary layer? Static evanescent Bloch waves
#
# A boundary-layer solution decays like a static Bloch wave (Allaire and Conca 1998 for Bloch waves in homogenization; used here only as a computational device) $e^{-\kappa y_1}p(y)$, $p$ periodic in $y_2$ and in $y_1$ up to the factor, solving $\nabla\!\cdot a\nabla u=0$. On one Q1 cell, open in $y_1$ and periodic in $y_2$, with Bloch factor $\lambda=e^{-\kappa}$ across the cell this is the quadratic eigenproblem $(\lambda^2A_1+\lambda A_0+A_{-1})v=0$, with $A_0=K_{aa}+S^TK_{bb}S$, $A_1=K_{ab}S$, $A_{-1}=S^TK_{ba}$ ($S$ picks the first node column, the others are interior columns and are eliminated exactly in `evanescent_exponents`). The roots with $|\lambda|<1$ are the decaying modes; their exponents $\kappa$ are real here (a symmetric cell), and the mirror $y_2\to1-y_2$ labels them even or odd. A given edge excites one parity: the data $-\chi_1$ on a generic left edge is even about the cell's $y_2$ mirror, the data $-\chi_2$ is odd.
#
# In a homogeneous medium the modes are $\cos(2\pi ky_2)$, with $\kappa=2\pi k$ in the continuum. The Q1 discretization changes this at $O(h^2)$ and so does the staircase of the inclusion, so exponents are only comparable at equal $m$. All tables below fix $m$.

# %%
rows = []
print("smallest decaying exponents kappa (even | odd parity), Q1 cells, r = 0.3 disk, a_out = 1")
print(f"{'m':>3s} {'a_in':>6s} {'kappa_even':>11s} {'kappa_odd':>10s} {'smallest':>9s} {'> hom?':>7s}")
KAP = {}
for m_ in (8, 16, 24):
    kh = ts.evanescent_exponents(np.ones((m_, m_)), 4)[0].real[0]
    KAP[(m_, 1.0)] = (kh, kh)
    print(f"{m_:3d} {'hom.':>6s} {kh:11.5f} {kh:10.5f} {kh:9.5f}")
    for ain in (0.1, 10.0, 100.0):
        kk, pp = ts.evanescent_exponents(cs.disk_pixels(m_, 0.3, ain, 1.0), 8)
        ke = kk.real[pp == 1][0]
        ko = kk.real[pp == -1][0]
        KAP[(m_, ain)] = (ke, ko)
        print(f"{m_:3d} {ain:6g} {ke:11.5f} {ko:10.5f} {min(ke, ko):9.5f} {str(min(ke, ko) > kh):>7s}")
kh16, kh8, kh24 = KAP[(16, 1.0)][0], KAP[(8, 1.0)][0], KAP[(24, 1.0)][0]
ratio_q = abs(kh16 - 2 * np.pi) / abs(kh8 - 2 * np.pi)
print(f"\nhomogeneous cell: kappa_1(m) - 2 pi = {kh8 - 2 * np.pi:.5f}, {kh16 - 2 * np.pi:.5f}, {kh24 - 2 * np.pi:.5f} for m = 8, 16, 24;  ratio 16 over 8 = {ratio_q:.4f} (O(h^2): 0.25)")
def kappa_hom_exact(m_):
    """Closed form of the lowest Q1 exponent of the homogeneous cell: m acosh(s/2), s = (8 - 2c)/(1 + 2c), c = cos(2 pi/m)."""
    c = np.cos(2 * np.pi / m_)
    return m_ * np.arccosh((8 - 2 * c) / (1 + 2 * c) / 2)


for m_ in (8, 16, 24):
    check(f"homogeneous QEP against the closed form m acosh(s/2), m = {m_}", KAP[(m_, 1.0)][0], 1e-8, kappa_hom_exact(m_))
print(f"closed form: {kappa_hom_exact(8):.6f}, {kappa_hom_exact(16):.6f}, {kappa_hom_exact(24):.6f}")
check_in("homogeneous QEP: |kappa_1(24) - 2 pi| / |kappa_1(16) - 2 pi| against (16/24)^2 = 0.444 (second order)", abs(kh24 - 2 * np.pi) / abs(kh16 - 2 * np.pi), 0.40, 0.48)
check_in("homogeneous QEP converges at second order: |kappa_1(16) - 2 pi| / |kappa_1(8) - 2 pi|", ratio_q, 0.2, 0.3)
for m_ in (8, 16, 24):
    for ain in (0.1, 10.0, 100.0):
        assert min(KAP[(m_, ain)]) > KAP[(m_, 1.0)][0]
MARG = {(m_, ain): min(KAP[(m_, ain)]) - KAP[(m_, 1.0)][0] for m_ in (8, 16, 24) for ain in (0.1, 10.0, 100.0)}
print("\nmargin of the smallest exponent over the homogeneous one at equal m (positive: thinner layer):")
print(f"{'a_in':>6s} " + " ".join(f"{'m = ' + str(m_):>9s}" for m_ in (8, 16, 24)))
for ain in (0.1, 10.0, 100.0):
    print(f"{ain:6g} " + " ".join(f"{MARG[(m_, ain)]:9.5f}" for m_ in (8, 16, 24)))
check_in("heterogeneity thins the layer at a_in in {0.1, 10, 100}: smallest margin over all m and contrasts at least 0.15 (a stated floor, below the smallest printed margin 0.188; the margins are not converged in m)",
         min(MARG.values()), 0.15, 10.0)
check_in("the margins vary with m: largest over smallest margin at a_in = 10 (m = 8, 16, 24) is above 1.2, so the value of the margin is not converged", max(MARG[(m_, 10.0)] for m_ in (8, 16, 24)) / min(MARG[(m_, 10.0)] for m_ in (8, 16, 24)), 1.2, 100.0)

# decay on the strips against the QEP exponent of the matching parity
kq, pq, dq = ts.evanescent_exponents(CELL, 6, return_details=True)
check_in("QEP residual (a_in = 10) below 1e-8", dq["residual"].max(), 0.0, 1e-8)
check_in("QEP parity expectation is +-1: max deviation below 1e-8", np.abs(np.abs(dq["parity_expectation"].real) - 1).max(), 0.0, 1e-8)
Nst = 10
PROF = {}
MATHLAB = {-1: r"odd: $\chi_2$, $s=0$", 1: r"even: $\chi_1$, $s=(1/4,0)$"}
print(f"\nstrip decay between depths 1 and 2 cells (L2 norm over the node column of beta - tail), a_in = 10, m = 16:")
print(f"{'data':>26s} {'strip rate':>11s} {'QEP kappa':>10s} {'rel diff':>9s} {'parity':>7s}")
for lab, shift, dcorr, par in (("odd: chi_2, s = 0", (0, 0), 1, -1), ("even: chi_1, s = (1/4, 0)", (4, 0), 0, 1)):
    a_ = ts.periodic_medium(CELL, Nst, shift)
    cn_ = np.stack([ts.periodic_nodal(CHI[k], Nst, shift) for k in range(2)])
    tl, fl = ts.edge_tails(a_, cn_, M, 6, 0, True)
    U = fl["left"][dcorr]
    rates = ts.strip_decay_rates(U, M, tl[dcorr, 0])
    kpar = kq.real[pq == par][0]
    print(f"{lab:>26s} {rates[1]:11.5f} {kpar:10.5f} {abs(rates[1] / kpar - 1):9.2e} {par:7d}")
    check_in(f"strip decay equals the QEP exponent of its parity: relative difference, {lab}", abs(rates[1] / kpar - 1), 0.0, 1e-2)
    PROF[lab] = (np.array([np.linalg.norm(U[k * M] - tl[dcorr, 0]) for k in range(6)]), kpar, MATHLAB[par])
print(f"\nattenuation per cell exp(-kappa) at the smallest exponent: homogeneous {np.exp(-kh16):.2e}, a_in = 10: {np.exp(-min(KAP[(16, 10.0)])):.2e}")
print("amplitude of the smallest layer mode at the interior box (depth N/4 cells), exp(-kappa N/4) with kappa = kappa_1(a_in = 10):")
print("   " + ", ".join(f"N = {N}: {np.exp(-min(KAP[(16, 10.0)]) * N / 4):.1e}" for N in Ns))

# sweep over the contrast: lowest exponent of each parity
AIN = np.geomspace(0.05, 200, 121)
KE, KO = [], []
for ain in AIN:
    kk, pp = ts.evanescent_exponents(cs.disk_pixels(16, 0.3, ain, 1.0), 8)
    KE.append(kk.real[pp == 1][0] if np.any(pp == 1) else np.nan)
    KO.append(kk.real[pp == -1][0] if np.any(pp == -1) else np.nan)
KE, KO = np.array(KE), np.array(KO)
print("\nweak contrast, m = 16: exponent minus the homogeneous value, even | odd")
WEAK = {}
for ain in (0.9, 0.95, 0.99, 1.01, 1.05, 1.1):
    kk, pp = ts.evanescent_exponents(cs.disk_pixels(16, 0.3, ain, 1.0), 8)
    WEAK[ain] = (kk.real[pp == 1][0] - kh16, kk.real[pp == -1][0] - kh16)
    print(f"   a_in = {ain:5.2f}: {WEAK[ain][0]:+.5f} | {WEAK[ain][1]:+.5f}")
check_in("weak contrast, a_in = 0.95: even above and odd below the homogeneous value (smaller of the two margins, floor 1e-6 = 1000 times the solver residual)", min(WEAK[0.95][0], -WEAK[0.95][1]), 1e-6, 1.0)
check_in("weak contrast, a_in = 1.05: even below and odd above the homogeneous value (smaller of the two margins, floor 1e-6)", min(-WEAK[1.05][0], WEAK[1.05][1]), 1e-6, 1.0)
check_in("the split is first order in (a_in - 1): odd shift per unit contrast at 0.99 over that at 0.95", (WEAK[0.99][1] / 0.01) / (WEAK[0.95][1] / 0.05), 0.8, 1.3)


def kap_pair(ain, m_=16):
    kk, pp = ts.evanescent_exponents(cs.disk_pixels(m_, 0.3, ain, 1.0), 8)
    return kk.real[pp == 1][0], kk.real[pp == -1][0]


# mean and splitting of the pair at small contrast: the mean shift is second order, the splitting first order
print("\nthe degenerate pair at weak contrast a_in = 1 + d, m = 16: splitting (odd - even) and mean shift (mean of the two minus the homogeneous value)")
print(f"{'d':>6s} {'split':>11s} {'mean shift':>12s}   {'split / d':>10s} {'mean / d^2':>11s}")
MS = {}
for d in (0.01, 0.02, 0.04):
    ke_, ko_ = kap_pair(1 + d)
    MS[d] = (ko_ - ke_, 0.5 * (ke_ + ko_) - kh16)
    print(f"{d:6.2f} {MS[d][0]:11.5f} {MS[d][1]:12.3e}   {MS[d][0] / d:10.4f} {MS[d][1] / d ** 2:11.4f}")
check_in("splitting doubles with d: split(0.02) / split(0.01) (first order gives 2)", MS[0.02][0] / MS[0.01][0], 1.8, 2.2)
check_in("the mean shift is second order: mean(0.02) / mean(0.01) (second order gives 4; a first-order mean would give 2)", MS[0.02][1] / MS[0.01][1], 3.4, 4.6)

# window of the smaller-than-homogeneous exponent: root finding and a scalar minimizer instead of the scan grid
f_min = lambda ain: min(kap_pair(ain)) - kh16
f_even = lambda ain: kap_pair(ain)[0] - kh16
f_odd = lambda ain: kap_pair(ain)[1] - kh16
W_LO = brentq(f_odd, 0.7, 0.8, xtol=1e-12)           # odd exponent crosses the homogeneous value
W_HI = brentq(f_even, 1.2, 1.3, xtol=1e-12)          # even exponent crosses it
D_LO = minimize_scalar(f_odd, bounds=(0.8, 0.97), method="bounded", options=dict(xatol=1e-9))
D_HI = minimize_scalar(f_even, bounds=(1.03, 1.2), method="bounded", options=dict(xatol=1e-9))
print(f"\nrefined by root finding and a bounded scalar minimizer (m = 16, homogeneous value {kh16:.5f}):")
print(f"   smallest exponent below the homogeneous one for {W_LO:.5f} < a_in < {W_HI:.5f} (a_in = 1 itself is the homogeneous cell)")
print(f"   a_in < 1: it is the ODD exponent; deepest dip {D_LO.fun:.4e} (relative {D_LO.fun / kh16:.2e}) at a_in = {D_LO.x:.5f}")
print(f"   a_in > 1: it is the EVEN exponent (the one a generic edge excites); deepest dip {D_HI.fun:.4e} (relative {D_HI.fun / kh16:.2e}) at a_in = {D_HI.x:.5f}")
for ain in (0.9, 0.8, 0.5, 0.2):
    print(f"   a_in = {ain}: even - hom = {f_even(ain):+.5f}, odd - hom = {f_odd(ain):+.5f}")
print(f"   the EVEN exponent is below the homogeneous one only for 1 < a_in < {W_HI:.5f}; for a_in < 1 it is above it for every contrast scanned, so the layer excited by a generic edge is thinner there")
KMIN = np.minimum(KE, KO)
dk_ = KMIN - kh16
low = AIN[dk_ < -1e-9]
print(f"scan grid of {len(AIN)} contrasts (log spacing {np.log(AIN[1] / AIN[0]):.3f}): grid points inside the window run from {low.min():.4g} to {low.max():.4g}; deepest grid value {dk_.min():.4e} at {AIN[np.argmin(dk_)]:.4g} (the grid understates the dip and the window)")
WEAK_LO, WEAK_HI = W_LO, W_HI
check_in("lower edge: the odd exponent minus the homogeneous one is negative 1e-4 inside and positive 1e-4 outside (smaller of the two magnitudes, floor 1e-8)", min(-f_odd(W_LO + 1e-4), f_odd(W_LO - 1e-4)), 1e-8, 1.0)
check_in("upper edge: the even exponent minus the homogeneous one is negative 1e-4 inside and positive 1e-4 outside (smaller of the two magnitudes, floor 1e-8)", min(-f_even(W_HI - 1e-4), f_even(W_HI + 1e-4)), 1e-8, 1.0)
fine_in = np.concatenate([np.linspace(W_LO + 2e-3, 0.999, 60), np.linspace(1.001, W_HI - 2e-3, 60)])
check_in("inside the refined window the smallest exponent is below the homogeneous one at 120 points (largest value of the difference)", max(f_min(a) for a in fine_in), -1.0, -1e-6)
check_in("the minimizer is at least as deep as the scan grid", dk_.min() - min(D_LO.fun, D_HI.fun), 0.0, 1.0)
check_in("the scan-grid window lies inside the refined window (margin of the smallest grid point inside, in a_in)", min(low.min() - W_LO, W_HI - low.max()), 0.0, 1.0)
check_in("even exponent above the homogeneous value for every scan point with a_in < 1 (smallest difference on the grid)", (KE - kh16)[AIN < 0.999].min(), 1e-6, 10.0)
check_in("every contrast outside [0.7, 1.5] has a larger smallest exponent than the homogeneous cell (smallest margin on the scan grid, floor 1e-6)", dk_[(AIN < 0.7) | (AIN > 1.5)].min(), 1e-6, 10.0)

# %%
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
dk = np.arange(6)
for i, (lab, (rr, kp, mlab)) in enumerate(PROF.items()):
    ci = 0 if mlab.startswith("even") else 1          # same colors as the right panel
    ax[0].semilogy(dk[:4], rr[:4], marker="o", color=CAT[ci], label=mlab)
    ax[0].semilogy(dk[:4], rr[0] * np.exp(-kp * dk[:4]), ":", color=INK, lw=1.2)
ax[0].set(xlabel="depth (cells)", ylabel=r"$\|\beta-\bar\beta\|_{L^2}$ on the node column", title=r"strip decay, $a_{in}=10$ (dotted: $e^{-\kappa k}$ from the QEP)")
ax[0].legend()
ax[1].plot(AIN, KE - kh16, color=CAT[0], marker="o", ms=3, label="even parity")
ax[1].plot(AIN, KO - kh16, color=CAT[1], marker="s", ms=3, label="odd parity")
ax[1].axhline(0.0, color=MUTED, lw=1, ls="--")
ax[1].set_xscale("log")
ax[1].set_yscale("symlog", linthresh=1e-3)
ax[1].set(xlabel=r"$a_{\rm in}$", ylabel=r"$\kappa-\kappa_{\rm hom}$ (symlog)", title="lowest exponent of each parity, m = 16, r = 0.3")
ax[1].axvspan(W_LO, W_HI, color=MUTED, alpha=0.18, lw=0)
ax[1].legend(loc="lower left")
ins = ax[1].inset_axes([0.52, 0.08, 0.45, 0.42])
aa = np.linspace(0.7, 1.3, 241)
kp_ = np.array([kap_pair(v) for v in aa])
ins.plot(aa, kp_[:, 0] - kh16, color=CAT[0], lw=1.2)
ins.plot(aa, kp_[:, 1] - kh16, color=CAT[1], lw=1.2)
ins.axhline(0.0, color=MUTED, lw=0.8, ls="--")
ins.axvspan(W_LO, W_HI, color=MUTED, alpha=0.18, lw=0)
ins.set_ylim(-4e-3, 4e-3)
ins.set_title("linear scale near $a_{in}=1$ (shaded: window)", fontsize=7.5)
ins.tick_params(labelsize=7)
plt.show()
print("Figure 4. Left: decay of the strip solution with depth for the two parities, with the decay predicted by the quadratic eigenproblem (dotted, almost hidden by the data). Right: the smallest decaying exponent of each parity against the inclusion contrast, as a difference from the discrete homogeneous value (symmetric log scale); the shaded band is the window in which the smaller exponent lies below the homogeneous value, and the inset shows it on a linear scale.")

# %% [markdown]
# **Second-order convergence.** The smallest exponent of the homogeneous Q1 cell is 6.36553 at $m=16$, above $2\pi$ by 0.08234; the offset is 0.35058 at $m=8$ and 0.03620 at $m=24$. The values equal the closed form $\kappa_1(m)=m\operatorname{arcosh}(s/2)$, $s=(8-2c)/(1+2c)$, $c=\cos(2\pi/m)$ (6.633768, 6.365526, 6.319383) to $10^{-14}$, which is an independent test of the eigensolver at these three sizes. The ratios, 0.235 from 8 to 16 and 0.440 from 16 to 24 (second order gives 0.25 and 0.444), show the $O(h^2)$ convergence. In the heterogeneous cells the staircase adds to this offset, which is why only values at equal $m$ can be compared.
#
# **Layer thickness against contrast.** For $a_\text{in}\in\{0.1,10,100\}$ the smallest exponent of either parity is above the homogeneous one at $m=8$, 16 and 24, so the layer is thinner: at $m=16$ the amplitude per cell is smaller by at least a factor $e^{-0.382}$ (0.382 is the smallest margin over the three contrasts; the attenuation per cell is $1.72\times10^{-3}$ for the homogeneous cell and $1.08\times10^{-3}$ for $a_\text{in}=10$). The margins in the table (0.188 to 0.791) change with $m$, by a factor 1.74 between $m=8$ and $m=16$ at $a_\text{in}=10$ (0.269, 0.468, 0.375 for $m=8,16,24$): the sign is stable and the value is not converged. That reverses the idea that high-index inclusions slow the layer, for these contrasts. At weak contrast it fails. The two members of the degenerate $k=1$ pair move in opposite directions at first order in $a_\text{in}-1$ (even above and odd below for $a_\text{in}<1$, the reverse for $a_\text{in}>1$), so the lower one is below the homogeneous value.
#
# *First-order reason.* Take $u=e^{-\kappa x}\cos\kappa y$ and $\sin\kappa y$ with $\kappa=2\pi$ and their growing partners $\tilde u=e^{+\kappa x}\cos\kappa y$, $\sin\kappa y$ (the adjoint modes of the static problem). A perturbation $a=1+\delta(x,y)$ moves the pair by the eigenvalues of the $2\times2$ matrix $\int\delta\,\nabla u_i\!\cdot\!\nabla\tilde u_j$ (up to a common normalization), and $\nabla u_c\!\cdot\!\nabla\tilde u_c=-\kappa^2\cos2\kappa y$, $\nabla u_s\!\cdot\!\nabla\tilde u_s=+\kappa^2\cos2\kappa y$, so the matrix has trace zero for every $\delta$: the mean first-order shift of the pair vanishes (the uniform scaling $a\to\lambda a$ leaves the equation unchanged, so a constant $\delta$ gives no shift either) and only the splitting survives, with sign $\pm\int\delta\cos 4\pi y$, which reverses with $\delta\to-\delta$. The off-diagonal element has $\sin 2\kappa y$ and integrates to zero for the centered disk. The printed table of the pair at $a_\text{in}=1+d$ agrees: the splitting divided by $d$ is constant (0.0755, 0.0749, 0.0737 at $d=0.01,0.02,0.04$) and the mean shift divided by $d^2$ is constant (0.167, 0.166, 0.162), so the mean is second order. This is the reasoning, supported by the scaling; the first-order coefficient itself was not compared with the matrix.
#
# *The window, by root finding.* A first reading of the 121-point log grid, whose spacing is 7%, gave the window 0.794 to 1.2 (dip 0.00221); this window is an artifact of the grid. Root finding on the odd exponent (lower edge) and on the even exponent (upper edge), and a bounded scalar minimizer, give: the smallest exponent of either parity is below the homogeneous value for $0.77493<a_\text{in}<1.23907$ (at $a_\text{in}=1$ the cell is homogeneous). For $a_\text{in}<1$ the smaller one is the odd exponent, with a dip of $-2.3779\times10^{-3}$ (relative $-3.74\times10^{-4}$) at $a_\text{in}=0.88149$; for $a_\text{in}>1$ it is the even exponent, $-2.0109\times10^{-3}$ ($-3.16\times10^{-4}$) at $1.11223$. The grid gave $-2.2144\times10^{-3}$ at 0.9114.
#
# *Which layer is thicker.* A generic edge excites the even parity (the data $-\chi_1$ are even), and the tails of section 4 belong to it. For the even exponent the layer is thicker only for $1<a_\text{in}<1.23907$; for every $a_\text{in}<1$ scanned it is thinner (even minus homogeneous is $+0.00606$ at 0.9, $+0.01779$ at 0.8, $+0.11865$ at 0.5, $+0.55343$ at 0.2). The odd window $0.77493<a_\text{in}<1$ matters only for data of the odd parity ($\chi_2$ on $x_1$ edges), whose tail vanishes, so it concerns the decaying part of the layer and not the $O(\varepsilon)$ interior error. The statement "thicker for $0.794\le a_\text{in}\le1.2$", read from the grid, judged by the smaller of the two exponents and overstated the effect on the layer that matters. Both dips are $3\times10^{-4}$ to $4\times10^{-4}$ in relative size. The expectation recorded before the computation (PLAN.md) was a thinner layer without a contrast condition; the computation shows that a condition on the contrast is needed.
#
# **Decay on the strips.** The decay between depths 1 and 2 of the strip solutions equals the exponent of the matching parity: 7.33872 (odd, $\chi_2$ data at $s=0$) and 6.83347 (even, $\chi_1$ data at $s=(\tfrac14,0)$), against 7.33872 and 6.83350 from the eigenproblem; the relative differences are $6\times10^{-8}$ and $4.5\times10^{-6}$, far below the $10^{-2}$ the tolerance allows. The solver's residuals and parity expectations are at $2.3\times10^{-9}$ and $10^{-14}$. The amplitude of the smallest mode at the inner box $[\tfrac14,\tfrac34]^2$ is $e^{-\kappa N/4}$: $1.1\times10^{-3}$ at $N=4$, $1.2\times10^{-6}$ at $N=8$ and negligible beyond. The boundary layer does not contaminate the interior by itself; what reaches the interior is the macroscopic field $\varepsilon\bar u_1$ that the tails generate.
#
# ## 6. Weak limits and the div-curl lemma
#
# With test functions $\varphi_k=\sin k\pi x\sin k\pi y$ (vanishing on $\partial\Omega$), $I_\text{flux}(\varepsilon)=\int\varphi\,a_\varepsilon|\nabla u_\varepsilon|^2\to\int\varphi\,\sigma^*|\nabla u_0|^2$. This is the div-curl lemma (Murat and Tartar; Tartar 2009): $a_\varepsilon\nabla u_\varepsilon$ is divergence controlled (its divergence is the fixed right-hand side), $\nabla u_\varepsilon$ is curl free, both converge weakly, so the scalar product converges to the product of the weak limits, $\sigma^*\nabla u_0\cdot\nabla u_0$. By contrast $I_\text{grad}(\varepsilon)=\int\varphi|\nabla u_\varepsilon|^2$ is the product of two curl-free factors and nothing controls the oscillations: it converges to $\int\varphi\,\nabla u_0\cdot M\nabla u_0$ with $M=\langle(I+\nabla\chi)^T(I+\nabla\chi)\rangle\ne I$ (two-scale convergence, Allaire 1992: $\nabla u_\varepsilon\approx(I+\nabla_y\chi)\nabla u_0$), and not to the naive limit $\int\varphi|\nabla u_0|^2$.
#
# Quadrature matters at the level that decides the slopes. The Q1 element-mean gradient (midpoint rule) of the fine solution has the cell limits $\langle a\,g_i\cdot g_j\rangle$ and $\langle g_i\cdot g_j\rangle$ built from element means, which differ from the exact element integrals $\sigma^*_h$ and $M$ by the twist energy of bilinear elements (about 1% at $m=16$); mixing midpoint integrals of the fine solution with the exact cell tensors leaves an $\varepsilon$-independent floor. The main comparison therefore uses the same quadrature on both sides: exact element energies ($2\times2$ Gauss points, exact for Q1) in $I_\text{flux}$, $I_\text{grad}$, $\sigma^*_h$ and $M$. The mixed version is printed to show the floor.

# %%
GRAM = ts.corrector_gram(CHI)
SM, MM = ts.midpoint_tensors(CELL, CHI)
print(f"cell tensors: sigma*_h = {SS:.6f}, M = {GRAM[0, 0]:.6f} (exact Q1 integrals, isotropic: M_xy = {GRAM[0, 1]:.1e}, M_xx - M_yy = {GRAM[0, 0] - GRAM[1, 1]:.1e});"
      f"   midpoint versions: {SM[0, 0]:.6f}, {MM[0, 0]:.6f}")
Wk = {}
for k in (1, 3):
    for N in Ns:
        n = N * M
        a, u = fine(CELL, M, N, 0)
        xc = (np.arange(n) + 0.5) / n
        phi = np.sin(k * np.pi * xc)[:, None] * np.sin(k * np.pi * xc)[None, :]
        _, cx, cy = ts.torsion(xc, xc, NT)
        cx, cy = cx / SS, cy / SS
        If, Ig = ts.weak_integrals(u, a, phi, exact_energy=True)
        lf, lg, nv = ts.weak_limits(phi, cx, cy, SIG, GRAM)
        Im, Igm = ts.weak_integrals(u, a, phi)
        lfm, lgm, nvm = ts.weak_limits(phi, cx, cy, SM, MM)
        lfx, lgx, _ = ts.weak_limits(phi, cx, cy, SIG, GRAM)          # exact tensors against midpoint integrals: the mixed version
        Wk[(k, N)] = dict(If=If, Ig=Ig, lf=lf, lg=lg, nv=nv, ef=If - lf, eg=Ig - lg, gap=Ig - nv, pgap=lg - nv,
                          efm=Im - lfm, egm=Igm - lgm, efx=Im - lfx, egx=Igm - lgx, nvx=nv)
GRAM_OWN = q1_energy_tensor(CELL, CHI, weighted=False)
print(f"corrector Gram matrix M: twoscale {GRAM[0, 0]:.6f}, own unweighted element integral {GRAM_OWN[0, 0]:.6f} (M_11 = 1 + <|grad chi_1|^2> + 2 <d_1 chi_1>, so M_11 - 1 = {GRAM[0, 0] - 1:.6f})")
check("corrector Gram matrix M_11 against an own element-integral recomputation (max abs difference over the tensor)", np.abs(GRAM - GRAM_OWN).max(), 1e-10)

# the same weak integrals for the generic cut s = 1/4 (the fine solves are cached), including N = 64
WkQ = {}
for k in (1, 3):
    for N in Ns + [64]:
        n = N * M
        a, u = fine(CELL, M, N, 4)
        xc = (np.arange(n) + 0.5) / n
        phi = np.sin(k * np.pi * xc)[:, None] * np.sin(k * np.pi * xc)[None, :]
        _, cx, cy = ts.torsion(xc, xc, NT)
        If, Ig = ts.weak_integrals(u, a, phi, exact_energy=True)
        lf, lg, nv = ts.weak_limits(phi, cx / SS, cy / SS, SIG, GRAM)
        WkQ[(k, N)] = (If - lf, Ig - lg)
print("\ngeneric cut s = 1/4, exact element energies: |I_flux - lim| and |I_grad - lim_M|, N = 4, 8, 16, 32, 64, and local rates")
for k in (1, 3):
    ef_q = np.array([abs(WkQ[(k, N)][0]) for N in Ns + [64]])
    eg_q = np.array([abs(WkQ[(k, N)][1]) for N in Ns + [64]])
    print(f"   phi_{k}: flux " + ", ".join(f"{v:.3e}" for v in ef_q) + f"   local rates " + ", ".join(f"{v:.3f}" for v in pairs(ef_q)))
    print(f"          grad " + ", ".join(f"{v:.3e}" for v in eg_q) + f"   local rates " + ", ".join(f"{v:.3f}" for v in pairs(eg_q)))
# the O(eps) term of the flux energy of the generic cut: u_eps ~ u0 + eps (chi.grad u0 + ubar_1), so the product of the weak limits
# (div-curl) gains the cross term 2 eps sigma* grad u0 . grad ubar_1; ubar_1 (from the tails) is the N = 32 field, independent of N
G1x, G1y = ts.element_gradients(RUNS[("1/4", 32)]["u1"])
xq = (np.arange(512) + 0.5) / 512
_, Cx, Cy = ts.torsion(xq, xq, NT)
Cx, Cy = Cx / SS, Cy / SS
CK = {}
for k in (1, 3):
    phi = np.sin(k * np.pi * xq)[:, None] * np.sin(k * np.pi * xq)[None, :]
    CK[k] = 2 * SS * np.mean(phi * (Cx * G1x + Cy * G1y))
print("predicted O(eps) coefficient of the flux energy, generic cut: C_k = 2 sigma* int phi_k grad u0 . grad ubar_1;  error against the model I_flux - lim = C_k eps + D_k eps^2")
for k in (1, 3):
    Nl = np.array(Ns[1:] + [64], float)
    err = np.array([WkQ[(k, int(N))][0] for N in Nl])
    rem = err - CK[k] / Nl
    print(f"   phi_{k}: C = {CK[k]:+.4e}; C/N at N = {', '.join(str(int(v)) for v in Nl)}: " + ", ".join(f"{v:+.2e}" for v in CK[k] / Nl)
          + "; error minus C/N: " + ", ".join(f"{v:+.2e}" for v in rem) + f"; times N^2: " + ", ".join(f"{v:+.3e}" for v in rem * Nl ** 2))
Nl = np.array([16.0, 32.0, 64.0])
rem3 = np.array([WkQ[(3, int(N))][0] for N in Nl]) - CK[3] / Nl
print(f"   phi_3 local rates of |error minus C/N|, N = 16-32, 32-64: " + ", ".join(f"{v:.3f}" for v in pairs(np.abs(rem3))))
check_in("generic cut, phi_3: after subtracting the predicted O(eps) term the flux error falls at rate 2 (last pair N = 32 to 64, within 0.2 of 2)", abs(pairs(np.abs(rem3))[-1] - 2.0), 0.0, 0.2)
check_in("generic cut: the predicted O(eps) coefficient is much larger for phi_3 than for phi_1 (ratio of magnitudes, at least 10)", abs(CK[3] / CK[1]), 10.0, 1e6)

for k in (1, 3):
    print(f"\nphi_{k} = sin({k} pi x) sin({k} pi y), s = 0, exact element energies")
    print(f"{'N':>4s} {'I_flux':>10s} {'lim_flux':>10s} {'|diff|':>10s} {'I_grad':>10s} {'lim_M':>10s} {'|diff|':>10s} {'naive':>10s} {'I_grad-naive':>13s} {'pred. gap':>10s} {'ratio':>7s}")
    for N in Ns:
        w = Wk[(k, N)]
        print(f"{N:4d} {w['If']:10.6f} {w['lf']:10.6f} {abs(w['ef']):10.3e} {w['Ig']:10.6f} {w['lg']:10.6f} {abs(w['eg']):10.3e} {w['nv']:10.6f} {w['gap']:13.4e} {w['pgap']:10.4e} {w['gap'] / w['pgap']:7.4f}")
    ef = np.array([abs(Wk[(k, N)]["ef"]) for N in Ns])
    eg = np.array([abs(Wk[(k, N)]["eg"]) for N in Ns])
    print(f"   slope of |I_flux - lim| over N = 4-32: {slope(Ns, ef):.3f} (last pair {pairs(ef)[-1]:.3f});  slope of |I_grad - lim_M|: {slope(Ns, eg):.3f} (last pair {pairs(eg)[-1]:.3f})")
    print(f"   local rates, flux: " + ", ".join(f"{v:.3f}" for v in pairs(ef)) + ";  grad: " + ", ".join(f"{v:.3f}" for v in pairs(eg)))
print("\nmixed version, k = 1: midpoint integrals against the exact tensors (floors) and the same-quadrature midpoint pair")
print(f"{'N':>4s} {'|I_flux - lim|':>15s} {'floor flux':>11s} {'ratio':>6s} | {'|I_grad - lim_M|':>17s} {'floor grad':>11s} {'ratio':>6s} | {'midpoint pair: flux':>20s} {'grad':>10s}")
for N in Ns:
    w = Wk[(1, N)]
    floor_f = (SS - SM[0, 0]) * w["nvx"]
    floor_g = (GRAM[0, 0] - MM[0, 0]) * w["nvx"]
    print(f"{N:4d} {abs(w['efx']):15.3e} {abs(floor_f):11.3e} {abs(w['efx'] / floor_f):6.3f} | {abs(w['egx']):17.3e} {abs(floor_g):11.3e} {abs(w['egx'] / floor_g):6.3f} | {abs(w['efm']):20.3e} {abs(w['egm']):10.3e}")
efx = np.array([abs(Wk[(1, N)]["efx"]) for N in Ns])
egx = np.array([abs(Wk[(1, N)]["egx"]) for N in Ns])
efm = np.array([abs(Wk[(1, N)]["efm"]) for N in Ns])
egm = np.array([abs(Wk[(1, N)]["egm"]) for N in Ns])
print(f"   mixed: overall slopes {slope(Ns, efx):.3f} (flux), {slope(Ns, egx):.3f} (grad); last pairs {pairs(efx)[-1]:.3f}, {pairs(egx)[-1]:.3f};  midpoint pair last pairs {pairs(efm)[-1]:.3f}, {pairs(egm)[-1]:.3f}")

for k in (1, 3):
    ef = np.array([abs(Wk[(k, N)]["ef"]) for N in Ns])
    eg = np.array([abs(Wk[(k, N)]["eg"]) for N in Ns])
    check_in(f"div-curl: slope of |I_flux - lim| over N = 4-32 at least 0.8, k = {k}", slope(Ns, ef), 0.8, 3.0)
    check_in(f"two-scale: slope of |I_grad - lim_M| over N = 4-32 at least 0.8, k = {k}", slope(Ns, eg), 0.8, 3.0)
    check_in(f"gap check: |I_grad - naive| / predicted gap at N = 32, k = {k} (expected within 10% of 1; restates the convergence of I_grad to lim_M, since lim_M - naive is the gap)", Wk[(k, 32)]["gap"] / Wk[(k, 32)]["pgap"], 0.9, 1.1)
check_in("exact quadrature, k = 1: last-pair rate of |I_flux - lim| is above 1.5", pairs([abs(Wk[(1, N)]["ef"]) for N in Ns])[-1], 1.5, 3.0)
check_in("exact quadrature, k = 1: last-pair rate of |I_grad - lim_M| is above 1.5", pairs([abs(Wk[(1, N)]["eg"]) for N in Ns])[-1], 1.5, 3.0)
check_in("the mixed version stalls: last-pair rate of |I_grad - lim_M| below 1", pairs(egx)[-1], -2.0, 1.0)
FLOOR_G = abs((GRAM[0, 0] - MM[0, 0]) * Wk[(1, 32)]["nvx"])
FLOOR_F = abs((SS - SM[0, 0]) * Wk[(1, 32)]["nvx"])
check_in("mixed gradient error at N = 32: its own floor (M - M_mid) times the naive limit, as a fraction of the error (the rest is the rate-2 midpoint error)", FLOOR_G / abs(Wk[(1, 32)]["egx"]), 0.6, 0.9)
check_in("mixed flux error at N = 32: its own floor (sigma*_h - Sigma_mid) times the naive limit, as a fraction of the error", FLOOR_F / abs(Wk[(1, 32)]["efx"]), 0.6, 0.9)
check_in("the floor is the 0.8% offset of the corrector Gram matrix: (M - M_mid) / M", (GRAM[0, 0] - MM[0, 0]) / GRAM[0, 0], 0.007, 0.009)
check_in("midpoint pair converges at rate 2: last-pair rate of the flux error", pairs(efm)[-1], 1.5, 3.0)

# %%
fig, ax = plt.subplots(1, 2, figsize=(9.6, 3.8))
Nn = np.array(Ns)
ax[0].loglog(Ns, [abs(Wk[(1, N)]["ef"]) for N in Ns], "o-", color=CAT[0], label=r"$|I_{\rm flux}-\int\varphi\sigma^*|\nabla u_0|^2|$")
ax[0].loglog(Ns, [abs(Wk[(1, N)]["eg"]) for N in Ns], "s-", color=CAT[1], label=r"$|I_{\rm grad}-\int\varphi\,\nabla u_0\!\cdot\! M\nabla u_0|$")
ax[0].loglog(Ns, efx, "o--", color=CAT[0], alpha=0.6, label="flux, midpoint vs exact tensor")
ax[0].loglog(Ns, egx, "s--", color=CAT[1], alpha=0.6, label="grad, midpoint vs exact tensor")
ax[0].loglog(Ns, 3 * abs(Wk[(1, 4)]["ef"]) * (Nn / 4.0) ** -2.0, ":", color=MUTED)
ax[0].set(xlabel="N (cells per side)", ylabel="weak-limit error", title=r"$\varphi_1$; dotted: $N^{-2}$")
ax[0].legend(fontsize=7.5)
ax[1].set_xscale("log")
ax[1].semilogy(Ns, [abs(Wk[(1, N)]["gap"]) for N in Ns], "o-", color=CAT[2], label=r"$I_{\rm grad}-\int\varphi|\nabla u_0|^2$ (measured)")
ax[1].semilogy(Ns, [abs(Wk[(1, N)]["pgap"]) for N in Ns], "k--", lw=1, label=r"$\int\varphi\,\nabla u_0\cdot(M-I)\nabla u_0$ (predicted)")
ax[1].semilogy(Ns, [abs(Wk[(1, N)]["eg"]) for N in Ns], "s-", color=CAT[1], label=r"$|I_{\rm grad}-\mathrm{lim}_M|$")
ax[1].set(xlabel="N (cells per side)", ylabel="energy difference", title=r"the plain gradient energy misses $\int\varphi|\nabla u_0|^2$ by a constant")
ax[1].legend(fontsize=8, loc="center right")
plain_ticks(ax[0], Ns)
plain_ticks(ax[1], Ns)
plt.show()
print("Figure 5. Left: convergence of the two weak limits with consistent (exact element) quadrature, and of the mixed midpoint version (dashed), which stalls. Right: the gap between the gradient energy and the naive limit stays at its predicted constant.")

# %% [markdown]
# **Consistent quadrature.** With exact element energies on both sides the flux energy converges to the div-curl limit $\int\varphi\,\sigma^*|\nabla u_0|^2$ for the mirror-line cut $s=0$, with slope 1.905 over $N=4$-$32$ and last-pair rate 1.984 for $\varphi_1$, and the gradient energy converges to the corrector-weighted limit with the same rates; for $\varphi_3$ (the even test function that changes sign; $\varphi_2$ integrates to zero by symmetry and says nothing) the overall slopes are 2.155 and 2.278 and the last pairs 2.073 and 2.099, above 2 because the values at $N=4$ are small by sign cancellation (pre-asymptotic), not a rate of 2.3. So the statement is: rate 2 for $s=0$ (a centrosymmetric cell cut on a mirror line) and test functions vanishing on $\partial\Omega$. A plausible reason, stated as a hypothesis and not tested: centrosymmetry removes the $O(\varepsilon)$ term and the layer enters at second order because $\varphi$ vanishes on the boundary.
#
# *The generic cut $s=\tfrac14$* (one more set of solves, cached from section 2, plus $N=64$). For $\varphi_1$ the flux and gradient errors also fall at rate about 1.9 (flux local rates 1.615, 1.888, 1.935, 1.913). For $\varphi_3$ they do not: the flux error is $2.85\times10^{-6}$, $4.14\times10^{-6}$, $2.77\times10^{-6}$ at $N=16,32,64$, with local rates $-0.538$ and $0.579$. This is not the torsion series (6001 terms instead of 2001 change the $\varphi_3$ errors by at most $1.2\times10^{-10}$ in a separate check) and it is not noise: the interior correction $\bar u_1$ of section 4 gives the flux energy the cross term $2\varepsilon\sigma^*\nabla u_0\cdot\nabla\bar u_1$ (product of weak limits), with coefficient $C_3=-2.2267\times10^{-4}$ for $\varphi_3$ and $C_1=+6.003\times10^{-6}$ for $\varphi_1$, 37 times smaller. After subtracting $C_k/N$ the remainder times $N^2$ is constant ($2.832$, $2.885$, $2.895\times10^{-3}$ for $\varphi_3$ at $N=16,32,64$) and the remainder falls at rates 1.973 and 1.995. So for the generic cut the weak limits have an $O(\varepsilon)$ error whose coefficient is predicted by the tails, small for the lowest test function by near-cancellation (the origin of the cancellation was not investigated). For the gradient energy the analogous coefficient was not computed.
#
# **What the plain gradient energy converges to.** $I_\text{grad}-\int\varphi|\nabla u_0|^2$ equals 0.9932 of the predicted constant $\int\varphi\nabla u_0\!\cdot\!(M-I)\nabla u_0=1.6142\times10^{-3}$ at $N=32$ (0.9943 for $\varphi_3$). Since $\lim_M-\text{naive}$ is that constant, this ratio is $1-(\lim_M-I_\text{grad})/\text{gap}$: it restates the convergence of $I_\text{grad}$ to $\lim_M$ at the rate of the weak limit and is not independent evidence. The gap is 50.5% of the naive limit itself, which is $M_{11}-1=0.504546$ for an isotropic $M$ (also not independent); $M_{11}$ itself is recomputed here from the correctors with own element integrals and agrees to $4\times10^{-16}$. The quantity $1-\text{ratio}$ shrinks by a factor of about 4 per doubling of $N$. The plain gradient energy does not approach $\int\varphi|\nabla u_0|^2$ at any $N$. The value 0.97 expected beforehand for this ratio was a midpoint-rule value with its 0.8% quadrature offset; the consistent value is 0.9932.
#
# **The mixed version.** Midpoint integrals of the fine solution against the exact cell tensors give overall slopes 1.116 and 1.217 (1.12 and 1.22 were expected beforehand, and pass the threshold of 0.8 set beforehand in PLAN.md) but last-pair rates 0.605 and 0.724. The errors stall at $5.4\times10^{-5}$ (flux) and $5.0\times10^{-5}$ (gradient). Each has its own predicted floor: for the flux, $(\sigma^*_h-\Sigma_\text{mid})$ times the naive limit, $4.43\times10^{-5}$, and for the gradient, $(M-M_\text{mid})$ times the naive limit, $3.87\times10^{-5}$. At $N=32$ the flux error is 1.21 times its floor and the gradient error 1.28 times its floor, that is, the floors are 0.82 and 0.78 of the errors (the rest is the rate-2 midpoint error). The relative offset $(M-M_\text{mid})/M$ is 0.8%, from the twist energy of bilinear elements. A slope above one in the mixed version comes from approaching this floor, not from convergence; the pair of midpoint quantities converges at rate 1.984 to the midpoint tensors 1.674409 and 1.492440, which differ from $\sigma^*_h$ and $M$.
#
# ## Caveats and open ends
#
# - **One cell, one domain, one right-hand side.** All results are for the staircase disk with $a_\text{in}=10$ (and $m=8$ as second sequence), the unit square and $f=1$. Rates for other cells, other contrasts and non-convex or curvilinear domains are not tested.
# - **The hierarchy is discrete.** The correctors are Q1 on the pixel grid of the cell and the fine problems tile the same pixel grid, so $u_\varepsilon$ is the solution of the discrete sequence, and the comparison excludes the Q1 error of the fine problem relative to the PDE with the staircase medium, which would be a fixed relative error of order $1/m$.
# - **Pre-asymptotic rate.** The $H^1$ rate of the generic cut is 0.548 at $N=64$ and 0.552 at $m=8$, $N=64$. The asymptotic value $1/2$ is consistent with the data (free-exponent fit $q=1.019\pm0.007$ with two degrees of freedom, out-of-sample prediction of $N=64$ to 0.50%) and is the standard theory's assertion; it is not an observed asymptote, and a fit with five points cannot exclude a slowly varying correction. The rates of the interior error (1 and 2) are established at the printed local rates, approaching 2 from below for mirror-line cuts. The margins of the exponents against the homogeneous cell (section 5) are not converged in $m$ (0.269, 0.468, 0.375 at $a_\text{in}=10$ for $m=8,16,24$); only the sign is stable. Only two resolutions ($m=8$, 16) were run for the rates.
# - **Conditions of the cut result.** The improvement for mirror-line cuts uses: a discrete cell mirror symmetric about the cut node lines (even $m$, centered staircase), centrosymmetry (so that no $O(\varepsilon)$ dispersive term enters), the same cell phase at opposite edges (integer $N$), and, for $\bar u_1$ harmonic, an isotropic $\sigma^*$. Corners of the torsion solution are not analyzed (they are believed to add at most logarithmic factors, a statement not verified here). Non-centrosymmetric cells, irrational or oblique cuts (for which the tail does not exist in this form), and second-order tails were not tried.
# - **Tolerances set after seeing results.** The window [0.45, 0.70] for the generic-cut last-pair rate (exploratory value 0.586); the torsion tolerance 5e-9 and the rate-ratio bound 114; the splitting ratio (0.8 to 1.3) and the weak-contrast scaling windows (1.8 to 2.2 and 3.4 to 4.6); the fit tolerance 2% for the two-term model, the 1% for the out-of-sample prediction (found 0.50%), the 5 standard errors for $B$ and the 0.25 band for $q-1$; the margin floor 0.15 (smallest printed margin 0.188) and the ratio 1.2 for the margin variation with $m$; the interval 0.6 to 0.9 for both floor fractions and the interval 0.007 to 0.009 for $(M-M_\text{mid})/M$; the gap-ratio window 0.9 to 1.1 (set beforehand, PLAN.md); the sensitivity bound 2e-3 for the 301-term series; the last-pair-rate bound 1.5 for exact quadrature and for the midpoint pair, and the bound 1 for the stalled mixed version; the windows [0.5, 0.65] and [0.5, 0.586887] for the $m=8$ and $m=16$ last-pair $H^1$ rates; the 0.2 band for the rate after subtracting the $O(\varepsilon)$ term and the ratio 10 of $|C_3/C_1|$ (found 37); the floors $10^{-6}$ and $10^{-8}$ of the sign checks (chosen as large multiples of the solver residual $2\times10^{-9}$); the local-rate monotonicity checks. The checks that are sign tests carry such floors rather than the windows $[0,1]$ or $[0,10]$. The tolerances set beforehand (PLAN.md) are used as stated everywhere else.
# - **Outcomes that differ from the expectations recorded beforehand.** PLAN.md records the design, the expectations written before the computations, and the corrections made after an independent review. The torsion tolerance 1e-10 is unreachable (3.3e-7 between 301 and 601 terms; $p,q\le2001$ used). The test function $\varphi_2$ gives a zero integral by symmetry and was replaced by $\varphi_3$. The thinner layer is not a general property of heterogeneity (weak contrasts give a thicker layer, for the even parity only above $a_\text{in}=1$). The ratio of the gap check is 0.9932 with consistent quadrature, not the 0.97 expected from the midpoint rule. The generic-cut local rate is not 0.586 but still falling (0.548 at $N=64$). The rate 2 of the weak limits does not hold for the generic cut at $\varphi_3$ (an $O(\varepsilon)$ term from $\bar u_1$). The area fraction of the staircase explains 2.50 of the 5.32 percentage points of $\sigma^*_h$ above the smooth disk, not all of it.
# - **Checks that cannot fail.** The checks on the isotropy of $\sigma^*$, on the zeros of $\chi$ on mirror lines and on the zero tails are tests of exact symmetries of the discrete problem (they pass to rounding or fail by a bug). The agreement of `cell_correctors` with `cellsolver.q1_effective` is an interface identity and is not tested. The nested-bracket check and the exact $\sqrt{31/13}$ inside the brackets are certified inequalities, not statistical agreements. The gap-ratio check restates the convergence of $I_\text{grad}$ (section 6).
# - **Module issue.** Above $a_\text{in}\approx400$ the reduced eigensolver returns a spurious exponent below $10^{-4}$ (the exponent-zero Jordan block of constants and linear solutions, split by rounding, passes the $10^{-6}$ filter); the contrast scan stops at 200. The module is used unchanged.
# - **Literature.** The statements about boundary-layer tails rest on the abstract of Allaire and Amar (1999) as recorded in the bibliographic check; the full text was not read, so the Dirichlet case and the interior rate are not verified there. The $\varepsilon^{1/2}$ rate for $u_\varepsilon-w_\varepsilon$ is the standard statement of the BLP and JKO books (from memory); the mirror-line improvement, the tail prediction and the $O(\varepsilon)$ term of the weak limits are measured here, and no published statement of them for this setting was located. The formula $\sqrt{(1+3h)/(3+h)}$ for the square inclusion of side $1/2$ is the case of the Mortola-Steffe formula for the square array at area fraction $1/4$ (a four-phase checkerboard with three equal tiles), conjectured by Mortola and Steffe (1985) and proved in 2001 by Craster and Obnosov and, independently, by Milton; the specialization is from memory, only the bibliographic records of Milton 2001 and Craster-Obnosov 2001 were checked (the original Mortola-Steffe record was not found), and the value is confirmed here by the certified brackets (the $8\times8$ refined bracket has half-width $7.4\times10^{-4}$) and not derived. `rayleigh.py` is imported only for the smooth-disk values (and the Rayleigh value at the staircase fraction).
#
# ## References
#
# Entries marked (checked) were confirmed against publisher or library records (bibliographic details only; the content was not read unless the notes in PLAN.md say so).
#
# - (checked) A. Bensoussan, J.-L. Lions, G. Papanicolaou, *Asymptotic Analysis for Periodic Structures*, North-Holland, Amsterdam, 1978 (AMS Chelsea reprint 2011).
# - (checked) V. V. Jikov, S. M. Kozlov, O. A. Oleinik, *Homogenization of Differential Operators and Integral Functionals*, Springer, Berlin, Heidelberg, 1994.
# - (checked) G. Allaire, Homogenization and two-scale convergence, SIAM J. Math. Anal. 23 (1992), 1482-1518.
# - (checked) G. Allaire, M. Amar, Boundary layer tails in periodic homogenization, ESAIM Control Optim. Calc. Var. 4 (1999), 209-243 (abstract only was read, see the caveats).
# - (checked, bibliographic record only) G. W. Milton, Proof of a conjecture on the conductivity of checkerboards, J. Math. Phys. 42 (2001), 4873-4882.
# - (checked, bibliographic record only) R. V. Craster, Yu. V. Obnosov, Four-phase checkerboard composites, SIAM J. Appl. Math. 61 (2001), 1839, DOI 10.1137/S0036139900371825.
# - (from memory; the original record was not found) S. Mortola, S. Steffe, A two-dimensional homogenization problem, Atti Accad. Naz. Lincei Rend. Cl. Sci. Fis. Mat. Natur. (8) 78 (1985) (the conjecture).
# - (checked) D. Gerard-Varet, N. Masmoudi, Homogenization and boundary layers, Acta Math. 209 (2012), 133-178; and Homogenization in polygonal domains, J. Eur. Math. Soc. 13 (2011), 1477-1503 (related work, not used quantitatively here).
# - (checked) F. Murat, L. Tartar, H-convergence, in: Topics in the Mathematical Modelling of Composite Materials, Birkhauser, Boston, 1997, pp. 21-43 (the div-curl lemma and the weak limit of the flux energy).
# - (checked) L. Tartar, *The General Theory of Homogenization: A Personalized Introduction*, Lecture Notes of the UMI 7, Springer, Berlin, Heidelberg, 2009 (record dated 2010).
# - (checked) Lord Rayleigh, On the influence of obstacles arranged in rectangular order upon the properties of a medium, Phil. Mag. (5) 34 (1892), 481-502 (smooth-disk reference value via `rayleigh.py`).
# - (checked) G. Allaire, C. Conca, Bloch wave homogenization and spectral asymptotic analysis, J. Math. Pures Appl. 77 (1998), 153-208 (bibliographic record only; static Bloch exponents are used here as a computational device, the content of the paper was not used).
# - In-repository: `cellsolver.py` and `twoscale.py` in this folder, and `PLAN.md`, the design record of the folder.

# %%
print(f"total run time of the notebook: {time.perf_counter() - T0:.1f} s (budget 120 s with 4 threads)")
