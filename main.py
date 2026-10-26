import os

import matplotlib.pyplot as plt
import numpy as np

c = 0.4        # non-linear damping parameter
alpha = 2.0    # u(0)
beta = -2.0    # u'(0)
T = 15.0       # final time
I = 100        # number of steps
N = 2          # dimension of the first-order system

Y0 = np.array([alpha, beta], dtype=float)   # always float, never int


def F(t, Y):
    Y1, Y2 = Y[0], Y[1]
    return np.array([Y2, c * (1.0 - Y1**2) * Y2 - Y1])


def dF(t, Y):                                     # Jacobian of F, for Newton
    Y1, Y2 = Y[0], Y[1]
    return np.array([[0.0, 1.0],
                     [-2.0 * c * Y1 * Y2 - 1.0, c * (1.0 - Y1**2)]])


def _setup(N, Y0, T, I):
    if I < 1:
        raise ValueError()
    Y0 = np.asarray(Y0, dtype=float).reshape(N)
    dt = T / I                                    # time step
    t = np.linspace(0.0, T, I + 1)                # t_0 = 0, ..., t_I = T exactly
    M = np.zeros((N, I + 1))                      # N rows, I+1 columns
    M[:, 0] = Y0                                  # column 0 = Y^0
    return dt, t, M


def PointMilieu(N, Y0, T, I, F):
    dt, t, M = _setup(N, Y0, T, I)
    for i in range(I):                            # i = 0, ..., I-1
        Yi = M[:, i]                              # Y^i on read-only
        K1 = F(t[i], Yi)                          # slope at t_i
        K2 = F(t[i] + dt / 2, Yi + dt / 2 * K1)   # slope at the midpoint
        M[:, i + 1] = Yi + dt * K2
    return M


def RK4(Y0, T, I, F):
    N = np.size(Y0)
    dt, t, M = _setup(N, Y0, T, I)
    for i in range(I):                            # i = 0, ..., I-1
        Yi = M[:, i]                              # Y^i on read-only
        K0 = F(t[i], Yi)                          # slope at the start
        K1 = F(t[i] + dt / 2, Yi + dt / 2 * K0)   # 1st midpoint slope
        K2 = F(t[i] + dt / 2, Yi + dt / 2 * K1)   # 2nd midpoint slope
        K3 = F(t[i] + dt, Yi + dt * K2)           # slope at the end
        M[:, i + 1] = Yi + dt / 6 * (K0 + 2 * K1 + 2 * K2 + K3)
    return M


def Bashforth(N, Y0, Y1, T, I, F):
    dt, t, M = _setup(N, Y0, T, I)
    M[:, 1] = np.asarray(Y1, dtype=float).reshape(N)  # column 1 = Y^1
    F_prev = F(t[0], M[:, 0])                     # F(t_0, Y^0)
    for i in range(1, I):                         # i = 1, ..., I-1
        F_curr = F(t[i], M[:, i])                 # the only new evaluation of the step
        M[:, i + 1] = M[:, i] + dt / 2 * (3 * F_curr - F_prev)
        F_prev = F_curr                           # reused as F(t_{i-1}, Y^{i-1}) next step
    return M


def _newton(G, JG, Y, tol=1e-12, kmax=50):
    Y = np.array(Y, dtype=float)                  # Y^0 = Y_i, copied so M is not modified
    for _ in range(kmax):
        G1, G2 = G(Y)
        (J11, J12), (J21, J22) = JG(Y)
        det = J11 * J22 - J21 * J12
        step = np.array([J22 * G1 - J12 * G2, -J21 * G1 + J11 * G2]) / det  # J^{-1} G
        Y -= step                                 # Y^{k+1} = Y^k - J^{-1}(Y^k) G(Y^k)
        if np.max(np.abs(step)) < tol:
            return Y
    raise RuntimeError(f"Newton did not converge in {kmax} iterations")


def Moulton(N, Y0, Y1, T, I, F, dF):
    dt, t, M = _setup(N, Y0, T, I)
    M[:, 1] = np.asarray(Y1, dtype=float).reshape(N)   # column 1 = Y^1
    a = 5 * dt / 12                                    # alpha = 5h/12 of the PDF, not u(0)
    F_prev = F(t[0], M[:, 0])                          # F(t_0, Y^0)
    for i in range(1, I):                              # i = 1, ..., I-1
        F_curr = F(t[i], M[:, i])
        P = M[:, i] + dt / 12 * (8 * F_curr - F_prev)  # P^i, the explicit part

        def G(Y):                                      # G^i, its zero is Y^{i+1}
            return a * F(t[i + 1], Y) - Y + P

        def JG(Y):                                     # Jacobian of G^i
            return a * dF(t[i + 1], Y) - np.eye(N)

        M[:, i + 1] = _newton(G, JG, M[:, i])
        F_prev = F_curr
    return M


def _solve_midpoint(Y0, T, I, F):
    return PointMilieu(np.size(Y0), Y0, T, I, F)


def _solve_bashforth(Y0, T, I, F):
    N = np.size(Y0)
    Y1 = PointMilieu(N, Y0, T / I, 1, F)[:, 1]    # Y^1: one midpoint step of size dt
    return Bashforth(N, Y0, Y1, T, I, F)


def _solve_moulton(Y0, T, I, F):
    N = np.size(Y0)
    Y1 = RK4(Y0, T / I, 1, F)[:, 1]               # Y^1: one RK4 step of size dt
    return Moulton(N, Y0, Y1, T, I, F, JACOBIAN[F])

SCHEMES = {
    "PointMilieu": {"solver": _solve_midpoint, "order": 2, "label": "midpoint"},
    "RK4": {"solver": RK4, "order": 4, "label": "RK4"},
    "Bashforth": {"solver": _solve_bashforth, "order": 2, "label": "Adams-Bashforth"},
    "Moulton": {"solver": _solve_moulton, "order": 3, "label": "Adams-Moulton"},
}

from scipy.integrate import solve_ivp


def reference(F, Y0, T, t_eval):
    sol = solve_ivp(F, (0.0, T), Y0, method="DOP853",
                    rtol=1e-13, atol=1e-13, t_eval=t_eval)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y


def _harmonic(t, Y):
    return np.array([Y[1], -Y[0]])


def _d_harmonic(t, Y):
    return np.array([[0.0, 1.0], [-1.0, 0.0]])


JACOBIAN = {F: dF, _harmonic: _d_harmonic}        # Adams-Moulton needs dF/dY for each F


def _test(name, I=(100, 200, 400)):
    scheme = SCHEMES[name]
    p = scheme["order"]
    Y_exact = np.array([2 * np.cos(T) - 2 * np.sin(T),
                          -2 * np.sin(T) - 2 * np.cos(T)])
    print(f"  {scheme['label']}  (expected ratio 2^{p} = {2**p})")
    e_prev = None
    for k in I:
        M = scheme["solver"](np.array([2.0, -2.0]), T, k, _harmonic)
        e = np.linalg.norm(M[:, -1] - Y_exact)
        ratio = "" if e_prev is None else f"   ratio = {e_prev / e:5.2f}"
        print(f"    I = {k:4d}   ||Y^I - Y(T)|| = {e:.3e}{ratio}")
        e_prev = e


FIG_DIR = "figures"                     # all figures are saved here

SCHEME_STYLE = {
    "PointMilieu": {"color": "#eb6834", "marker": "o", "prefix": "pm"},
    "RK4": {"color": "#2a78d6", "marker": "s", "prefix": "rk4"},
    "Bashforth": {"color": "#1baf7a", "marker": "^", "prefix": "ab2"},
    "Moulton": {"color": "#eda100", "marker": "D", "prefix": "am"},
}
INK, MUTED = "#0b0b0b", "#8a8985"
MAX_MARKERS = 100                       # more markers merge into a band hiding the reference


def apply_style():
    plt.rcParams.update({
        "axes.edgecolor": MUTED, "axes.labelcolor": "#2b2a28",
        "xtick.color": "#52514e", "ytick.color": "#52514e",
        "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "lines.linewidth": 2,
    })


def save_figure(fig, filename):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, filename), dpi=200)
    print(f"  saved {FIG_DIR}/{filename}")


def show_if_possible():
    if plt.get_backend().lower() != "agg":
        plt.show()


def marker_step(M, sep=", "):
    every = max(1, (M.shape[1] - 1) // MAX_MARKERS)
    return every, "" if every == 1 else f"{sep}1 marker every {every} steps"


def plot_solution(t, M, name, t_fine, Y_fine):
    s, label = SCHEME_STYLE[name], SCHEMES[name]["label"]
    dt = t[1] - t[0]
    every, note = marker_step(M)
    fig, ax = plt.subplots(2, 1, figsize=(8, 6.4), sharex=True)
    for k, y_label in enumerate(["u(t)", "u'(t)"]):
        ax[k].plot(t_fine, Y_fine[k], color=INK, lw=1.2, label="reference (DOP853)")
        ax[k].plot(t[::every], M[k, ::every], s["marker"], ms=4, color=s["color"],
                   label=f"{label}, I = {len(t) - 1}, Δt = {dt:.3g}{note}")
        ax[k].set_ylabel(y_label)
    ax[1].set_xlabel("t")
    fig.suptitle(f"Van der Pol (c = {c:g}): {label} solution")
    handles, labels = ax[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.95))
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    save_figure(fig, f"{s['prefix']}.png")


def limit_cycle(t_end=100.0, t_start=80.0):
    t_long = np.linspace(0.0, t_end, 20001)
    Y_long = reference(F, Y0, t_end, t_long)
    return Y_long[:, t_long >= t_start]


def plot_phase(M, name, Y_fine, cycle):
    s, label = SCHEME_STYLE[name], SCHEMES[name]["label"]
    every, note = marker_step(M, sep="\n")
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(cycle[0], cycle[1], "--", color=MUTED, lw=2, label="limit cycle (t ∈ [80, 100])")
    ax.plot(Y_fine[0], Y_fine[1], color=INK, lw=1, label=f"reference, t ∈ [0, {T:g}]")
    ax.plot(M[0, ::every], M[1, ::every], s["marker"], ms=4, color=s["color"],
            label=f"{label}, I = {M.shape[1] - 1}{note}")
    ax.plot(*Y0, "*", ms=14, color=INK, label=f"Y0 = ({Y0[0]:g}, {Y0[1]:g})")
    ax.set_xlabel("u")
    ax.set_ylabel("u'")
    ax.set_title("Phase portrait (u, u')")
    ax.set_aspect("equal")
    ax.legend(loc="center", fontsize=8)          # the inside of the cycle is empty
    fig.tight_layout()
    save_figure(fig, f"{s['prefix']}_phase.png")


def plot_error_over_time(t, errors):
    dt = t[1] - t[0]
    fig, ax = plt.subplots(figsize=(8, 4))
    for name, e in errors.items():
        ax.semilogy(t[1:], e[1:], color=SCHEME_STYLE[name]["color"],
                    label=f"{SCHEMES[name]['label']} (order {SCHEMES[name]['order']})")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$\|Y^i - Y(t_i)\|_2$")
    ax.set_title(f"Error along the trajectory, same step Δt = {dt:.3g}")
    ax.grid(True, which="both")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, "error_over_time.png")


def convergence(names, I_list=(25, 50, 100, 200, 400, 800, 1600)):
    I_list = np.asarray(I_list)
    dts = T / I_list
    Y_ref = reference(F, Y0, T, [T])[:, -1]
    with np.errstate(over="ignore", invalid="ignore"):
        err = {name: np.array([np.linalg.norm(SCHEMES[name]["solver"](Y0, T, int(k), F)[:, -1]
                                              - Y_ref)
                               for k in I_list])
               for name in names}

    print("  observed order = log2( e(dt) / e(dt/2) )")
    print("      I       dt" + "".join(f"  {SCHEMES[n]['label']:>15}  order" for n in err))
    for k in range(len(I_list)):
        row = f"  {I_list[k]:5d}  {dts[k]:.5f}"
        for e in err.values():
            if not np.isfinite(e[k]):
                row += f"  {'diverged':>15}  {'':>5}"
                continue
            p = "" if k == 0 or not np.isfinite(e[k - 1]) else f"{np.log2(e[k - 1] / e[k]):5.2f}"
            row += f"  {e[k]:15.3e}  {p:>5}"
        print(row)

    # fit only on I >= 100: larger steps are not yet in the regime e ~ C dt^p
    asymptotic = I_list >= 100
    slopes = {n: np.polyfit(np.log(dts[asymptotic]), np.log(e[asymptotic]), 1)[0]
              for n, e in err.items()}
    for n, sl in slopes.items():
        print(f"  fitted slope, {SCHEMES[n]['label']}: {sl:.2f}"
              f"  (theory {SCHEMES[n]['order']})")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    for name, e in err.items():
        s = SCHEME_STYLE[name]
        ax.loglog(dts, e, s["marker"] + "-", ms=6, color=s["color"],
                  label=f"{SCHEMES[name]['label']} (slope {slopes[name]:.2f})")
    drawn = set()
    for name, e in sorted(err.items(), key=lambda ne: SCHEMES[ne[0]]["order"]):  # guide lines through the last point
        p = SCHEMES[name]["order"]
        ax.loglog(dts, e[-1] * (dts / dts[-1])**p, {2: ":", 3: "-.", 4: "--"}[p],
                  color=MUTED, lw=1.2, label=None if p in drawn else f"slope {p}")
        drawn.add(p)
    ax.set_xlabel(r"$\Delta t = T / I$")
    ax.set_ylabel(r"$\|Y^I - Y(T)\|_2$")
    ax.set_title("Convergence at t = T")
    ax.grid(True, which="both")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save_figure(fig, "convergence.png")
    return slopes


def _counted(f):
    def g(t, Y):
        g.calls += 1
        return f(t, Y)
    g.calls = 0
    return g


def _moulton_cost(I):
    Fc, dFc = _counted(F), _counted(dF)
    Y1 = RK4(Y0, T / I, 1, Fc)[:, 1]
    M = Moulton(N, Y0, Y1, T, I, Fc, dFc)
    return M, Fc.calls + dFc.calls                # evaluations of F and of its Jacobian


def same_cost_comparison(I_list=(100, 200, 400)):
    Y_ref = reference(F, Y0, T, [T])[:, -1]
    for k in I_list:
        e_rk4 = np.linalg.norm(RK4(Y0, T, k, F)[:, -1] - Y_ref)
        e_pm = np.linalg.norm(PointMilieu(N, Y0, T, 2 * k, F)[:, -1] - Y_ref)
        e_ab = np.linalg.norm(_solve_bashforth(Y0, T, 4 * k, F)[:, -1] - Y_ref)
        print(f"  {4 * k:5d} evaluations:  RK4 {e_rk4:.2e}   midpoint {e_pm:.2e}"
              f"   Adams-Bashforth {e_ab:.2e}   (RK4 {e_pm / e_rk4:.0f}x more accurate"
              f" than midpoint, {e_ab / e_rk4:.0f}x than Adams-Bashforth)")
        I_guess = 4 * k // 9                      # ~4 Newton iterations: 1 + 2 * 4 per step
        _, cost = _moulton_cost(I_guess)
        I_am = round(I_guess * 4 * k / cost)      # rescale I to hit the 4k budget
        M_am, cost = _moulton_cost(I_am)
        while cost > 4 * k:                       # never more evaluations than the others
            I_am -= 1
            M_am, cost = _moulton_cost(I_am)
        e_am = np.linalg.norm(M_am[:, -1] - Y_ref)
        newton = ((cost - 5) / (I_am - 1) - 1) / 2   # cost = 4 (RK4 for Y^1) + 1 + (I-1)(1 + 2 it)
        print(f"{'':20}Adams-Moulton {e_am:.2e}   (I = {I_am}, {cost} evaluations of F and dF,"
              f" {newton:.1f} Newton iterations per step, RK4 {e_am / e_rk4:.0f}x more accurate)")


def main():
    apply_style()
    np.set_printoptions(precision=6)

    print("0. Exact-solution test (c = 0)")
    for name in SCHEMES:
        _test(name)

    print(f"\n1. Van der Pol: c = {c}, alpha = {alpha}, beta = {beta}, "
          f"T = {T}, I = {I} (dt = {T / I})")
    results = {
        "PointMilieu": PointMilieu(N, Y0, T, I, F),
        "RK4": RK4(Y0, T, I, F),
        "Bashforth": _solve_bashforth(Y0, T, I, F),
        "Moulton": _solve_moulton(Y0, T, I, F),
    }
    t = np.linspace(0.0, T, I + 1)
    Y_ref = reference(F, Y0, T, t)
    errors = {}
    for name, M in results.items():
        errors[name] = np.linalg.norm(M - Y_ref, axis=0)   # ||Y^i - Y(t_i)||_2
        print(f"  {SCHEMES[name]['label']:>15}:  M.shape = {M.shape}   Y^I = {M[:, -1]}"
              f"   max_i ||Y^i - Y(t_i)|| = {errors[name].max():.3e}")

    print("\n2. Figures")
    t_fine = np.linspace(0.0, T, 3001)
    Y_fine = reference(F, Y0, T, t_fine)
    cycle = limit_cycle()
    for name, M in results.items():
        plot_solution(t, M, name, t_fine, Y_fine)
        plot_phase(M, name, Y_fine, cycle)
    plot_error_over_time(t, errors)

    print("\n3. Convergence at t = T")
    convergence(list(SCHEMES))

    print("\n4. Same number of F evaluations (4I)")
    same_cost_comparison()

    show_if_possible()


if __name__ == "__main__":
    main()
