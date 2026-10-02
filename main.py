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
        Yi = M[:, i]                              # Y^i (read only)
        K1 = F(t[i], Yi)                          # slope at t_i
        K2 = F(t[i] + dt / 2, Yi + dt / 2 * K1)   # slope at the midpoint
        M[:, i + 1] = Yi + dt * K2
    return M


def _solve_midpoint(Y0, T, I, F):
    return PointMilieu(np.size(Y0), Y0, T, I, F)

SCHEMES = {
    "PointMilieu": dict(solver=_solve_midpoint, order=2, label="midpoint"),
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

def main():
    np.set_printoptions(precision=6)

    print("0. Exact-solution test (c = 0)")
    for name in SCHEMES:
        _test(name)

    print(f"\n1. Van der Pol: c = {c}, alpha = {alpha}, beta = {beta}, "
          f"T = {T}, I = {I} (dt = {T / I})")
    results = {
        "PointMilieu": PointMilieu(N, Y0, T, I, F),   # call exactly as in the PDF
    }
    t = np.linspace(0.0, T, I + 1)
    Y_ref = reference(F, Y0, T, t)
    errors = {}
    for name, M in results.items():
        errors[name] = np.linalg.norm(M - Y_ref, axis=0)   # ||Y^i - Y(t_i)||_2
        print(f"  {SCHEMES[name]['label']:>8}:  M.shape = {M.shape}   Y^I = {M[:, -1]}"
              f"   max_i ||Y^i - Y(t_i)|| = {errors[name].max():.3e}")


if __name__ == "__main__":
    main()
