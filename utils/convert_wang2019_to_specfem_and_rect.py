import numpy as np


def qp_qs_to_qkappa_qmu(vp, vs, Qp, Qs):
    """
    Qp, Qs -> Qkappa, Qmu
    для SPECFEM2D, Cartesian 2D plane strain.
    """
    r = (vs / vp) ** 2

    Qmu = Qs

    inv_Qkappa = (
        1.0 / Qp
        - r / Qmu
    ) / (1.0 - r)

    if inv_Qkappa <= 0.0:
        raise ValueError(
            "Получено Qkappa <= 0. "
            "Проверьте Vp, Vs, Qp и Qs."
        )

    Qkappa = 1.0 / inv_Qkappa

    return Qkappa, Qmu


def compute_relaxation_times(Q, f0, N_SLS):
    """
    Расчёт tau_epsilon и tau_sigma
    по алгоритму SPECFEM2D при USE_SOLVOPT = .false.

    Используется диапазон:
        f_min = f0 / 10
        f_max = f0 * 10
    """

    if N_SLS < 1:
        raise ValueError("N_SLS must be >= 1")

    f_min = f0 / 10.0
    f_max = f0 * 10.0

    # ---------------------------------------------------------
    # Частоты релаксационных механизмов
    # ---------------------------------------------------------

    if N_SLS == 1:
        f_relax = np.array([
            np.sqrt(f_min * f_max)
        ])
    else:
        f_relax = np.geomspace(
            f_min,
            f_max,
            N_SLS,
        )

    theta = 2.0 * np.pi * f_relax

    # tau_sigma_l = 1 / theta_l
    tau_sigma = 1.0 / theta

    # ---------------------------------------------------------
    # Частоты, на которых аппроксимируется 1/Q
    # ---------------------------------------------------------

    number_fit_points = 2 * N_SLS - 1

    if number_fit_points == 1:
        f_fit = np.array([
            np.sqrt(f_min * f_max)
        ])
    else:
        f_fit = np.geomspace(
            f_min,
            f_max,
            number_fit_points,
        )

    omega = 2.0 * np.pi * f_fit

    # ---------------------------------------------------------
    # Матрица системы A w ~= b
    #
    # A_kl =
    #
    # omega_k * (theta_l - omega_k / Q)
    # -----------------------------------
    #        theta_l^2 + omega_k^2
    #
    # ---------------------------------------------------------

    A = np.empty(
        (number_fit_points, N_SLS),
        dtype=float,
    )

    for k, w in enumerate(omega):
        for l, th in enumerate(theta):
            A[k, l] = (
                w * (th - w / Q)
                / (th**2 + w**2)
            )

    b = np.full(
        number_fit_points,
        1.0 / Q,
    )

    # Linear least squares
    weights, *_ = np.linalg.lstsq(
        A,
        b,
        rcond=None,
    )

    # ---------------------------------------------------------
    # tau_epsilon_l
    # ---------------------------------------------------------

    tau_epsilon = (
        tau_sigma
        * (1.0 + N_SLS * weights)
    )

    return tau_epsilon, tau_sigma


def modulus_factors(
    tau_epsilon,
    tau_sigma,
    f0,
):
    """
    Возвращает:

        A = M_unrelaxed / M_relaxed

    и фактор

        F = M_unrelaxed / M(f0)

    для данного набора SLS.
    """

    N = len(tau_sigma)

    omega0 = 2.0 * np.pi * f0

    a = (
        tau_epsilon / tau_sigma
        - 1.0
    )

    # M_U / M_R
    A = 1.0 + np.sum(a) / N

    # знаменатель для модуля при f0
    B = 1.0 + np.sum(
        a
        / (
            1.0
            + 1.0
            / (omega0 * tau_sigma) ** 2
        )
    ) / N

    # M_U / M(f0)
    factor_unrelaxed = A / B

    return A, B, factor_unrelaxed


def convert_material(
    rho,
    vp_f0,
    vs_f0,
    Qp,
    Qs,
    f0,
    N_SLS,
):
    # =========================================================
    # 1. Qp, Qs -> Qkappa, Qmu
    # =========================================================

    Qkappa, Qmu = qp_qs_to_qkappa_qmu(
        vp=vp_f0,
        vs=vs_f0,
        Qp=Qp,
        Qs=Qs,
    )

    # =========================================================
    # 2. Relaxation times
    # =========================================================

    # Bulk / dilatational part
    tau1_e, tau1_s = compute_relaxation_times(
        Q=Qkappa,
        f0=f0,
        N_SLS=N_SLS,
    )

    # Shear part
    tau2_e, tau2_s = compute_relaxation_times(
        Q=Qmu,
        f0=f0,
        N_SLS=N_SLS,
    )

    # =========================================================
    # 3. Moduli at reference frequency f0
    #
    # 2D plane strain:
    #
    # mu = rho * Vs^2
    # kappa = lambda + mu
    #       = rho * (Vp^2 - Vs^2)
    # =========================================================

    mu_f0 = rho * vs_f0**2

    kappa_f0 = rho * (
        vp_f0**2
        - vs_f0**2
    )

    # =========================================================
    # 4. SLS scaling factors
    # =========================================================

    (
        A_kappa,
        B_kappa,
        factor_kappa,
    ) = modulus_factors(
        tau_epsilon=tau1_e,
        tau_sigma=tau1_s,
        f0=f0,
    )

    (
        A_mu,
        B_mu,
        factor_mu,
    ) = modulus_factors(
        tau_epsilon=tau2_e,
        tau_sigma=tau2_s,
        f0=f0,
    )

    # =========================================================
    # 5. Unrelaxed moduli
    #
    # These are the internal SPECFEM moduli.
    # =========================================================

    kappa_unrelaxed = (
        kappa_f0 * factor_kappa
    )

    mu_unrelaxed = (
        mu_f0 * factor_mu
    )

    # =========================================================
    # 6. Relaxed moduli
    #
    # M_R = M_U / A
    #
    # equivalently:
    #
    # M_R = M(f0) / B
    # =========================================================

    kappa_relaxed = (
        kappa_unrelaxed / A_kappa
    )

    mu_relaxed = (
        mu_unrelaxed / A_mu
    )

    # =========================================================
    # 7. Velocities
    # =========================================================

    vs_unrelaxed = np.sqrt(
        mu_unrelaxed / rho
    )

    vp_unrelaxed = np.sqrt(
        (
            kappa_unrelaxed
            + mu_unrelaxed
        )
        / rho
    )

    vs_relaxed = np.sqrt(
        mu_relaxed / rho
    )

    vp_relaxed = np.sqrt(
        (
            kappa_relaxed
            + mu_relaxed
        )
        / rho
    )

    return {
        "rho": rho,

        "vp_f0": vp_f0,
        "vs_f0": vs_f0,

        "Qp": Qp,
        "Qs": Qs,

        "Qkappa": Qkappa,
        "Qmu": Qmu,

        "vp_unrelaxed": vp_unrelaxed,
        "vs_unrelaxed": vs_unrelaxed,

        "vp_relaxed": vp_relaxed,
        "vs_relaxed": vs_relaxed,

        "tau1_e": tau1_e,
        "tau1_s": tau1_s,

        "tau2_e": tau2_e,
        "tau2_s": tau2_s,
    }


def fmt(value):
    if abs(value) >= 0.1:
        return f"{value:.15f}"

    return f"{value:.15e}"


def print_material(name, material):
    print("=" * 70)
    print(name)
    print("=" * 70)

    print()
    print("SPECFEM2D:")
    print(f"rho = {material['rho']}")
    print(f"Vp(f0) = {material['vp_f0']:.10f}")
    print(f"Vs(f0) = {material['vs_f0']:.10f}")
    print(f"QKappa = {material['Qkappa']:.15g}")
    print(f"Qmu = {material['Qmu']:.15g}")

    print()
    print("SPECFEM internal unrelaxed velocities:")
    print(
        f"Vp_unrelaxed = "
        f"{material['vp_unrelaxed']:.10f}"
    )
    print(
        f"Vs_unrelaxed = "
        f"{material['vs_unrelaxed']:.10f}"
    )

    print()
    print("RECT:")

    print(
        f"c1 = "
        f"{material['vp_relaxed']:.10f}"
    )
    print(
        f"c2 = "
        f"{material['vs_relaxed']:.10f}"
    )

    rho = material["rho"]

    if float(rho).is_integer():
        print(f"rho = {int(rho)}")
    else:
        print(f"rho = {rho}")

    for i, value in enumerate(
        material["tau1_e"],
        start=1,
    ):
        print(
            f"tau1_e_{i} = {fmt(value)}"
        )

    for i, value in enumerate(
        material["tau1_s"],
        start=1,
    ):
        print(
            f"tau1_s_{i} = {fmt(value)}"
        )

    for i, value in enumerate(
        material["tau2_e"],
        start=1,
    ):
        print(
            f"tau2_e_{i} = {fmt(value)}"
        )

    for i, value in enumerate(
        material["tau2_s"],
        start=1,
    ):
        print(
            f"tau2_s_{i} = {fmt(value)}"
        )

    print()


if __name__ == "__main__":
    f0 = 1.0
    N_SLS = 3

    materials = [
        {
            "name": "Layer 1",
            "rho": 2100.0,
            "vp_f0": 6000.0,
            "vs_f0": 3500.0,
            "Qp": 160.0,
            "Qs": 80.0,
        },
        {
            "name": "Layer 2",
            "rho": 2300.0,
            "vp_f0": 7200.0,
            "vs_f0": 4200.0,
            "Qp": 300.0,
            "Qs": 150.0,
        },
    ]

    for params in materials:
        name = params.pop("name")

        material = convert_material(
            **params,
            f0=f0,
            N_SLS=N_SLS,
        )

        print_material(
            name,
            material,
        )
