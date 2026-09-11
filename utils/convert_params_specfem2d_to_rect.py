import numpy as np


def compute_relaxation_times(Q, f0, N_SLS):
    """
    Вычисляет tau_sigma и tau_epsilon для набора SLS-механизмов
    по алгоритму, соответствующему SPECFEM2D при стандартном
    частотном диапазоне [f0/10, 10*f0].
    """

    f_min = f0 / 10.0
    f_max = f0 * 10.0

    # Частоты релаксационных механизмов
    f_relax = np.geomspace(f_min, f_max, N_SLS)
    theta = 2.0 * np.pi * f_relax

    # tau_sigma_l = 1 / theta_l
    tau_sigma = 1.0 / theta

    # Частоты, на которых аппроксимируется Q:
    # всего 2*N_SLS - 1 точек
    f_fit = np.geomspace(f_min, f_max, 2 * N_SLS - 1)
    omega = 2.0 * np.pi * f_fit

    # Система A w ~= b
    A = np.zeros((2 * N_SLS - 1, N_SLS))

    for i, w in enumerate(omega):
        for l, th in enumerate(theta):
            A[i, l] = (
                w * (th - w / Q)
                / (th**2 + w**2)
            )

    b = np.full(2 * N_SLS - 1, 1.0 / Q)

    # Метод наименьших квадратов
    weights, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    # tau_epsilon_l
    tau_epsilon = tau_sigma * (1.0 + N_SLS * weights)

    return tau_epsilon, tau_sigma


def compute_material(
    rho,
    vp_unrelaxed,
    vs_unrelaxed,
    QKappa,
    Qmu,
    f0,
    N_SLS,
):
    # ------------------------------------------------------------
    # 1. Времена релаксации
    # ------------------------------------------------------------

    # Объёмная часть
    tau1_e, tau1_s = compute_relaxation_times(
        Q=QKappa,
        f0=f0,
        N_SLS=N_SLS,
    )

    # Сдвиговая часть
    tau2_e, tau2_s = compute_relaxation_times(
        Q=Qmu,
        f0=f0,
        N_SLS=N_SLS,
    )

    # ------------------------------------------------------------
    # 2. Отношение unrelaxed / relaxed модулей
    # ------------------------------------------------------------

    M_kappa = np.mean(tau1_e / tau1_s)
    M_mu = np.mean(tau2_e / tau2_s)

    # ------------------------------------------------------------
    # 3. Unrelaxed модули
    #
    # Для 2D plane strain:
    #
    # mu = rho * Vs^2
    # kappa = lambda + mu = rho * (Vp^2 - Vs^2)
    #
    # Для 3D формула другая!
    # ------------------------------------------------------------

    mu_unrelaxed = rho * vs_unrelaxed**2

    kappa_unrelaxed = rho * (
        vp_unrelaxed**2 - vs_unrelaxed**2
    )

    # ------------------------------------------------------------
    # 4. Relaxed модули
    # ------------------------------------------------------------

    mu_relaxed = mu_unrelaxed / M_mu
    kappa_relaxed = kappa_unrelaxed / M_kappa

    # ------------------------------------------------------------
    # 5. Relaxed скорости
    # ------------------------------------------------------------

    vs_relaxed = np.sqrt(
        mu_relaxed / rho
    )

    vp_relaxed = np.sqrt(
        (kappa_relaxed + mu_relaxed) / rho
    )

    return {
        "rho": rho,
        "c1": vp_relaxed,
        "c2": vs_relaxed,
        "tau1_e": tau1_e,
        "tau1_s": tau1_s,
        "tau2_e": tau2_e,
        "tau2_s": tau2_s,
    }


def format_tau(value):
    """
    Форматирование как в требуемом примере:
    значения >= 0.1 печатаются десятичной дробью,
    меньшие значения — в экспоненциальной форме.
    """
    if abs(value) >= 0.1:
        return f"{value:.15f}"

    return f"{value:.15e}"


def print_material(material):
    print(f"c1 = {material['c1']:.10f}")
    print(f"c2 = {material['c2']:.10f}")

    rho = material["rho"]

    if float(rho).is_integer():
        print(f"rho = {int(rho)}")
    else:
        print(f"rho = {rho}")

    for i, value in enumerate(material["tau1_e"], start=1):
        print(f"tau1_e_{i} = {format_tau(value)}")

    for i, value in enumerate(material["tau1_s"], start=1):
        print(f"tau1_s_{i} = {format_tau(value)}")

    for i, value in enumerate(material["tau2_e"], start=1):
        print(f"tau2_e_{i} = {format_tau(value)}")

    for i, value in enumerate(material["tau2_s"], start=1):
        print(f"tau2_s_{i} = {format_tau(value)}")


if __name__ == "__main__":
    rho = 2500.0

    vp_unrelaxed = 3957.419
    vs_unrelaxed = 2444.790

    QKappa = 44.6
    Qmu = 30.0

    f0 = 18.0
    N_SLS = 3

    material = compute_material(
        rho=rho,
        vp_unrelaxed=vp_unrelaxed,
        vs_unrelaxed=vs_unrelaxed,
        QKappa=QKappa,
        Qmu=Qmu,
        f0=f0,
        N_SLS=N_SLS,
    )

    print_material(material)
