import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from decimal import Decimal
from pathlib import Path


# def gen_imp(eta, epsilone, t0, f0, dt, steps, path_save, filename):
#     T = dt*steps

#     t = np.arange(0, T, dt)
#     imp = np.exp(-eta*f0**2 * np.power((t-t0), 2)) * np.cos(epsilone*np.pi*f0*(t-t0))


#     delta_size = 0
#     # mpl.rcParams.update({
#     # 'font.family': 'serif',
#     # 'font.serif': ['Times New Roman'],
#     # 'font.size': 18 + delta_size,
#     # 'axes.titlesize': 16 + delta_size,
#     # 'axes.labelsize': 16 + delta_size,
#     # 'xtick.labelsize': 14 + delta_size,
#     # 'ytick.labelsize': 14 + delta_size,
#     # 'legend.fontsize': 12 + delta_size,
#     # 'figure.titlesize': 18 + delta_size
#     # })
#     # fig, ax = plt.subplots(figsize=(4,3))
#     plt.figure()
#     plt.plot(t, imp)
#     # plt.xlim((0, T_impulse))
#     # plt.grid()
#     plt.xlabel("t, с")
#     plt.ylabel("impulse")
#     plt.title("Форма импульса")
#     plt.tight_layout()
#     plt.savefig(path_save/f"{filename}.png", dpi=300)
#     plt.close()

#     to_save = np.vstack((t, imp)).T
#     # print(t.shape, amp.shape, to_save.shape)
#     # print(to_save[:10])
#     decimals = abs(Decimal(str(dt)).as_tuple().exponent)
#     fmt_time = f'%.{decimals}f'
#     np.savetxt(path_save/f"{filename}.txt", to_save, fmt=[fmt_time, '%.18e'])
#     return path_save


def gen_ricker_imp_specfem(
    t0: float,
    f0: float,
    dt: float,
    steps: int,
    path_save: Path,
    filename: str,
    force_amplitude: float = 1.0,
) -> Path:
    """
    Создаёт классический импульс Рикера:

        impulse(t) = F * (1 - 2*pi^2*f0^2*(t-t0)^2)
                       * exp(-pi^2*f0^2*(t-t0)^2)

    Parameters
    ----------
    t0 : float
        Момент времени, в котором расположен максимум импульса, с.
    f0 : float
        Центральная частота импульса, Гц.
    dt : float
        Шаг по времени, с.
    steps : int
        Количество временных шагов.
    path_save : Path
        Каталог для сохранения результатов.
    filename : str
        Имя файлов без расширения.
    force_amplitude : float, optional
        Амплитуда силы F. По умолчанию равна 1.

    Returns
    -------
    Path
        Путь к каталогу с сохранёнными файлами.
    """
    if dt <= 0:
        raise ValueError("dt должен быть положительным")

    if f0 <= 0:
        raise ValueError("f0 должен быть положительным")

    if steps <= 0:
        raise ValueError("steps должен быть положительным")

    path_save = Path(path_save)
    path_save.mkdir(parents=True, exist_ok=True)

    # Такая запись гарантирует получение ровно steps отсчётов.
    t = np.arange(steps, dtype=np.float64) * dt

    tau = t - t0
    a = np.pi**2 * f0**2

    impulse = (
        force_amplitude
        * (1.0 - 2.0 * a * tau**2)
        * np.exp(-a * tau**2)
    )

    plt.figure()
    plt.plot(t, impulse)
    plt.xlabel("t, с")
    plt.ylabel("impulse")
    plt.title("Классический импульс Рикера")
    plt.tight_layout()
    plt.savefig(path_save / f"{filename}.png", dpi=300)
    plt.close()

    data_to_save = np.column_stack((t, impulse))

    decimals = max(
        0,
        -Decimal(str(dt)).normalize().as_tuple().exponent,
    )
    fmt_time = f"%.{decimals}f"

    np.savetxt(
        path_save / f"{filename}.txt",
        data_to_save,
        fmt=[fmt_time, "%.18e"],
    )

    return path_save


if __name__ == "__main__":
    # eta = 0.5
    # epsilone = 1
    # t0 = 0.06
    # f0 = 50
    # dt = 5e-4
    # steps = 1001
    # gen_imp(eta, epsilone, t0, f0, dt, steps, Path("."), "impulse")

    f0 = 18.0
    t0 = 1.2 / f0

    gen_ricker_imp_specfem(
        t0=t0,
        f0=f0,
        dt=1.0e-4,
        steps=2000,
        path_save=Path("."),
        filename="impulse",
    )