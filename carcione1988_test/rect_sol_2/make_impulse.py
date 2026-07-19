import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from decimal import Decimal
from pathlib import Path


def gen_imp(eta, epsilone, t0, f0, dt, steps, path_save, filename):
    T = dt*steps

    t = np.arange(0, T, dt)
    imp = np.exp(-eta*f0**2 * np.power((t-t0), 2)) * np.cos(epsilone*np.pi*f0*(t-t0))


    delta_size = 0
    # mpl.rcParams.update({
    # 'font.family': 'serif',
    # 'font.serif': ['Times New Roman'],
    # 'font.size': 18 + delta_size,
    # 'axes.titlesize': 16 + delta_size,
    # 'axes.labelsize': 16 + delta_size,
    # 'xtick.labelsize': 14 + delta_size,
    # 'ytick.labelsize': 14 + delta_size,
    # 'legend.fontsize': 12 + delta_size,
    # 'figure.titlesize': 18 + delta_size
    # })
    # fig, ax = plt.subplots(figsize=(4,3))
    plt.figure()
    plt.plot(t, imp)
    # plt.xlim((0, T_impulse))
    # plt.grid()
    plt.xlabel("t, с")
    plt.ylabel("impulse")
    plt.title("Форма импульса")
    plt.tight_layout()
    plt.savefig(path_save/f"{filename}.png", dpi=300)
    plt.close()

    to_save = np.vstack((t, imp)).T
    # print(t.shape, amp.shape, to_save.shape)
    # print(to_save[:10])
    decimals = abs(Decimal(str(dt)).as_tuple().exponent)
    fmt_time = f'%.{decimals}f'
    np.savetxt(path_save/f"{filename}.txt", to_save, fmt=[fmt_time, '%.18e'])
    return path_save

if __name__ == "__main__":
    eta = 0.5
    epsilone = 1
    t0 = 0.06
    f0 = 50
    dt = 5e-4
    steps = 1001
    gen_imp(eta, epsilone, t0, f0, dt, steps, Path("."), "impulse")
