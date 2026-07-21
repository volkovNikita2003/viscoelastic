from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def read_an_sol(filename):
    return pd.read_csv(
        filename,
        sep=r"\s+",
        header=None,
        names=["x", "y"],
        engine="python",
    )


def read_station_rect(filename):
    df = pd.read_csv(
        filename,
        sep=r"\s+",
        comment="#",
        names=["T", "vx", "vy", "sxx", "syy", "sxy"],
        engine="python",
    )

    t = df["T"].to_numpy()
    vx = df["vx"].to_numpy()
    vy = df["vy"].to_numpy()

    ux = np.zeros_like(t, dtype=float)
    uy = np.zeros_like(t, dtype=float)

    dt = np.diff(t)

    ux[1:] = np.cumsum(
        0.5 * (vx[:-1] + vx[1:]) * dt
    )
    uy[1:] = np.cumsum(
        0.5 * (vy[:-1] + vy[1:]) * dt
    )

    df["ux"] = ux
    df["uy"] = uy

    return df


def plot_accuracy_graph(error_norms, filename_suffix=""):
    rows = error_norms
    rows = sorted(
        rows,
        key=lambda x: x["h"]
    )

    h_vals = [r["h"] for r in rows]

    l1_vals = [r["L1"] for r in rows]
    l2_vals = [r["L2"] for r in rows]
    linf_vals = [r["Linf"] for r in rows]


    log_h = np.log(h_vals)
    p_l1, c_l1 = np.polyfit(log_h, np.log(l1_vals), 1)
    p_l2, c_l2 = np.polyfit(log_h, np.log(l2_vals), 1)
    p_linf, c_linf = np.polyfit(log_h, np.log(linf_vals), 1)
    print(
        f"{station} {component}: "
        f"p(L1)={p_l1:.3f}, "
        f"p(L2)={p_l2:.3f}, "
        f"p(Linf)={p_linf:.3f}"
    )

    plt.figure(figsize=(7,5))

    plt.loglog(
        h_vals,
        l1_vals,
        "o-",
        color="blue",
        label=f"L1 (p={p_l1:.2f})",
    )
    plt.loglog(
        h_vals,
        l2_vals,
        "s-",
        color="red",
        label=f"L2 (p={p_l2:.2f})",
    )
    plt.loglog(
        h_vals,
        linf_vals,
        "^-",
        color="green",
        label=f"L∞ (p={p_linf:.2f})",
    )

    fit_l1 = np.exp(c_l1) * np.array(h_vals) ** p_l1
    fit_l2 = np.exp(c_l2) * np.array(h_vals) ** p_l2
    fit_linf = np.exp(c_linf) * np.array(h_vals) ** p_linf
    plt.loglog(
        h_vals,
        fit_l1,
        "--",
        color="blue",
        linewidth=1,
    )
    plt.loglog(
        h_vals,
        fit_l2,
        "--",
        color="red",
        linewidth=1,
    )
    plt.loglog(
        h_vals,
        fit_linf,
        "--",
        color="green",
        linewidth=1,
    )

    plt.xlabel("h")
    plt.ylabel("error norm")

    plt.grid(True, which="both")
    plt.legend()
    plt.title(
        f"{station} {component}"
    )

    outfile = OUTPUT_DIR / f"{station}-{component}-error-norms{filename_suffix}.png"
    plt.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Сохранён {outfile}")





BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "graphs"
OUTPUT_DIR.mkdir(exist_ok=True)
cases_h = [10, 4, 2, 1, 0.5, 0.25]
# c1u = 3297.849
rho = 2000


# ----- аналитическое решение -----
print(f"Аналитическое решение")
ANALYTICAL_DIR = BASE_DIR / "analytical_sol"
SPECFEM_DIR = BASE_DIR / "specfem_sol/check_absolute_amplitude_of_force_source_seismograms_viscoelastic_auto_nx_88"



# stations_an = {
#     "_500.0_500.0": "station_1",
#     "_0.0_500.0": "station_2",
#     "_500.0_0.0": "station_3",
#     "_-500.0_500.0": "station_4",
# }

# analytical_files = [
#     "spectrum_of_the_source_used.gnu",
# ]
# # analytical_files += [f"Vx_time_analytical_solution_viscoelastic{i}.dat" for i in stations_an.keys()]
# # analytical_files += [f"Vz_time_analytical_solution_viscoelastic{i}.dat" for i in stations_an.keys()]

# analytical_files += [f"Ux_time_analytical_solution_viscoelastic.dat"]
# analytical_files += [f"Uz_time_analytical_solution_viscoelastic.dat"]

# analytical_files_elastic = [
#     f"Ux_time_analytical_solution_elastic.dat",
#     f"Uz_time_analytical_solution_elastic.dat"
# ]

# ----- RECT -----
print(f"\nЧисленное решение")
RECT_SOL_DIR = BASE_DIR / "rect_sol"


# ----- сравнение численного и аналитического решений -----
print(f"\nСравнение численного и аналитического решений")

stations_an = {
    "station_1": "",
    # "station_2": "_0.0_500.0",
    # "station_3": "_500.0_0.0",
    # "station_4": "_-500.0_500.0",
}

analytical = {}
analytical_elastic = {}
specfem_sol = {}

for station, suffix in stations_an.items():
    ux = read_an_sol(ANALYTICAL_DIR / f"Ux_time_analytical_solution_viscoelastic{suffix}.dat")
    uy = read_an_sol(ANALYTICAL_DIR / f"Uz_time_analytical_solution_viscoelastic{suffix}.dat")
    analytical[(station, "ux")] = ux
    analytical[(station, "uy")] = uy

    ux = read_an_sol(ANALYTICAL_DIR / f"Ux_time_analytical_solution_elastic{suffix}.dat")
    uy = read_an_sol(ANALYTICAL_DIR / f"Uz_time_analytical_solution_elastic{suffix}.dat")
    analytical_elastic[(station, "ux")] = ux
    analytical_elastic[(station, "uy")] = uy

    ux = np.loadtxt(SPECFEM_DIR/"OUTPUT_FILES/AA.S0001.BXX.semd")
    uy = np.loadtxt(SPECFEM_DIR/"OUTPUT_FILES/AA.S0001.BXZ.semd")
    specfem_sol[(station, "ux")] = ux
    specfem_sol[(station, "uy")] = uy


scale_factors_num = {}
scale_factors_an = {}
for h in cases_h:
    rect_dir = RECT_SOL_DIR / f"viscoelastic_schema-h_{h}/result/txt"
    num = read_station_rect(
        rect_dir / "station_1.txt"
    )
    amp_num = np.max(np.abs(num["ux"]))

    an = analytical[("station_1", "ux")]
    amp_an = np.max(np.abs(an["y"]))
    scale_factors_num[h] = amp_an / amp_num
    # scale_factors_an[h] = c1u**2 * rho * 2*np.pi / h**2  # c^2*rho*2pi/h^2
    scale_factors_an[h] = 1 / rho / h**2
    print(
        f"viscoelastic: h={h}, "
        f"amp_num={amp_num:.4e}, "
        f"amp_an={amp_an:.4e}, "
        f"k_num={scale_factors_num[h]:.4e}, "
        f"k_an={scale_factors_an[h]:.4e}"
    )


f0 = 18.0
t0 = 1.2 / f0
# сдивг из-за реализации specfem
t_shift = t0
error_norms_rect = {}
error_norms_specfem = {}
for h in cases_h:
    rect_dir = RECT_SOL_DIR / f"viscoelastic_schema-h_{h}/result/txt"
    # k = scale_factors_num[h]
    # k = scale_factors_an[h]
    k = 1
    for station in stations_an.keys():
        num = read_station_rect(rect_dir / f"{station}.txt")
        for component in ["ux", "uy"]:
            t_num = num["T"].to_numpy()
            u_num = (num[component].to_numpy()* k)

            specfem = specfem_sol[(station, component)]
            t_specfem = specfem[:, 0] + t_shift
            u_specfem = specfem[:, 1]
            # решение specfem на сетке численного решения
            u_specfem_interp = np.interp(t_num, t_specfem, u_specfem,)
            
            an = analytical[(station, component)]
            t_an = an["x"].to_numpy() + t_shift
            u_an = an["y"].to_numpy()

            an_elastic = analytical_elastic[(station, component)]
            t_an_elastic = an_elastic["x"].to_numpy() + t_shift
            u_an_elastic = an_elastic["y"].to_numpy()

            # аналитика на сетке численного решения
            u_an_interp = np.interp(t_num, t_an, u_an,)
            err = u_num - u_an_interp
            l1 = np.mean(np.abs(err))
            l2 = np.sqrt(np.mean(err**2))
            linf = np.max(np.abs(err))
            error_norms_rect.setdefault(
                (station, component),
                []
            ).append(
                {
                    "h": h,
                    "L1": l1,
                    "L2": l2,
                    "Linf": linf,
                }
            )

            # аналитика на сетке численного решения specfem
            u_an_interp_specfem = np.interp(t_specfem, t_an, u_an,)
            err_specfem = u_specfem - u_an_interp_specfem
            l1 = np.mean(np.abs(err_specfem))
            l2 = np.sqrt(np.mean(err_specfem**2))
            linf = np.max(np.abs(err_specfem))
            error_norms_specfem.setdefault(
                (station, component),
                []
            ).append(
                {
                    "h": h,
                    "L1": l1,
                    "L2": l2,
                    "Linf": linf,
                }
            )

            fig, axes = plt.subplots(
                1,
                1,
                figsize=(8, 6),
                sharex=True,
            )

            axes.plot(
                t_num,
                u_num,
                label=f"rect h={h}",
            )
            # axes.plot(
            #     t_num,
            #     u_an_interp,
            #     "--",
            #     label="analytical",
            # )
            # axes.plot(
            #     t_num,
            #     u_specfem_interp,
            #     ":",
            #     label="specfem",
            # )
            axes.plot(
                t_an,
                u_an,
                "--",
                label="analytical",
            )
            axes.plot(
                t_specfem,
                u_specfem,
                ":",
                label="specfem",
            )
            axes.plot(
                t_an_elastic,
                u_an_elastic,
                "-.",
                label="analytical elastic",
            )
            # axes.plot(
            #     t_num,
            #     err*10,
            #     ":",
            #     label="error*10"
            # )
            axes.grid(True)
            axes.legend()
            axes.set_ylabel(component)
            axes.set_xlabel("t, s")
            axes.set_xlim(0, 0.6)

            fig.suptitle(
                f"{station}, {component}, h={h}"
            )
            fig.tight_layout()
            outfile = (
                OUTPUT_DIR /
                f"{station}-{component}-h_{h}-compare.png"
            )
            fig.savefig(
                outfile,
                dpi=300,
                bbox_inches="tight",
            )
            plt.close(fig)
            print(f"Сохранён {outfile}")


print(f"\nГрафики сходимости")
for station, component in error_norms_rect:
    plot_accuracy_graph(error_norms_rect[(station, component)], "-rect")
    plot_accuracy_graph(error_norms_specfem[(station, component)], "-specfem")


