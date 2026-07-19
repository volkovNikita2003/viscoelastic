from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def velocity_to_displacement(df):
    t = df["T"].to_numpy()
    ux = np.zeros_like(t)
    uy = np.zeros_like(t)
    dt = np.diff(t)
    ux[1:] = np.cumsum(
        0.5 * (df["vx"].to_numpy()[:-1] + df["vx"].to_numpy()[1:]) * dt
    )
    uy[1:] = np.cumsum(
        0.5 * (df["vy"].to_numpy()[:-1] + df["vy"].to_numpy()[1:]) * dt
    )
    return ux, uy


BASE_DIR = Path(__file__).parent
OUTPUT_DIR = BASE_DIR / "plots"
OUTPUT_DIR.mkdir(exist_ok=True)
cases_h = [10, 4, 2, 1, 0.5, 0.25]
rho = 2000


# ----- аналитическое решение -----
print(f"Аналитическое решение")
ANALYTICAL_DIR = BASE_DIR / "analytical_sol"

def read_two_columns(filename):
    return pd.read_csv(
        filename,
        sep=r"\s+",
        header=None,
        names=["x", "y"],
        engine="python",
    )

stations_an = {
    "_500.0_500.0": "station_1",
    "_0.0_500.0": "station_2",
    "_500.0_0.0": "station_3",
    "_-500.0_500.0": "station_4",
}

analytical_files = [
    "spectrum_of_the_source_used.gnu",
]
analytical_files += [f"Vx_time_analytical_solution_viscoelastic{i}.dat" for i in stations_an.keys()]
analytical_files += [f"Vz_time_analytical_solution_viscoelastic{i}.dat" for i in stations_an.keys()]

for filename in analytical_files:
    filepath = ANALYTICAL_DIR / filename
    if not filepath.exists():
        print(f"Файл отсутствует: {filepath}")
        continue

    data = read_two_columns(filepath)

    plt.figure(figsize=(8, 5))
    plt.plot(
        data["x"],
        data["y"],
        linewidth=1,
    )

    if "spectrum" in filename:
        plt.ylim(0, data["y"].max() * 1.05)
    else:
        ymax = max(np.abs(data["y"]).max(), 1e-12)
        plt.ylim(-ymax, ymax)

    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid(True)

    plt.title(filepath.stem)

    # ----------------------------
    # Формирование имени файла
    # ----------------------------

    if filename == "spectrum_of_the_source_used.gnu":
        outname = "source-spectrum-an.png"
    else:
        station_name = None
        for suffix, station in stations_an.items():
            if suffix in filename:
                station_name = station
                break
        if station_name is None:
            print(f"Не удалось определить станцию для {filename}")
            continue
        if filename.startswith("Vx_"):
            component = "vx"
        elif filename.startswith("Vz_"):
            component = "vy"
        else:
            component = "unknown"
        outname = f"{station_name}-{component}-an.png"

    outfile = OUTPUT_DIR / outname

    plt.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"Сохранён {outfile}")


# ----- RECT -----
print(f"\nЧисленное решение")
RECT_SOL_DIR = BASE_DIR / "rect_sol"

def read_station(filename):
    return pd.read_csv(
        filename,
        sep=r"\s+",
        comment="#",
        names=["T", "vx", "vy", "sxx", "syy", "sxy"],
        engine="python",
    )


for h in cases_h:
    ELASTIC_DIR = RECT_SOL_DIR / f"elastic-h_{h}/result/txt"
    VISCOELASTIC_DIR = RECT_SOL_DIR / f"viscoelastic-h_{h}/result/txt"
    stations_rect = [
        "station_1",
        "station_2",
        "station_3",
        "station_4",
    ]

    for station_name in stations_rect:
        elastic_file = ELASTIC_DIR / f"{station_name}.txt"
        elastic = read_station(elastic_file)

        visco_file = VISCOELASTIC_DIR / f"{station_name}.txt"
        if not visco_file.exists():
            print(f"Файл отсутствует: {visco_file}")
            continue
        visco = read_station(visco_file)

        
        fig, axes = plt.subplots(
            1,
            2,
            figsize=(10, 5),
            sharex=True,
        )

        # ---------- vx ----------
        axes[0].plot(
            elastic["T"],
            elastic["vx"],
            "--",
            linewidth=1,
            label="elastic",
        )

        axes[0].plot(
            visco["T"],
            visco["vx"],
            linewidth=1,
            label="viscoelastic",
        )

        vx_max = max(
            elastic["vx"].abs().max(),
            visco["vx"].abs().max(),
            1e-12,
        )

        axes[0].set_ylim(-vx_max, vx_max)
        axes[0].set_xlabel("t, s")
        axes[0].set_ylabel("vx")
        axes[0].grid(True)
        axes[0].legend()

        # ---------- vy ----------
        axes[1].plot(
            elastic["T"],
            elastic["vy"],
            "--",
            linewidth=1,
            label="elastic",
        )

        axes[1].plot(
            visco["T"],
            visco["vy"],
            linewidth=1,
            label="viscoelastic",
        )

        vy_max = max(
            elastic["vy"].abs().max(),
            visco["vy"].abs().max(),
            1e-12,
        )

        axes[1].set_ylim(-vy_max, vy_max)
        axes[1].set_xlabel("t, s")
        axes[1].set_ylabel("vy")
        axes[1].grid(True)
        axes[1].legend()

        fig.suptitle(f"{station_name}, h={h}")
        fig.tight_layout()

        outfile = OUTPUT_DIR / f"{station_name}-h_{h}.png"
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Сохранён {outfile}")




        elastic_ux, elastic_uy = velocity_to_displacement(elastic)
        visco_ux, visco_uy = velocity_to_displacement(visco)


        fig, axes = plt.subplots(
            1,
            2,
            figsize=(10, 5),
            sharex=True,
        )

        # ---------- ux ----------
        axes[0].plot(
            elastic["T"],
            elastic_ux,
            "--",
            linewidth=1,
            label="elastic",
        )

        axes[0].plot(
            visco["T"],
            visco_ux,
            linewidth=1,
            label="viscoelastic",
        )

        ux_max = max(
            np.abs(elastic_ux).max(),
            np.abs(visco_ux).max(),
            1e-12,
        )

        axes[0].set_ylim(-ux_max, ux_max)
        # axes[0].set_ylim(-1.3e-8, 1.3e-8)
        # axes[0].set_yticks(ticks=[-1.3e-8, -0.65e-8, 0, 0.65e-8, 1.3e-8])
        axes[0].set_xlabel("t, s")
        axes[0].set_ylabel("ux")
        axes[0].grid(True)
        axes[0].legend()

        # ---------- uy ----------
        axes[1].plot(
            elastic["T"],
            elastic_uy,
            "--",
            linewidth=1,
            label="elastic",
        )

        axes[1].plot(
            visco["T"],
            visco_uy,
            linewidth=1,
            label="viscoelastic",
        )

        uy_max = max(
            np.abs(elastic_uy).max(),
            np.abs(visco_uy).max(),
            1e-12,
        )

        axes[1].set_ylim(-uy_max, uy_max)
        # axes[1].set_ylim(-1.3e-8, 1.3e-8)
        # axes[1].set_yticks(ticks=[-1.3e-8, -0.65e-8, 0, 0.65e-8, 1.3e-8])
        axes[1].set_xlabel("t, s")
        axes[1].set_ylabel("uy")
        axes[1].grid(True)
        axes[1].legend()

        fig.suptitle(f"{station_name} displacement, h={h}")
        fig.tight_layout()

        outfile = OUTPUT_DIR / f"{station_name}-displacement-h_{h}.png"
        fig.savefig(outfile, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Сохранён {outfile}")



# --- сравнение решений с разным шагом ---
stations_rect = [
    "station_1",
    # "station_2",
    # "station_3",
    # "station_4",
]
for station_name in stations_rect:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5),
        sharex=True,
    )
    vx_max = 1e-16
    vy_max = 1e-16

    for h in cases_h:
        ELASTIC_DIR = RECT_SOL_DIR / f"elastic-h_{h}/result/txt"

        elastic_file = ELASTIC_DIR / f"{station_name}.txt"
        elastic = read_station(elastic_file)

        # ---------- vx ----------
        axes[0].plot(
            elastic["T"],
            elastic["vx"],
            "--",
            linewidth=1,
            label=f"h={h}",
        )
        vx_max = max(
            elastic["vx"].abs().max(),
            vx_max,
        )

        # ---------- vy ----------
        axes[1].plot(
            elastic["T"],
            elastic["vy"],
            "--",
            linewidth=1,
            label=f"h={h}",
        )
        vy_max = max(
            elastic["vy"].abs().max(),
            vy_max,
        )

    axes[0].set_ylim(-vx_max, vx_max)
    axes[0].set_xlabel("t, s")
    axes[0].set_ylabel("vx")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_ylim(-vy_max, vy_max)
    axes[1].set_xlabel("t, s")
    axes[1].set_ylabel("vy")
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle(f"elastic, {station_name}")
    fig.tight_layout()

    outfile = OUTPUT_DIR / f"elastic-{station_name}.png"
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Сохранён {outfile}")

for station_name in stations_rect:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(10, 5),
        sharex=True,
    )
    vx_max = 1e-16
    vy_max = 1e-16

    for h in cases_h:
        coef = 1/(rho*h**2)
        VISCOELASTIC_DIR = RECT_SOL_DIR / f"viscoelastic-h_{h}/result/txt"

        visco_file = VISCOELASTIC_DIR / f"{station_name}.txt"
        if not visco_file.exists():
            print(f"Файл отсутствует: {visco_file}")
            continue
        visco = read_station(visco_file)
        visco["vx"] *= coef
        visco["vy"] *= coef

        # ---------- vx ----------
        axes[0].plot(
            visco["T"],
            visco["vx"],
            linewidth=1,
            label=f"h={h}",
        )

        vx_max = max(
            visco["vx"].abs().max(),
            vx_max,
        )

        # ---------- vy ----------
        axes[1].plot(
            visco["T"],
            visco["vy"],
            linewidth=1,
            label=f"h={h}",
        )

        vy_max = max(
            visco["vy"].abs().max(),
            vy_max,
        )

    axes[0].set_ylim(-vx_max, vx_max)
    axes[0].set_xlabel("t, s")
    axes[0].set_ylabel("vx")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].set_ylim(-vy_max, vy_max)
    axes[1].set_xlabel("t, s")
    axes[1].set_ylabel("vy")
    axes[1].grid(True)
    axes[1].legend()

    fig.suptitle(f"viscoelastic, {station_name}")
    fig.tight_layout()

    outfile = OUTPUT_DIR / f"viscoelastic-{station_name}.png"
    fig.savefig(outfile, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Сохранён {outfile}")





# ----- сравнение численного и аналитического решений -----
print(f"\nСравнение численного и аналитического решений")

stations_an = {
    "station_1": "_500.0_500.0",
    "station_2": "_0.0_500.0",
    "station_3": "_500.0_0.0",
    "station_4": "_-500.0_500.0",
}

analytical = {}

for station, suffix in stations_an.items():
    vx = read_two_columns(
        ANALYTICAL_DIR /
        f"Vx_time_analytical_solution_viscoelastic{suffix}.dat"
    )
    vy = read_two_columns(
        ANALYTICAL_DIR /
        f"Vz_time_analytical_solution_viscoelastic{suffix}.dat"
    )
    analytical[(station, "vx")] = vx
    analytical[(station, "vy")] = vy


scale_factors_num = {}
scale_factors_an = {}
for h in cases_h:
    rect_dir = RECT_SOL_DIR / f"viscoelastic-h_{h}/result/txt"
    num = read_station(
        rect_dir / "station_3.txt"
    )
    an = analytical[("station_3", "vy")]
    amp_num = np.max(np.abs(num["vy"]))
    amp_an = np.max(np.abs(an["y"]))
    scale_factors_num[h] = amp_an / amp_num
    scale_factors_an[h] = 1/(rho*h**2)
    print(
        f"h={h}, "
        f"amp_num={amp_num:.4e}, "
        f"amp_an={amp_an:.4e}, "
        f"k_num={scale_factors_num[h]:.4e}, "
        f"k_an={scale_factors_an[h]:.4e}"
    )


error_norms = {}
for h in cases_h:
    rect_dir = RECT_SOL_DIR / f"viscoelastic-h_{h}/result/txt"
    # k = scale_factors_num[h]
    k = scale_factors_an[h]
    for station in stations_an.keys():
        num = read_station(rect_dir / f"{station}.txt")
        for component in ["vx", "vy"]:
            an = analytical[(station, component)]
            t_num = num["T"].to_numpy()
            u_num = (num[component].to_numpy()* k)
            t_an = an["x"].to_numpy()
            u_an = an["y"].to_numpy()
            # аналитика на сетке численного решения
            u_an_interp = np.interp(t_num, t_an, u_an,)
            err = u_num - u_an_interp

            l1 = np.mean(np.abs(err))
            l2 = np.sqrt(np.mean(err**2))
            linf = np.max(np.abs(err))

            error_norms.setdefault(
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
            axes.plot(
                t_num,
                u_an_interp,
                "--",
                label="analytical",
            )
            axes.plot(
                t_num,
                err*10,
                ":",
                label="error*10"
            )
            axes.grid(True)
            axes.legend()
            axes.set_ylabel(component)
            axes.set_xlabel("t, s")

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
for station, component in error_norms:
    rows = error_norms[(station, component)]
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

    outfile = OUTPUT_DIR / f"{station}-{component}-error-norms.png"
    plt.savefig(
        outfile,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()
    print(f"Сохранён {outfile}")


exit()













# for filename in analytical_files:
#     filepath = ANALYTICAL_DIR / filename

#     if not filepath.exists():
#         print(f"Файл отсутствует: {filepath}")
#         continue

#     data = read_two_columns(filepath)

#     plt.figure(figsize=(8, 5))

#     plt.plot(
#         data["x"],
#         data["y"],
#         linewidth=1,
#     )

#     # ymax = max(np.abs(data["y"]).max(), 1e-12)
#     # plt.ylim(-ymax, ymax)
#     if "spectrum" in filename:
#         plt.ylim(0, data["y"].max() * 1.05)
#     else:
#         ymax = max(np.abs(data["y"]).max(), 1e-12)
#         plt.ylim(-ymax, ymax)

#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.grid(True)

#     plt.title(filepath.stem)

#     outfile = OUTPUT_DIR / f"{filepath.stem}.png"
#     plt.savefig(outfile, dpi=300, bbox_inches="tight")
#     plt.close()

#     print(f"Сохранён {outfile}")




visco_file = VISCO_DIR / "station_1.txt"
visco = read_station(visco_file)

filepath = ANALYTICAL_DIR / "Vx_time_analytical_solution_viscoelastic.dat"
data = read_two_columns(filepath)

index_max_rect = np.argmax(visco["vx"].to_numpy())
index_max_an = np.argmax(data["y"].to_numpy())
coef = visco["vx"][index_max_rect] / data["y"][index_max_an]
print(f"{coef=}")
shift_t = visco["T"][index_max_rect] - data["x"][index_max_an]
shift_t = 0
print(shift_t)

analytical_sol_t = data["x"]+shift_t
analytical_sol = data["y"]*coef

from scipy.interpolate import interp1d
an_sol = interp1d(analytical_sol_t, analytical_sol, bounds_error=False, fill_value=0)
an_to_rect_time = an_sol(visco["T"])
delta_sol = visco["vx"] - an_to_rect_time
index = np.argmax(np.abs(delta_sol))
print(visco["vx"][index], an_to_rect_time[index], delta_sol[index], visco["vx"][index]-an_to_rect_time[index])
print(f"max_error = {delta_sol[index]/np.max(np.abs(an_to_rect_time))}%")

plt.figure(figsize=(8, 5))
plt.plot(
    visco["T"],
    visco["vx"],
    linewidth=1,
    # label="viscoelastic",
    label="rect"
)

# plt.plot(
#     data["x"]+shift_t,
#     data["y"]*coef,
#     linewidth=1,
#     linestyle="--",
#     label=f"analytical\nshift_t={shift_t}\ncoef={coef:.1e}"
# )

plt.plot(
    visco["T"],
    an_to_rect_time,
    linewidth=1,
    linestyle="--",
    label=f"an_to_rect_time"
)

plt.plot(
    visco["T"],
    delta_sol*10,
    linewidth=1,
    linestyle="--",
    label=f"delta_sol*10"
)

# ymax = max(np.abs(data["y"]).max(), 1e-12)
# plt.ylim(-ymax, ymax)
# if "spectrum" in filename:
#     plt.ylim(0, data["y"].max() * 1.05)
# else:
#     ymax = max(np.abs(data["y"]).max(), 1e-12)
#     plt.ylim(-ymax, ymax)

plt.xlabel("t, s")
plt.ylabel("vx")
plt.grid(True)
plt.legend()

plt.title("viscoelastic")

outfile = OUTPUT_DIR / f"visco_vx.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close()

print(f"Сохранён {outfile}")






visco_file = VISCO_DIR2 / "station_1.txt"
visco = read_station(visco_file)

filepath = ANALYTICAL_DIR / "Vx_time_analytical_solution_viscoelastic.dat"
data = read_two_columns(filepath)

index_max_rect = np.argmax(visco["vx"].to_numpy())
index_max_an = np.argmax(data["y"].to_numpy())
coef = visco["vx"][index_max_rect] / data["y"][index_max_an]
print(f"{coef=}")
shift_t = visco["T"][index_max_rect] - data["x"][index_max_an]
shift_t = 0
print(shift_t)

analytical_sol_t = data["x"]+shift_t
analytical_sol = data["y"]*coef

from scipy.interpolate import interp1d
an_sol = interp1d(analytical_sol_t, analytical_sol, bounds_error=False, fill_value=0)
an_to_rect_time = an_sol(visco["T"])
delta_sol = visco["vx"] - an_to_rect_time
index = np.argmax(np.abs(delta_sol))
print(visco["vx"][index], an_to_rect_time[index], delta_sol[index], visco["vx"][index]-an_to_rect_time[index])
print(f"max_error = {delta_sol[index]/np.max(np.abs(an_to_rect_time))}%")

plt.figure(figsize=(8, 5))
plt.plot(
    visco["T"],
    visco["vx"],
    linewidth=1,
    # label="viscoelastic",
    label="rect"
)

# plt.plot(
#     data["x"]+shift_t,
#     data["y"]*coef,
#     linewidth=1,
#     linestyle="--",
#     label=f"analytical\nshift_t={shift_t}\ncoef={coef:.1e}"
# )

plt.plot(
    visco["T"],
    an_to_rect_time,
    linewidth=1,
    linestyle="--",
    label=f"an_to_rect_time"
)

plt.plot(
    visco["T"],
    delta_sol*10,
    linewidth=1,
    linestyle="--",
    label=f"delta_sol*10"
)

# ymax = max(np.abs(data["y"]).max(), 1e-12)
# plt.ylim(-ymax, ymax)
# if "spectrum" in filename:
#     plt.ylim(0, data["y"].max() * 1.05)
# else:
#     ymax = max(np.abs(data["y"]).max(), 1e-12)
#     plt.ylim(-ymax, ymax)

plt.xlabel("t, s")
plt.ylabel("vx")
plt.grid(True)
plt.legend()

plt.title("viscoelastic")

outfile = OUTPUT_DIR / f"visco_vx-2.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close()

print(f"Сохранён {outfile}")








visco_file = VISCO_DIR / "station_1.txt"
visco = read_station(visco_file)

filepath = ANALYTICAL_DIR / "Vz_time_analytical_solution_viscoelastic.dat"
data = read_two_columns(filepath)

index_max_rect = np.argmax(visco["vy"].to_numpy())
index_max_an = np.argmax(data["y"].to_numpy())
coef = visco["vy"][index_max_rect] / data["y"][index_max_an]
print(f"{coef=}")
shift_t = visco["T"][index_max_rect] - data["x"][index_max_an]
shift_t = 0
print(shift_t)

analytical_sol_t = data["x"]+shift_t
analytical_sol = data["y"]*coef

from scipy.interpolate import interp1d
an_sol = interp1d(analytical_sol_t, analytical_sol, bounds_error=False, fill_value=0)
an_to_rect_time = an_sol(visco["T"])
delta_sol = visco["vy"] - an_to_rect_time
index = np.argmax(np.abs(delta_sol))
print(visco["vy"][index], an_to_rect_time[index], delta_sol[index], visco["vy"][index]-an_to_rect_time[index])
print(f"max_error = {delta_sol[index]/np.max(np.abs(an_to_rect_time))}%")

plt.figure(figsize=(8, 5))
plt.plot(
    visco["T"],
    visco["vy"],
    linewidth=1,
    # label="viscoelastic",
    label="rect"
)

# plt.plot(
#     data["x"]+shift_t,
#     data["y"]*coef,
#     linewidth=1,
#     linestyle="--",
#     label=f"analytical\nshift_t={shift_t}\ncoef={coef:.1e}"
# )

plt.plot(
    visco["T"],
    an_to_rect_time,
    linewidth=1,
    linestyle="--",
    label=f"an_to_rect_time"
)

plt.plot(
    visco["T"],
    delta_sol*10,
    linewidth=1,
    linestyle="--",
    label=f"delta_sol*10"
)

# ymax = max(np.abs(data["y"]).max(), 1e-12)
# plt.ylim(-ymax, ymax)
# if "spectrum" in filename:
#     plt.ylim(0, data["y"].max() * 1.05)
# else:
#     ymax = max(np.abs(data["y"]).max(), 1e-12)
#     plt.ylim(-ymax, ymax)

plt.xlabel("t, s")
plt.ylabel("vy")
plt.grid(True)
plt.legend()

plt.title("viscoelastic")

outfile = OUTPUT_DIR / f"visco_vy.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close()

print(f"Сохранён {outfile}")






visco_file = VISCO_DIR2 / "station_1.txt"
visco = read_station(visco_file)

filepath = ANALYTICAL_DIR / "Vz_time_analytical_solution_viscoelastic.dat"
data = read_two_columns(filepath)

index_max_rect = np.argmax(visco["vy"].to_numpy())
index_max_an = np.argmax(data["y"].to_numpy())
coef = visco["vy"][index_max_rect] / data["y"][index_max_an]
print(f"{coef=}")
shift_t = visco["T"][index_max_rect] - data["x"][index_max_an]
shift_t = 0
print(shift_t)

analytical_sol_t = data["x"]+shift_t
analytical_sol = data["y"]*coef

from scipy.interpolate import interp1d
an_sol = interp1d(analytical_sol_t, analytical_sol, bounds_error=False, fill_value=0)
an_to_rect_time = an_sol(visco["T"])
delta_sol = visco["vy"] - an_to_rect_time
index = np.argmax(np.abs(delta_sol))
print(visco["vy"][index], an_to_rect_time[index], delta_sol[index], visco["vy"][index]-an_to_rect_time[index])
print(f"max_error = {delta_sol[index]/np.max(np.abs(an_to_rect_time))}%")

plt.figure(figsize=(8, 5))
plt.plot(
    visco["T"],
    visco["vy"],
    linewidth=1,
    # label="viscoelastic",
    label="rect"
)

# plt.plot(
#     data["x"]+shift_t,
#     data["y"]*coef,
#     linewidth=1,
#     linestyle="--",
#     label=f"analytical\nshift_t={shift_t}\ncoef={coef:.1e}"
# )

plt.plot(
    visco["T"],
    an_to_rect_time,
    linewidth=1,
    linestyle="--",
    label=f"an_to_rect_time"
)

plt.plot(
    visco["T"],
    delta_sol*10,
    linewidth=1,
    linestyle="--",
    label=f"delta_sol*10"
)

# ymax = max(np.abs(data["y"]).max(), 1e-12)
# plt.ylim(-ymax, ymax)
# if "spectrum" in filename:
#     plt.ylim(0, data["y"].max() * 1.05)
# else:
#     ymax = max(np.abs(data["y"]).max(), 1e-12)
#     plt.ylim(-ymax, ymax)

plt.xlabel("t, s")
plt.ylabel("vy")
plt.grid(True)
plt.legend()

plt.title("viscoelastic")

outfile = OUTPUT_DIR / f"visco_vy-2.png"
plt.savefig(outfile, dpi=300, bbox_inches="tight")
plt.close()

print(f"Сохранён {outfile}")




# h = 2
# rho = 2000
# c = 3190.226435988779
# magic = c**2 * rho / h**2 * np.pi / 41064586571.27474
# print(f'magic = {magic}')
# scale = 10e13 
# rect_scale = scale * magic
# specfem_scale = scale

# magic = c**2 * rho / h**2 * np.pi
# print(f'magic = {magic}')


