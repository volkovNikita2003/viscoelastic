#!/usr/bin/env python3
"""Plot RECT and SPECFEM2D solutions and their convergence graphs."""

from pathlib import Path
import csv
import re

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
ANALYTICAL_DIR = BASE_DIR / "analytical_sol"
RECT_SOL_DIR = BASE_DIR / "rect_sol"
SPECFEM_SOL_DIR = BASE_DIR / "specfem_sol"
OUTPUT_DIR = BASE_DIR / "graphs"

F0 = 18.0
TIME_SHIFT = 1.2 / F0
TIME_LIMITS = (0.0, 0.6)

# Station numbers are identical in all three solutions:
# analytical ``*_station_N.dat``, RECT ``station_N.txt``, and
# SPECFEM2D ``AA.S000N.*.semd``.
STATIONS = {
    f"station_{number}": {
        "analytical_station": number,
        "specfem_station": f"S{number:04d}",
    }
    for number in range(1, 9)
}

COMPONENTS = {
    "ux": {"analytical_component": "Ux", "specfem_component": "BXX"},
    "uy": {"analytical_component": "Uz", "specfem_component": "BXZ"},
}

SPECFEM_PROJECT_PREFIX = (
    "check_absolute_amplitude_of_force_source_seismograms_viscoelastic_auto_nx_"
)
RECT_PROJECT_PREFIX = "viscoelastic_schema_nx_"
RECT_ELASTIC_PROJECT_PREFIX = "elastic_nx_"
TIMING_SUFFIX = "_timing"
FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"


def read_two_column_file(filename: Path) -> tuple[np.ndarray, np.ndarray]:
    values = np.loadtxt(filename, ndmin=2)
    if values.shape[1] < 2:
        raise ValueError(f"В файле {filename} ожидается не менее двух столбцов")
    return values[:, 0], values[:, 1]


def read_station_rect(filename: Path) -> dict[str, np.ndarray]:
    """Read RECT velocities and integrate them to displacements."""
    values = np.loadtxt(filename, comments="#", ndmin=2)
    if values.shape[1] < 3:
        raise ValueError(f"В файле {filename} ожидается не менее трёх столбцов")

    time = values[:, 0]
    dt = np.diff(time)
    data = {"T": time}
    for velocity, displacement in ((values[:, 1], "ux"), (values[:, 2], "uy")):
        integrated = np.zeros_like(time, dtype=float)
        integrated[1:] = np.cumsum(
            0.5 * (velocity[:-1] + velocity[1:]) * dt
        )
        data[displacement] = integrated
    return data


def discover_rect_cases() -> list[tuple[int, Path]]:
    cases = []
    for path in RECT_SOL_DIR.glob(f"{RECT_PROJECT_PREFIX}*"):
        if not path.is_dir():
            continue
        try:
            nx = int(path.name.removeprefix(RECT_PROJECT_PREFIX))
        except ValueError:
            # Timing projects end with "_timing" and have no saved solution.
            continue
        cases.append((nx, path))
    return sorted(cases, key=lambda case: case[0])


def discover_specfem_cases() -> list[tuple[int, Path]]:
    cases = []
    for path in SPECFEM_SOL_DIR.glob(f"{SPECFEM_PROJECT_PREFIX}*"):
        if not path.is_dir():
            continue
        try:
            nx = int(path.name.removeprefix(SPECFEM_PROJECT_PREFIX))
        except ValueError:
            continue
        cases.append((nx, path))
    return sorted(cases, key=lambda case: case[0])


def read_timing_file(filename: Path) -> dict[str, float]:
    values = {}
    for line in filename.read_text(encoding="utf-8").splitlines():
        if "=" not in line:
            continue
        name, value = line.split("=", maxsplit=1)
        values[name.strip()] = float(value.strip())

    required = {"real_s", "user_s", "sys_s"}
    missing = required - values.keys()
    if missing:
        raise ValueError(f"В {filename} отсутствуют значения: {sorted(missing)}")
    return {name: values[name] for name in ("real_s", "user_s", "sys_s")}


def read_main_loop_time(filename: Path, solver: str) -> float:
    contents = filename.read_text(encoding="utf-8")
    if solver == "rect":
        pattern = rf"^Solver::step\s+\d+\s+({FLOAT_PATTERN})"
    elif solver == "specfem":
        pattern = (
            rf"date and time of the system\s*:\s*({FLOAT_PATTERN})\s+s"
        )
    else:
        raise ValueError(f"Неизвестная программа для чтения времени: {solver}")

    matches = re.findall(pattern, contents, flags=re.MULTILINE | re.IGNORECASE)
    if len(matches) != 1:
        raise ValueError(
            f"В {filename} найдено значений времени основного цикла: "
            f"{len(matches)}, ожидалось одно"
        )
    return float(matches[0])


def discover_timing_cases(
    directory: Path, prefix: str, solver: str
) -> list[dict[str, float]]:
    rows = []
    for path in directory.glob(f"{prefix}*{TIMING_SUFFIX}"):
        if not path.is_dir():
            continue
        nx_text = path.name.removeprefix(prefix).removesuffix(TIMING_SUFFIX)
        try:
            nx = int(nx_text)
        except ValueError:
            continue
        timing_file = path / "timing.txt"
        if not timing_file.is_file():
            print(f"Пропущен проект без timing.txt: {path}")
            continue
        log_file = path / "log_run.txt"
        if not log_file.is_file():
            raise FileNotFoundError(f"Не найден журнал расчёта: {log_file}")
        rows.append(
            {
                "nx": nx,
                **read_timing_file(timing_file),
                "main_loop_s": read_main_loop_time(log_file, solver),
            }
        )
    return sorted(rows, key=lambda row: row["nx"])


def save_timing_table(rows: list[dict[str, float]], filename: Path) -> None:
    with filename.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["nx", "real_s", "user_s", "sys_s", "main_loop_s"],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Сохранена таблица {filename}")


def load_analytical_solutions(
    station: str, component: str
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    station_config = STATIONS[station]
    component_name = COMPONENTS[component]["analytical_component"]
    analytical_station = station_config["analytical_station"]

    viscoelastic_file = (
        ANALYTICAL_DIR
        / (
            f"{component_name}_time_analytical_solution_viscoelastic_"
            f"station_{analytical_station}.dat"
        )
    )
    elastic_file = (
        ANALYTICAL_DIR
        / (
            f"{component_name}_time_analytical_solution_elastic_"
            f"station_{analytical_station}.dat"
        )
    )
    time_viscoelastic, value_viscoelastic = read_two_column_file(viscoelastic_file)
    time_elastic, value_elastic = read_two_column_file(elastic_file)
    return (
        (time_viscoelastic + TIME_SHIFT, value_viscoelastic),
        (time_elastic + TIME_SHIFT, value_elastic),
    )


def calculate_error_norms(
    numerical_time: np.ndarray,
    numerical_value: np.ndarray,
    analytical_time: np.ndarray,
    analytical_value: np.ndarray,
) -> dict[str, float]:
    """Calculate errors inside TIME_LIMITS and the analytical time range."""
    time_min = max(analytical_time.min(), TIME_LIMITS[0])
    time_max = min(analytical_time.max(), TIME_LIMITS[1])
    common = (
        (numerical_time >= time_min)
        & (numerical_time <= time_max)
    )
    if not np.any(common):
        raise ValueError("Временные диапазоны решений не пересекаются")

    numerical_common = numerical_value[common]
    analytical_interpolated = np.interp(
        numerical_time[common], analytical_time, analytical_value
    )
    error = numerical_common - analytical_interpolated
    return {
        "L1": float(np.mean(np.abs(error))),
        "L2": float(np.sqrt(np.mean(error**2))),
        "Linf": float(np.max(np.abs(error))),
    }


def plot_solution(
    numerical_time: np.ndarray,
    numerical_value: np.ndarray,
    analytical_viscoelastic: tuple[np.ndarray, np.ndarray],
    analytical_elastic: tuple[np.ndarray, np.ndarray],
    numerical_label: str,
    title: str,
    filename: Path,
) -> None:
    analytical_time, analytical_value = analytical_viscoelastic
    common = (
        (numerical_time >= analytical_time.min())
        & (numerical_time <= analytical_time.max())
    )
    if not np.any(common):
        raise ValueError("Временные диапазоны решений не пересекаются")

    difference_time = numerical_time[common]
    analytical_interpolated = np.interp(
        difference_time, analytical_time, analytical_value
    )
    difference = numerical_value[common] - analytical_interpolated

    fig, (solution_axis, difference_axis) = plt.subplots(
        2, 1, figsize=(8, 10), sharex=True
    )
    solution_axis.plot(numerical_time, numerical_value, label=numerical_label, zorder=4)
    solution_axis.plot(
        *analytical_viscoelastic, "--", label="quasi-analytical viscoelastic", zorder=5
    )
    solution_axis.plot(
        *analytical_elastic, "-.", label="quasi-analytical elastic", zorder=3
    )
    solution_axis.axhline(0.0, color="black", linewidth=0.8, zorder=2)
    solution_axis.set_xlabel("t, s")
    solution_axis.set_ylabel("displacement")
    solution_axis.tick_params(axis="x", labelbottom=True)
    solution_axis.grid(True)
    solution_axis.legend()
    solution_axis.set_title(title)

    difference_axis.plot(
        difference_time,
        difference,
        zorder=3,
        label="numerical − quasi-analytical viscoelastic",
    )
    difference_axis.axhline(0.0, color="black", linewidth=0.8, zorder=2)
    difference_axis.set_xlabel("t, s")
    difference_axis.set_ylabel("displacement difference")
    difference_axis.set_xlim(*TIME_LIMITS)
    difference_axis.grid(True)
    difference_axis.legend()
    difference_axis.set_title("Difference")

    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранён {filename}")


def convergence_fit(
    x_values: np.ndarray, errors: np.ndarray
) -> tuple[float, float] | None:
    valid = (x_values > 0.0) & (errors > 0.0) & np.isfinite(errors)
    if np.count_nonzero(valid) < 2:
        return None
    slope, intercept = np.polyfit(
        np.log(x_values[valid]), np.log(errors[valid]), 1
    )
    return float(slope), float(intercept)


def plot_error_graph(
    rows: list[dict[str, float]],
    parameter: str,
    station: str,
    component: str,
    solver: str,
) -> None:
    rows = sorted(rows, key=lambda row: row[parameter])
    x_values = np.array([row[parameter] for row in rows], dtype=float)

    fig, axis = plt.subplots(figsize=(7, 5))
    styles = {"L1": "o-", "L2": "s-", "Linf": "^-"}
    for norm, style in styles.items():
        errors = np.array([row[norm] for row in rows], dtype=float)
        fit = convergence_fit(x_values, errors)
        label = norm if fit is None else f"{norm} (slope={fit[0]:.2f})"
        data_line = axis.loglog(x_values, errors, style, label=label)[0]
        if fit is not None:
            slope, intercept = fit
            fitted_errors = np.exp(intercept) * x_values**slope
            axis.loglog(
                x_values,
                fitted_errors,
                "--",
                color=data_line.get_color(),
                linewidth=1.2,
                label="_nolegend_",
            )
            print(f"{solver} {station} {component}: slope({norm})={slope:.3f}")

    axis.set_xlabel(parameter)
    axis.set_ylabel("error norm")
    axis.grid(True, which="both")
    axis.legend()
    axis.set_title(f"{solver}: {station}, {component}")
    fig.tight_layout()
    filename = OUTPUT_DIR / f"{solver}-{station}-{component}-error-vs-{parameter}.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранён {filename}")


def save_error_table(
    errors_by_station: dict[tuple[str, str], list[dict[str, float]]],
    filename: Path,
) -> None:
    fieldnames = [
        "station",
        "component",
        "nx",
        "L1",
        "L2",
        "Linf",
        "L1_slope",
        "L2_slope",
        "Linf_slope",
    ]
    table_rows = []
    for (station, component), rows in sorted(errors_by_station.items()):
        rows = sorted(rows, key=lambda row: row["nx"])
        nx = np.array([row["nx"] for row in rows], dtype=float)
        slopes = {}
        for norm in ("L1", "L2", "Linf"):
            values = np.array([row[norm] for row in rows], dtype=float)
            fit = convergence_fit(nx, values)
            slopes[norm] = "" if fit is None else fit[0]

        for row in rows:
            table_rows.append(
                {
                    "station": station,
                    "component": component,
                    "nx": row["nx"],
                    "L1": row["L1"],
                    "L2": row["L2"],
                    "Linf": row["Linf"],
                    "L1_slope": slopes["L1"],
                    "L2_slope": slopes["L2"],
                    "Linf_slope": slopes["Linf"],
                }
            )

    with filename.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"Сохранена таблица {filename}")


def plot_calculation_time_vs_nx(
    series: list[tuple[str, list[dict[str, float]], str]],
    title: str,
    filename: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(7, 5))
    for label, rows, style in series:
        if not rows:
            continue
        nx = np.array([row["nx"] for row in rows], dtype=float)
        real_time = np.array([row["real_s"] for row in rows], dtype=float)
        axis.loglog(nx, real_time, style, label=label)

    axis.set_xlabel("nx")
    axis.set_ylabel("calculation time, s")
    axis.grid(True, which="both")
    axis.legend()
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранён {filename}")


def plot_error_vs_calculation_time(
    rect_errors: dict[tuple[str, str], list[dict[str, float]]],
    specfem_errors: dict[tuple[str, str], list[dict[str, float]]],
    rect_timings: list[dict[str, float]],
    specfem_timings: list[dict[str, float]],
) -> None:
    timing_maps = {
        "RECT viscoelastic": {row["nx"]: row["real_s"] for row in rect_timings},
        "SPECFEM2D": {row["nx"]: row["real_s"] for row in specfem_timings},
    }
    error_sets = {
        "RECT viscoelastic": rect_errors,
        "SPECFEM2D": specfem_errors,
    }
    colors = {"L1": "tab:blue", "L2": "tab:orange", "Linf": "tab:green"}
    solver_styles = {"RECT viscoelastic": "-", "SPECFEM2D": "--"}
    markers = {"L1": "o", "L2": "s", "Linf": "^"}

    keys = sorted(set(rect_errors) | set(specfem_errors))
    for station, component in keys:
        fig, axis = plt.subplots(figsize=(8, 6))
        plotted = False
        for solver, solver_errors in error_sets.items():
            rows = solver_errors.get((station, component), [])
            timing_map = timing_maps[solver]
            matched = sorted(
                (
                    (timing_map[row["nx"]], row)
                    for row in rows
                    if row["nx"] in timing_map
                ),
                key=lambda item: item[0],
            )
            if not matched:
                continue
            calculation_times = np.array([item[0] for item in matched])
            for norm in ("L1", "L2", "Linf"):
                errors = np.array([item[1][norm] for item in matched])
                axis.loglog(
                    calculation_times,
                    errors,
                    color=colors[norm],
                    linestyle=solver_styles[solver],
                    marker=markers[norm],
                    label=f"{solver} {norm}",
                )
                plotted = True

        if not plotted:
            plt.close(fig)
            continue
        axis.set_xlabel("calculation time, s")
        axis.set_ylabel("error norm")
        axis.grid(True, which="both")
        axis.legend()
        axis.set_title(f"Error vs calculation time: {station}, {component}")
        fig.tight_layout()
        filename = OUTPUT_DIR / (
            f"specfem-rect-{station}-{component}-error-vs-calculation-time.png"
        )
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Сохранён {filename}")


def process_rect() -> dict[tuple[str, str], list[dict[str, float]]]:
    cases = discover_rect_cases()
    if not cases:
        raise FileNotFoundError(f"Не найдены расчёты RECT в {RECT_SOL_DIR}")

    errors: dict[tuple[str, str], list[dict[str, float]]] = {}
    for nx, case_dir in cases:
        for station in STATIONS:
            station_file = case_dir / "result" / "txt" / f"{station}.txt"
            numerical = read_station_rect(station_file)
            numerical_time = numerical["T"]

            for component in COMPONENTS:
                numerical_value = numerical[component]
                analytical_viscoelastic, analytical_elastic = (
                    load_analytical_solutions(station, component)
                )
                filename = OUTPUT_DIR / (
                    f"rect-{station}-{component}-nx_{nx}-solution.png"
                )
                plot_solution(
                    numerical_time,
                    numerical_value,
                    analytical_viscoelastic,
                    analytical_elastic,
                    f"RECT nx={nx}",
                    f"RECT: {station}, {component}, nx={nx}",
                    filename,
                )
                norms = calculate_error_norms(
                    numerical_time,
                    numerical_value,
                    *analytical_viscoelastic,
                )
                errors.setdefault((station, component), []).append(
                    {"nx": nx, **norms}
                )

    for (station, component), rows in errors.items():
        plot_error_graph(rows, "nx", station, component, "rect")
    return errors


def process_specfem() -> dict[tuple[str, str], list[dict[str, float]]]:
    cases = discover_specfem_cases()
    if not cases:
        raise FileNotFoundError(f"Не найдены расчёты SPECFEM2D в {SPECFEM_SOL_DIR}")

    errors: dict[tuple[str, str], list[dict[str, float]]] = {}
    for nx, case_dir in cases:
        for station, station_config in STATIONS.items():
            station_code = station_config["specfem_station"]
            for component, component_config in COMPONENTS.items():
                trace_file = case_dir / "OUTPUT_FILES" / (
                    f"AA.{station_code}.{component_config['specfem_component']}.semd"
                )
                numerical_time, numerical_value = read_two_column_file(trace_file)
                numerical_time = numerical_time + TIME_SHIFT
                analytical_viscoelastic, analytical_elastic = (
                    load_analytical_solutions(station, component)
                )
                filename = OUTPUT_DIR / (
                    f"specfem-{station}-{component}-nx_{nx}-solution.png"
                )
                plot_solution(
                    numerical_time,
                    numerical_value,
                    analytical_viscoelastic,
                    analytical_elastic,
                    f"SPECFEM2D nx={nx}",
                    f"SPECFEM2D: {station}, {component}, nx={nx}",
                    filename,
                )
                norms = calculate_error_norms(
                    numerical_time,
                    numerical_value,
                    *analytical_viscoelastic,
                )
                errors.setdefault((station, component), []).append(
                    {"nx": nx, **norms}
                )

    for (station, component), rows in errors.items():
        plot_error_graph(rows, "nx", station, component, "specfem")
    return errors


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    rect_errors = process_rect()
    specfem_errors = process_specfem()

    save_error_table(rect_errors, OUTPUT_DIR / "rect-errors-vs-nx.csv")
    save_error_table(specfem_errors, OUTPUT_DIR / "specfem-errors-vs-nx.csv")

    specfem_timings = discover_timing_cases(
        SPECFEM_SOL_DIR, SPECFEM_PROJECT_PREFIX, "specfem"
    )
    rect_viscoelastic_timings = discover_timing_cases(
        RECT_SOL_DIR, RECT_PROJECT_PREFIX, "rect"
    )
    rect_elastic_timings = discover_timing_cases(
        RECT_SOL_DIR, RECT_ELASTIC_PROJECT_PREFIX, "rect"
    )

    save_timing_table(specfem_timings, OUTPUT_DIR / "specfem-timings.csv")
    save_timing_table(
        rect_viscoelastic_timings,
        OUTPUT_DIR / "rect-viscoelastic-timings.csv",
    )
    save_timing_table(
        rect_elastic_timings, OUTPUT_DIR / "rect-elastic-timings.csv"
    )

    plot_calculation_time_vs_nx(
        [
            ("SPECFEM2D", specfem_timings, "o-"),
            ("RECT viscoelastic", rect_viscoelastic_timings, "s-"),
        ],
        "Calculation time: SPECFEM2D and RECT viscoelastic",
        OUTPUT_DIR / "specfem-rect-viscoelastic-time-vs-nx.png",
    )
    plot_calculation_time_vs_nx(
        [
            ("RECT viscoelastic", rect_viscoelastic_timings, "o-"),
            ("RECT elastic", rect_elastic_timings, "s-"),
        ],
        "RECT calculation time",
        OUTPUT_DIR / "rect-viscoelastic-elastic-time-vs-nx.png",
    )
    plot_error_vs_calculation_time(
        rect_errors,
        specfem_errors,
        rect_viscoelastic_timings,
        specfem_timings,
    )


if __name__ == "__main__":
    main()
