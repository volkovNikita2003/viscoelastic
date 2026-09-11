#!/usr/bin/env python3
"""Analyze RECT and SPECFEM2D velocity solutions for the half-space test."""

from pathlib import Path
import csv
import re

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = Path(__file__).resolve().parent
RECT_SOL_DIR = BASE_DIR / "rect_sol"
SPECFEM_SOL_DIR = BASE_DIR / "specfem_sol"
OUTPUT_DIR = BASE_DIR / "results"

F0 = 18.0
SPECFEM_TIME_SHIFT = 1.2 / F0
TIME_LIMITS = (0.0, 1.2)

STATIONS = {
    f"station_{number}": f"S{number:04d}"
    for number in range(1, 11)
}

# RECT calls the vertical coordinate y, while SPECFEM2D calls it z.  Use the
# RECT component name ``vy`` in plots and tables.
COMPONENTS = {
    "vx": {"rect_column": 1, "specfem_component": "BXX"},
    "vy": {"rect_column": 2, "specfem_component": "BXZ"},
}

SPECFEM_PROJECT_PREFIX = (
    "check_absolute_amplitude_of_force_source_seismograms_viscoelastic_auto_nx_"
)
RECT_PROJECT_PREFIX = "viscoelastic_schema_nx_"
TIMING_SUFFIX = "_timing"
FLOAT_PATTERN = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[Ee][+-]?\d+)?"
NORMS = ("L1", "L2", "Linf")

Trace = tuple[np.ndarray, np.ndarray]
SolutionRow = dict[str, int | np.ndarray]
Solutions = dict[tuple[str, str], list[SolutionRow]]


def read_two_column_file(filename: Path) -> Trace:
    values = np.loadtxt(filename, ndmin=2)
    if values.shape[1] < 2:
        raise ValueError(f"В файле {filename} ожидается не менее двух столбцов")
    return values[:, 0], values[:, 1]


def read_station_rect(filename: Path) -> dict[str, np.ndarray]:
    """Read time and velocity components saved by RECT."""
    values = np.loadtxt(filename, comments="#", ndmin=2)
    if values.shape[1] < 3:
        raise ValueError(f"В файле {filename} ожидается не менее трёх столбцов")
    return {
        "time": values[:, 0],
        **{
            component: values[:, config["rect_column"]]
            for component, config in COMPONENTS.items()
        },
    }


def discover_cases(directory: Path, prefix: str) -> list[tuple[int, Path]]:
    cases = []
    for path in directory.glob(f"{prefix}*"):
        if not path.is_dir():
            continue
        nx_text = path.name.removeprefix(prefix)
        try:
            nx = int(nx_text)
        except ValueError:
            # Timing projects end with "_timing" and contain no traces.
            continue
        cases.append((nx, path))
    return sorted(cases)


def load_rect_solutions() -> Solutions:
    cases = discover_cases(RECT_SOL_DIR, RECT_PROJECT_PREFIX)
    if not cases:
        raise FileNotFoundError(f"Не найдены расчёты RECT в {RECT_SOL_DIR}")

    solutions: Solutions = {}
    for nx, case_dir in cases:
        for station in STATIONS:
            filename = case_dir / "result" / "txt" / f"{station}.txt"
            trace = read_station_rect(filename)
            for component in COMPONENTS:
                solutions.setdefault((station, component), []).append(
                    {"nx": nx, "time": trace["time"], "value": trace[component]}
                )
    return solutions


def load_specfem_solutions() -> Solutions:
    cases = discover_cases(SPECFEM_SOL_DIR, SPECFEM_PROJECT_PREFIX)
    if not cases:
        raise FileNotFoundError(
            f"Не найдены расчёты SPECFEM2D в {SPECFEM_SOL_DIR}"
        )

    solutions: Solutions = {}
    for nx, case_dir in cases:
        for station, station_code in STATIONS.items():
            for component, config in COMPONENTS.items():
                filename = case_dir / "OUTPUT_FILES" / (
                    f"AA.{station_code}.{config['specfem_component']}.semv"
                )
                time, value = read_two_column_file(filename)
                # SPECFEM2D starts a Ricker trace at -1.2/f0, whereas RECT
                # starts at zero and places the Ricker maximum at +1.2/f0.
                time = time + SPECFEM_TIME_SHIFT
                solutions.setdefault((station, component), []).append(
                    {"nx": nx, "time": time, "value": value}
                )
    return solutions


def common_interval(source_time: np.ndarray, reference_time: np.ndarray) -> tuple[float, float]:
    time_min = max(TIME_LIMITS[0], source_time.min(), reference_time.min())
    time_max = min(TIME_LIMITS[1], source_time.max(), reference_time.max())
    if time_min >= time_max:
        raise ValueError("Временные диапазоны решений не пересекаются")
    return float(time_min), float(time_max)


def difference_on_source_grid(source: Trace, reference: Trace) -> Trace:
    source_time, source_value = source
    reference_time, reference_value = reference
    time_min, time_max = common_interval(source_time, reference_time)
    mask = (source_time >= time_min) & (source_time <= time_max)
    time = source_time[mask]
    reference_interpolated = np.interp(
        time, reference_time, reference_value
    )
    return time, source_value[mask] - reference_interpolated


def calculate_error_norms(source: Trace, reference: Trace) -> dict[str, float]:
    _, error = difference_on_source_grid(source, reference)
    return {
        "L1": float(np.mean(np.abs(error))),
        "L2": float(np.sqrt(np.mean(error**2))),
        "Linf": float(np.max(np.abs(error))),
    }


def plot_velocity_series(
    solutions: Solutions,
    solver: str,
    filename_prefix: str,
) -> None:
    for (station, component), rows in sorted(solutions.items()):
        fig, axis = plt.subplots(figsize=(8, 5))
        for row in sorted(rows, key=lambda item: int(item["nx"])):
            axis.plot(
                row["time"],
                row["value"],
                label=f"nx={row['nx']}",
            )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_xlim(*TIME_LIMITS)
        axis.set_xlabel("t, s")
        axis.set_ylabel(f"{component}, m/s")
        axis.set_title(f"{solver}: {station}, {component}")
        axis.grid(True)
        axis.legend()
        fig.tight_layout()
        filename = OUTPUT_DIR / (
            f"{filename_prefix}-{station}-{component}-velocities.png"
        )
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Сохранён {filename}")


def plot_finest_method_comparison(
    rect_solutions: Solutions,
    specfem_solutions: Solutions,
) -> None:
    for key in sorted(set(rect_solutions) & set(specfem_solutions)):
        station, component = key
        rect = max(rect_solutions[key], key=lambda row: int(row["nx"]))
        specfem = max(specfem_solutions[key], key=lambda row: int(row["nx"]))
        rect_trace = rect["time"], rect["value"]
        specfem_trace = specfem["time"], specfem["value"]
        difference_time, difference = difference_on_source_grid(
            rect_trace, specfem_trace
        )

        fig, (solution_axis, difference_axis) = plt.subplots(
            2, 1, figsize=(8, 9), sharex=True
        )
        solution_axis.plot(
            *rect_trace, label=f"RECT nx={rect['nx']}", zorder=3
        )
        solution_axis.plot(
            *specfem_trace,
            "--",
            label=f"SPECFEM2D nx={specfem['nx']}",
            zorder=4,
        )
        solution_axis.axhline(0.0, color="black", linewidth=0.8)
        solution_axis.set_ylabel(f"{component}, m/s")
        solution_axis.set_title(
            f"Finest-grid velocities: {station}, {component}"
        )
        solution_axis.grid(True)
        solution_axis.legend()
        solution_axis.tick_params(axis="x", labelbottom=True)
        solution_axis.set_xlabel("t, s")

        difference_axis.plot(
            difference_time,
            difference,
            label="RECT − SPECFEM2D",
        )
        difference_axis.axhline(0.0, color="black", linewidth=0.8)
        difference_axis.set_xlim(*TIME_LIMITS)
        difference_axis.set_xlabel("t, s")
        difference_axis.set_ylabel("velocity difference, m/s")
        difference_axis.set_title("Difference")
        difference_axis.grid(True)
        difference_axis.legend()

        fig.tight_layout()
        filename = OUTPUT_DIR / (
            f"rect-specfem-{station}-{component}-finest-comparison.png"
        )
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Сохранён {filename}")


def calculate_errors_against_specfem_reference(
    solutions: Solutions,
    specfem_solutions: Solutions,
    *,
    exclude_specfem_reference: bool,
) -> dict:
    """Calculate errors against the finest SPECFEM2D trace."""
    errors = {}
    for key, rows in sorted(solutions.items()):
        rows = sorted(rows, key=lambda row: int(row["nx"]))
        if key not in specfem_solutions:
            raise KeyError(f"В SPECFEM2D отсутствует решение для {key}")
        reference = max(
            specfem_solutions[key], key=lambda row: int(row["nx"])
        )
        reference_trace = reference["time"], reference["value"]
        error_rows = []
        for row in rows:
            if exclude_specfem_reference and row is reference:
                continue
            norms = calculate_error_norms(
                (row["time"], row["value"]), reference_trace
            )
            error_rows.append(
                {
                    "nx": int(row["nx"]),
                    "reference_solver": "SPECFEM2D",
                    "reference_nx": int(reference["nx"]),
                    **norms,
                }
            )
        errors[key] = error_rows
    return errors


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


def plot_convergence(errors: dict, solver: str, filename_prefix: str) -> None:
    styles = {"L1": "o-", "L2": "s-", "Linf": "^-"}
    for (station, component), rows in sorted(errors.items()):
        if not rows:
            continue
        rows = sorted(rows, key=lambda row: row["nx"])
        nx = np.array([row["nx"] for row in rows], dtype=float)
        fig, axis = plt.subplots(figsize=(7, 5))
        plotted = False
        for norm, style in styles.items():
            values = np.array([row[norm] for row in rows], dtype=float)
            fit = convergence_fit(nx, values)
            valid = (nx > 0.0) & (values > 0.0) & np.isfinite(values)
            if not np.any(valid):
                continue
            label = norm if fit is None else f"{norm} (slope={fit[0]:.2f})"
            line = axis.loglog(nx[valid], values[valid], style, label=label)[0]
            plotted = True
            if fit is not None:
                slope, intercept = fit
                axis.loglog(
                    nx[valid],
                    np.exp(intercept) * nx[valid]**slope,
                    "--",
                    color=line.get_color(),
                    linewidth=1.2,
                    label="_nolegend_",
                )
        if not plotted:
            plt.close(fig)
            print(
                f"Пропущен нулевой график ошибки: "
                f"{solver}, {station}, {component}"
            )
            continue
        axis.set_xlabel("nx")
        axis.set_ylabel("velocity error norm, m/s")
        axis.set_title(f"{solver}: {station}, {component}")
        axis.grid(True, which="both")
        axis.legend()
        fig.tight_layout()
        filename = OUTPUT_DIR / (
            f"{filename_prefix}-{station}-{component}-error-vs-nx.png"
        )
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Сохранён {filename}")


def save_error_table(errors: dict, filename: Path) -> None:
    fieldnames = [
        "station",
        "component",
        "nx",
        "reference_solver",
        "reference_nx",
        "L1",
        "L2",
        "Linf",
        "L1_slope",
        "L2_slope",
        "Linf_slope",
    ]
    table_rows = []
    for (station, component), rows in sorted(errors.items()):
        rows = sorted(rows, key=lambda row: row["nx"])
        nx = np.array([row["nx"] for row in rows], dtype=float)
        slopes = {}
        for norm in NORMS:
            values = np.array([row[norm] for row in rows], dtype=float)
            fit = convergence_fit(nx, values)
            slopes[norm] = "" if fit is None else fit[0]
        for row in rows:
            table_rows.append(
                {
                    "station": station,
                    "component": component,
                    **row,
                    **{f"{norm}_slope": slopes[norm] for norm in NORMS},
                }
            )

    with filename.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)
    print(f"Сохранена таблица {filename}")


def read_timing_file(filename: Path) -> dict[str, float]:
    values = {}
    for line in filename.read_text(encoding="utf-8").splitlines():
        if "=" in line:
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
        pattern = rf"date and time of the system\s*:\s*({FLOAT_PATTERN})\s+s"
    else:
        raise ValueError(f"Неизвестная программа: {solver}")
    matches = re.findall(pattern, contents, flags=re.MULTILINE | re.IGNORECASE)
    if len(matches) != 1:
        raise ValueError(
            f"В {filename} найдено времён основного цикла: {len(matches)}, "
            "ожидалось одно"
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
        log_file = path / "log_run.txt"
        if not timing_file.is_file():
            print(f"Пропущен проект без timing.txt: {path}")
            continue
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


def plot_calculation_time(
    rect_timings: list[dict[str, float]],
    specfem_timings: list[dict[str, float]],
) -> None:
    fig, axis = plt.subplots(figsize=(7, 5))
    for label, rows, style in (
        ("RECT", rect_timings, "o-"),
        ("SPECFEM2D", specfem_timings, "s-"),
    ):
        if rows:
            axis.loglog(
                [row["nx"] for row in rows],
                [row["real_s"] for row in rows],
                style,
                label=label,
            )
    axis.set_xlabel("nx")
    axis.set_ylabel("calculation time, s")
    axis.set_title("Calculation time")
    axis.grid(True, which="both")
    axis.legend()
    fig.tight_layout()
    filename = OUTPUT_DIR / "rect-specfem-time-vs-nx.png"
    fig.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Сохранён {filename}")


def plot_error_vs_calculation_time(
    error_sets: dict[str, dict],
    timing_sets: dict[str, list[dict[str, float]]],
) -> None:
    norm_styles = {
        "L1": {"color": "tab:blue", "marker": "o"},
        "L2": {"color": "tab:orange", "marker": "s"},
        "Linf": {"color": "tab:green", "marker": "^"},
    }
    solver_line_styles = {
        "RECT": "-",
        "SPECFEM2D": "--",
    }
    for key in sorted(set().union(*(errors.keys() for errors in error_sets.values()))):
        station, component = key
        fig, axis = plt.subplots(figsize=(8, 6))
        plotted = False
        for solver, errors in error_sets.items():
            timing_map = {
                row["nx"]: row["real_s"] for row in timing_sets[solver]
            }
            rows = [
                row for row in errors.get(key, [])
                if row["nx"] in timing_map
            ]
            for norm in NORMS:
                if rows:
                    axis.loglog(
                        [timing_map[row["nx"]] for row in rows],
                        [row[norm] for row in rows],
                        color=norm_styles[norm]["color"],
                        marker=norm_styles[norm]["marker"],
                        linestyle=solver_line_styles[solver],
                        label=f"{solver} {norm}",
                    )
                    plotted = True
        if not plotted:
            plt.close(fig)
            continue
        axis.set_xlabel("calculation time, s")
        axis.set_ylabel("velocity error norm, m/s")
        axis.set_title(f"Error vs calculation time: {station}, {component}")
        axis.grid(True, which="both")
        axis.legend()
        fig.tight_layout()
        filename = OUTPUT_DIR / (
            f"rect-specfem-{station}-{component}-error-vs-time.png"
        )
        fig.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Сохранён {filename}")


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)

    rect_solutions = load_rect_solutions()
    specfem_solutions = load_specfem_solutions()
    plot_velocity_series(rect_solutions, "RECT", "rect")
    plot_velocity_series(specfem_solutions, "SPECFEM2D", "specfem")
    plot_finest_method_comparison(rect_solutions, specfem_solutions)

    rect_errors = calculate_errors_against_specfem_reference(
        rect_solutions,
        specfem_solutions,
        exclude_specfem_reference=False,
    )
    specfem_errors = calculate_errors_against_specfem_reference(
        specfem_solutions,
        specfem_solutions,
        exclude_specfem_reference=True,
    )
    plot_convergence(rect_errors, "RECT", "rect")
    plot_convergence(specfem_errors, "SPECFEM2D", "specfem")
    save_error_table(rect_errors, OUTPUT_DIR / "rect-errors-vs-nx.csv")
    save_error_table(specfem_errors, OUTPUT_DIR / "specfem-errors-vs-nx.csv")

    rect_timings = discover_timing_cases(
        RECT_SOL_DIR, RECT_PROJECT_PREFIX, "rect"
    )
    specfem_timings = discover_timing_cases(
        SPECFEM_SOL_DIR, SPECFEM_PROJECT_PREFIX, "specfem"
    )
    save_timing_table(rect_timings, OUTPUT_DIR / "rect-timings.csv")
    save_timing_table(specfem_timings, OUTPUT_DIR / "specfem-timings.csv")
    plot_calculation_time(rect_timings, specfem_timings)
    plot_error_vs_calculation_time(
        {"RECT": rect_errors, "SPECFEM2D": specfem_errors},
        {"RECT": rect_timings, "SPECFEM2D": specfem_timings},
    )


if __name__ == "__main__":
    main()
