#!/usr/bin/env python3
"""Plot RECT and SPECFEM2D solutions and their convergence graphs."""

from pathlib import Path

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
RECT_DOMAIN_SIZE_X = 2500.0

# analytical_suffix is appended to analytical filenames.  SPECFEM station names
# use the receiver number written to files such as AA.S0001.BXX.semd.
STATIONS = {
    "station_1": {"analytical_suffix": "", "specfem_station": "S0001"},
    # "station_2": {
    #     "analytical_suffix": "_0.0_500.0",
    #     "specfem_station": "S0002",
    # },
}

COMPONENTS = {
    "ux": {"analytical_component": "Ux", "specfem_component": "BXX"},
    "uy": {"analytical_component": "Uz", "specfem_component": "BXZ"},
}

SPECFEM_PROJECT_PREFIX = (
    "check_absolute_amplitude_of_force_source_seismograms_viscoelastic_auto_nx_"
)
RECT_PROJECT_PREFIX = "viscoelastic_schema_nx_"


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


def load_analytical_solutions(
    station: str, component: str
) -> tuple[tuple[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]:
    station_config = STATIONS[station]
    component_name = COMPONENTS[component]["analytical_component"]
    suffix = station_config["analytical_suffix"]

    viscoelastic_file = (
        ANALYTICAL_DIR
        / f"{component_name}_time_analytical_solution_viscoelastic{suffix}.dat"
    )
    elastic_file = (
        ANALYTICAL_DIR
        / f"{component_name}_time_analytical_solution_elastic{suffix}.dat"
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
    """Calculate errors on numerical points within the analytical time range."""
    common = (
        (numerical_time >= analytical_time.min())
        & (numerical_time <= analytical_time.max())
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


def process_rect() -> None:
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
                    {"nx": nx, "h": RECT_DOMAIN_SIZE_X / nx, **norms}
                )

    for (station, component), rows in errors.items():
        plot_error_graph(rows, "nx", station, component, "rect")
        plot_error_graph(rows, "h", station, component, "rect")


def process_specfem() -> None:
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


def main() -> None:
    OUTPUT_DIR.mkdir(exist_ok=True)
    process_rect()
    process_specfem()


if __name__ == "__main__":
    main()
