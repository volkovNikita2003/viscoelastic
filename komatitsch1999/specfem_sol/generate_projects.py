#!/usr/bin/env python3
"""Create and configure the SPECFEM2D project with the requested mesh step."""

from pathlib import Path
import math
import re
import shutil

import numpy as np
from numpy.polynomial.legendre import Legendre


BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / (
    "check_absolute_amplitude_of_force_source_seismograms_viscoelastic_template"
)
PROJECT_NAME_PREFIX = (
    "check_absolute_amplitude_of_force_source_seismograms_viscoelastic_auto"
)
NX_VALUES = (100, 80, 60, 50)
CFL = 0.5
VP = 3297.849
T_TOTAL_S = 0.65
DOMAIN_SIZE_M = 2500.0
SPECTRAL_POLYNOMIAL_DEGREE = 4
TIMING_PROJECT_SUFFIX = "_timing"

TEMPLATE_FILES = (
    "Par_file_no_attenuation_2D_at_the_corner_between_several_spectral_elements",
    "interfaces_attenuation_analytic.dat",
)

TIMING_OUTPUT_PARAMETERS = {
    "SAVE_FORWARD": ".false.",
    "SAVE_MODEL": "default",
    "write_moving_sources_database": ".false.",
    "save_ASCII_seismograms": ".false.",
    "save_binary_seismograms_single": ".false.",
    "save_binary_seismograms_double": ".false.",
    "SU_FORMAT": ".false.",
    "save_ASCII_kernels": ".false.",
    "output_grid_Gnuplot": ".false.",
    "output_grid_ASCII": ".false.",
    "OUTPUT_ENERGY": ".false.",
    "COMPUTE_INTEGRATED_ENERGY_FIELD": ".false.",
    "output_color_image": ".false.",
    "output_postscript_snapshot": ".false.",
    "output_wavefield_dumps": ".false.",
}


def project_name(nx: int) -> str:
    return f"{PROJECT_NAME_PREFIX}_nx_{nx}"


def timing_project_name(nx: int) -> str:
    return f"{project_name(nx)}{TIMING_PROJECT_SUFFIX}"


def calculate_min_gll_distance(h_se: float, degree: int) -> float:
    """Calculate the minimum distance between GLL points in one element.

    GLL points consist of the endpoints -1 and 1 and the roots of the
    derivative of the Legendre polynomial of the requested degree.  The
    points are mapped from [-1, 1] to an element of length ``h_se``.
    """
    if h_se <= 0:
        raise ValueError("The spectral element size h_se must be positive")
    if degree < 1:
        raise ValueError("The spectral polynomial degree must be at least 1")

    interior_points = Legendre.basis(degree).deriv().roots()
    reference_points = np.concatenate(([-1.0], interior_points, [1.0]))
    element_points = 0.5 * h_se * (reference_points + 1.0)
    return float(np.min(np.diff(element_points)))


def calculate_parameters(nx: int) -> dict[str, int | float]:
    h_se = DOMAIN_SIZE_M / nx  # h of spectral element
    h_min = calculate_min_gll_distance(h_se, SPECTRAL_POLYNOMIAL_DEGREE)
    dt = CFL * h_min / VP
    nstep = math.ceil(T_TOTAL_S / dt)
    return {
        "NSTEP": nstep,
        "DT": dt,
        "NTSTEP_BETWEEN_OUTPUT_SEISMOS": nstep,
        "nx": nx,
        "nz": nx,
        "NTSTEP_BETWEEN_OUTPUT_INFO": nstep // 10,
        "NTSTEP_BETWEEN_OUTPUT_IMAGES": nstep // 10,
        "h_se": h_se,
        "h_min": h_min,
    }


def format_template(path: Path, parameters: dict[str, int | float]) -> None:
    """Replace ``str.format`` fields in a copied template file."""
    contents = path.read_text(encoding="utf-8")
    path.write_text(contents.format(**parameters), encoding="utf-8")


def disable_outputs(par_file: Path, nstep: int) -> None:
    """Disable optional SPECFEM2D output in a project used for timing."""
    contents = par_file.read_text(encoding="utf-8")
    replacements = {
        **TIMING_OUTPUT_PARAMETERS,
        "NTSTEP_BETWEEN_OUTPUT_INFO": str(nstep + 1),
        "NTSTEP_BETWEEN_OUTPUT_IMAGES": str(nstep + 1),
        "NTSTEP_BETWEEN_OUTPUT_SEISMOS": str(nstep + 1),
    }
    for parameter, value in replacements.items():
        pattern = re.compile(
            rf"^(\s*{re.escape(parameter)}\s*=\s*)\S+(.*)$", re.MULTILINE
        )
        contents, count = pattern.subn(rf"\g<1>{value}\g<2>", contents)
        if count != 1:
            raise ValueError(
                f"Параметр {parameter} найден в {par_file} {count} раз вместо одного"
            )
    par_file.write_text(contents, encoding="utf-8")


def write_script(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def main() -> None:
    project_dirs = [BASE_DIR / project_name(nx) for nx in NX_VALUES]
    timing_project_dirs = [BASE_DIR / timing_project_name(nx) for nx in NX_VALUES]
    all_project_dirs = project_dirs + timing_project_dirs
    existing_projects = [path for path in all_project_dirs if path.exists()]
    if existing_projects:
        project_list = ", ".join(str(path) for path in existing_projects)
        raise FileExistsError(
            f"Projects already exist: {project_list}. "
            "Run clean_specfem.sh before generating them again."
        )

    for nx, project_dir in zip(NX_VALUES, project_dirs):
        parameters = calculate_parameters(nx)
        shutil.copytree(TEMPLATE_DIR, project_dir)
        print(f"Создан проект: {project_dir}")

        for filename in TEMPLATE_FILES:
            format_template(project_dir / "DATA" / filename, parameters)
        print(
            "Параметры проекта записаны: "
            f"nx={nx}, nz={nx}, h_se={parameters['h_se']}, "
            f"h_min={parameters['h_min']}, DT={parameters['DT']}, "
            f"NSTEP={parameters['NSTEP']}, "
            f"NTSTEP_BETWEEN_OUTPUT_INFO={parameters['NTSTEP_BETWEEN_OUTPUT_INFO']}"
        )

        timing_project_dir = BASE_DIR / timing_project_name(nx)
        shutil.copytree(project_dir, timing_project_dir)
        timing_par_file = timing_project_dir / "DATA" / TEMPLATE_FILES[0]
        disable_outputs(timing_par_file, int(parameters["NSTEP"]))
        print(
            "Создан проект для замера времени без сохранения результатов: "
            f"{timing_project_dir}"
        )

    bash_projects = " ".join(f'"{project_name(nx)}"' for nx in NX_VALUES)
    bash_timing_projects = " ".join(
        f'"{timing_project_name(nx)}"' for nx in NX_VALUES
    )

    write_script(
        BASE_DIR / "calc_specfem.sh",
        f"""#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)
PROJECTS=({bash_projects})
TIMING_PROJECTS=({bash_timing_projects})

for project in "${{PROJECTS[@]}}"; do
    cd "$SCRIPT_DIR/$project"
    echo "Запуск расчета в $PWD"
    ./run_this_example.sh > log_run.txt 2>&1
    echo "Расчет завершен. Журнал: $PWD/log_run.txt"
done

TIMEFORMAT=$'real_s=%R\nuser_s=%U\nsys_s=%S'
for project in "${{TIMING_PROJECTS[@]}}"; do
    cd "$SCRIPT_DIR/$project"
    echo "Запуск замера времени в $PWD"
    {{ time ./run_this_example.sh > log_run.txt 2>&1; }} 2> timing.txt
    echo "Замер завершен. Результат: $PWD/timing.txt"
    cat timing.txt
done
""",
    )
    write_script(
        BASE_DIR / "clean_specfem.sh",
        f"""#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)
PROJECTS=({bash_projects})
TIMING_PROJECTS=({bash_timing_projects})

for project in "${{PROJECTS[@]}}" "${{TIMING_PROJECTS[@]}}"; do
    echo "Удаление проекта: $SCRIPT_DIR/$project"
    rm -rf -- "$SCRIPT_DIR/$project"
    echo "Проект удален: $project"
done
""",
    )
    print(f"Создан скрипт запуска расчета: {BASE_DIR / 'calc_specfem.sh'}")
    print(f"Создан скрипт удаления проектов: {BASE_DIR / 'clean_specfem.sh'}")


if __name__ == "__main__":
    main()
