#!/usr/bin/env python3
"""Create and configure the SPECFEM2D project with the requested mesh step."""

from pathlib import Path
import math
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
NX_VALUES = (22, 44, 88)
CFL = 0.5
VP = 3297.8490000000002
T_TOTAL_S = 1.0
DOMAIN_SIZE_M = 2000.0
SPECTRAL_POLYNOMIAL_DEGREE = 4

TEMPLATE_FILES = (
    "Par_file_no_attenuation_2D_at_the_corner_between_several_spectral_elements",
    "interfaces_attenuation_analytic.dat",
)


def project_name(nx: int) -> str:
    return f"{PROJECT_NAME_PREFIX}_nx_{nx}"


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
    h_se = DOMAIN_SIZE_M / nx
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


def write_script(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def main() -> None:
    project_dirs = [BASE_DIR / project_name(nx) for nx in NX_VALUES]
    existing_projects = [path for path in project_dirs if path.exists()]
    if existing_projects:
        project_list = ", ".join(str(path) for path in existing_projects)
        raise FileExistsError(
            f"Projects already exist: {project_list}. "
            "Run clear_specfem.sh before generating it again."
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

    bash_projects = " ".join(f'"{project_name(nx)}"' for nx in NX_VALUES)

    write_script(
        BASE_DIR / "calc_specfem.sh",
        f"""#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)
PROJECTS=({bash_projects})

for project in "${{PROJECTS[@]}}"; do
    cd "$SCRIPT_DIR/$project"
    echo "Запуск расчёта в $PWD"
    ./run_this_example.sh > log_run.txt 2>&1
    echo "Расчёт завершён. Журнал: $PWD/log_run.txt"
done
""",
    )
    write_script(
        BASE_DIR / "clean_specfem.sh",
        f"""#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)
PROJECTS=({bash_projects})

for project in "${{PROJECTS[@]}}"; do
    echo "Удаление проекта: $SCRIPT_DIR/$project"
    rm -rf -- "$SCRIPT_DIR/$project"
    echo "Проект удалён: $project"
done
""",
    )
    print(f"Создан скрипт запуска расчёта: {BASE_DIR / 'calc_specfem.sh'}")
    print(f"Создан скрипт удаления проекта: {BASE_DIR / 'clear_specfem.sh'}")


if __name__ == "__main__":
    main()
