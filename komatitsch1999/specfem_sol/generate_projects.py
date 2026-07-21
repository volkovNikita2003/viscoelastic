#!/usr/bin/env python3
"""Create and configure the SPECFEM2D project with the requested mesh step."""

from pathlib import Path
import shutil


BASE_DIR = Path(__file__).resolve().parent
TEMPLATE_DIR = BASE_DIR / (
    "check_absolute_amplitude_of_force_source_seismograms_viscoelastic_template"
)
PROJECT_NAME = (
    "check_absolute_amplitude_of_force_source_seismograms_viscoelastic_auto_test"
)
PROJECT_DIR = BASE_DIR / PROJECT_NAME

PARAMETERS = {
    "NSTEP": 1400,
    "DT": 4e-4,
    "NTSTEP_BETWEEN_OUTPUT_SEISMOS": 1400,
    "nx": 44,
    "nz": 44,
    "NTSTEP_BETWEEN_OUTPUT_INFO": 200,
    "NTSTEP_BETWEEN_OUTPUT_IMAGES": 200,
}

TEMPLATE_FILES = (
    "Par_file_no_attenuation_2D_at_the_corner_between_several_spectral_elements",
    "interfaces_attenuation_analytic.dat",
)


def format_template(path: Path) -> None:
    """Replace ``str.format`` fields in a copied template file."""
    contents = path.read_text(encoding="utf-8")
    path.write_text(contents.format(**PARAMETERS), encoding="utf-8")


def write_script(path: Path, contents: str) -> None:
    path.write_text(contents, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def main() -> None:
    if PROJECT_DIR.exists():
        raise FileExistsError(
            f"Project already exists: {PROJECT_DIR}. "
            "Run clear_specfem.sh before generating it again."
        )

    shutil.copytree(TEMPLATE_DIR, PROJECT_DIR)
    print(f"Создан проект: {PROJECT_DIR}")

    for filename in TEMPLATE_FILES:
        format_template(PROJECT_DIR / "DATA" / filename)
    print("Параметры проекта записаны в шаблонные файлы DATA")

    write_script(
        BASE_DIR / "calc_specfem.sh",
        f"""#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)
cd "$SCRIPT_DIR/{PROJECT_NAME}"
echo "Запуск расчёта в $PWD"
./run_this_example.sh > log_run.txt 2>&1
echo "Расчёт завершён. Журнал: $PWD/log_run.txt"
""",
    )
    write_script(
        BASE_DIR / "clear_specfem.sh",
        f"""#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)
echo "Удаление проекта: $SCRIPT_DIR/{PROJECT_NAME}"
rm -rf -- "$SCRIPT_DIR/{PROJECT_NAME}"
echo "Проект удалён"
""",
    )
    print(f"Создан скрипт запуска расчёта: {BASE_DIR / 'calc_specfem.sh'}")
    print(f"Создан скрипт удаления проекта: {BASE_DIR / 'clear_specfem.sh'}")


if __name__ == "__main__":
    main()
