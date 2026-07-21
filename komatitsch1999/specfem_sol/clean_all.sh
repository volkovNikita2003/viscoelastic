#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
PROJECT_PREFIX="check_absolute_amplitude_of_force_source_seismograms_viscoelastic_auto"

shopt -s nullglob
generated_projects=("$SCRIPT_DIR/$PROJECT_PREFIX"*)

for project in "${generated_projects[@]}"; do
    if [[ -d "$project" ]]; then
        echo "Удаление автоматически созданного проекта: $project"
        rm -rf -- "$project"
    fi
done

for generated_script in calc_specfem.sh clear_specfem.sh; do
    generated_path="$SCRIPT_DIR/$generated_script"
    if [[ -f "$generated_path" ]]; then
        echo "Удаление автоматически созданного файла: $generated_path"
        rm -- "$generated_path"
    fi
done

echo "Очистка автоматически созданных файлов и директорий завершена"
