#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

shopt -s nullglob
generated_files=(
    "$SCRIPT_DIR"/Ux_time_analytical_solution_*.dat
    "$SCRIPT_DIR"/Uz_time_analytical_solution_*.dat
    "$SCRIPT_DIR"/spectrum_of_the_source_used.gnu
    "$SCRIPT_DIR"/bin/analytical_solution_elastic
    "$SCRIPT_DIR"/bin/analytical_solution_viscoelastic
)

for generated_file in "${generated_files[@]}"; do
    if [[ -f "$generated_file" ]]; then
        echo "Удаление автоматически созданного файла: $generated_file"
        rm -- "$generated_file"
    fi
done

if [[ -d "$SCRIPT_DIR/bin" ]]; then
    rmdir --ignore-fail-on-non-empty "$SCRIPT_DIR/bin"
fi

echo "Очистка квазианалитического решения завершена"
