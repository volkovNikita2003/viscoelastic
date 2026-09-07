#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

shopt -s nullglob
generated_projects=(
    "$SCRIPT_DIR"/viscoelastic_schema_nx_*
    "$SCRIPT_DIR"/elastic_nx_*
)

for project in "${generated_projects[@]}"; do
    if [[ -d "$project" ]]; then
        echo "Удаление автоматически созданного проекта: $project"
        rm -rf -- "$project"
    fi
done

for generated_script in calc_rect.sh clean_rect.sh; do
    generated_path="$SCRIPT_DIR/$generated_script"
    if [[ -f "$generated_path" ]]; then
        echo "Удаление автоматически созданного файла: $generated_path"
        rm -- "$generated_path"
    fi
done

echo "Очистка автоматически созданных файлов и директорий RECT завершена"
