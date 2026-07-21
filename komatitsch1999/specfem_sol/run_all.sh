#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

echo
echo "Очистка ранее созданных проектов и файлов"
"$SCRIPT_DIR/clean_all.sh"

echo
echo "Генерация расчётных проектов"
python3 "$SCRIPT_DIR/generate_projects.py"

echo
echo "Запуск всех расчётов SPECFEM2D"
"$SCRIPT_DIR/calc_specfem.sh"

echo
echo "Все расчёты SPECFEM2D завершены"
