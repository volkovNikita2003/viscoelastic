#!/usr/bin/env bash
set -Eeuo pipefail

CURRENT_STAGE="инициализация"

on_error() {
    exit_code=$?
    line_number=$1
    printf 'Ошибка: общий запуск не завершился в нормальном режиме. Этап: %s; строка: %s; код: %s\n' \
        "$CURRENT_STAGE" "$line_number" "$exit_code" >&2
    exit "$exit_code"
}

trap 'on_error $LINENO' ERR

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

CURRENT_STAGE="полная очистка проекта"
echo
echo "Полная очистка проекта полупространства"
"$SCRIPT_DIR/clean_all.sh"

CURRENT_STAGE="численное решение RECT"
echo
echo "Запуск всех расчётов RECT"
"$SCRIPT_DIR/rect_sol/run_all.sh"

CURRENT_STAGE="численное решение SPECFEM2D"
echo
echo "Запуск всех расчётов SPECFEM2D"
"$SCRIPT_DIR/specfem_sol/run_all.sh"

CURRENT_STAGE="построение графиков и таблиц"
echo
echo "Построение графиков и таблиц"
python3 "$SCRIPT_DIR/analyze_results.py"

CURRENT_STAGE="завершение"
echo
echo "Все этапы проекта двух полупространств успешно завершены"
