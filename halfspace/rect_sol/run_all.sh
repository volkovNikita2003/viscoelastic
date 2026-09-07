#!/usr/bin/env bash
set -Eeuo pipefail

on_error() {
    exit_code=$?
    line_number=$1
    printf 'Ошибка: run.sh не завершился в нормальном режиме (строка %s, код %s)\n' \
        "$line_number" "$exit_code" >&2
    exit "$exit_code"
}

trap 'on_error $LINENO' ERR

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

echo
echo "Очистка ранее созданных проектов RECT"
"$SCRIPT_DIR/clean_all.sh"

echo
echo "Генерация проектов RECT"
cd "$SCRIPT_DIR"
python3 generate_projects.py

echo
echo "Запуск расчётов RECT и замеров времени"
"$SCRIPT_DIR/calc_all.sh"

echo
echo "Все расчёты RECT завершены"
