#!/usr/bin/env bash
set -Eeuo pipefail

CURRENT_STAGE="инициализация"

on_error() {
    exit_code=$?
    line_number=$1
    printf 'Ошибка: полная очистка проекта не завершилась в нормальном режиме. Этап: %s; строка: %s; код: %s\n' \
        "$CURRENT_STAGE" "$line_number" "$exit_code" >&2
    exit "$exit_code"
}

trap 'on_error $LINENO' ERR

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

CURRENT_STAGE="очистка квазианалитического решения"
echo
echo "Очистка квазианалитического решения"
"$SCRIPT_DIR/analytical_sol/clean_all.sh"

CURRENT_STAGE="очистка RECT"
echo
echo "Очистка проектов RECT"
"$SCRIPT_DIR/rect_sol/clean_all.sh"

CURRENT_STAGE="очистка SPECFEM2D"
echo
echo "Очистка проектов SPECFEM2D"
"$SCRIPT_DIR/specfem_sol/clean_all.sh"

CURRENT_STAGE="удаление графиков и таблиц"
if [[ -d "$SCRIPT_DIR/graphs" ]]; then
    echo
    echo "Удаление графиков и CSV-таблиц: $SCRIPT_DIR/graphs"
    rm -rf -- "$SCRIPT_DIR/graphs"
fi

CURRENT_STAGE="удаление Python-кэшей"
for cache_dir in \
    "$SCRIPT_DIR/__pycache__" \
    "$SCRIPT_DIR/analytical_sol/__pycache__" \
    "$SCRIPT_DIR/rect_sol/__pycache__" \
    "$SCRIPT_DIR/specfem_sol/__pycache__"
do
    if [[ -d "$cache_dir" ]]; then
        echo "Удаление Python-кэша: $cache_dir"
        rm -rf -- "$cache_dir"
    fi
done

CURRENT_STAGE="завершение"
echo
echo "Полная очистка проекта Komatitsch 1999 завершена"
