#!/usr/bin/env bash
set -Eeuo pipefail

CURRENT_STAGE="инициализация"

on_error() {
    exit_code=$?
    line_number=$1
    printf 'Ошибка: запуск квазианалитического решения не завершился в нормальном режиме. Этап: %s; строка: %s; код: %s\n' \
        "$CURRENT_STAGE" "$line_number" "$exit_code" >&2
    exit "$exit_code"
}

trap 'on_error $LINENO' ERR

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

CURRENT_STAGE="очистка"
echo
echo "Очистка результатов предыдущего квазианалитического расчёта"
"$SCRIPT_DIR/clean_all.sh"

CURRENT_STAGE="упругое решение"
echo
echo "Сборка и запуск квазианалитического упругого решения"
"$SCRIPT_DIR/build_and_run_analytical_solution_elastic.sh"

CURRENT_STAGE="вязкоупругое решение"
echo
echo "Сборка и запуск квазианалитического вязкоупругого решения"
"$SCRIPT_DIR/build_and_run_analytical_solution_viscoelastic.sh"

CURRENT_STAGE="завершение"
echo
echo "Все квазианалитические расчёты успешно завершены"
