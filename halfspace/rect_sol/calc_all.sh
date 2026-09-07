#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
CALC_SCRIPT="$SCRIPT_DIR/calc_rect.sh"

if [[ ! -x "$CALC_SCRIPT" ]]; then
    echo "Не найден исполняемый $CALC_SCRIPT. Сначала запустите generate_projects.py" >&2
    exit 1
fi

echo "Запуск обычных расчётов RECT и замеров времени"
"$CALC_SCRIPT"
echo "Все расчёты RECT завершены"
