#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_file="analytical_solution_viscoelastic_2D_plane_strain_Carcione_correct_with_1_over_L.f90"
executable="bin/analytical_solution_viscoelastic"

cd "$script_dir"
mkdir -p bin

FC="${FC:-gfortran}"

if ! command -v "$FC" >/dev/null 2>&1; then
  echo "Ошибка: компилятор '$FC' не найден." >&2
  echo "Установите gfortran или задайте компилятор через переменную FC." >&2
  exit 1
fi

echo "Компиляция $source_file ..."
# The bundled legacy FFTPACK code intentionally aliases COMPLEX/REAL and
# INTEGER/REAL arrays. Modern gfortran requires the compatibility flag below.
"$FC" \
  -O2 \
  -std=legacy \
  -fallow-argument-mismatch \
  -w \
  "$source_file" \
  -o "$executable"

echo "Запуск $executable ..."
exec "./$executable"
