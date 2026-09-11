import math
import numpy as np
import shutil
import re
from pathlib import Path
from copy import deepcopy
from make_impulse import gen_ricker_imp_specfem


DIR_TEMPLATE = Path("templates")

# количество отрезков (элементов) по направлению X
# NX_VALUES = (600, 900, 1200, 1800, 2400, 4800, 9600)
NX_VALUES = (600, 900, 1200,)
# NX_VALUES = (900,)

RHO_SOURCE = 2000
VP = 3957.419

# импульс
F0 = 18.0
T0 = 1.2 / F0

CFL = 0.5
T_TOTAL_S = 1.2
DOMAIN_SIZE_Z_M = 3000.
TIMING_PROJECT_SUFFIX = "_timing"
OMP_NUM_THREADS = 1
RECEIVER_OFFSET_M = 500.0

DOMAIN_SIZE_X_M = 2 * DOMAIN_SIZE_Z_M
DOMAIN_SIZE_M = np.array([DOMAIN_SIZE_X_M, DOMAIN_SIZE_Z_M], dtype=np.float64)
SOURCE_COORD_M = np.array(
    [DOMAIN_SIZE_X_M / 2, DOMAIN_SIZE_Z_M * 2 / 3], dtype=np.float64
)
RECEIVERS = (
    {"name": "station_1", "coord_m": SOURCE_COORD_M + [-2*RECEIVER_OFFSET_M, RECEIVER_OFFSET_M], "grids": "viscoelastic_schema_1"},
    {"name": "station_2", "coord_m": SOURCE_COORD_M + [-RECEIVER_OFFSET_M, RECEIVER_OFFSET_M], "grids": "viscoelastic_schema_1"},
    {"name": "station_3", "coord_m": SOURCE_COORD_M + [0.0, RECEIVER_OFFSET_M], "grids": "viscoelastic_schema_1"},
    {"name": "station_4", "coord_m": SOURCE_COORD_M + [RECEIVER_OFFSET_M, RECEIVER_OFFSET_M], "grids": "viscoelastic_schema_1"},
    {"name": "station_5", "coord_m": SOURCE_COORD_M + [2*RECEIVER_OFFSET_M, RECEIVER_OFFSET_M], "grids": "viscoelastic_schema_1"},

    # эти приемники в о второй сетке, координаты приемников даны в координатах относительно origin этой сетки, 
    # а не в глобальной системе координат. По-хорошему, стоит переписать код 
    {"name": "station_6", "coord_m": [SOURCE_COORD_M[0] - 2*RECEIVER_OFFSET_M, RECEIVER_OFFSET_M], "grids": "viscoelastic_schema_2"},
    {"name": "station_7", "coord_m": [SOURCE_COORD_M[0] - RECEIVER_OFFSET_M, RECEIVER_OFFSET_M], "grids": "viscoelastic_schema_2"},
    {"name": "station_8", "coord_m": [SOURCE_COORD_M[0], RECEIVER_OFFSET_M], "grids": "viscoelastic_schema_2"},
    {"name": "station_9", "coord_m": [SOURCE_COORD_M[0] + RECEIVER_OFFSET_M, RECEIVER_OFFSET_M], "grids": "viscoelastic_schema_2"},
    {"name": "station_10", "coord_m": [SOURCE_COORD_M[0] + 2*RECEIVER_OFFSET_M, RECEIVER_OFFSET_M], "grids": "viscoelastic_schema_2"},
)


params_base = {
    "dt": None,
    "steps": None,
    "grid_spacing": None,
    "grid_size": None,
    "grid_origin": np.array([0, 0], dtype=np.float64),
    "impulse_coord_m": SOURCE_COORD_M,
    "impulse_index": None,
    "force_coef": None,
    "save_vtk_num": 1,
    "save_vtk": None,
    **{
        f"{receiver['name']}_coord_m": receiver["coord_m"]
        for receiver in RECEIVERS
    },
    **{
        f"{receiver['name']}_grids": receiver["grids"]
        for receiver in RECEIVERS
    },
}


def add_station_position_dict(params: dict) -> dict:
    pattern = re.compile(r"station_(\d+)_coord_m")

    station_numbers = sorted(
        int(match.group(1))
        for key in list(params)
        if (match := pattern.fullmatch(key))
    )

    for station_number in station_numbers:
        coord_key = f"station_{station_number}_coord_m"
        grids_key = f"station_{station_number}_grids"
        index_key = f"station_{station_number}"

        params[index_key] = (
            get_point_index(
                params[coord_key],
                params["grid_origin"],
                params["grid_spacing"],
            ),
            params[grids_key]
        )
    return params

def get_point_index(
        point_coord,
        origin_coord,
        h,
        *,
        rtol: float = 1e-9,
        atol: float = 1e-12,
):
    """
    Вычисляет индекс точки относительно origin_coord с шагом h.

    Параметры могут быть числами или NumPy-массивами. Если вычисленный
    индекс не является целым числом с заданной точностью, выбрасывается
    ValueError.

    Возвращает:
        int, если результат скалярный;
        np.ndarray с целочисленным dtype, если результат является массивом.
    """
    point_coord = np.asarray(point_coord)
    origin_coord = np.asarray(origin_coord)
    h = np.asarray(h)

    if np.any(h == 0):
        raise ValueError("Шаг h не может быть равен нулю")

    index = (point_coord - origin_coord) / h
    rounded_index = np.rint(index)

    is_integer = np.isclose(
        index,
        rounded_index,
        rtol=rtol,
        atol=atol,
    )

    if not np.all(is_integer):
        invalid_indices = index[~is_integer] if index.ndim > 0 else index

        raise ValueError(
            "Не все координаты соответствуют узлам сетки. "
            f"Получены нецелые индексы: {invalid_indices}"
        )

    result = rounded_index.astype(np.int64)

    if result.ndim == 0:
        return int(result)

    return result

def get_case_params(nx):
    if not isinstance(nx, int) or isinstance(nx, bool):
        raise TypeError("nx must be integer")
    if nx <= 0:
        raise ValueError("nx must be positive")
    if nx % 6 != 0:
        raise ValueError(
            "nx must be divisible by 6 so that nx/2 is an integer and the "
            "source at (Lx/2, 2*Lz/3) lies at a grid node"
        )
    for receiver in RECEIVERS:
        coord = np.asarray(receiver["coord_m"])
        # if np.any(coord < 0.0) or np.any(coord > DOMAIN_SIZE_M):
        #     raise ValueError(
        #         f"Receiver {receiver['name']} at {coord.tolist()} is outside "
        #         f"the domain [0, {DOMAIN_SIZE_M[0]}] x [0, {DOMAIN_SIZE_M[1]}]"
        #     )
    nz = nx // 2
    n_elements = np.array([nx, nz], dtype=np.int64)

    grid_spacing = DOMAIN_SIZE_M / n_elements
    h = np.min(grid_spacing)
    grid_size = n_elements + 1
    dt = CFL * h / VP
    steps = math.ceil(T_TOTAL_S / dt)

    impule_index = get_point_index(
        params_base["impulse_coord_m"],
        params_base["grid_origin"],
        grid_spacing,
    )

    params = deepcopy(params_base)
    params["dt"] = dt
    params["steps"] = steps
    params["grid_spacing"] = grid_spacing
    params["grid_size"] = grid_size
    params["impulse_index"] = impule_index
    params["force_coef"] = 1/(RHO_SOURCE*grid_spacing[0]*grid_spacing[1])
    params["save_vtk"] = math.ceil((steps-1)/params_base["save_vtk_num"])
    params = add_station_position_dict(params)
    return params

def convert_params_to_format_dict(params: dict):
    format_params = {
        "dt": params["dt"],
        "steps": params["steps"],
        "grid_spacing": f"{params['grid_spacing'][0]}, {params['grid_spacing'][1]}",
        "grid_size": f"{params['grid_size'][0]}, {params['grid_size'][1]}",
        "grid_origin": f"{params['grid_origin'][0]}, {params['grid_origin'][1]}",
        "impulse_index": f"{params['impulse_index'][0]}, {params['impulse_index'][1]}, 0",
        "save_vtk": params["save_vtk"],
        "force_coef": params["force_coef"],
        "station_savers": format_station_savers(params),
    }

    station_pattern = re.compile(r"station_(\d+)")
    station_keys = sorted(
        (
            (int(match.group(1)), key)
            for key in params
            if (match := station_pattern.fullmatch(key))
        ),
        key=lambda item: item[0],
    )

    for station_number, station_key in station_keys:
        station_index = params[station_key]

        format_params[station_key] = (
            f"{station_index[0]}, "
            f"{station_index[1]}, 0"
        )
    
    return format_params


def format_station_savers(params: dict) -> str:
    station_pattern = re.compile(r"station_(\d+)")
    station_keys = sorted(
        (
            (int(match.group(1)), key)
            for key in params
            if (match := station_pattern.fullmatch(key))
        ),
        key=lambda item: item[0],
    )

    blocks = []
    for _, station_name in station_keys:
        station_index, station_grids = params[station_name]
        index = f"{station_index[0]}, {station_index[1]}, 0"
        blocks.append(
            f"""    [saver]
        name = SinglePointSaver
        path = result/txt/{station_name}.txt
        grids = {station_grids}
        order = 1
        save = 1
        params = vx, vy, sxx, syy, sxy
        norms = 0, 0, 0, 0, 0
        index = {index}
        eps = 1e-6
    [/saver]"""
        )
    return "\n".join(blocks)


def write_script(path: Path, contents: str) -> None:
    path = Path(path)
    path.write_text(contents, encoding="utf-8")
    path.chmod(path.stat().st_mode | 0o111)


def comment_line(line: str) -> str:
    """Comment a configuration line while preserving its indentation."""
    indentation = line[: len(line) - len(line.lstrip())]
    return f"{indentation}# {line.lstrip()}"


def disable_savers(config_path: Path) -> None:
    """Disable every saver and the source VTK output in a RECT config."""
    lines = config_path.read_text(encoding="utf-8").splitlines(keepends=True)
    result = []
    inside_savers = False
    savers_sections = 0
    source_saves = 0

    for line in lines:
        stripped = line.strip()
        if stripped == "[savers]":
            inside_savers = True
            savers_sections += 1
            result.append(line)
            continue

        if inside_savers:
            if stripped == "[/savers]":
                inside_savers = False
                result.append(line)
            else:
                result.append(comment_line(line))
            continue

        if re.fullmatch(r"save\s*=\s*result/vtk/source\.vtk", stripped):
            result.append(comment_line(line))
            source_saves += 1
            continue

        result.append(line)

    if inside_savers:
        raise ValueError(f"В {config_path} не закрыта секция [savers]")
    if savers_sections != 1:
        raise ValueError(
            f"В {config_path} найдено секций [savers]: {savers_sections}, ожидалась одна"
        )
    if source_saves != 1:
        raise ValueError(
            f"В {config_path} найдено сохранений source.vtk: {source_saves}, ожидалось одно"
        )

    config_path.write_text("".join(result), encoding="utf-8")


def add_timing_project(
    dir_case: Path,
    case_name: str,
) -> Path:
    """Copy a project and disable output in its timing configuration."""
    timing_dir = dir_case.with_name(f"{dir_case.name}{TIMING_PROJECT_SUFFIX}")
    shutil.copytree(dir_case, timing_dir)
    disable_savers(timing_dir / case_name)

    print(f"Создан проект для замера времени: {timing_dir}")
    return timing_dir


def main():
    visco_schema_template = ''
    with open(DIR_TEMPLATE/"viscoelastic_schema_template.conf", "r") as f:
        visco_schema_template = f.read()


    project_names = []
    timing_project_names = []

    for nx in NX_VALUES:
        params = get_case_params(nx)
        format_params = convert_params_to_format_dict(params)
        

        # ----- viscoelastic_schema -----
        dir_case = Path(f"viscoelastic_schema_nx_{nx}/")
        # if dir_case.exists():
            # shutil.rmtree(dir_case)
        dir_case.mkdir(parents=True, exist_ok=False)
    
        case_name = f"viscoelastic_schema_nx_{nx}.conf"
        with open(dir_case/case_name, "w") as f:
            f.write(
                visco_schema_template.format(**format_params)
            )
    
        gen_ricker_imp_specfem(
            t0=T0,
            f0=F0,
            dt=params["dt"],
            steps=params["steps"],
            path_save=dir_case,
            filename="impulse",
        )
        
        dir_res_vtk = dir_case/"result/vtk"
        dir_res_vtk.mkdir(parents=True, exist_ok=True)
        dir_res_txt = dir_case/"result/txt"
        dir_res_txt.mkdir(parents=True, exist_ok=True)
    
        timing_dir = add_timing_project(dir_case, case_name)
        project_names.append(dir_case.name)
        timing_project_names.append(timing_dir.name)


    bash_projects = " ".join(f'"{name}"' for name in project_names)
    bash_timing_projects = " ".join(
        f'"{name}"' for name in timing_project_names
    )

    bash_calc_rect_sh = f"""#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)
RECT_EXECUTABLE="$SCRIPT_DIR/bin/rect"
PROJECTS=({bash_projects})
TIMING_PROJECTS=({bash_timing_projects})

if [[ ! -x "$RECT_EXECUTABLE" ]]; then
    echo "Ошибка: не найден исполняемый файл RECT: $RECT_EXECUTABLE" >&2
    echo "Создайте ссылку rect_sol/bin/rect согласно README.md" >&2
    exit 1
fi

for project in "${{PROJECTS[@]}}"; do
    cd "$SCRIPT_DIR/$project"
    echo "Запуск расчета в $PWD"
    "$RECT_EXECUTABLE" "$project.conf" > log_run.txt 2>&1
    echo "Расчет завершен. Журнал: $PWD/log_run.txt"
done

TIMEFORMAT=$'real_s=%R\nuser_s=%U\nsys_s=%S'
for project in "${{TIMING_PROJECTS[@]}}"; do
    cd "$SCRIPT_DIR/$project"
    config_name="${{project%{TIMING_PROJECT_SUFFIX}}}.conf"
    echo "Запуск замера времени в $PWD"
    {{ time OMP_NUM_THREADS={OMP_NUM_THREADS} "$RECT_EXECUTABLE" "$config_name" > log_run.txt 2>&1; }} 2> timing.txt
    echo "Замер завершен. Результат: $PWD/timing.txt"
    cat timing.txt
done
"""

    bash_clean_rect_sh = f"""#!/usr/bin/env bash
set -e

SCRIPT_DIR=$(cd -- "$(dirname -- "${{BASH_SOURCE[0]}}")" && pwd)
PROJECTS=({bash_projects})
TIMING_PROJECTS=({bash_timing_projects})

for project in "${{PROJECTS[@]}}" "${{TIMING_PROJECTS[@]}}"; do
    echo "Удаление проекта: $SCRIPT_DIR/$project"
    rm -rf -- "$SCRIPT_DIR/$project"
    echo "Проект удален: $project"
done
"""

    write_script("clean_rect.sh", bash_clean_rect_sh)
    write_script("calc_rect.sh", bash_calc_rect_sh)

if __name__ == "__main__":
    main()
