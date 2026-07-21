import numpy as np
import shutil
from pathlib import Path
from copy import deepcopy
from make_impulse import gen_imp


DIR_TEMPLATE = Path("templates")
DIR_RECT = Path("~/rect_git/rect/build/")
rho = 2000

# импульс
eta = 0.5
epsilone = 1
t0 = 0.06
f0 = 50

params_base = {
    "dt": 5e-4,
    "steps": 1000,
    "grid_spaing": np.array([2, 2], dtype=int),
    "grid_size": np.array([1001, 1001], dtype=np.float64),
    "impulse_index": np.array([500, 500], dtype=int),
    "force_coef": 1.0,
    "save_vtk": 10,
    "station_1_rel_m": np.array([500, 500], dtype=int),
    "station_2_rel_m": np.array([0, 500], dtype=int),
    "station_3_rel_m": np.array([500, 0], dtype=int),
    "station_4_rel_m": np.array([-500, 500], dtype=int)
}

def get_station_position_ind(impulse_ind, position_rel_m, grid_spaing):
    x_imp_ind = impulse_ind[0]
    y_imp_ind = impulse_ind[1]
    h_x_m = grid_spaing[0]
    h_y_m = grid_spaing[1]
    station_pos_rel_x_m = position_rel_m[0]
    station_pos_rel_y_m = position_rel_m[1]
    if (station_pos_rel_x_m % h_x_m != 0 or 
        station_pos_rel_y_m % h_y_m != 0):
        raise Exception(f"Числа должны быть кратными: {station_pos_rel_x_m}%{h_x_m}={station_pos_rel_x_m % h_x_m}")
    station_pos_x_ind = int(x_imp_ind + int(station_pos_rel_x_m // h_x_m))
    station_pos_y_ind = int(y_imp_ind + int(station_pos_rel_y_m // h_y_m))
    return np.array([station_pos_x_ind, station_pos_y_ind], dtype=int)

def add_station_position_dict(params: dict):
    for i in range(1, 5, 1):
        params[f"station_{i}"] = get_station_position_ind(
            params["impulse_index"],
            params[f"station_{i}_rel_m"],
            params["grid_spaing"]
        )
    return params

def get_case_params(h):
    params = deepcopy(params_base)
    h_base = params_base["grid_spaing"][0]
    coef = h / h_base
    params["dt"] = params_base["dt"]*coef
    params["steps"] = int(params_base["steps"]/coef)
    params["grid_spaing"] = params["grid_spaing"]*coef
    params["grid_size"] = (params["grid_size"]-1)/coef + 1
    params["impulse_index"] = params["impulse_index"]/coef
    params["force_coef"] = 1/(rho*h**2)
    params["save_vtk"] = int(params_base["save_vtk"]/coef)
    params = add_station_position_dict(params)
    return params

def convert_params_to_format_dict(params: dict):
    format_params = {
        "dt": params["dt"],
        "steps": int(params["steps"]),
        "grid_spaing": f"{params['grid_spaing'][0]}, {params['grid_spaing'][1]}",
        "grid_size": f"{int(params['grid_size'][0])}, {int(params['grid_size'][1])}",
        "impulse_index": f"{int(params['impulse_index'][0])}, {int(params['impulse_index'][1])}, 0",
        "save_vtk": int(params["save_vtk"]),
        "force_coef": params["force_coef"],
        "station_1": f"{int(params['station_1'][0])}, {int(params['station_1'][1])}, 0",
        "station_2": f"{int(params['station_2'][0])}, {int(params['station_2'][1])}, 0",
        "station_3": f"{int(params['station_3'][0])}, {int(params['station_3'][1])}, 0",
        "station_4": f"{int(params['station_4'][0])}, {int(params['station_4'][1])}, 0",
    }
    return format_params


visco_template = ''
with open(DIR_TEMPLATE/"viscoelastic-template.conf", "r") as f:
    visco_template = f.read()

visco_2_template = ''
with open(DIR_TEMPLATE/"viscoelastic_2-template.conf", "r") as f:
    visco_2_template = f.read()

visco_schema_template = ''
with open(DIR_TEMPLATE/"viscoelastic_schema-template.conf", "r") as f:
    visco_schema_template = f.read()

elastic_template = ''
with open(DIR_TEMPLATE/"elastic-template.conf", "r") as f:
    elastic_template = f.read()


clean_all_sh = "#!/bin/bash\n\n"
bash_calc_all_sh = "#!/bin/bash\n\n"

# h_arr = [10, 4, 2, 1, 0.5, 0.25]
h_arr = [0.5]
for h in h_arr:
    params = get_case_params(h)
    format_params = convert_params_to_format_dict(params)
    
    # # ----- viscoelastic -----
    # dir_case = Path(f"viscoelastic-h_{h}/")
    # # if dir_case.exists():
    #     # shutil.rmtree(dir_case)
    # dir_case.mkdir(parents=True, exist_ok=False)

    # case_name = f"viscoelastic-h_{h}.conf"
    # with open(dir_case/case_name, "w") as f:
    #     f.write(
    #         visco_template.format(**format_params)
    #     )

    # gen_imp(eta, epsilone, t0, f0, params["dt"], params["steps"], dir_case, "impulse")
    
    # dir_res_vtk = dir_case/"result/vtk"
    # dir_res_vtk.mkdir(parents=True, exist_ok=True)
    # dir_res_txt = dir_case/"result/txt"
    # dir_res_txt.mkdir(parents=True, exist_ok=True)

    # clean_all_sh += f"rm -rf {dir_case}\n"
    # bash_calc_all_sh += f"cd {dir_case.relative_to(".")}\n"
    # bash_calc_all_sh += f"{DIR_RECT}/rect {case_name}\n"
    # bash_calc_all_sh += f"cd ..\n\n"


    # # ----- viscoelastic_2 -----
    # dir_case = Path(f"viscoelastic_2-h_{h}/")
    # # if dir_case.exists():
    #     # shutil.rmtree(dir_case)
    # dir_case.mkdir(parents=True, exist_ok=False)

    # case_name = f"viscoelastic_2-h_{h}.conf"
    # with open(dir_case/case_name, "w") as f:
    #     f.write(
    #         visco_2_template.format(**format_params)
    #     )

    # gen_imp(eta, epsilone, t0, f0, params["dt"], params["steps"], dir_case, "impulse")
    
    # dir_res_vtk = dir_case/"result/vtk"
    # dir_res_vtk.mkdir(parents=True, exist_ok=True)
    # dir_res_txt = dir_case/"result/txt"
    # dir_res_txt.mkdir(parents=True, exist_ok=True)

    # clean_all_sh += f"rm -rf {dir_case}\n"
    # bash_calc_all_sh += f"cd {dir_case.relative_to(".")}\n"
    # bash_calc_all_sh += f"{DIR_RECT}/rect {case_name}\n"
    # bash_calc_all_sh += f"cd ..\n\n"


    # ----- viscoelastic_schema -----
    dir_case = Path(f"viscoelastic_schema-h_{h}/")
    # if dir_case.exists():
        # shutil.rmtree(dir_case)
    dir_case.mkdir(parents=True, exist_ok=False)

    case_name = f"viscoelastic_schema-h_{h}.conf"
    with open(dir_case/case_name, "w") as f:
        f.write(
            visco_schema_template.format(**format_params)
        )

    gen_imp(eta, epsilone, t0, f0, params["dt"], params["steps"], dir_case, "impulse")
    
    dir_res_vtk = dir_case/"result/vtk"
    dir_res_vtk.mkdir(parents=True, exist_ok=True)
    dir_res_txt = dir_case/"result/txt"
    dir_res_txt.mkdir(parents=True, exist_ok=True)

    clean_all_sh += f"rm -rf {dir_case}\n"
    bash_calc_all_sh += f"cd {dir_case.relative_to(".")}\n"
    bash_calc_all_sh += f"time OMP_NUM_THREADS=1 {DIR_RECT}/rect {case_name}\n"
    bash_calc_all_sh += f"cd ..\n\n"


    # ----- elastic -----
    dir_case = Path(f"elastic-h_{h}/")
    # if dir_case.exists():
        # shutil.rmtree(dir_case)
    dir_case.mkdir(parents=True, exist_ok=False)

    case_name = f"elastic-h_{h}.conf"
    with open(dir_case/case_name, "w") as f:
        f.write(
            elastic_template.format(**format_params)
        )

    gen_imp(eta, epsilone, t0, f0, params["dt"], params["steps"], dir_case, "impulse")
    
    dir_res_vtk = dir_case/"result/vtk"
    dir_res_vtk.mkdir(parents=True, exist_ok=True)
    dir_res_txt = dir_case/"result/txt"
    dir_res_txt.mkdir(parents=True, exist_ok=True)

    clean_all_sh += f"rm -rf {dir_case}\n"
    bash_calc_all_sh += f"cd {dir_case.relative_to(".")}\n"
    bash_calc_all_sh += f"time OMP_NUM_THREADS=1 {DIR_RECT}/rect {case_name}\n"
    bash_calc_all_sh += f"cd ..\n\n"


with open("clean-all.sh", "w") as f:
    f.write(clean_all_sh)

with open("calc-all.sh", "w") as f:
    f.write(bash_calc_all_sh)
