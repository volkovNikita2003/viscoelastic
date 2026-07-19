import numpy as np
from math import sqrt

def c1(rho, la, mu):
    return sqrt((la + 2*mu)/rho)

def c2(rho, la, mu):
    return sqrt(mu/rho)

def mu(rho, c1, c2):
    return rho * c2**2

def la(rho, c1, c2):
    return rho * (c1**2 - 2*c2**2)

def M_uv(Mv, tau_e_v_arr, tau_s_v_arr):
    res = 1
    for i in range(len(tau_e_v_arr)):
        res -= (1 - tau_e_v_arr[i] / tau_s_v_arr[i])
    res *= Mv
    return res

def phi_0(M, tau_e, tau_s):
    return M / tau_s * (1 - tau_e / tau_s)

n = 2
M1 = 20e9
M2 = 16e9
rho = 2000
tau1_e = [0.0325305, 0.0032530]
tau1_s = [0.0311465, 0.0031146]
tau2_e = [0.0332577, 0.0033257]
tau2_s = [0.0304655, 0.0030465]

M_u1 = M_uv(M1, tau1_e, tau1_s)
M_u2 = M_uv(M2, tau2_e, tau2_s)
print(f"{M_u1=}")
print(f"{M_u2=}")
lambda_u = (M_u1 - M_u2) / n
mu_u = M_u2 / 2
c1_u = c1(rho, lambda_u, mu_u)
c2_u = c2(rho, lambda_u, mu_u)
print(f"{c1_u=}")
print(f"{c2_u=}")

phi_1_l = [phi_0(M1, tau_e, tau_s) for (tau_e, tau_s) in zip(tau1_e, tau1_s)]
phi_2_l = [phi_0(M2, tau_e, tau_s) for (tau_e, tau_s) in zip(tau2_e, tau2_s)]
print(f"{phi_1_l=}")
print(f"{phi_2_l=}")
