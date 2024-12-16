import numpy as np
from scipy.optimize import minimize

def objective_function(e_values, initial_params):
    init_c_g, init_c_p, init_eps_total, init_q, init_t_s, MULTIPLIER = initial_params

    e_g, e_p, e_ev, e_cd = e_values   # variables
    c_g, c_p = init_c_g, init_c_p

    e_t = init_eps_total
    q0 = init_q / MULTIPLIER
    t_s = init_t_s
    a_g, a_ev, a_p, a_cd = e_g / e_t, e_ev / e_t, e_p / e_t, e_cd / e_t
    c_eps = (1 / a_g + 1 / a_ev - e_t) / c_g + (1 / a_p + 1 / a_cd - e_t) / c_p

    return q0 * (c_eps*q0 + e_t*(1-t_s)) / (e_t*t_s - c_eps*q0)


def objective_function_ir_ratio(e_values, initial_params):
    init_c_g, init_c_p, init_eps_total, init_q, init_t_s, init_I, MULTIPLIER = initial_params

    e_g, e_p, e_ev, e_cd = e_values   # variables
    c_g, c_p = init_c_g, init_c_p
    
    e_t = init_eps_total
    q0 = init_q / MULTIPLIER
    t_s = init_t_s
    I = init_I
    a_g, a_ev, a_p, a_cd = e_g/e_t, e_ev/e_t, e_p/e_t, e_cd/e_t
    c_eps = (1/a_g + 1/a_ev - e_t)/(c_g*I) + (1/a_p + 1/a_cd - e_t)/c_p
    
    return q0 * (I*c_eps*q0 + e_t*(I-t_s)) / (e_t*t_s - I*c_eps*q0)


def objective_function_ep_rate(e_values, initial_params):
    init_c_g, init_c_p, init_eps_total, init_q, init_t_s, init_s, MULTIPLIER = initial_params

    e_g, e_p, e_ev, e_cd = e_values   # variables
    c_g, c_p = init_c_g, init_c_p
    
    e_t = init_eps_total
    q0 = init_q / MULTIPLIER
    t_s = init_t_s
    s = init_s / MULTIPLIER
    a_g, a_ev, a_p, a_cd = e_g/e_t, e_ev/e_t, e_p/e_t, e_cd/e_t
    c_eps = (1/a_g + 1/a_ev - e_t)/c_g + (1/a_p + 1/a_cd - e_t)/c_p - s*(1/a_g + 1/a_ev - e_t)*(1/a_p + 1/a_cd - e_t)/(c_p*c_g*e_t)
    c_A = e_t - s*(1/a_g + 1/a_ev - e_t)/c_g
    c_B = e_t - s*(1/a_p + 1/a_cd - e_t)/c_p

    return (q0 * (c_eps*q0 + c_A - c_B*t_s) + s*t_s*e_t) / (c_B*t_s - c_eps*q0)
 


def constraint(e_values, initial_eps_total):
    # e_values are variables
    return initial_eps_total - sum(e_values)


def find_minimum(obj_function, initial_params):
    _, _, initial_eps_total, *_ = initial_params

    # Initial guesses and bounds
    x0 = [0.5, 0.5, 0.5, 0.5]
    b = (0.1, 1)
    bnds = (b, b, b, b)

    # Constraints
    con1 = {'type': 'ineq', 'fun': lambda x: constraint(x, initial_eps_total)}
    cons = [con1]

    # Perform minimization
    result = minimize(obj_function, x0, args=(initial_params,), 
                      bounds=bnds, constraints=cons, tol=10**(-16))
    return result