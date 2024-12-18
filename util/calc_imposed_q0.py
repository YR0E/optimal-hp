import numpy as np
import streamlit as st
from scipy.optimize import minimize


@st.cache_data
def objective_function(x, initial_params, config):
    if config=='e':
        e_g, e_p, e_ev, e_cd = x   # variables
        c_g, c_p, e_t, q0, t_s, MULTIPLIER = initial_params
    elif config=='c':
        c_g, c_p = x   # variables
        e_g, e_p, _, q0, t_s, MULTIPLIER = initial_params
        e_ev, e_cd = e_g, e_p
        e_t = e_g + e_p + e_ev + e_cd
    elif config=='sae':
        e_g, e_p, e_ev, e_cd, c_g, c_p = x   # variables
        e_t, _, q0, t_s, MULTIPLIER = initial_params
    
    q0 = q0 / MULTIPLIER
    a_g, a_ev, a_p, a_cd = e_g / e_t, e_ev / e_t, e_p / e_t, e_cd / e_t
    c_eps = (1 / a_g + 1 / a_ev - e_t) / c_g + (1 / a_p + 1 / a_cd - e_t) / c_p

    return q0 * (c_eps*q0 + e_t*(1-t_s)) / (e_t*t_s - c_eps*q0)


@st.cache_data
def objective_function_ir_ratio(x, initial_params, config):
    if config=='e':
        e_g, e_p, e_ev, e_cd = x   # variables
        c_g, c_p, e_t, q0, t_s, I, MULTIPLIER = initial_params
    elif config=='c':
        c_g, c_p = x   # variables
        e_g, e_p, _, q0, t_s, I, MULTIPLIER = initial_params
        e_ev, e_cd = e_g, e_p
        e_t = e_g + e_p + e_ev + e_cd
    elif config=='sae':
        e_g, e_p, e_ev, e_cd, c_g, c_p = x   # variables
        e_t, _, q0, t_s, I, MULTIPLIER = initial_params
    
    q0 = q0 / MULTIPLIER
    a_g, a_ev, a_p, a_cd = e_g/e_t, e_ev/e_t, e_p/e_t, e_cd/e_t
    c_eps = (1/a_g + 1/a_ev - e_t)/(c_g*I) + (1/a_p + 1/a_cd - e_t)/c_p
    
    return q0 * (I*c_eps*q0 + e_t*(I-t_s)) / (e_t*t_s - I*c_eps*q0)


@st.cache_data
def objective_function_ep_rate(x, initial_params, config):
    if config=='e':
        e_g, e_p, e_ev, e_cd = x   # variables
        c_g, c_p, e_t, q0, t_s, s, MULTIPLIER = initial_params
    elif config=='c':
        c_g, c_p = x   # variables
        e_g, e_p, _, q0, t_s, s, MULTIPLIER = initial_params
        e_ev, e_cd = e_g, e_p
        e_t = e_g + e_p + e_ev + e_cd
    elif config=='sae':
        e_g, e_p, e_ev, e_cd, c_g, c_p = x   # variables
        e_t, _, q0, t_s, s, MULTIPLIER = initial_params
    
    q0 = q0 / MULTIPLIER
    s = s / MULTIPLIER
    a_g, a_ev, a_p, a_cd = e_g/e_t, e_ev/e_t, e_p/e_t, e_cd/e_t
    c_eps = (1/a_g + 1/a_ev - e_t)/c_g + (1/a_p + 1/a_cd - e_t)/c_p - s*(1/a_g + 1/a_ev - e_t)*(1/a_p + 1/a_cd - e_t)/(c_p*c_g*e_t)
    c_A = e_t - s*(1/a_g + 1/a_ev - e_t)/c_g
    c_B = e_t - s*(1/a_p + 1/a_cd - e_t)/c_p

    return (q0 * (c_eps*q0 + c_A - c_B*t_s) + s*t_s*e_t) / (c_B*t_s - c_eps*q0)
 


def constraint(x, initial_var_total, config):
    if config=='e':
        # e variables
        return initial_var_total - sum(x)
    elif config=='c':
        # c variables
        return initial_var_total - sum(x)
    
    elif config=='sae1':
        # e variables
        *x, _, _ = x
        return initial_var_total - sum(x)
    elif config=='sae2':
        # e variables
        _, _, _, _, *x = x
        return initial_var_total - sum(x)

@st.cache_data
def find_minimum(_obj_function, initial_params, config):
    if config=='e':
        initial_eps_total = initial_params[2]

        # Initial guesses and bounds
        x0 = [0.5, 0.5, 0.5, 0.5]
        b = (0.1, 1)
        bnds = (b, b, b, b)

        # Constraints
        con1 = {'type': 'ineq', 'fun': lambda x: constraint(x, initial_eps_total, config)}
        cons = [con1]

    elif config=='c':
        init_c_total = initial_params[2]

        # Initial guesses and bounds
        x0 = [0.1, 0.1]
        b = (0.05, 0.5)
        bnds = (b, b)

        # Constraints
        con1 = {'type': 'ineq', 'fun': lambda x: constraint(x, init_c_total, config)}
        cons = [con1]

    elif config=='sae':
        initial_eps_total = initial_params[0]
        initial_c_total = initial_params[1]

        # Initial guesses and bounds
        x0 = [0.5, 0.5, 0.5, 0.5, 0.2, 0.2]
        b1 = (0, 1)
        b2 = (0, 1)
        bnds = (b1, b1, b1, b1, b2, b2)
        con1 = {'type': 'ineq', 'fun': lambda x: constraint(x, initial_eps_total, 'sae1')}
        con2 = {'type': 'ineq', 'fun': lambda x: constraint(x, initial_c_total, 'sae2')}
        cons = [con1, con2]


    # Perform minimization
    result = minimize(_obj_function, x0, args=(initial_params, config), 
                      bounds=bnds, constraints=cons, tol=10**(-16))
    return result