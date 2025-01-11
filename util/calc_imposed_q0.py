import numpy as np
import streamlit as st
from scipy.optimize import minimize


def objective_function(x, initial_params, config):
    if config=='e':
        e_g, e_p, e_ev, e_cd = x   # variables
        c_g, c_p, e_t, q0, t_s, MULTIPLIER = initial_params
    elif config=='c':
        c_g, c_p = x   # variables
        e_g, e_p, _, q0, t_s, MULTIPLIER = initial_params
        e_ev, e_cd = e_g, e_p
        e_t = e_g + e_p + e_ev + e_cd
    elif config[0]=='sa':
        e_g, e_p, e_ev, e_cd, c_g, c_p = x   # variables
        e_t, _, q0, t_s, MULTIPLIER = initial_params
    
    q0 = q0 / MULTIPLIER
    a_g, a_ev, a_p, a_cd = e_g / e_t, e_ev / e_t, e_p / e_t, e_cd / e_t
    c_eps = (1 / a_g + 1 / a_ev - e_t) / c_g + (1 / a_p + 1 / a_cd - e_t) / c_p

    return q0 * (c_eps*q0 + e_t*(1-t_s)) / (e_t*t_s - c_eps*q0)


def objective_function_ir_ratio(x, initial_params, config):
    if config=='e':
        e_g, e_p, e_ev, e_cd = x   # variables
        c_g, c_p, e_t, q0, t_s, I, MULTIPLIER = initial_params
    elif config=='c':
        c_g, c_p = x   # variables
        e_g, e_p, _, q0, t_s, I, MULTIPLIER = initial_params
        e_ev, e_cd = e_g, e_p
        e_t = e_g + e_p + e_ev + e_cd
    elif config[0]=='sa':
        e_g, e_p, e_ev, e_cd, c_g, c_p = x   # variables
        e_t, _, q0, t_s, I, MULTIPLIER = initial_params
    
    q0 = q0 / MULTIPLIER
    a_g, a_ev, a_p, a_cd = e_g/e_t, e_ev/e_t, e_p/e_t, e_cd/e_t
    c_eps = (1/a_g + 1/a_ev - e_t)/(c_g*I) + (1/a_p + 1/a_cd - e_t)/c_p
    
    return q0 * (I*c_eps*q0 + e_t*(I-t_s)) / (e_t*t_s - I*c_eps*q0)


def objective_function_ep_rate(x, initial_params, config):
    if config=='e':
        e_g, e_p, e_ev, e_cd = x   # variables
        c_g, c_p, e_t, q0, t_s, s, MULTIPLIER = initial_params
    elif config=='c':
        c_g, c_p = x   # variables
        e_g, e_p, _, q0, t_s, s, MULTIPLIER = initial_params
        e_ev, e_cd = e_g, e_p
        e_t = e_g + e_p + e_ev + e_cd
    elif config[0]=='sa':
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
    
    elif config=='sa-e':
        # e variables
        *x, _, _ = x
        return initial_var_total - sum(x)
    elif config=='sa-c':
        # c variables
        _, _, _, _, *x = x
        return initial_var_total - sum(x)


def find_minimum(_obj_function, initial_params, config):
    """
    Finds the minimum of the objective function for a given configuration.

    Parameters:
    - _obj_function: Objective function to minimize (e.g., objective_function).
    - initial_params: Tuple containing initial parameters for the minimization.
    - config: String specifying the configuration ('e', 'c', 'sae').

    Returns:
    - result: Optimization result object from scipy.optimize.minimize.
    """
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

    elif config[0]=='sa':
        initial_eps_total = initial_params[0]
        initial_c_total = initial_params[1]

        # Initial guesses and bounds
        x0 = [0.5, 0.5, 0.5, 0.5, 0.2, 0.2]
        b1 = (0, 1)
        b2 = (0, 1)     # (0.05, 0.95)
        bnds = (b1, b1, b1, b1, b2, b2)
        con1 = {'type': 'ineq', 'fun': lambda x: constraint(x, initial_eps_total, 'sa-e')}
        con2 = {'type': 'ineq', 'fun': lambda x: constraint(x, initial_c_total, 'sa-c')}
        cons = [con1, con2]


    # Perform minimization
    result = minimize(_obj_function, x0, args=(initial_params, config), 
                      bounds=bnds, constraints=cons, tol=10**(-16))
    return result



def find_minimum_vectorized(obj_func, opt_var, opt_config, *params):
    """
    Vectorizes the find_minimum function for a given objective function.
    
    Parameters:
    - obj_func: Objective function to minimize (e.g., objective_function).
    - opt_var: Array of values over which to minimize.
    - opt_config: Optimization configuration (e.g., ('sa', 'e')).
    - *params: Additional parameters passed to the objective function.
    
    Returns:
    - result: Array of results for each value in opt_var.
    """
    def wrapper(var, opt_config, *params):
        if opt_config[1] == 'e':
            initial_params = (var, *params)
        elif opt_config[1] == 'c':
            initial_params = (params[0], var, *params[1:])
        elif opt_config[1] == 'q':
            initial_params = (*params[:2], var, *params[2:])
        elif opt_config[1] == 't':
            initial_params = (*params[:3], var, *params[3:])
        elif opt_config[1] in ['I', 's']:
            initial_params = (*params[:4], var, params[-1])

        return find_minimum(obj_func, initial_params, opt_config)
    
    vectorized_func = np.vectorize(wrapper, excluded=[1, 2])  # Exclude opt_var and params
    return vectorized_func(opt_var, opt_config, *params)