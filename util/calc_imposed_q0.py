import numpy as np
from scipy.optimize import minimize


def parse_config(x, initial_params, config):
    """
    Extracts variables and parameters based on the configuration.
    """
    if config == 'e':
        e_g, e_p, e_ev, e_cd = x     # variables
        c_g, c_p, e_t, q0, t_s, *others = initial_params
    elif config == 'c':
        c_g, c_p = x                 # variables
        e_g, e_p, _, q0, t_s, *others = initial_params
        e_ev, e_cd = e_g, e_p
        e_t = e_g + e_p + e_ev + e_cd
    elif config[0] == 'sa':
        e_g, e_p, e_ev, e_cd, c_g, c_p = x   # variables
        e_t, _, q0, t_s, *others = initial_params
    else:
        raise ValueError(f"Unsupported config: {config}")
    return e_g, e_p, e_ev, e_cd, c_g, c_p, e_t, q0, t_s, others


def create_bounds_and_constraints(config, initial_total):
    """
    Creates bounds and constraints based on configuration.
    """
    if config == 'e':
        x0 = [0.5] * 4
        bounds = [(0.1, 1)] * 4
        constraints = [{'type': 'ineq', 'fun': lambda x: initial_total - sum(x)}]  # sum(e_i) <= e_t
    elif config == 'c':
        x0 = [0.1] * 2
        bounds = [(0.05, 0.5)] * 2
        constraints = [{'type': 'ineq', 'fun': lambda x: initial_total - sum(x)}]  # sum(c_i) <= c_t
    elif config[0] == 'sa':
        x0 = [0.5] * 4 + [0.2] * 2
        bounds = [(0, 1)] * 4 + [(0, 1)] * 2
        constraints = [
            {'type': 'ineq', 'fun': lambda x: initial_total[0] - sum(x[:4])},  # sum(e_i) <= e_t
            {'type': 'ineq', 'fun': lambda x: initial_total[1] - sum(x[4:])},  # sum(c_i) <= c_t
        ]
    else:
        raise ValueError(f"Unsupported config: {config}")
    return x0, bounds, constraints


def objective_function(x, initial_params, config):
    e_g, e_p, e_ev, e_cd, c_g, c_p, e_t, q0, t_s, others = parse_config(x, initial_params, config)
    
    q0 = q0 / others[0]  # /MULTIPLIER
    a_g, a_ev, a_p, a_cd = e_g / e_t, e_ev / e_t, e_p / e_t, e_cd / e_t
    c_eps = (1 / a_g + 1 / a_ev - e_t) / c_g + (1 / a_p + 1 / a_cd - e_t) / c_p

    return q0 * (c_eps*q0 + e_t*(1-t_s)) / (e_t*t_s - c_eps*q0)


def objective_function_ir_ratio(x, initial_params, config):
    e_g, e_p, e_ev, e_cd, c_g, c_p, e_t, q0, t_s, others = parse_config(x, initial_params, config)
    
    I = others[0]
    q0 = q0 / others[1]  # /MULTIPLIER
    a_g, a_ev, a_p, a_cd = e_g/e_t, e_ev/e_t, e_p/e_t, e_cd/e_t
    c_eps = (1/a_g + 1/a_ev - e_t)/(c_g*I) + (1/a_p + 1/a_cd - e_t)/c_p
    
    return q0 * (I*c_eps*q0 + e_t*(I-t_s)) / (e_t*t_s - I*c_eps*q0)


def objective_function_ep_rate(x, initial_params, config):
    e_g, e_p, e_ev, e_cd, c_g, c_p, e_t, q0, t_s, others = parse_config(x, initial_params, config)
    
    s = others[0] / others[1]  # s / MULTIPLIER
    q0 = q0 / others[1]        # /MULTIPLIER
    a_g, a_ev, a_p, a_cd = e_g/e_t, e_ev/e_t, e_p/e_t, e_cd/e_t
    c_eps = (
        (1/a_g + 1/a_ev - e_t)/c_g
         + (1/a_p + 1/a_cd - e_t)/c_p
         - s*(1/a_g + 1/a_ev - e_t)*(1/a_p + 1/a_cd - e_t)/(c_p*c_g*e_t)
    )
    c_A = e_t - s*(1/a_g + 1/a_ev - e_t)/c_g
    c_B = e_t - s*(1/a_p + 1/a_cd - e_t)/c_p

    return (q0 * (c_eps*q0 + c_A - c_B*t_s) + s*t_s*e_t) / (c_B*t_s - c_eps*q0)
 

def find_minimum(_obj_func, initial_params, config):
    """
    Finds the minimum of the objective function for a given configuration.
    """
    if config == 'e':
        initial_total = initial_params[2]  # e_total
    elif config == 'c':
        initial_total = initial_params[2]  # c_total
    elif config[0] == 'sa':
        initial_total = (initial_params[0], initial_params[1])  # (e_total, c_total)
    else:
        raise ValueError(f"Unsupported config: {config}")

    x0, bounds, constraints = create_bounds_and_constraints(config, initial_total)

    # Perform minimization
    result = minimize(
        _obj_func,
        x0,
        args=(initial_params, config),
        bounds=bounds,
        constraints=constraints,
        tol=1e-16,
    )
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