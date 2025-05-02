import numpy as np
from scipy.optimize import minimize


def parse_config(x, initial_params, config):
    """
    Parse input variables and parameters based on the given configuration.

    Parameters:
    x (list): Variables to be optimized.
    initial_params (list): Parameters' initial values for the optimization.
    config (str): Configuration string. 'e' for e_total, 'c' for c_total,
                  and 'sa' for sensitivity analysis.

    Returns:
    e_i (float): Effectiveness of the i HX.
    c_i (float): Flow capacity of the i circuit.
    e_t (float): Total effectiveness of HX.
    q0 (float): Imposed heat extraction.
    t_s (float): Temperature.
    others (list): Other parameters (I, s, etc).

    Raises:
    ValueError: If the configuration string is not supported.
    """

    if config == "e":
        e_g, e_p, e_ev, e_cd = x  # variables
        c_g, c_p, e_t, q0, t_s, *others = initial_params
    elif config == "c":
        c_g, c_p = x  # variables
        e_g, e_p, _, q0, t_s, *others = initial_params
        e_ev, e_cd = e_g, e_p
        e_t = e_g + e_p + e_ev + e_cd
    elif config[0] == "sa":
        e_g, e_p, e_ev, e_cd, c_g, c_p = x  # variables
        e_t, _, q0, t_s, *others = initial_params
    else:
        raise ValueError(f"Unsupported config: {config}")

    return e_g, e_p, e_ev, e_cd, c_g, c_p, e_t, q0, t_s, others


def create_bounds_and_constraints(config, initial_total):
    """
    Creates initial guesses, bounds, and constraints for the optimization problem
    based on the given configuration.

    Parameters:
    config (str): Configuration string. 'e' for e_total, 'c' for c_total,
                  and 'sa' for sensitivity analysis.
    initial_total (list or float): Initial values for the 'total' variables.

    Returns:
    x0 (list): Initial guesses for the variables.
    bounds (list): Bounds for the variables.
    constraints (list): Constraints for the optimization problem.
    """

    if config == "e":
        x0 = [0.5] * 4
        bounds = [(0, 1)] * 4
        constraints = [
            {"type": "ineq", "fun": lambda x: initial_total - sum(x)}
        ]  # sum(e_i) <= e_t
    elif config == "c":
        x0 = [0.2] * 2
        bounds = [(0, 1)] * 2
        constraints = [
            {"type": "ineq", "fun": lambda x: initial_total - sum(x)}
        ]  # sum(c_i) <= c_t
    elif config[0] == "sa":  # can be removed later
        x0 = [0.5] * 4 + [0.2] * 2
        bounds = [(0, 1)] * 6
        constraints = [
            {
                "type": "ineq",
                "fun": lambda x: initial_total[0] - sum(x[:4]),
            },  # sum(e_i) <= e_t
            {
                "type": "ineq",
                "fun": lambda x: initial_total[1] - sum(x[4:]),
            },  # sum(c_i) <= c_t
        ]
    else:
        raise ValueError(f"Unsupported config: {config}")

    return x0, bounds, constraints


def objective_function(x, initial_params, config):
    """
    Calculates the objective function for optimization.

    Parameters:
    x (list): Values of the variables (e_g, e_p, e_ev, e_cd, c_g, c_p) .
    initial_params (list): Parameters (e_total, c_total, q0, t_s, ...).
    config (str): Configuration string. 'e' for e_total, 'c' for c_total, and 'sa' for sensitivity analysis.

    Returns:
    float: The value of the objective function.
    """

    e_g, e_p, e_ev, e_cd, c_g, c_p, e_t, q0, t, others = parse_config(
        x, initial_params, config
    )

    q0 = q0 / others[0]  # /MULTIPLIER
    # a_g, a_ev, a_p, a_cd = e_g / e_t, e_ev / e_t, e_p / e_t, e_cd / e_t
    # c_eps = (1 / a_g + 1 / a_ev - e_t) / c_g + (1 / a_p + 1 / a_cd - e_t) / c_p
    alpha_g = (1 / e_g + 1 / e_ev - 1) / c_g
    alpha_p = (1 / e_p + 1 / e_cd - 1) / c_p
    alpha = alpha_g + alpha_p

    # return q0 * (c_eps * q0 + e_t * (1 - t_s)) / (e_t * t_s - c_eps * q0)
    return q0 * (1 / (t - q0 * alpha) - 1)


def objective_function_ir_ratio(x, initial_params, config):
    """
    Calculates the objective function for optimization with irreversibility ratio.

    Parameters:
    x (list): Values of the variables (e_g, e_p, e_ev, e_cd, c_g, c_p) .
    initial_params (list): Parameters (e_total, c_total, q0, t_s, ...).
    config (str): Configuration string. 'e' for e_total, 'c' for c_total, and 'sa' for sensitivity analysis.

    Returns:
    float: The value of the objective function.
    """

    e_g, e_p, e_ev, e_cd, c_g, c_p, e_t, q0, t, others = parse_config(
        x, initial_params, config
    )

    I = others[0]
    q0 = q0 / others[1]  # /MULTIPLIER
    # a_g, a_ev, a_p, a_cd = e_g / e_t, e_ev / e_t, e_p / e_t, e_cd / e_t
    # c_eps = (1 / a_g + 1 / a_ev - e_t) / (c_g * I) + (1 / a_p + 1 / a_cd - e_t) / c_p

    alpha_g = (1 / e_g + 1 / e_ev - 1) / c_g
    alpha_p = (1 / e_p + 1 / e_cd - 1) / c_p
    alpha_I = alpha_g + I * alpha_p

    # return q0 * (I * c_eps * q0 + e_t * (I - t_s)) / (e_t * t_s - I * c_eps * q0)
    return q0 * (I / (t - q0 * alpha_I) - 1)


def objective_function_ep_rate(x, initial_params, config):
    """
    Calculates the objective function for optimization with entropy production rate.

    Parameters:
    x (list): Values of the variables (e_g, e_p, e_ev, e_cd, c_g, c_p) .
    initial_params (list): Parameters (e_total, c_total, q0, t_s, ...).
    config (str): Configuration string. 'e' for e_total, 'c' for c_total, and 'sa' for sensitivity analysis.

    Returns:
    float: The value of the objective function.
    """
    e_g, e_p, e_ev, e_cd, c_g, c_p, e_t, q0, t, others = parse_config(
        x, initial_params, config
    )

    s = others[0] / others[1]  # s / MULTIPLIER
    q0 = q0 / others[1]  # /MULTIPLIER
    # a_g, a_ev, a_p, a_cd = e_g / e_t, e_ev / e_t, e_p / e_t, e_cd / e_t
    # c_eps = (
    #     (1 / a_g + 1 / a_ev - e_t) / c_g
    #     + (1 / a_p + 1 / a_cd - e_t) / c_p
    #     - s
    #     * (1 / a_g + 1 / a_ev - e_t)
    #     * (1 / a_p + 1 / a_cd - e_t)
    #     / (c_p * c_g * e_t)
    # )
    # c_A = e_t - s * (1 / a_g + 1 / a_ev - e_t) / c_g
    # c_B = e_t - s * (1 / a_p + 1 / a_cd - e_t) / c_p
    alpha_g = (1 / e_g + 1 / e_ev - 1) / c_g
    alpha_p = (1 / e_p + 1 / e_cd - 1) / c_p
    alpha_sg = 1 - s * alpha_g
    alpha_sp = 1 - s * alpha_p
    alpha_s = alpha_g + alpha_p - s * alpha_g * alpha_p

    # return (q0 * (c_eps * q0 + c_A - c_B * t_s) + s * t_s * e_t) / (
    #     c_B * t_s - c_eps * q0
    # )
    return q0 * (alpha_sg / (t * alpha_sp - q0 * alpha_s) - 1) + s * t / (
        t * alpha_sp - q0 * alpha_s
    )


def find_minimum(obj_func, initial_params, config):
    """
    Finds the minimum of the objective function.

    Parameters:
    _obj_func (callable): Objective function to be minimized.
    initial_params (list): Parameters (e_total, c_total, q0, t_s, ...).
    config (str): Configuration string. 'e' for e_total, 'c' for c_total, and 'sa' for sensitivity analysis.

    Returns:
    result (scipy.optimize.OptimizeResult): Result of the minimization.

    Raises:
    ValueError: If the configuration string is not supported.
    """

    if config == "e":
        initial_total = initial_params[2]  # e_total
    elif config == "c":
        initial_total = initial_params[2]  # c_total
    elif config[0] == "sa":
        initial_total = (initial_params[0], initial_params[1])  # (e_total, c_total)
    else:
        raise ValueError(f"Unsupported config: {config}")

    x0, bounds, constraints = create_bounds_and_constraints(config, initial_total)

    # Perform minimization
    result = minimize(
        obj_func,
        x0,
        args=(initial_params, config),
        bounds=bounds,
        constraints=constraints,
        # method='SLSQP',
        tol=1e-16,
    )
    return result


def find_minimum_vectorized(
    obj_func,
    opt_var,
    opt_config,
    guess_bound,
    *params,
    method="SLSQP",
    tol=1e-16,
    warm_start=True,
):
    """
    Finds the minimum of the objective function for a vector of variables.

    Parameters:
    obj_func (callable): Objective function to be minimized.
    opt_var (list): List of values of the variable (e.g., e_total, c_total, q0, t_s, ...).
    opt_config (str): Configuration string. 'e' for e_total, 'c' for c_total, and 'sa' for sensitivity analysis.
    *params (list): Additional parameters (e.g., I, s, ...).

    Returns:
    result (numpy.array): Array of the minimized function values.

    Notes:
    The function uses the previous result as the initial guess for the next iteration.
    """
    x1, x2 = guess_bound  # e*_i (x0, xmin, xmax), c*_i (...)
    results = []
    s_values = []

    for i, var in enumerate(opt_var):
        # initial_params must be (e_total, c_total, q0, t_s, I or s, MULTIPLIER)
        if opt_config[1] == "e":
            initial_params = (var, *params)
        elif opt_config[1] == "c":
            initial_params = (params[0], var, *params[1:])
        elif opt_config[1] == "q":
            initial_params = (*params[:2], var, *params[2:])
        elif opt_config[1] == "t":
            initial_params = (*params[:3], var, *params[3:])
        elif opt_config[1] in ["I", "s"]:
            initial_params = (*params[:4], var, params[-1])

        # Use the previous result as the initial guess
        if (i > 0 and results[i - 1].success) and warm_start:
            x0 = results[i - 1].x  # Update initial guess with the previous result
            # _, bounds, constraints = create_bounds_and_constraints(opt_config, initial_params)
        else:
            # x0, bounds, constraints = create_bounds_and_constraints(opt_config, initial_params)
            x0 = [x1[0]] * 4 + [x2[0]] * 2

        bounds = [(x1[1], x1[2])] * 4 + [(x2[1], x2[2])] * 2
        constraints = [
            {
                "type": "ineq",
                "fun": lambda x: initial_params[0] - sum(x[:4]),
            },  # sum(e_i) <= e_t
            {
                "type": "ineq",
                "fun": lambda x: initial_params[1] - sum(x[4:]),
            },  # sum(c_i) <= c_t
        ]

        if obj_func.__name__ == "objective_function_ep_rate":
            e_total = initial_params[0]
            c_total = initial_params[1]
            q0 = initial_params[2]
            t_s = initial_params[3]
            I = initial_params[4]
            MULTIPLIER = initial_params[-1]

            s = (
                ((I - 1) * q0 / MULTIPLIER)
                / (t_s - (2 * q0 / (MULTIPLIER * c_total)) * (8 / e_total - 1))
                * MULTIPLIER
            )
            initial_params = (e_total, c_total, q0, t_s, s, MULTIPLIER)

            s_values.append(s)

        # Perform minimization
        result = minimize(
            obj_func,
            x0,
            args=(initial_params, opt_config),
            bounds=bounds,
            constraints=constraints,
            method=method,
            tol=tol,
        )

        results.append(result)  # Store the minimized function value

    if obj_func.__name__ == "objective_function_ep_rate":
        return np.array(results), np.array(s_values)
    else:
        return np.array(results)
