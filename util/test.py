import numpy as np
import matplotlib.pyplot as plt
from calc_imposed_q0 import find_minimum, objective_function


# Define initial parameters
init_c_g = 0.2
init_c_p = 0.2
init_eps_total = 2
init_q = 10
init_t_s = 0.9

MULTIPLIER = 10**4
initial_params = [init_c_g, init_c_p, init_eps_total, init_q, init_t_s, MULTIPLIER]



# Perform optimization
optimization_result = find_minimum(initial_params)

# Extract optimized variables
optimized_e_g, optimized_e_p, optimized_e_ev, optimized_e_cd = optimization_result.x
minimum_objective_value = optimization_result.fun * MULTIPLIER


# Display results
print(
    f"- Optimized e* values: e_g = {optimized_e_g:.2f}, e_p = {optimized_e_p:.2f}, "
    f"e_ev = {optimized_e_ev:.2f}, e_cd = {optimized_e_cd:.2f}  --> "
    f"Minimum Objective Value = {minimum_objective_value:.3f}"
)
print(optimization_result)


# Generate data for surface plot
step_size = 0.01
e_g_values = np.arange(0.1, 1.0001, step_size)
e_p_values = np.arange(0.1, 1.0001, step_size)
X, Y = np.meshgrid(e_g_values, e_p_values)
Z = objective_function([X, Y, X, Y], initial_params) * MULTIPLIER


fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
surf = ax.plot_surface(
    X, Y, Z, cmap='viridis', edgecolor='royalblue', lw=0.5, rstride=8, cstride=8, alpha=0.3
)
ax.set_xlabel('e_g')
ax.set_ylabel('e_p')
ax.set_zlabel('Objective Value')
ax.set_title('Objective Function Surface')
plt.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
plt.show()