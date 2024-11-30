import numpy as np
from scipy.optimize import minimize

# Curzon-Ahlborn : imposed q0 - e_total
# with all sliders and chechbox

def obj(x):
    if len(x)>5: e_g, e_p, e_ev, e_cd, c_g, c_p = x
    else:
        e_g, e_p, e_ev, e_cd = x
        c_g, c_p = init_c_g, init_c_p
    
    e_t = init_eps_total
    q0 = init_q*10**(-4)
    t_s = init_t_s
    a_g, a_ev, a_p, a_cd = e_g/e_t, e_ev/e_t, e_p/e_t, e_cd/e_t
    c_eps = (1/a_g + 1/a_ev - e_t)/c_g + (1/a_p + 1/a_cd - e_t)/c_p
    
    return q0 * (c_eps*q0 + e_t*(1-t_s)) / (e_t*t_s - c_eps*q0)


def obj_i(x):
    if len(x)>5: e_g, e_p, e_ev, e_cd, c_g, c_p = x
    else:
        e_g, e_p, e_ev, e_cd = x
        c_g, c_p = init_c_g, init_c_p
    
    e_t = init_eps_total
    q0 = init_q*10**(-4)
    t_s = init_t_s
    I = init_I
    a_g, a_ev, a_p, a_cd = e_g/e_t, e_ev/e_t, e_p/e_t, e_cd/e_t
    c_eps = (1/a_g + 1/a_ev - e_t)/(c_g*I) + (1/a_p + 1/a_cd - e_t)/c_p
    
    return q0 * (I*c_eps*q0 + e_t*(I-t_s)) / (e_t*t_s - I*c_eps*q0)
 

def obj_s(x):
    if len(x)>5: e_g, e_p, e_ev, e_cd, c_g, c_p = x
    else:
        e_g, e_p, e_ev, e_cd = x
        c_g, c_p = init_c_g, init_c_p
    
    e_t = init_eps_total
    q0 = init_q*10**(-4)
    t_s = init_t_s
    s = init_s*10**(-4)
    a_g, a_ev, a_p, a_cd = e_g/e_t, e_ev/e_t, e_p/e_t, e_cd/e_t
    c_eps = (1/a_g + 1/a_ev - e_t)/c_g + (1/a_p + 1/a_cd - e_t)/c_p - s*(1/a_g + 1/a_ev - e_t)*(1/a_p + 1/a_cd - e_t)/(c_p*c_g*e_t)
    c_A = e_t - s*(1/a_g + 1/a_ev - e_t)/c_g
    c_B = e_t - s*(1/a_p + 1/a_cd - e_t)/c_p

    return (q0 * (c_eps*q0 + c_A - c_B*t_s) + s*t_s*e_t) / (c_B*t_s - c_eps*q0)
 

def constraint1(x):
    e_g, e_p, e_ev, e_cd = x
    return init_eps_total - e_g - e_p - e_ev - e_cd


x = np.arange(0.1, 1.0001, 0.01)    # e_g
y = np.arange(0.1, 1.0001, 0.01)    # e_p
X, Y = np.meshgrid(x, y)

# Define initial parameters
init_c_g = 0.2
init_c_p = 0.2
init_eps_total = 2
init_q = 10
init_t_s = 0.9
init_I = 1.01
init_s = 0.1

# constraints
x0 = [0.5, 0.5, 0.5, 0.5]
b = (0.1, 1)
bnds = (b, b, b, b)

con1 = {'type': 'ineq', 'fun': constraint1}
cons = [con1]


result = minimize(obj, x0, bounds=bnds, constraints=cons, tol=10**(-16))
min_x, min_y, min_x2, min_y2 = np.meshgrid(result.x[0], result.x[1], result.x[2], result.x[3])
min_z = obj(np.stack([min_x, min_y, min_x2, min_y2]))*10**4

result_i = minimize(obj_i, x0, bounds=bnds, constraints=cons, tol=10**(-16))
min_x_i, min_y_i, min_x2_i, min_y2_i = np.meshgrid(result_i.x[0], result_i.x[1], result_i.x[2], result_i.x[3])
min_z_i = obj_i(np.stack([min_x_i, min_y_i, min_x2_i, min_y2_i]))*10**4

result_s = minimize(obj_s, x0, bounds=bnds, constraints=cons, tol=10**(-16))
min_x_s, min_y_s, min_x2_s, min_y2_s = np.meshgrid(result_s.x[0], result_s.x[1], result_s.x[2], result_s.x[3])
min_z_s = obj_s(np.stack([min_x_s, min_y_s, min_x2_s, min_y2_s]))*10**4

print(f'-  e* = {min_x.item():0.2f}, {min_y.item():.2f}, {min_x2.item():0.2f}, {min_y2.item():.2f}  --> min = {min_z.item():.3f}')
print(f'\t{min_x_i.item():.2f}, {min_y_i.item():.2f}, {min_x2_i.item():.2f}, {min_y2_i.item():.2f} \t\t  {min_z_i.item():.3f}')
print(f'\t{min_x_s.item():.2f}, {min_y_s.item():.2f}, {min_x2_s.item():.2f}, {min_y2_s.item():.2f} \t\t  {min_z_s.item():.3f}')

