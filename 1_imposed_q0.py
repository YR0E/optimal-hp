import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from util.calc_imposed_q0 import find_minimum, objective_function, objective_function_ir_ratio, objective_function_ep_rate

st.set_page_config(layout='wide', initial_sidebar_state='expanded', page_title='min(w)')


st.write("Welcome to Page 1")
st.warning("Work in progress...")
st.markdown('***')

st.markdown("### Navigation:")
st.page_link("0_home.py", label="Home page", icon=":material/home:")
st.page_link("2_imposed_w0.py", label="Imposed w0: Find maximum heat extraction max(q)", icon=":material/function:")
st.page_link("3_imposed_COP0.py", label="Imposed COP0: Find maximum heat extraction max(q)", icon=":material/function:")

st.markdown('***')
st.title("About")




# Initialize session state for the slider values
MULTIPLIER = 10**4
POWER_OF_10 = np.log10(MULTIPLIER)
DEFAULT_VALUES_EPS = {
    'eps_total': 2.0,
    'c_g': 0.2,
    'c_p': 0.2,
    'q0': 10.0,
    't_s': 0.9,
    'I': 1.01,
    's': 0.1
}
DEFAULT_VALUES_C = {
    'e_g': 0.5,
    'e_p': 0.5,
    'c_total': 0.8,
    'q0_c': 10.0,
    't_s_c': 0.9,
    'I_c': 1.01,
    's_c': 0.1
}

for key, value in DEFAULT_VALUES_EPS.items():
    if key not in st.session_state:
        st.session_state[key] = value
for key, value in DEFAULT_VALUES_C.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Function to reset the sliders
def reset_sliders(default):
    for key, value in default.items():
        st.session_state[key] = value

def init_slider(varname, key, minval, maxval, step, fmt="%.2f", help=None):
    col1, col2 = st.columns((0.15, 0.85))
    col1.markdown(varname, help=help)
    init_val = col2.slider(label=key, label_visibility="collapsed", key=key,
                           min_value=minval, max_value=maxval, step=step, format=fmt)
    return init_val


st.info(r'Choose a variable $(\varepsilon_{total}$ or $c_{total})$ to minimize $f(w)$')
tab_eps_total, tab_c_total = st.tabs([r'$\varepsilon_{total}$', '$c_{total}$'])

#===========EPS_TOTAL================
with tab_eps_total:
    st.write(r"Curzon-Ahlborn model: imposed $q_0 - \varepsilon_{total}$: find minimum power consumption $\min(w)$")

    col_control, _, col_plot = st.columns((0.34, 0.02, 0.64))

    #========SLIDERS========
    with col_control:
    # with col_control.form("sliders", border=False):
        init_eps_total = init_slider(r'$\varepsilon_{total}:$', 'eps_total', 
                                     0.4, 4.0, 0.1, fmt="%.1f")
        init_c_g = init_slider('$c_{g}:$', 'c_g', 0.05, 0.5, 0.01)
        init_c_p = init_slider('$c_{p}:$', 'c_p', 0.05, 0.5, 0.01)
        init_q = init_slider('$q_{0}:$', 'q0', 1.0, 50.0, 0.1, fmt="%.1f",
                             help=fr'$q_{{0}} \times 10^{{-{POWER_OF_10:.0f}}}$')
        init_t_s = init_slider('$t_{s}:$', 't_s', 0.8, 1.0, 0.01)
        init_I = init_slider('$I:$', 'I', 1.0, 3.0, 0.01)
        init_s = init_slider('$s:$', 's', 0.1, 30.0, 0.01, 
                             help=fr'$s \times 10^{{-{POWER_OF_10:.0f}}}$')

        # Submit and Reset Buttons
        # col_btn1, col_btn2, _ = st.columns(3)
        # submit = col_btn1.form_submit_button("Submit")
        # reset = col_btn2.form_submit_button("Reset", on_click=reset_sliders)
        st.button("Reset", on_click=lambda: reset_sliders(DEFAULT_VALUES_EPS))


    initial_params = [init_c_g, init_c_p, init_eps_total, init_q, init_t_s, MULTIPLIER]
    initial_params_ir_ratio = [init_c_g, init_c_p, init_eps_total, init_q, init_t_s, init_I, MULTIPLIER]
    initial_params_ep_rate = [init_c_g, init_c_p, init_eps_total, init_q, init_t_s, init_s, MULTIPLIER]


    # Perform optimization
    opt_var = 'e'
    res = find_minimum(objective_function, initial_params, opt_var)
    res_ir_ratio = find_minimum(objective_function_ir_ratio, initial_params_ir_ratio, opt_var)
    res_ep_rate = find_minimum(objective_function_ep_rate, initial_params_ep_rate, opt_var)


    df = pd.DataFrame({
        'I': [np.NaN, init_I, np.NaN],
        f's [e-{POWER_OF_10:.0f}]': [np.NaN, np.NaN, init_s],
        'e*_g': [res.x[0], res_ir_ratio.x[0], res_ep_rate.x[0]],
        'e*_p': [res.x[1], res_ir_ratio.x[1], res_ep_rate.x[1]],
        'e*_ev': [res.x[2], res_ir_ratio.x[2], res_ep_rate.x[2]],
        'e*_cd': [res.x[3], res_ir_ratio.x[3], res_ep_rate.x[3]],
        f'min(w) [e-{POWER_OF_10:.0f}]': np.array([res.fun, res_ir_ratio.fun, res_ep_rate.fun]) * MULTIPLIER
        }, 
        index=['reversibility', 'irreversibility ratio', 'entropy production rate']
    )



    #=======PLOT=======
    # Generate data for surface plot
    step_size = 0.01
    e_g_values = np.arange(0.1, 1.0001, step_size)
    e_p_values = np.arange(0.1, 1.0001, step_size)
    X, Y = np.meshgrid(e_g_values, e_p_values)
    Z = objective_function([X, Y, X, Y], initial_params, opt_var) * MULTIPLIER
    Z_ir = objective_function_ir_ratio([X, Y, X, Y], initial_params_ir_ratio, opt_var) * MULTIPLIER
    Z_ep = objective_function_ep_rate([X, Y, X, Y], initial_params_ep_rate, opt_var) * MULTIPLIER

    # contraint line
    if init_eps_total/2 <= 1.1: 
        X_line = np.linspace(0.1, init_eps_total/2 - 0.1, 21)
    else: X_line = np.linspace(init_eps_total/2 - 1, 1, 21)

    min_line_lim = min(np.min(Z), np.min(Z_ir), np.min(Z_ep))
    max_line_lim = max(np.max(Z), np.max(Z_ir), np.max(Z_ep))

    X_line, Z_line = np.meshgrid(X_line, np.linspace(min_line_lim, max_line_lim, 21))
    Y_line = init_eps_total/2 - X_line

    # minimum points
    x_r, y_r, z_r = res.x[0], res.x[1], res.fun * MULTIPLIER
    x_ir, y_ir, z_ir = res_ir_ratio.x[0], res_ir_ratio.x[1], res_ir_ratio.fun * MULTIPLIER
    x_ep, y_ep, z_ep = res_ep_rate.x[0], res_ep_rate.x[1], res_ep_rate.fun * MULTIPLIER


    # Set the configuration for the Plotly chart, including the resolution settings
    config = {
        "toImageButtonOptions": {
            "format": "png",  # The format of the exported image (png, svg, etc.)
            "filename": "surface_plot",  # Default filename
            # "height": 1080,  # Image height
            # "width": 1920,   # Image width
            "scale": 3       # Increase the resolution (scales up the image)
        }
    }
    contours = dict(
        x=dict(
            show=True,
            usecolormap=True,
            highlight=True,
            highlightcolor="white",
        ),
        y=dict(
            show=True,
            usecolormap=True,
            highlight=True,
            highlightcolor="white",
        )
    )

    fig = go.Figure()

    fig.add_trace(go.Surface(
        z=Z, x=X, y=Y, 
        name='reversibility', legendgroup='reversibility', 
        colorscale='Viridis', showlegend=True, showscale=False, opacity=0.75,
        contours=contours
    ))
    fig.add_trace(go.Surface(
        z=Z_ir, x=X, y=Y, 
        name='irreversibility ratio', legendgroup='irreversibility',
        colorscale='RdBu_r', showlegend=True, showscale=False, opacity=0.75,
        contours=contours
    ))
    fig.add_trace(go.Surface(
        z=Z_ep, x=X, y=Y, 
        name='entropy production rate', legendgroup='entropy production', 
        colorscale='rdylgn_r', showlegend=True, showscale=False, opacity=0.75,
        contours=contours
    ))
    fig.add_trace(go.Surface(
        z=Z_line, x=X_line, y=Y_line, 
        name='constraint', legendgroup='constraint', 
        colorscale=[[0, 'red'], [1, 'red']], showlegend=True, showscale=False, opacity=0.1,
    ))
    fig.add_trace(go.Scatter3d(
        x=[x_r], y=[y_r], z=[z_r],
        mode='markers',
        name='minimum', legendgroup='minimum', showlegend=True,
        marker=dict(size=5, color='red', symbol='circle'),
    ))
    fig.add_trace(go.Scatter3d(
        x=[x_ir], y=[y_ir], z=[z_ir],
        mode='markers',
        name='minimum', legendgroup='minimum', showlegend=False,
        marker=dict(size=5, color='red', symbol='circle'),
    ))
    fig.add_trace(go.Scatter3d(
        x=[x_ep], y=[y_ep], z=[z_ep],
        mode='markers',
        name='minimum', legendgroup='minimum', showlegend=False,
        marker=dict(size=5, color='red', symbol='circle'),
    ))

    fig.update_layout(
        title=dict(text='3D Plot'), 
        autosize=False,
        width=600, height=500,
        margin=dict(l=10, r=10, b=10, t=40),
        scene=dict(
            xaxis_title='<i>ε<sub>g</sub></i>',
            yaxis_title='<i>ε<sub>p</sub></i>',
            zaxis_title=f'<i>w · 10<sup>−{POWER_OF_10:.0f}</sup></i>',
            xaxis_title_font=dict(family='STIX Two Math'),
            yaxis_title_font=dict(family='STIX Two Math'),
            zaxis_title_font=dict(family='STIX Two Math'),
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                eye=dict(x=2, y=1, z=0.5)
            )
        ),
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="center",
            x=0.5,
            orientation="h"
        )
    )


    with col_plot:
        st.plotly_chart(fig, use_container_width=True, config=config)

    st.write('#### Results:')
    st.dataframe(df)




#===========C_TOTAL================
with tab_c_total:
    st.write("Curzon-Ahlborn model: imposed $q_0 - c_{total}$: find minimum power consumption $\min(w)$")

    col_control, _, col_plot = st.columns((0.34, 0.02, 0.64))

    #========SLIDERS========
    with col_control:
        init_c_total = init_slider('$c_{total}:$', 'c_total', 0.1, 1.0, 0.01)
        init_e_g = init_slider(r'$\varepsilon_{g}:$', 'e_g', 0.1, 1.0, 0.01)
        init_e_p = init_slider(r'$\varepsilon_{p}:$', 'e_p', 0.1, 1.0, 0.01)
        init_q = init_slider('$q_{0}:$', 'q0_c', 1.0, 50.0, 0.1, fmt="%.1f",
                             help=fr'$q_{{0}} \times 10^{{-{POWER_OF_10:.0f}}}$')
        init_t_s = init_slider('$t_{s}:$', 't_s_c', 0.8, 1.0, 0.01)
        init_I = init_slider('$I:$', 'I_c', 1.0, 3.0, 0.01)
        init_s = init_slider('$s:$', 's_c', 0.1, 30.0, 0.01, 
                             help=fr'$s \times 10^{{-{POWER_OF_10:.0f}}}$')

        st.button("Reset", on_click=lambda: reset_sliders(DEFAULT_VALUES_C), key='btn_c')


    initial_params_c = [init_e_g, init_e_p, init_c_total, init_q, init_t_s, MULTIPLIER]
    initial_params_ir_c = [init_e_g, init_e_p, init_c_total, init_q, init_t_s, init_I, MULTIPLIER]
    initial_params_ep_c = [init_e_g, init_e_p, init_c_total, init_q, init_t_s, init_s, MULTIPLIER]


    # Perform optimization
    opt_var_c = 'c'
    res_c = find_minimum(objective_function, initial_params_c, opt_var_c)
    res_ir_c = find_minimum(objective_function_ir_ratio, initial_params_ir_c, opt_var_c)
    res_ep_c = find_minimum(objective_function_ep_rate, initial_params_ep_c, opt_var_c)


    df = pd.DataFrame({
        'I': [np.NaN, init_I, np.NaN],
        f's [e-{POWER_OF_10:.0f}]': [np.NaN, np.NaN, init_s],
        'c*_g': [res_c.x[0], res_ir_c.x[0], res_ep_c.x[0]],
        'c*_p': [res_c.x[1], res_ir_c.x[1], res_ep_c.x[1]],
        f'min(w) [e-{POWER_OF_10:.0f}]': np.array([res_c.fun, res_ir_c.fun, res_ep_c.fun]) * MULTIPLIER
        }, 
        index=['reversibility', 'irreversibility ratio', 'entropy production rate']
    )



    # #=======PLOT=======
    # # Generate data for surface plot
    step_size = 0.005
    c_g_values = np.arange(0.05, 0.50001, step_size)
    c_p_values = np.arange(0.05, 0.50001, step_size)
    X, Y = np.meshgrid(c_g_values, c_p_values)
    Z = objective_function([X, Y], initial_params_c, opt_var_c) * MULTIPLIER
    Z_ir = objective_function_ir_ratio([X, Y], initial_params_ir_c, opt_var_c) * MULTIPLIER
    Z_ep = objective_function_ep_rate([X, Y], initial_params_ep_c, opt_var_c) * MULTIPLIER

    # contraint line
    if init_c_total <= 0.55: 
        X_line = np.linspace(0.05, init_c_total - 0.05, 21)
    else: X_line = np.linspace(init_c_total - 0.5, 0.5, 21)

    min_line_lim = min(np.min(Z), np.min(Z_ir), np.min(Z_ep))
    max_line_lim = max(np.max(Z), np.max(Z_ir), np.max(Z_ep))

    X_line, Z_line = np.meshgrid(X_line, np.linspace(min_line_lim, max_line_lim, 21))
    Y_line = init_c_total - X_line

    # minimum points
    x_r, y_r, z_r = res_c.x[0], res_c.x[1], res_c.fun * MULTIPLIER
    x_ir, y_ir, z_ir = res_ir_c.x[0], res_ir_c.x[1], res_ir_c.fun * MULTIPLIER
    x_ep, y_ep, z_ep = res_ep_c.x[0], res_ep_c.x[1], res_ep_c.fun * MULTIPLIER


    # Set the configuration for the Plotly chart, including the resolution settings
    config = {
        "toImageButtonOptions": {
            "format": "png",  # The format of the exported image (png, svg, etc.)
            "filename": "surface_plot",  # Default filename
            # "height": 1080,  # Image height
            # "width": 1920,   # Image width
            "scale": 3       # Increase the resolution (scales up the image)
        }
    }
    contours = dict(
        x=dict(
            show=True,
            usecolormap=True,
            highlight=True,
            highlightcolor="white",
        ),
        y=dict(
            show=True,
            usecolormap=True,
            highlight=True,
            highlightcolor="white",
        )
    )

    fig = go.Figure()

    fig.add_trace(go.Surface(
        z=Z, x=X, y=Y, 
        name='reversibility', legendgroup='reversibility', 
        colorscale='Viridis', showlegend=True, showscale=False, opacity=0.75,
        contours=contours
    ))
    fig.add_trace(go.Surface(
        z=Z_ir, x=X, y=Y, 
        name='irreversibility ratio', legendgroup='irreversibility',
        colorscale='RdBu_r', showlegend=True, showscale=False, opacity=0.75,
        contours=contours
    ))
    fig.add_trace(go.Surface(
        z=Z_ep, x=X, y=Y, 
        name='entropy production rate', legendgroup='entropy production', 
        colorscale='rdylgn_r', showlegend=True, showscale=False, opacity=0.75,
        contours=contours
    ))
    fig.add_trace(go.Surface(
        z=Z_line, x=X_line, y=Y_line, 
        name='constraint', legendgroup='constraint', 
        colorscale=[[0, 'red'], [1, 'red']], showlegend=True, showscale=False, opacity=0.1,
    ))
    fig.add_trace(go.Scatter3d(
        x=[x_r], y=[y_r], z=[z_r],
        mode='markers',
        name='minimum', legendgroup='minimum', showlegend=True,
        marker=dict(size=5, color='red', symbol='circle'),
    ))
    fig.add_trace(go.Scatter3d(
        x=[x_ir], y=[y_ir], z=[z_ir],
        mode='markers',
        name='minimum', legendgroup='minimum', showlegend=False,
        marker=dict(size=5, color='red', symbol='circle'),
    ))
    fig.add_trace(go.Scatter3d(
        x=[x_ep], y=[y_ep], z=[z_ep],
        mode='markers',
        name='minimum', legendgroup='minimum', showlegend=False,
        marker=dict(size=5, color='red', symbol='circle'),
    ))

    fig.update_layout(
        title=dict(text='3D Plot'), 
        autosize=False,
        width=600, height=500,
        margin=dict(l=10, r=10, b=10, t=40),
        scene=dict(
            xaxis_title='<i>c<sub>g</sub></i>',
            yaxis_title='<i>c<sub>p</sub></i>',
            zaxis_title=f'<i>w · 10<sup>−{POWER_OF_10:.0f}</sup></i>',
            xaxis_title_font=dict(family='STIX Two Math'),
            yaxis_title_font=dict(family='STIX Two Math'),
            zaxis_title_font=dict(family='STIX Two Math'),
            aspectratio=dict(x=1, y=1, z=1),
            camera=dict(
                eye=dict(x=2, y=1, z=0.5)
            )
        ),
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="center",
            x=0.5,
            orientation="h"
        )
    )


    with col_plot:
        st.plotly_chart(fig, use_container_width=True, config=config, key='plotly_c')

    st.write('#### Results:')
    st.dataframe(df)
