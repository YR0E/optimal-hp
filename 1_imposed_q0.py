import streamlit as st
import numpy as np
import plotly.graph_objects as go
from util.calc_imposed_q0 import find_minimum, objective_function, objective_function_ir_ratio, objective_function_ep_rate

st.set_page_config(layout='wide', initial_sidebar_state='expanded', page_title='min(w)')


st.write("Welcome to Page 1")

st.markdown("### Navigation:")
st.page_link("page0_home.py", label="Home page", icon=":material/home:")
st.page_link("page2.py", label="Page 2", icon=":material/function:")

st.markdown('***')
st.title("About")
st.write(r"Curzon-Ahlborn : imposed $q_0 - \varepsilon_{total}$: only reversible case: find $\min(w)$")
st.markdown('***')




# Initialize session state for the slider values
MULTIPLIER = 10**4
POWER_OF_10 = np.log10(MULTIPLIER)
DEFAULT_VALUES = {
    'eps_total': 2.0,
    'c_g': 0.2,
    'c_p': 0.2,
    'q0': 10.0,
    't_s': 0.9,
    'I': 1.01,
    's': 0.1
}

for key, value in DEFAULT_VALUES.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Function to reset the sliders
def reset_sliders():
    for key, value in DEFAULT_VALUES.items():
        st.session_state[key] = value



col_control, _, col_plot = st.columns((0.4, 0.01, 0.59))

with col_control:
    # Layout with sliders
    col1, col2 = st.columns((0.15, 0.85))
    col1.write(r'$\varepsilon_{total}:$')
    init_eps_total = col2.slider('e_tot', min_value=0.4, max_value=4.0, step=0.1,
                                format="%.1f", label_visibility="collapsed", key="eps_total")

    col1, col2 = st.columns((0.15, 0.85))
    col1.write('$c_{g}:$')
    init_c_g = col2.slider('c_g', min_value=0.05, max_value=0.5, step=0.01,
                        format="%.2f", label_visibility="collapsed", key="c_g")

    col1, col2 = st.columns((0.15, 0.85))
    col1.write('$c_{p}:$')
    init_c_p = col2.slider('c_p', min_value=0.05, max_value=0.5, step=0.01,
                        format="%.2f", label_visibility="collapsed", key="c_p")

    col1, col2 = st.columns((0.15, 0.85))
    col1.markdown('$q_{0}:$', help=fr'$q_{{0}} \times 10^{{-{POWER_OF_10:.0f}}}$')
    init_q = col2.slider('q0', min_value=1.0, max_value=50.0, step=0.1,
                        format="%.1f", label_visibility="collapsed", key="q0")

    col1, col2 = st.columns((0.15, 0.85))
    col1.write('$t_{s}:$')
    init_t_s = col2.slider('t_s', min_value=0.8, max_value=1.0, step=0.01,
                        format="%.2f", label_visibility="collapsed", key="t_s")

    col1, col2 = st.columns((0.15, 0.85))
    col1.write('$I:$')
    init_I = col2.slider('I', min_value=1.0, max_value=2.001, step=0.01,
                        format="%.2f", label_visibility="collapsed", key="I")

    col1, col2 = st.columns((0.15, 0.85))
    col1.markdown('$\dot{s}:$', help=fr'$\dot{{s}} \times 10^{{-{POWER_OF_10:.0f}}}$')
    init_s = col2.slider('s', min_value=0.1, max_value=30.0, step=0.01,
                        format="%.2f", label_visibility="collapsed", key="s")

    # Add a reset button with a callback
    st.button("Reset", on_click=reset_sliders)



initial_params = [init_c_g, init_c_p, init_eps_total, init_q, init_t_s, MULTIPLIER]
initial_params_ir_ratio = [init_c_g, init_c_p, init_eps_total, init_q, init_t_s, init_I, MULTIPLIER]
initial_params_ep_rate = [init_c_g, init_c_p, init_eps_total, init_q, init_t_s, init_s, MULTIPLIER]


# Perform optimization
optimization_result = find_minimum(objective_function, initial_params)
optimization_result_ir_ratio = find_minimum(objective_function_ir_ratio, initial_params_ir_ratio)
optimization_result_ep_rate = find_minimum(objective_function_ep_rate, initial_params_ep_rate)

# Extract optimized variables
optimized_e_g, optimized_e_p, optimized_e_ev, optimized_e_cd = optimization_result.x
minimum_objective_value = optimization_result.fun * MULTIPLIER



# Generate data for surface plot
step_size = 0.01
e_g_values = np.arange(0.1, 1.0001, step_size)
e_p_values = np.arange(0.1, 1.0001, step_size)
X, Y = np.meshgrid(e_g_values, e_p_values)
Z = objective_function([X, Y, X, Y], initial_params) * MULTIPLIER
Z_ir = objective_function_ir_ratio([X, Y, X, Y], initial_params_ir_ratio) * MULTIPLIER
Z_ep = objective_function_ep_rate([X, Y, X, Y], initial_params_ep_rate) * MULTIPLIER



if init_eps_total/2 <= 1.1: 
    X_line = np.linspace(0.1, init_eps_total/2 - 0.1, 21)
else: X_line = np.linspace(init_eps_total/2 - 1, 1, 21)

min_line_lim = min(np.min(Z), np.min(Z_ir), np.min(Z_ep))
max_line_lim = max(np.max(Z), np.max(Z_ir), np.max(Z_ep))

X_line, Z_line = np.meshgrid(X_line, np.linspace(min_line_lim, max_line_lim, 21))
Y_line = init_eps_total/2 - X_line



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


fig = go.Figure()

fig.add_trace(go.Surface(
    z=Z, x=X, y=Y, 
    name='reversibility', legendgroup='reversibility', 
    colorscale='Viridis', showlegend=True, showscale=False, opacity=0.75,
    contours=dict(
        x=dict(
            show=True,
            usecolormap=True,
            width=2,
            highlight=True,
            highlightcolor="gray",
            highlightwidth=5
        ),
        y=dict(
            show=True,
            usecolormap=True,
            width=2,
            highlight=True,
            highlightcolor="gray",
            highlightwidth=2
        )
    )
))
fig.add_trace(go.Surface(
    z=Z_ir, x=X, y=Y, 
    name='irreversibility ratio', legendgroup='irreversibility',
    colorscale='RdBu_r', showlegend=True, showscale=False, opacity=0.75,
    contours=dict(
        x=dict(
            show=True,
            usecolormap=True,
            width=2,
            highlight=True,
            highlightcolor="gray",
            highlightwidth=5
        ),
        y=dict(
            show=True,
            usecolormap=True,
            width=2,
            highlight=True,
            highlightcolor="gray",
            highlightwidth=2
        )
    )
))
fig.add_trace(go.Surface(
    z=Z_ep, x=X, y=Y, 
    name='entropy production rate', legendgroup='entropy production', 
    colorscale='rdylgn_r', showlegend=True, showscale=False, opacity=0.75,
    contours=dict(
        x=dict(
            show=True,
            usecolormap=True,
            width=2,
            highlight=True,
            highlightcolor="gray",
            highlightwidth=5
        ),
        y=dict(
            show=True,
            usecolormap=True,
            width=2,
            highlight=True,
            highlightcolor="gray",
            highlightwidth=2
        )
    )
))
fig.add_trace(go.Surface(
    z=Z_line, x=X_line, y=Y_line, 
    name='constraint', legendgroup='constraint', 
    colorscale=[[0, 'red'], [1, 'red']], showlegend=True, showscale=False, opacity=0.15,
))
fig.add_trace(go.Scatter3d(
    x=[optimization_result.x[0]], y=[optimization_result.x[1]], z=[optimization_result.fun * MULTIPLIER],
    mode='markers',
    name='minimum', legendgroup='minimum', showlegend=True,
    marker=dict(size=5, color='red', symbol='circle'),
))
fig.add_trace(go.Scatter3d(
    x=[optimization_result_ir_ratio.x[0]], y=[optimization_result_ir_ratio.x[1]], z=[optimization_result_ir_ratio.fun * MULTIPLIER],
    mode='markers',
    name='minimum', legendgroup='minimum', showlegend=False,
    marker=dict(size=5, color='red', symbol='circle'),
))
fig.add_trace(go.Scatter3d(
    x=[optimization_result_ep_rate.x[0]], y=[optimization_result_ep_rate.x[1]], z=[optimization_result_ep_rate.fun * MULTIPLIER],
    mode='markers',
    name='minimum', legendgroup='minimum', showlegend=False,
    marker=dict(size=5, color='red', symbol='circle'),
))

fig.update_layout(
    title=dict(text='Objective Function Surface Plot'), 
    autosize=False,
    width=700, height=540,
    margin=dict(l=10, r=10, b=20, t=40),
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
    uirevision='constant',
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

    st.write(
        r'Optimized $\varepsilon^*$ values: $\varepsilon_g = $', optimized_e_g.round(2), r', $\varepsilon_p = $', optimized_e_p.round(2),
        r', $\varepsilon_{ev} = $', optimized_e_ev.round(2), r', $\varepsilon_{cd} = $', optimized_e_cd.round(2))
    
    st.write('Minimum $w = $', minimum_objective_value.round(3), fr' $ \times 10^{{-{POWER_OF_10:.0f}}}$')

st.write('Optimization Result:', optimization_result)
st.write('Optimization Result of irreversibility ratio:', optimization_result_ir_ratio)
st.write('Optimization Result of entropy production rate:', optimization_result_ep_rate)
