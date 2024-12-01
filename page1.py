import streamlit as st
import numpy as np
import plotly.graph_objects as go
from util.calc_imposed_q0 import find_minimum, objective_function


st.write("Welcome to Page 1")

st.page_link("page0_home.py", label="Home page")
st.page_link("page2.py", label="Page 2")

st.markdown('***')
st.title("About")
st.write("Curzon-Ahlborn : imposed $q_0$ - e_total: only reversible case: find $\min(w)$")
st.markdown('***')




# Initialize session state for the slider values
default_values = {
    'eps_total': 2.0,
    'c_g': 0.2,
    'c_p': 0.2,
    'q0': 10.0,
    't_s': 0.9
}

for key, value in default_values.items():
    if key not in st.session_state:
        st.session_state[key] = value

# Function to reset the sliders
def reset_sliders():
    for key, value in default_values.items():
        st.session_state[key] = value

# Layout with sliders
col1, col2 = st.columns((0.15, 0.85))
col1.write(r'$\varepsilon_{total}:$')
init_eps_total = col2.slider('e_tot', min_value=0.1, max_value=4.0, step=0.1,
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
col1.write(r'$q_{0} \times 10^{-4}:$')
init_q = col2.slider('q0', min_value=1.0, max_value=50.0, step=0.1,
                     format="%.1f", label_visibility="collapsed", key="q0")

col1, col2 = st.columns((0.15, 0.85))
col1.write('$t_{s}:$')
init_t_s = col2.slider('t_s', min_value=0.8, max_value=1.0, step=0.01,
                       format="%.2f", label_visibility="collapsed", key="t_s")

# Add a reset button with a callback
st.button("Reset", on_click=reset_sliders)



MULTIPLIER = 10**4
initial_params = [init_c_g, init_c_p, init_eps_total, init_q, init_t_s, MULTIPLIER]


# Perform optimization
optimization_result = find_minimum(initial_params)

# Extract optimized variables
optimized_e_g, optimized_e_p, optimized_e_ev, optimized_e_cd = optimization_result.x
minimum_objective_value = optimization_result.fun * MULTIPLIER



# Generate data for surface plot
step_size = 0.01
e_g_values = np.arange(0.1, 1.0001, step_size)
e_p_values = np.arange(0.1, 1.0001, step_size)
X, Y = np.meshgrid(e_g_values, e_p_values)
Z = objective_function([X, Y, X, Y], initial_params) * MULTIPLIER


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


fig = go.Figure(data=[go.Surface(z=Z, x=X, y=Y)])
fig.update_layout(title=dict(text='Objective Function Surface Plot'), autosize=False,
                  width=500, height=500,
                  margin=dict(l=65, r=50, b=65, t=90),
                  scene=dict(
                      xaxis_title='<i>ε<sub>g</sub></i>',
                      yaxis_title='<i>ε<sub>p</sub></i>',
                      zaxis_title='<i>w · 10<sup>−4</sup></i>',
                      xaxis_title_font=dict(family='STIX Two Math'),
                      yaxis_title_font=dict(family='STIX Two Math'),
                      zaxis_title_font=dict(family='STIX Two Math'),
                      aspectratio=dict(x=1, y=1, z=1)
                      ),
                  )

st.plotly_chart(fig, use_container_width=True, config=config)


st.write(
    r'Optimized $\varepsilon^*$ values: $\varepsilon_g = $', optimized_e_g.round(2), r', $\varepsilon_p = $', optimized_e_p.round(2),
    r', $\varepsilon_{ev} = $', optimized_e_ev.round(2), r', $\varepsilon_{cd} = $', optimized_e_cd.round(2))
    
st.write('Minimum $w = $', minimum_objective_value.round(3), r' $ \times 10^{-4}$')

st.write('Optimization Result:', optimization_result)