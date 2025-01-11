import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from streamlit_theme import st_theme
from util.calc_imposed_q0 import find_minimum, find_minimum_vectorized
from util.calc_imposed_q0 import objective_function, objective_function_ir_ratio, objective_function_ep_rate
from util.plot import plotting3D, plotting_sensitivity
from util.navigation import link_to_pages


st.set_page_config(layout='wide', initial_sidebar_state='expanded', page_title='min(w)')
theme = st_theme()
if theme is not None and theme['base']=='dark':
    pio.templates.default = "plotly_dark"
    theme_session = 'streamlit'      # to handle some streamlit issue with rendering plotly template
else:
    pio.templates.default = "plotly"
    theme_session = None


st.write("Welcome to Page 1")
st.warning("Work in progress...")
st.markdown('***')

st.markdown("### Navigation:")
link_to_pages(pages=[0, 2, 3])


#==========PREP for MAIN CALCULATION===========
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

def init_session_state(default):
    """
    Initializes the session state with default values if they are not already set.

    Parameters:
    - default (dict): A dictionary of default values, where each key is the name 
      of the session state variable, and each value is the default value to set.
    """

    for key, value in default.items():
        if key not in st.session_state:
            st.session_state[key] = value

def reset_sliders(default):
    """
    Resets all slider values to their default values.

    Parameters:
    - default (dict): A dictionary of default values, keyed by the session state key.
    """
    
    for key, value in default.items():
        st.session_state[key] = value

def init_slider(varname, key, minval, maxval, step, fmt="%.2f", help=None):
    """
    Initializes a slider.

    Parameters:
    - varname (str): The display name for the slider variable.
    - key (str): The unique key for the slider in the Streamlit session state.
    - minval (float): The minimum value the slider can take.
    - maxval (float): The maximum value the slider can take.
    - step (float): The step size for slider increments.
    - fmt (str, optional): The format string for displaying slider values. Default is "%.2f".
    - help (str, optional): A tooltip that provides additional information about the slider.

    Returns:
    - slider_val: The value of the slider.
    """

    col1, col2 = st.columns((0.15, 0.85))
    col1.markdown(varname, help=help)
    slider_val = col2.slider(label=key, label_visibility="collapsed", key=key,
                           min_value=minval, max_value=maxval, step=step, format=fmt)
    return slider_val

@st.fragment
def tab_eps_total_plane():
    init_session_state(DEFAULT_VALUES_EPS)

    #=====INFO=====
    st.write(r"Curzon-Ahlborn model: imposed $q_0 - \varepsilon_{total}$: find minimum power consumption $\min(w)$")


    col_control, _, col_plot = st.columns((0.34, 0.02, 0.64))

    #=====SLIDERS=====
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


    # Perform optimization
    initial_params = {
        'r': [init_c_g, init_c_p, init_eps_total, init_q, init_t_s, MULTIPLIER],
        'ir': [init_c_g, init_c_p, init_eps_total, init_q, init_t_s, init_I, MULTIPLIER],
        'ep': [init_c_g, init_c_p, init_eps_total, init_q, init_t_s, init_s, MULTIPLIER]
    }
    opt_var = 'e'
    results = {
        'r': find_minimum(objective_function, initial_params['r'], opt_var),
        'ir': find_minimum(objective_function_ir_ratio, initial_params['ir'], opt_var),
        'ep': find_minimum(objective_function_ep_rate, initial_params['ep'], opt_var)
    }

    #=====PLOT=====
    with col_plot:
        plotting3D(results, initial_params, opt_var)

    #=====RESULT INFO=====
    st.write('#### Results:')
    df = pd.DataFrame({
        'I': [np.NaN, init_I, np.NaN],
        f's [e-{POWER_OF_10:.0f}]': [np.NaN, np.NaN, init_s],
        'e*_g': [results['r'].x[0], results['ir'].x[0], results['ep'].x[0]],
        'e*_p': [results['r'].x[1], results['ir'].x[1], results['ep'].x[1]],
        'e*_ev': [results['r'].x[2], results['ir'].x[2], results['ep'].x[2]],
        'e*_cd': [results['r'].x[3], results['ir'].x[3], results['ep'].x[3]],
        f'min(w) [e-{POWER_OF_10:.0f}]': np.array([results['r'].fun, results['ir'].fun, results['ep'].fun]) * MULTIPLIER
        }, 
        index=['reversibility', 'irreversibility ratio', 'entropy production rate']
    )
    st.dataframe(df)

@st.fragment
def tab_c_total_plane():
    init_session_state(DEFAULT_VALUES_C)

    #=====INFO=====
    st.write("Curzon-Ahlborn model: imposed $q_0 - c_{total}$: find minimum power consumption $\min(w)$")


    col_control, _, col_plot = st.columns((0.34, 0.02, 0.64))

    #=====SLIDERS=====
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


    # Perform optimization
    initial_params = {
        'r': [init_e_g, init_e_p, init_c_total, init_q, init_t_s, MULTIPLIER],
        'ir': [init_e_g, init_e_p, init_c_total, init_q, init_t_s, init_I, MULTIPLIER],
        'ep': [init_e_g, init_e_p, init_c_total, init_q, init_t_s, init_s, MULTIPLIER]
    }
    opt_var = 'c'
    results = {
        'r': find_minimum(objective_function, initial_params['r'], opt_var),
        'ir': find_minimum(objective_function_ir_ratio, initial_params['ir'], opt_var),
        'ep': find_minimum(objective_function_ep_rate, initial_params['ep'], opt_var)
    }


    #=====PLOT=====
    with col_plot:
        plotting3D(results, initial_params, opt_var)


    #=====RESULT INFO=====
    st.write('#### Results:')
    df = pd.DataFrame({
        'I': [np.NaN, init_I, np.NaN],
        f's [e-{POWER_OF_10:.0f}]': [np.NaN, np.NaN, init_s],
        'c*_g': [results['r'].x[0], results['ir'].x[0], results['ep'].x[0]],
        'c*_p': [results['r'].x[1], results['ir'].x[1], results['ep'].x[1]],
        f'min(w) [e-{POWER_OF_10:.0f}]': np.array([results['r'].fun, results['ir'].fun, results['ep'].fun]) * MULTIPLIER
        }, 
        index=['reversibility', 'irreversibility ratio', 'entropy production rate']
    )
    st.dataframe(df)


#============MAIN CALCULATION==============
st.markdown('***')
st.markdown("## Minimum power consumption")
st.markdown('Text about...')

st.info(r'Choose a variable $(\varepsilon_{total}$ or $c_{total})$ to minimize $f(w)$')
tab_eps_total, tab_c_total = st.tabs([r'$\varepsilon_{total}$', '$c_{total}$'])

with tab_eps_total:
    tab_eps_total_plane()

with tab_c_total:
    tab_c_total_plane()



#=========PREP for SENSITIVITY ANALYSIS==========
DEFAULT_VALUES_SA_E = {
    'e_t_sae': (1.0, 4.0),
    'c_t_sae': 0.7,
    'q0_sae': 10.0,
    't_s_sae': 0.9,
    'I_sae': 1.1,
    's_sae': 1.1
}
def results_to_df(results, param, param_name):
    """
    Converts optimization results into a list of DataFrames.
    
    Parameters:
        results (dict): Dictionary containing optimization results.
        param (list): List of parameter values.
        param_name (str): Name of the parameter (used as index).
        
    Returns:
        list: A list containing DataFrames for each key in the results dictionary.
    """
    dataframes = []
    columns = [param_name, 'minw', 'ε*_g', 'ε*_p', 'ε*_ev', 'ε*_cd', 'c*_g', 'c*_p']
    
    for key, result_list in results.items():
        df = pd.DataFrame.from_records(
            [(e, r.fun * MULTIPLIER, *r.x) for e, r in zip(param, result_list)],
            columns=columns
        ).set_index(param_name)
        
        # Set values out of range (<0 or >1000) to None (to handle optimization errors)
        df[(df < 0) | (df > 1000)] = None
        dataframes.append(df)
    
    return dataframes

@st.fragment
def tab_e_total_sa():
    init_session_state(DEFAULT_VALUES_SA_E)
    col_control, _, col_plot = st.columns((0.28, 0.02, 0.70))
    
    #========SLIDERS========
    with col_control:
        init_eps_t = init_slider(r'$\varepsilon_{total}:$', 'e_t_sae', 1.0, 4.0, 0.1, fmt="%.1f")
        init_c_t = init_slider('$c_{total}:$', 'c_t_sae', 0.1, 1.0, 0.01)
        init_q = init_slider('$q_{0}:$', 'q0_sae', 1.0, 100.0, 0.1, fmt="%.1f",
                             help=fr'$q_{{0}} \times 10^{{-{POWER_OF_10:.0f}}}$')
        init_t_s = init_slider('$t_{s}:$', 't_s_sae', 0.8, 1.0, 0.01)
        init_I = init_slider('$I:$', 'I_sae', 1.0, 3.0, 0.01)
        init_s = init_slider('$s:$', 's_sae', 0.1, 20.0, 0.01,
                             help=fr'$s \times 10^{{-{POWER_OF_10:.0f}}}$')

        st.button("Reset", on_click=lambda: reset_sliders(DEFAULT_VALUES_SA_E), key='btn_sae')

    step_size = 0.1
    e_total = np.arange(init_eps_t[0], init_eps_t[1]+step_size, step_size)
    initial_params = {
        'r': [init_c_t, init_q, init_t_s, MULTIPLIER],
        'ir': [init_c_t, init_q, init_t_s, init_I, MULTIPLIER],
        'ep': [init_c_t, init_q, init_t_s, init_s, MULTIPLIER]
    }

    # Perform optimization
    with st.spinner("Calculating..."):
        opt_var = 'sae'
        results = {
            'r': find_minimum_vectorized(objective_function, e_total, opt_var, *initial_params['r']),
            'ir': find_minimum_vectorized(objective_function_ir_ratio, e_total, opt_var, *initial_params['ir']),
            'ep': find_minimum_vectorized(objective_function_ep_rate, e_total, opt_var, *initial_params['ep']),
        }
        df1, df2, df3 = results_to_df(results, e_total, 'ε_t')


    with col_plot:
        plotting_sensitivity(
            [df1, df2, df3], 
            ['reversibility', 'irrevers. ratio', 'entropy prod. rate'], 
            POWER_OF_10,
            theme_session
        )


    st.write('#### Results')
    col1, col2, col3 = st.columns((1, 1, 1))
    with col1:
        st.write("Reversibility:")
        st.dataframe(df1, height=210)
    with col2:
        st.write("Irreversibility ratio:")
        st.dataframe(df2, height=210)
    with col3:
        st.write("Entropy production rate:")
        st.dataframe(df3, height=210)


#==============SENSITIVITY ANALYSIS==============
st.markdown('***')
st.markdown('## Sensitivity analysis')
st.markdown('Text about...')

st.info(r'Choose a variable/parameter to analyse its impact on the minimum power consumption $min(w)$')
tab_e, tab_c, tab_q, tab_t, tab_i, tab_s = st.tabs([r'$\varepsilon_{total}$', '$c_{total}$', 
                                                    '&nbsp;&nbsp;&nbsp;&nbsp;$q_0$&nbsp;&nbsp;&nbsp;&nbsp;', 
                                                    '&nbsp;&nbsp;&nbsp;&nbsp;$t_0$&nbsp;&nbsp;&nbsp;&nbsp;', 
                                                    '&nbsp;&nbsp;&nbsp;&nbsp;$I$&nbsp;&nbsp;&nbsp;&nbsp;', 
                                                    '&nbsp;&nbsp;&nbsp;&nbsp;$s$&nbsp;&nbsp;&nbsp;&nbsp;'])

    # Define a wrapper for vectorized call

with tab_e:
    tab_e_total_sa()

with tab_c:
    st.write('c')

with tab_q:
    st.write('q')

with tab_t:
    st.write('t')

with tab_i:
    st.write('i')
    
with tab_s:
    st.write('s')