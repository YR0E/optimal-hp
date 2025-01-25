import timeit
import streamlit as st
import numpy as np
import pandas as pd
import plotly.io as pio
from streamlit_theme import st_theme
from util.calc_imposed_q0 import find_minimum, find_minimum_vectorized, find_minimum_loop, find_minimum_warm_start
from util.calc_imposed_q0 import objective_function, objective_function_ir_ratio, objective_function_ep_rate
from util.plot import plotting3D, plotting_sensitivity
# from util.navigation import link_to_pages


st.set_page_config(layout='wide', initial_sidebar_state='expanded', page_title='min(w)')

theme = st_theme()
dark_mode = False
if theme is not None and theme['base']=='dark':
    dark_mode = True
    pio.templates.default = "plotly_dark"
    theme_session = 'streamlit'    # to handle some streamlit issue with rendering plotly template
else:
    pio.templates.default = "plotly"
    theme_session = None



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

    start = timeit.default_timer()
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
    stop = timeit.default_timer()
    st.write(f'Optimization took {stop - start:.4f} seconds.')

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
_, col_img, _ = st.columns([1, 6, 1])
image_path = "img/gshp-dark.png" if dark_mode else "img/gshp.png"

col_img.image(image_path, caption="Ground source heat pump")
st.markdown("## Minimum power consumption")
st.markdown('Text about...')

st.info(r'Choose a variable $(\varepsilon_{total}$ or $c_{total})$ to minimize $f(w)$')
tab_eps_total, tab_c_total = st.tabs([r'$\varepsilon_{total}$', '$c_{total}$'])

with tab_eps_total:
    st.write(r"Curzon-Ahlborn model: imposed $q_0 - \varepsilon_{total}$: find minimum power consumption $\min(w)$")
    tab_eps_total_plane()

with tab_c_total:
    st.write("Curzon-Ahlborn model: imposed $q_0 - c_{total}$: find minimum power consumption $\min(w)$")
    tab_c_total_plane()



#=========PREP for SENSITIVITY ANALYSIS==========
DEFAULT_VALUES_SA_E = {
    'e_t_sae': (2.0, 4.0),
    'c_t_sae': 0.7,
    'q0_sae': 10.0,
    't_s_sae': 0.9,
    'I_sae': 1.1,
    's_sae': 1.1
}
DEFAULT_VALUES_SA_C = {
    'e_t_sac': 2.0,
    'c_t_sac': (0.4, 1.0),
    'q0_sac': 10.0,
    't_s_sac': 0.9,
    'I_sac': 1.1,
    's_sac': 1.1
}
DEFAULT_VALUES_SA_Q = {
    'e_t_saq': 2.0,
    'c_t_saq': 0.5,
    'q0_saq': (10.0, 80.0),
    't_s_saq': 0.9,
    'I_saq': 1.1,
    's_saq': 1.1
}
DEFAULT_VALUES_SA_T = {
    'e_t_sat': 2.0,
    'c_t_sat': 0.5,
    'q0_sat': 10.0,
    't_s_sat': (0.85, 0.95),
    'I_sat': 1.1,
    's_sat': 1.1
}
DEFAULT_VALUES_SA_I = {
    'e_t_sai': 2.0,
    'c_t_sai': 0.5,
    'q0_sai': 10.0,
    't_s_sai': 0.9,
    'I_sai': (1.0, 2.0),
}
DEFAULT_VALUES_SA_S = {
    'e_t_sas': 2.0,
    'c_t_sas': 0.5,
    'q0_sas': 10.0,
    't_s_sas': 0.9,
    's_sas': (0.0, 25.0),
}
def results_to_df(results, param, param_name, fix=True):
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
        
        # Set values out of range (<0 or >9999) to None (to handle optimization errors)
        if fix:
            df[(df < 0) | (df > 9999)] = None
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
    e_total = np.arange(init_eps_t[0], init_eps_t[1]+0.0001, step_size)
    initial_params = {
        'r': [init_c_t, init_q, init_t_s, MULTIPLIER],
        'ir': [init_c_t, init_q, init_t_s, init_I, MULTIPLIER],
        'ep': [init_c_t, init_q, init_t_s, init_s, MULTIPLIER]
    }

    with st.spinner("Calculating..."):
        opt_config = ('sa', 'e')
        results = {
            'r': find_minimum_vectorized(objective_function, e_total[:5], opt_config, *initial_params['r']),
            'ir': find_minimum_vectorized(objective_function_ir_ratio, e_total[:5], opt_config, *initial_params['ir']),
            'ep': find_minimum_vectorized(objective_function_ep_rate, e_total[:5], opt_config, *initial_params['ep']),
        }
        

    # np.vectorized
    start = timeit.default_timer()
    # Perform optimization
    with st.spinner("Calculating..."):
        opt_config = ('sa', 'e')
        results = {
            'r': find_minimum_vectorized(objective_function, e_total, opt_config, *initial_params['r']),
            'ir': find_minimum_vectorized(objective_function_ir_ratio, e_total, opt_config, *initial_params['ir']),
            'ep': find_minimum_vectorized(objective_function_ep_rate, e_total, opt_config, *initial_params['ep']),
        }
        df4, df5, df6 = results_to_df(results, e_total, 'ε_t')
    stop = timeit.default_timer()
    st.write(f"Vect took: {stop - start:.4f} seconds. Each calc took: {(stop - start)/(len(e_total)):.4f} seconds")
    
    
    # for loop
    start = timeit.default_timer()
    # Perform optimization
    with st.spinner("Calculating..."):
        opt_config = ('sa', 'e')
        results = {
            'r': find_minimum_loop(objective_function, e_total, opt_config, *initial_params['r']),
            'ir': find_minimum_loop(objective_function_ir_ratio, e_total, opt_config, *initial_params['ir']),
            'ep': find_minimum_loop(objective_function_ep_rate, e_total, opt_config, *initial_params['ep']),
        }
        df1, df2, df3 = results_to_df(results, e_total, 'ε_t')
    stop = timeit.default_timer()
    st.write(f"Loop took: {stop - start:.4f} seconds. Each calc took: {(stop - start)/(len(e_total)):.4f} seconds")


    # warm start loop
    start = timeit.default_timer()
    # Perform optimization
    with st.spinner("Calculating..."):
        opt_config = ('sa', 'e')
        results = {
            'r': find_minimum_warm_start(objective_function, e_total, opt_config, *initial_params['r']),
            'ir': find_minimum_warm_start(objective_function_ir_ratio, e_total, opt_config, *initial_params['ir']),
            'ep': find_minimum_warm_start(objective_function_ep_rate, e_total, opt_config, *initial_params['ep']),
        }
        df11, df22, df33 = results_to_df(results, e_total, 'ε_t')
    stop = timeit.default_timer()
    st.write(f"Warm took: {stop - start:.4f} seconds. Each calc took: {(stop - start)/(len(e_total)):.4f} seconds")



    st.write('Difference in minw:', (df1['minw']-df4['minw']).mean(), (df2['minw']-df5['minw']).mean(), (df3['minw']-df6['minw']).mean())
    st.write('Difference in minw - warmstart:', (df1['minw']-df11['minw']).mean(), (df2['minw']-df22['minw']).mean(), (df3['minw']-df33['minw']).mean())

    temp_df = pd.concat([df1['minw'], df11['minw'], df2['minw'], df22['minw'], df3['minw'], df33['minw']], axis=1, keys=['loop', 'warm', 'loop-ir', 'warm-ir', 'loop-ep', 'warm-ep'])
    temp_df['diff'] = temp_df['warm'] - temp_df['loop']
    temp_df['diff-ir'] = temp_df['warm-ir'] - temp_df['loop-ir']
    temp_df['diff-ep'] = temp_df['warm-ep'] - temp_df['loop-ep']

    left, right = st.columns(2)
    left.line_chart(temp_df[['loop', 'warm', 'loop-ir', 'warm-ir']], y_label='minw')
    right.line_chart(temp_df[['diff', 'diff-ir', 'diff-ep']], y_label='difference')
    
    st.write(temp_df)


    with col_plot:
        plotting_sensitivity(
            [df11, df22, df33], 
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

@st.fragment
def tab_c_total_sa():
    init_session_state(DEFAULT_VALUES_SA_C)
    col_control, _, col_plot = st.columns((0.28, 0.02, 0.70))
    
    #========SLIDERS========
    with col_control:
        init_c_t = init_slider('$c_{total}:$', 'c_t_sac', 0.2, 1.0, 0.02)
        init_eps_t = init_slider(r'$\varepsilon_{total}:$', 'e_t_sac', 0.4, 4.0, 0.01)
        init_q = init_slider('$q_{0}:$', 'q0_sac', 1.0, 100.0, 0.1, fmt="%.1f",
                             help=fr'$q_{{0}} \times 10^{{-{POWER_OF_10:.0f}}}$')
        init_t_s = init_slider('$t_{s}:$', 't_s_sac', 0.8, 1.0, 0.01)
        init_I = init_slider('$I:$', 'I_sac', 1.0, 3.0, 0.01)
        init_s = init_slider('$s:$', 's_sac', 0.1, 30.0, 0.01,
                             help=fr'$s \times 10^{{-{POWER_OF_10:.0f}}}$')

        st.button("Reset", on_click=lambda: reset_sliders(DEFAULT_VALUES_SA_C), key='btn_sac')

    step_size = 0.025
    c_total = np.arange(init_c_t[0], init_c_t[1]+0.0001, step_size)
    initial_params = {
        'r': [init_eps_t, init_q, init_t_s, MULTIPLIER],
        'ir': [init_eps_t, init_q, init_t_s, init_I, MULTIPLIER],
        'ep': [init_eps_t, init_q, init_t_s, init_s, MULTIPLIER]
    }
    
    # Perform optimization
    with st.spinner("Calculating..."):
        opt_config = ('sa', 'c')
        results = {
            'r': find_minimum_vectorized(objective_function, c_total, opt_config, *initial_params['r']),
            'ir': find_minimum_vectorized(objective_function_ir_ratio, c_total, opt_config, *initial_params['ir']),
            'ep': find_minimum_vectorized(objective_function_ep_rate, c_total, opt_config, *initial_params['ep']),
        }
        df1, df2, df3 = results_to_df(results, c_total, 'c_t')


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

@st.fragment
def tab_q0_sa():
    init_session_state(DEFAULT_VALUES_SA_Q)
    col_control, _, col_plot = st.columns((0.28, 0.02, 0.70))
    
    #========SLIDERS========
    with col_control:
        init_q = init_slider('$q_{0}:$', 'q0_saq', 0.0, 100.0, 0.1, fmt="%.1f",
                             help=fr'$q_{{0}} \times 10^{{-{POWER_OF_10:.0f}}}$')
        init_eps_t = init_slider(r'$\varepsilon_{total}:$', 'e_t_saq', 0.4, 4.0, 0.01)
        init_c_t = init_slider('$c_{total}:$', 'c_t_saq', 0.1, 1.0, 0.01)
        init_t_s = init_slider('$t_{s}:$', 't_s_saq', 0.8, 1.0, 0.01)
        init_I = init_slider('$I:$', 'I_saq', 1.0, 3.0, 0.01)
        init_s = init_slider('$s:$', 's_saq', 0.1, 30.0, 0.01,
                             help=fr'$s \times 10^{{-{POWER_OF_10:.0f}}}$')

        st.button("Reset", on_click=lambda: reset_sliders(DEFAULT_VALUES_SA_Q), key='btn_saq')

    step_size = 2.0
    q0 = np.arange(init_q[0], init_q[1]+0.0001, step_size)
    initial_params = {
        'r': [init_eps_t, init_c_t, init_t_s, MULTIPLIER],
        'ir': [init_eps_t, init_c_t, init_t_s, init_I, MULTIPLIER],
        'ep': [init_eps_t, init_c_t, init_t_s, init_s, MULTIPLIER]
    }

    
    # Perform optimization
    with st.spinner("Calculating..."):
        opt_config = ('sa', 'q')
        results = {
            'r': find_minimum_vectorized(objective_function, q0, opt_config, *initial_params['r']),
            'ir': find_minimum_vectorized(objective_function_ir_ratio, q0, opt_config, *initial_params['ir']),
            'ep': find_minimum_vectorized(objective_function_ep_rate, q0, opt_config, *initial_params['ep']),
        }
        df1, df2, df3 = results_to_df(results, q0, 'q0')


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

@st.fragment
def tab_ts_sa():
    init_session_state(DEFAULT_VALUES_SA_T)
    col_control, _, col_plot = st.columns((0.28, 0.02, 0.70))
    
    #========SLIDERS========
    with col_control:
        init_t_s = init_slider('$t_{s}:$', 't_s_sat', 0.8, 1.0, 0.01)
        init_eps_t = init_slider(r'$\varepsilon_{total}:$', 'e_t_sat', 0.4, 4.0, 0.01)
        init_c_t = init_slider('$c_{total}:$', 'c_t_sat', 0.1, 1.0, 0.01)
        init_q = init_slider('$q_{0}:$', 'q0_sat', 1.0, 100.0, 0.1, fmt="%.1f",
                             help=fr'$q_{{0}} \times 10^{{-{POWER_OF_10:.0f}}}$')
        init_I = init_slider('$I:$', 'I_sat', 1.0, 3.0, 0.01)
        init_s = init_slider('$s:$', 's_sat', 0.1, 30.0, 0.01,
                             help=fr'$s \times 10^{{-{POWER_OF_10:.0f}}}$')

        st.button("Reset", on_click=lambda: reset_sliders(DEFAULT_VALUES_SA_T), key='btn_sat')

    step_size = 0.005
    t_s = np.arange(init_t_s[0], init_t_s[1]+0.0001, step_size)
    initial_params = {
        'r': [init_eps_t, init_c_t, init_q, MULTIPLIER],
        'ir': [init_eps_t, init_c_t, init_q, init_I, MULTIPLIER],
        'ep': [init_eps_t, init_c_t, init_q, init_s, MULTIPLIER]
    }

    
    # Perform optimization
    with st.spinner("Calculating..."):
        opt_config = ('sa', 't')
        results = {
            'r': find_minimum_vectorized(objective_function, t_s, opt_config, *initial_params['r']),
            'ir': find_minimum_vectorized(objective_function_ir_ratio, t_s, opt_config, *initial_params['ir']),
            'ep': find_minimum_vectorized(objective_function_ep_rate, t_s, opt_config, *initial_params['ep']),
        }
        df1, df2, df3 = results_to_df(results, t_s, 't_s')


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

@st.fragment
def tab_i_sa():
    init_session_state(DEFAULT_VALUES_SA_I)
    col_control, _, col_plot = st.columns((0.28, 0.02, 0.70))
    
    #========SLIDERS========
    with col_control:
        init_I = init_slider('$I:$', 'I_sai', 1.0, 3.0, 0.01)
        init_eps_t = init_slider(r'$\varepsilon_{total}:$', 'e_t_sai', 0.4, 4.0, 0.01)
        init_c_t = init_slider('$c_{total}:$', 'c_t_sai', 0.1, 1.0, 0.01)
        init_q = init_slider('$q_{0}:$', 'q0_sai', 1.0, 100.0, 0.1, fmt="%.1f",
                             help=fr'$q_{{0}} \times 10^{{-{POWER_OF_10:.0f}}}$')
        init_t_s = init_slider('$t_{s}:$', 't_s_sai', 0.8, 1.0, 0.01)

        st.button("Reset", on_click=lambda: reset_sliders(DEFAULT_VALUES_SA_I), key='btn_sai')

    step_size = 0.02
    I = np.arange(init_I[0], init_I[1]+step_size, step_size)
    initial_params = {
        'ir': [init_eps_t, init_c_t, init_q, init_t_s, MULTIPLIER],
    }

    # Perform optimization
    with st.spinner("Calculating..."):
        opt_config = ('sa', 'I')
        results = {
            'ir': find_minimum_vectorized(objective_function_ir_ratio, I, opt_config, *initial_params['ir']),
        }
        df1,  = results_to_df(results, I, 'I')


    with col_plot:
        plotting_sensitivity(
            [df1], 
            ['irrevers. ratio'], 
            POWER_OF_10,
            theme_session
        )


    st.write('#### Results')
    st.write("Irreversibility ratio:")
    st.dataframe(df1, height=210)

@st.fragment
def tab_s_sa():
    init_session_state(DEFAULT_VALUES_SA_S)
    col_control, _, col_plot = st.columns((0.28, 0.02, 0.70))
    
    #========SLIDERS========
    with col_control:
        init_s = init_slider('$s:$', 's_sas', 0.0, 50.0, 0.5, fmt="%.1f",
                             help=fr'$s \times 10^{{-{POWER_OF_10:.0f}}}$')
        init_eps_t = init_slider(r'$\varepsilon_{total}:$', 'e_t_sas', 0.4, 4.0, 0.01)
        init_c_t = init_slider('$c_{total}:$', 'c_t_sas', 0.1, 1.0, 0.01)
        init_q = init_slider('$q_{0}:$', 'q0_sas', 1.0, 100.0, 0.1, fmt="%.1f",
                             help=fr'$q_{{0}} \times 10^{{-{POWER_OF_10:.0f}}}$')
        init_t_s = init_slider('$t_{s}:$', 't_s_sas', 0.8, 1.0, 0.01)

        st.button("Reset", on_click=lambda: reset_sliders(DEFAULT_VALUES_SA_S), key='btn_sas')

    step_size = 0.5
    s = np.arange(init_s[0], init_s[1]+0.0001, step_size)
    initial_params = {
        'ep': [init_eps_t, init_c_t, init_q, init_t_s, MULTIPLIER],
    }


    # Perform optimization
    with st.spinner("Calculating..."):
        opt_config = ('sa', 's')
        results = {
            'ep': find_minimum_vectorized(objective_function_ep_rate, s, opt_config, *initial_params['ep']),
        }
        df1,  = results_to_df(results, s, 's')


    with col_plot:
        plotting_sensitivity(
            [df1], 
            ['entropy prod. rate'], 
            POWER_OF_10,
            theme_session
        )


    st.write('#### Results')
    st.write("Entropy production rate:")
    st.dataframe(df1, height=210)

@st.fragment
def sensitivity_analysis():
    st.info(r'Choose a variable/parameter to analyze its impact on the minimum power consumption $min(w)$')
    tab_e, tab_c, tab_q, tab_t, tab_i, tab_s = st.tabs([r'$\varepsilon_{total}$', '$c_{total}$', 
                                                        '&nbsp;&nbsp;&nbsp;&nbsp;$q_0$&nbsp;&nbsp;&nbsp;&nbsp;', 
                                                        '&nbsp;&nbsp;&nbsp;&nbsp;$t_0$&nbsp;&nbsp;&nbsp;&nbsp;', 
                                                        '&nbsp;&nbsp;&nbsp;&nbsp;$I$&nbsp;&nbsp;&nbsp;&nbsp;', 
                                                        '&nbsp;&nbsp;&nbsp;&nbsp;$s$&nbsp;&nbsp;&nbsp;&nbsp;'])


    with tab_e:
        st.markdown(r'''Minimum power consumption $\min(w)$ as a function of $\varepsilon_{total}$, with parameters $c_{total}$, $q_0$, $t_0$, $I$, and $s$.  
                    Parameters $I$ and $s$ are for the irreversibility ratio and entropy production rate, respectively.
                    ''')
        tab_e_total_sa()

    with tab_c:
        st.markdown(r'''Minimum power consumption $\min(w)$ as a function of $c_{total}$, with parameters $\varepsilon_{total}$, $q_0$, $t_0$, $I$, and $s$.  
                    Parameters $I$ and $s$ are for the irreversibility ratio and entropy production rate, respectively.
                    ''')
        tab_c_total_sa()

    with tab_q:
        st.markdown(r'''Minimum power consumption $\min(w)$ as a function of $q_0$, with parameters $\varepsilon_{total}$, $c_{total}$, $t_0$, $I$, and $s$.  
                    Parameters $I$ and $s$ are for the irreversibility ratio and entropy production rate, respectively.
                    ''')
        tab_q0_sa()

    with tab_t:
        st.markdown(r'''Minimum power consumption $\min(w)$ as a function of $t_s$, with parameters $\varepsilon_{total}$, $c_{total}$, $q_0$, $I$, and $s$.  
                    Parameters $I$ and $s$ are for the irreversibility ratio and entropy production rate, respectively.
                    ''')
        tab_ts_sa()

    with tab_i:
        st.markdown(r'''Minimum power consumption $\min(w)$ as a function of $I$, with parameters $\varepsilon_{total}$, $c_{total}$, $q_0$, and $t_s$.  
                    Parameter $I$ is for the irreversibility ratio.
                    ''')
        tab_i_sa()

    with tab_s:
        st.markdown(r'''Minimum power consumption $\min(w)$ as a function of $s$, with parameters $\varepsilon_{total}$, $c_{total}$, $q_0$, and $t_s$.  
                    Parameter $s$ is for the entropy production rate.
                    ''')
        tab_s_sa()


#==============SENSITIVITY ANALYSIS==============
st.markdown('***')
st.markdown('## Sensitivity analysis')
st.markdown('Here you can perform a sensitivity analysis.')

if st.button("Analyze", type="primary"):
    with st.spinner("Calculating..."):
        sensitivity_analysis()

        st.toast('Sensitivity analysis done!', icon=':material/done_all:')