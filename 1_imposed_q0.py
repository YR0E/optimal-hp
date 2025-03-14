import time
import streamlit as st
import numpy as np
import pandas as pd
import plotly.io as pio
from streamlit_theme import st_theme
from util.calc_imposed_q0 import find_minimum, find_minimum_vectorized
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
    'e_t_e': 2.4,
    'c_g_e': 0.4,
    'c_p_e': 0.4,
    'q0_e': 20.0,
    't_s_e': 0.9,
    'I_e': 1.05,
    # 's_e': 0.1
}
DEFAULT_VALUES_C = {
    'e_g_c': 0.6,
    'e_p_c': 0.6,
    'c_t_c': 0.8,
    'q0_c': 20.0,
    't_s_c': 0.9,
    'I_c': 1.05,
    # 's_c': 0.1
}

def init_session_state(default):
    """
    Initializes the session state with default values if they are not already set.

    Parameters:
    - default (dict): A dictionary of default values, where each key is the name 
      of the session state variable, and each value is the default value to set.
    """

    def recursive_update(prefix, d):
        for key, value in d.items():
            full_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                recursive_update(full_key, value)
            elif full_key not in st.session_state:
                st.session_state[full_key] = value
                
    recursive_update(None, default)

def reset_sliders(default):
    """
    Resets all slider values to their default values.

    Parameters:
    - default (dict): A dictionary of default values, keyed by the session state key.
    """
    
    def recursive_update(prefix, d):
        for key, value in d.items():
            full_key = f"{prefix}_{key}" if prefix else key
            if isinstance(value, dict):
                recursive_update(full_key, value)
            
            st.session_state[full_key] = value
                
    recursive_update(None, default)

def init_slider(varname, key, minval, maxval, step, fmt="%.2f", value=None, help=None):
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
    slider_val = col2.slider(
        label=key, label_visibility="collapsed", key=key,
        min_value=minval, max_value=maxval, value=value,
        step=step, format=fmt
    )
    
    return slider_val

@st.fragment
def tab_eps_total_plane():
    init_session_state(DEFAULT_VALUES_EPS)
    col_control, _, col_plot = st.columns((0.34, 0.02, 0.64))

    #=====SLIDERS=====
    with col_control:
        init_eps_total = init_slider(
            r'$\varepsilon_{total}:$', 'e_t_e', 
            0.4, 4.0, 0.1, fmt="%.1f"
        )
        init_c_g = init_slider('$c_{g}:$', 'c_g_e', 0.01, 1.0, 0.01)
        init_c_p = init_slider('$c_{p}:$', 'c_p_e', 0.01, 1.0, 0.01)
        init_q = init_slider(
            '$q_{0}:$', 'q0_e', 1.0, 100.0, 0.1, fmt="%.1f",
            help=fr'$q_{{0}} \times 10^{{-{POWER_OF_10:.0f}}}$'
        )
        init_t_s = init_slider('$t_{s}:$', 't_s_e', 0.8, 1.0, 0.01)
        init_I = init_slider('$I:$', 'I_e', 1.0, 2.0, 0.01)
        
        s_value = ((init_I-1)*init_q/MULTIPLIER)/(init_t_s - (2*init_q/(MULTIPLIER*(init_c_g+init_c_p)))*(8/init_eps_total - 1))*MULTIPLIER
        init_s = init_slider(
            '$s:$', 's_e', 0.0, 100.0, 0.01, value=s_value,
            help=fr'$s \times 10^{{-{POWER_OF_10:.0f}}},\quad s=f(I, q_{{0}}, t_s, \varepsilon, c) $'
        )

        st.button("Reset", on_click=lambda: reset_sliders(DEFAULT_VALUES_EPS), key='btn_e')


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
    df = pd.DataFrame(
        {
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
    st.dataframe(df, width=640)

@st.fragment
def tab_c_total_plane():
    init_session_state(DEFAULT_VALUES_C)
    col_control, _, col_plot = st.columns((0.34, 0.02, 0.64))

    #=====SLIDERS=====
    with col_control:
        init_c_total = init_slider('$c_{total}:$', 'c_t_c', 0.01, 1.0, 0.01)
        init_e_g = init_slider(r'$\varepsilon_{g}:$', 'e_g_c', 0.01, 1.0, 0.01)
        init_e_p = init_slider(r'$\varepsilon_{p}:$', 'e_p_c', 0.01, 1.0, 0.01)
        init_q = init_slider(
            '$q_{0}:$', 'q0_c', 1.0, 100.0, 0.1, fmt="%.1f",
            help=fr'$q_{{0}} \times 10^{{-{POWER_OF_10:.0f}}}$'
        )
        init_t_s = init_slider('$t_{s}:$', 't_s_c', 0.8, 1.0, 0.01)
        init_I = init_slider('$I:$', 'I_c', 1.0, 2.0, 0.01)
        
        s_value = ((init_I-1)*init_q/MULTIPLIER)/(init_t_s - (2*init_q/(MULTIPLIER*init_c_total))*(8/(init_e_g+init_e_p) - 1))*MULTIPLIER
        init_s = init_slider(
            '$s:$', 's_c', 0.0, 100.0, 0.01, value=s_value,
            help=fr'$s \times 10^{{-{POWER_OF_10:.0f}}},\quad s=f(I, q_{{0}}, t_s, \varepsilon, c) $'
        )

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
    df = pd.DataFrame(
        {
        'I': [np.NaN, init_I, np.NaN],
        f's [e-{POWER_OF_10:.0f}]': [np.NaN, np.NaN, init_s],
        'c*_g': [results['r'].x[0], results['ir'].x[0], results['ep'].x[0]],
        'c*_p': [results['r'].x[1], results['ir'].x[1], results['ep'].x[1]],
        f'min(w) [e-{POWER_OF_10:.0f}]': np.array([results['r'].fun, results['ir'].fun, results['ep'].fun]) * MULTIPLIER
        }, 
        index=['reversibility', 'irreversibility ratio', 'entropy production rate']
    )
    st.dataframe(df, width=520)


#============MAIN CALCULATION==============
_, col_img, _ = st.columns([1, 6, 1])
image_path = "img/gshp-dark.svg" if dark_mode else "img/gshp.svg"

col_img.image(image_path, caption="Ground source heat pump")
st.markdown("## Minimum power consumption")
st.markdown('Text about...')

st.info(r'Choose a variable $(\varepsilon_{total}$ or $c_{total})$ to minimize $f(w)$')
tab_eps_total, tab_c_total = st.tabs([r'$\varepsilon_{total}$', '$c_{total}$'])

with tab_eps_total:
    st.write(r"Curzon-Ahlborn model: imposed $q_0 - \varepsilon_{total}$: find minimum power consumption $\min(w)$")
    st.write('')
    tab_eps_total_plane()

with tab_c_total:
    st.write("Curzon-Ahlborn model: imposed $q_0 - c_{total}$: find minimum power consumption $\min(w)$")
    st.write('')
    tab_c_total_plane()



#=========PREP for SENSITIVITY ANALYSIS==========
DEFAULT_SETTING_GLOBAL = {
    'opt_method': 'SLSQP',
    'tol': -16,
    'e_0': 0.5,
    'e_bnds': (0.0, 1.0),
    'c_0': 0.1,
    'c_bnds': (0.0, 1.0),
    'warm_start': True,
    'cut_off': True,
    'plot_param': 'Hot loop'
}
DEFAULT_SETTING_VALUES = {
    'e': {
        'name': r'$\varepsilon_{total}$',
        'step': 0.1,
        'step_widget': (0.01, 0.20, 0.01),
        'range': (2.0, 4.0),
        'range_widget': (1.0, 4.0),
        **DEFAULT_SETTING_GLOBAL
    }, 

    'c': {
        'name': '$c_{total}$',
        'step': 0.02,
        'step_widget': (0.005, 0.1, 0.005),
        'range': (0.4, 1.0),
        'range_widget': (0.1, 1.0),
        **DEFAULT_SETTING_GLOBAL,
        'tol': -21,
    },

    'q': {
        'name': '$q_0$',
        'step': 2.0,
        'step_widget': (0.05, 5.0, 0.05),
        'range': (10.0, 80.0), 
        'range_widget': (0.0, 100.0),
        **DEFAULT_SETTING_GLOBAL,
        'tol': -21,
    },
    
    't': {
        'name': '$t_s$',
        'step': 0.005,
        'step_widget': (0.001, 0.02, 0.001),
        'range': (0.85, 0.95),
        'range_widget': (0.8, 1.0),
        **DEFAULT_SETTING_GLOBAL,
        'tol': -21,
    },

    'I': {
        'name': '$I$',
        'step': 0.02,
        'step_widget': (0.005, 0.1, 0.005),
        'range': (1.0, 2.0),
        'range_widget': (1.0, 3.0),
        **DEFAULT_SETTING_GLOBAL
    },

    's': {
        'name': '$s$',
        'step': 0.5,
        'step_widget': (0.05, 2.0, 0.05),
        'range': (0.0, 25.0),
        'range_widget': (0.0, 50.0),
        **DEFAULT_SETTING_GLOBAL
    }
}
DEFAULT_SA_SLIDERS = {
    'e': {
        'e_t': (2.0, 4.0),
        'c_t': 0.8,
        'q0': 50.0,
        't_s': 0.9,
        'I': 1.1,
        # 's': 3.0
    },

    'c': {
        'e_t': 3.2,
        'c_t': (0.4, 1.0),
        'q0': 50.0,
        't_s': 0.9,
        'I': 1.1,
        # 's': 3.0
    },

    'q': {
        'e_t': 3.2,
        'c_t': 0.8,
        'q0': (10.0, 80.0),
        't_s': 0.9,
        'I': 1.1,
        # 's': 3.0
    },

    't': {
        'e_t': 3.2,
        'c_t': 0.8,
        'q0': 50.0,
        't_s': (0.85, 0.95),
        'I': 1.1,
        # 's': 3.0
    },
    
    'I': {
        'e_t': 3.2,
        'c_t': 0.8,
        'q0': 50.0,
        't_s': 0.9,
        'I': (1.0, 2.0),
    },

    's': {
        'e_t': 3.2,
        'c_t': 0.8,
        'q0': 50.0,
        't_s': 0.9,
        's': (0.0, 25.0)
    }
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
        if key=='ep':   # results['ep'] = [result, s_array] 
            df = pd.DataFrame.from_records(
                    [(p, r.fun * MULTIPLIER, *r.x) for p, r in zip(param, result_list[0])],
                    columns=columns
                ).set_index(param_name)
            df['s'] = result_list[1]
        else:
            df = pd.DataFrame.from_records(
                [(p, r.fun * MULTIPLIER, *r.x) for p, r in zip(param, result_list)],
                columns=columns
            ).set_index(param_name)
        
        # Set values out of range (<0 or >9999) to None (to handle optimization errors)
        if fix:
            df[(df < 0) | (df > 9999)] = None
            
        dataframes.append(df)
    
    return dataframes

def display_results(dfs):
    """
    Displays optimization results in Streamlit columns.

    Parameters:
        dfs (list): List of DataFrames containing optimization results.
    """

    height = 210

    st.write('')
    st.markdown(
        fr'''
        #### Results
        '''
    )

    if len(dfs) == 3:
        df1, df2, df3 = dfs
        col1, col2, col3 = st.columns((1, 1, 1))
        with col1:
            st.write("Reversibility:")
            st.dataframe(df1, height=height)
        with col2:
            st.write("Irreversibility ratio:")
            st.dataframe(df2, height=height)
        with col3:
            st.write("Entropy production rate:")
            st.dataframe(df3, height=height)
    elif len(dfs) == 2:
        df1, df2 = dfs
        col1, col2, _ = st.columns((1, 1, 1))
        with col1:
            st.write("Irreversibility ratio:")
            st.dataframe(df1, height=height)
        with col2:
            st.write("Entropy production rate:")
            st.dataframe(df2, height=height)

def display_info(x0_bounds, info, plot_param):
    """
    Displays optimization information.

    Parameters:
        x0_bounds (tuple): Tuple of tuples containing initial guesses and bounds for the parameters.
        info (str, optional): Additional information to display in the Streamlit Markdown. Defaults to ''.
    """

    e_0, e_min, e_max = x0_bounds[0]
    c_0, c_min, c_max = x0_bounds[1]

    st.write('')
    st.markdown(
        fr'''
        #### Info

        Initial guesses {info}: $\quad \varepsilon^*_{{i,0}} = $`{e_0:.2f}`; $\,\, c^*_{{i,0}} = $`{c_0:.2f}`  
        Bounds: $\quad \varepsilon^*_{{i}} \in [$`{e_min}, {e_max}`$]$;
        $\,\, c^*_{{i}} \in [$`{c_min}, {c_max}`$]$ 

        Plots with respect to `{plot_param}`
        '''
    )

def runtime_info(st_container, start_time, info):
    st_container.markdown(
        f"""
        <p style="font-size:13px; opacity:0.6;"> 
            Run time {info}: {time.time() - start_time:.3f} s
        </p>
        """, 
        unsafe_allow_html=True
    )

def settings_popover(var, defaults):
    """Display a Streamlit popover with settings for sensitivity analysis optimization.

    Parameters:
        var (str): The chosen parameter (variable).
        defaults (dict): A dictionary containing default values for the sensitivity analysis parameters.

    Returns:
        tuple: A tuple containing the set values for the sensitivity analysis parameters.
    """

    columns_size = (1.7, 0.3, 3)
    methods_list = ['COBYLA', 'SLSQP', 'trust-constr']

    with st.popover('Sensitivity analysis settings', icon=":material/tune:", help='Set sensitivity analysis parameters'):
        st.write(
            f'''
            Here you can set sensitivity analysis parameters for the optimization process.   
            The chosen parameter (variable) is {defaults[var]['name']}
            '''
        )
        
        st.button(
            "Reset", on_click=lambda: reset_sliders(defaults), key=f'reset_set_{var}', 
            icon=":material/reset_settings:", help='Reset Settings to Defaults'
        )
        
        left, _, right = st.columns(columns_size, vertical_alignment='top')
        step_min, step_max, step = defaults[var]['step_widget']
        step_size = left.number_input(
            "Variable step size:", 
            min_value=step_min, max_value=step_max, 
            step=step, format="%.3f", key=f'{var}_step'
        )
        range_min, range_max = defaults[var]['range_widget']
        var_range = right.slider(
            "Variable range:", 
            min_value=range_min, max_value=range_max, 
            step=step_size, key=f'{var}_range'
        )
        
        left, _, right = st.columns(columns_size, vertical_alignment='top')
        opt_method = left.selectbox("Opt. method:", methods_list, disabled=True, key=f'{var}_opt_method')
        tolerance = right.slider("Tolerance, $10^{x}$:", min_value=-24, max_value=-6, key=f'{var}_tol')
        tolerance = 10**tolerance
        
        left, _, right = st.columns(columns_size, vertical_alignment='top')
        e_0 = left.number_input(
            r"Initial guess of $\varepsilon^*_{i}$:", 
            min_value=0.0, max_value=1.0, 
            step=0.1, key=f'{var}_e_0'
        )
        e_bnds = right.slider(
            r"Bounds on $\varepsilon^*_{i}$:", 
            min_value=0.0, max_value=1.0, 
            step=0.1, key=f'{var}_e_bnds'
        )
        
        left, _, right = st.columns(columns_size, vertical_alignment='top')
        c_0 = left.number_input(
            r"Initial guess of $c^*_{i}$:", 
            min_value=0.0, max_value=1.0, 
            step=0.1, key=f'{var}_c_0'
        )
        c_bnds = right.slider(
            r"Bounds on $c^*_{i}$:", 
            min_value=0.0, max_value=1.0, 
            step=0.01, key=f'{var}_c_bnds'
        )
        guess_bound = (e_0, *e_bnds), (c_0, *c_bnds)
        
        warm_start = st.toggle(
            "Vectorize with warm starting*", key=f'{var}_warm_start', 
            help='Use previous results as initial guess for next iteration'
        )
        cuttoff_outliers = st.toggle("Outliers to `None`", key=f'{var}_cut_off')
        plot_param = st.radio('Plots w.r.t.:', ['Hot loop', 'Cold loop'], index=0, key=f'{var}_plot_param')

        return step_size, var_range, opt_method, tolerance, guess_bound, warm_start, cuttoff_outliers, plot_param

def calculate_gradient(dfs):  
    
    columns = [col for col in dfs[0].columns if not (col.endswith('ev') or col.endswith('cd'))]

    for df in dfs:
        M1 = MULTIPLIER if df.index.name=='q0' or df.index.name=='s' else 1
        for col in columns:
            M2 = MULTIPLIER if col=='minw' else 1
            df[f"grad_{col}"] = np.gradient(df[col]/M2, df.index/M1)

@st.fragment
def tab_e_total_sa():
    col_control, _, col_info = st.columns((0.34, 0.02, 0.64))
    
    with col_info:
        # settings
        step_size, var_range, *opt_settings = settings_popover('e', DEFAULT_SETTING_VALUES)
        opt_method, tolerance, guess_bound, warm_start, cuttoff_outliers, plot_param = opt_settings
        txt = f'(warm starting*)' if warm_start else ''
        param_index = 'p' if plot_param=='Hot loop' else 'g'
        
        display_info(guess_bound, txt, plot_param)


    with col_control:
        # sliders
        init_c_t = init_slider('$c_{total}:$', 'e_c_t', 0.1, 1.0, 0.01)
        init_q = init_slider(
            '$q_{0}:$', 'e_q0', 1.0, 100.0, 0.1, fmt="%.1f",
            help=fr'$q_{{0}} \times 10^{{-{POWER_OF_10:.0f}}}$'
        )
        init_t_s = init_slider('$t_{s}:$', 'e_t_s', 0.8, 1.0, 0.01)
        init_I = init_slider('$I:$', 'e_I', 1.0, 2.0, 0.01)
        # init_s = init_slider(
        #     '$s:$', 'e_s', 0.1, 20.0, 0.01,
        #     help=fr'$s \times 10^{{-{POWER_OF_10:.0f}}}$'
        # )

        st.button(
            "Reset", on_click=lambda: reset_sliders(DEFAULT_SA_SLIDERS), 
            key='btn_sae', help='Reset Parameters to Defaults'
        )
        st_runtime_info = st.empty()

    
    e_total = np.arange(var_range[0], var_range[1]+0.0001, step_size)
    initial_params = {
        'r': [init_c_t, init_q, init_t_s, MULTIPLIER],
        'ir': [init_c_t, init_q, init_t_s, init_I, MULTIPLIER],
        'ep': [init_c_t, init_q, init_t_s, init_I, MULTIPLIER]     # s=f(I,q,...) in find_minimum_vectorized
    }
 
    # Perform optimization
    start_time = time.time()
    with st.spinner("Calculating..."):
        opt_config = ('sa', 'e')
        results = {
            'r': find_minimum_vectorized(
                objective_function, e_total, opt_config, 
                guess_bound, *initial_params['r'],
                method=opt_method, tol=tolerance,
                warm_start=warm_start
                ),
            'ir': find_minimum_vectorized(
                objective_function_ir_ratio, e_total, opt_config, 
                guess_bound, *initial_params['ir'],
                method=opt_method, tol=tolerance,
                warm_start=warm_start
                ),
            'ep': find_minimum_vectorized(
                objective_function_ep_rate, e_total, opt_config, 
                guess_bound, *initial_params['ep'],
                method=opt_method, tol=tolerance,
                warm_start=warm_start
                ),
        }
        df1, df2, df3 = results_to_df(results, e_total, 'ε_t', fix=cuttoff_outliers)
    
    runtime_info(st_runtime_info, start_time, txt)


    calculate_gradient([df1, df2, df3])

    plotting_sensitivity(
        [df1, df2, df3], 
        ['reversibility', 'irrevers. ratio', 'entropy prod. rate'],     # labels
        param_index,                                                    # parameter index (e.g. 'p' for hot loop)
        POWER_OF_10,
        theme_session
    )

    display_results([df1, df2, df3])

@st.fragment
def tab_c_total_sa():
    col_control, _, col_info = st.columns((0.34, 0.02, 0.64))
    
    with col_info:
        # settings
        step_size, var_range, *opt_settings = settings_popover('c', DEFAULT_SETTING_VALUES)
        opt_method, tolerance, guess_bound, warm_start, cuttoff_outliers, plot_param = opt_settings
        txt = f'(warm starting*)' if warm_start else ''
        param_index = 'p' if plot_param=='Hot loop' else 'g'

        display_info(guess_bound, txt, plot_param)


    with col_control:
        # sliders
        init_eps_t = init_slider(r'$\varepsilon_{total}:$', 'c_e_t', 0.4, 4.0, 0.01)
        init_q = init_slider(
            '$q_{0}:$', 'c_q0', 1.0, 100.0, 0.1, fmt="%.1f",
            help=fr'$q_{{0}} \times 10^{{-{POWER_OF_10:.0f}}}$'
        )
        init_t_s = init_slider('$t_{s}:$', 'c_t_s', 0.8, 1.0, 0.01)
        init_I = init_slider('$I:$', 'c_I', 1.0, 2.0, 0.01)
        # init_s = init_slider(
        #     '$s:$', 'c_s', 0.1, 30.0, 0.01,
        #     help=fr'$s \times 10^{{-{POWER_OF_10:.0f}}}$'
        # )

        st.button(
            "Reset", on_click=lambda: reset_sliders(DEFAULT_SA_SLIDERS), 
            key='btn_sac', help='Reset Parameters to Defaults'
        )
        st_runtime_info = st.empty()

    
    c_total = np.arange(var_range[0], var_range[1]+0.0001, step_size)
    initial_params = {
        'r': [init_eps_t, init_q, init_t_s, MULTIPLIER],
        'ir': [init_eps_t, init_q, init_t_s, init_I, MULTIPLIER],
        'ep': [init_eps_t, init_q, init_t_s, init_I, MULTIPLIER]     # s=f(I,q,...) in find_minimum_vectorized
    }
    
    # Perform optimization
    start_time = time.time()
    with st.spinner("Calculating..."):
        opt_config = ('sa', 'c')
        results = {
            'r': find_minimum_vectorized(
                objective_function, c_total, opt_config, 
                guess_bound, *initial_params['r'],
                method=opt_method, tol=tolerance,
                warm_start=warm_start
                ),
            'ir': find_minimum_vectorized(
                objective_function_ir_ratio, c_total, opt_config,
                guess_bound, *initial_params['ir'],
                method=opt_method, tol=tolerance,
                warm_start=warm_start
                ),
            'ep': find_minimum_vectorized(
                objective_function_ep_rate, c_total, opt_config,
                guess_bound, *initial_params['ep'],
                method=opt_method, tol=tolerance,
                warm_start=warm_start
                ),
        }
        df1, df2, df3 = results_to_df(results, c_total, 'c_t', fix=cuttoff_outliers)

    runtime_info(st_runtime_info, start_time, txt)

    calculate_gradient([df1, df2, df3])

    plotting_sensitivity(
        [df1, df2, df3], 
        ['reversibility', 'irrevers. ratio', 'entropy prod. rate'],     # labels
        param_index,                                                    # parameter index (e.g. 'p' for hot loop)
        POWER_OF_10,
        theme_session
    )

    display_results([df1, df2, df3])

@st.fragment
def tab_q0_sa():
    col_control, _, col_info = st.columns((0.34, 0.02, 0.64))
    
    with col_info:
        # settings
        step_size, var_range, *opt_settings = settings_popover('q', DEFAULT_SETTING_VALUES)
        opt_method, tolerance, guess_bound, warm_start, cuttoff_outliers, plot_param = opt_settings
        txt = f'(warm starting*)' if warm_start else ''
        param_index = 'p' if plot_param=='Hot loop' else 'g'

        display_info(guess_bound, txt, plot_param)
    

    with col_control:
        # sliders
        init_eps_t = init_slider(r'$\varepsilon_{total}:$', 'q_e_t', 0.4, 4.0, 0.01)
        init_c_t = init_slider('$c_{total}:$', 'q_c_t', 0.1, 1.0, 0.01)
        init_t_s = init_slider('$t_{s}:$', 'q_t_s', 0.8, 1.0, 0.01)
        init_I = init_slider('$I:$', 'q_I', 1.0, 2.0, 0.01)
        # init_s = init_slider(
        #     '$s:$', 'q_s', 0.1, 30.0, 0.01,
        #     help=fr'$s \times 10^{{-{POWER_OF_10:.0f}}}$'
        # )

        st.button(
            "Reset", on_click=lambda: reset_sliders(DEFAULT_SA_SLIDERS), 
            key='btn_saq', help='Reset Parameters to Defaults'
        )
        st_runtime_info = st.empty()


    q0 = np.arange(var_range[0], var_range[1]+0.0001, step_size)
    initial_params = {
        'r': [init_eps_t, init_c_t, init_t_s, MULTIPLIER],
        'ir': [init_eps_t, init_c_t, init_t_s, init_I, MULTIPLIER],
        'ep': [init_eps_t, init_c_t, init_t_s, init_I, MULTIPLIER]     # s=f(I,q,...) in find_minimum_vectorized
    }

    # Perform optimization
    start_time = time.time()
    with st.spinner("Calculating..."):
        opt_config = ('sa', 'q')
        results = {
            'r': find_minimum_vectorized(
                objective_function, q0, opt_config,
                guess_bound, *initial_params['r'],
                method=opt_method, tol=tolerance,
                warm_start=warm_start
                ),
            'ir': find_minimum_vectorized(
                objective_function_ir_ratio, q0, opt_config,
                guess_bound, *initial_params['ir'],
                method=opt_method, tol=tolerance,
                warm_start=warm_start
                ),
            'ep': find_minimum_vectorized(
                objective_function_ep_rate, q0, opt_config,
                guess_bound, *initial_params['ep'],
                method=opt_method, tol=tolerance,
                warm_start=warm_start
                ),
        }
        df1, df2, df3 = results_to_df(results, q0, 'q0', fix=cuttoff_outliers)

    runtime_info(st_runtime_info, start_time, txt)

    calculate_gradient([df1, df2, df3])

    plotting_sensitivity(
        [df1, df2, df3], 
        ['reversibility', 'irrevers. ratio', 'entropy prod. rate'],     # labels
        param_index,                                                    # parameter index (e.g. 'p' for hot loop)
        POWER_OF_10,
        theme_session
    )

    display_results([df1, df2, df3])

@st.fragment
def tab_ts_sa():
    col_control, _, col_info = st.columns((0.34, 0.02, 0.64))
    
    with col_info:
        # settings
        step_size, var_range, *opt_settings = settings_popover('t', DEFAULT_SETTING_VALUES)
        opt_method, tolerance, guess_bound, warm_start, cuttoff_outliers, plot_param = opt_settings
        txt = f'(warm starting*)' if warm_start else ''
        param_index = 'p' if plot_param=='Hot loop' else 'g'
        
        display_info(guess_bound, txt, plot_param)


    with col_control:
        # sliders
        init_eps_t = init_slider(r'$\varepsilon_{total}:$', 't_e_t', 0.4, 4.0, 0.01)
        init_c_t = init_slider('$c_{total}:$', 't_c_t', 0.1, 1.0, 0.01)
        init_q = init_slider(
            '$q_{0}:$', 't_q0', 1.0, 100.0, 0.1, fmt="%.1f",
            help=fr'$q_{{0}} \times 10^{{-{POWER_OF_10:.0f}}}$'
        )
        init_I = init_slider('$I:$', 't_I', 1.0, 2.0, 0.01)
        # init_s = init_slider(
        #     '$s:$', 't_s', 0.1, 30.0, 0.01,
        #     help=fr'$s \times 10^{{-{POWER_OF_10:.0f}}}$'
        # )

        st.button(
            "Reset", on_click=lambda: reset_sliders(DEFAULT_SA_SLIDERS), 
            key='btn_sat', help='Reset Parameters to Defaults'
        )
        st_runtime_info = st.empty()


    t_s = np.arange(var_range[0], var_range[1]+0.0001, step_size)
    initial_params = {
        'r': [init_eps_t, init_c_t, init_q, MULTIPLIER],
        'ir': [init_eps_t, init_c_t, init_q, init_I, MULTIPLIER],
        'ep': [init_eps_t, init_c_t, init_q, init_I, MULTIPLIER]      # s=f(I,q,...) in find_minimum_vectorized
    }

    
    # Perform optimization
    start_time = time.time()
    with st.spinner("Calculating..."):
        opt_config = ('sa', 't')
        results = {
            'r': find_minimum_vectorized(
                objective_function, t_s, opt_config, 
                guess_bound, *initial_params['r'],
                method=opt_method, tol=tolerance,
                warm_start=warm_start
                ),
            'ir': find_minimum_vectorized(
                objective_function_ir_ratio, t_s, opt_config,
                guess_bound, *initial_params['ir'],
                method=opt_method, tol=tolerance,
                warm_start=warm_start
                ),
            'ep': find_minimum_vectorized(
                objective_function_ep_rate, t_s, opt_config,
                guess_bound, *initial_params['ep'],
                method=opt_method, tol=tolerance,
                warm_start=warm_start
                ),
        }
        df1, df2, df3 = results_to_df(results, t_s, 't_s', fix=cuttoff_outliers)

    runtime_info(st_runtime_info, start_time, txt)

    calculate_gradient([df1, df2, df3])

    plotting_sensitivity(
        [df1, df2, df3], 
        ['reversibility', 'irrevers. ratio', 'entropy prod. rate'],     # labels
        param_index,                                                    # parameter index (e.g. 'p' for hot loop)
        POWER_OF_10,
        theme_session
    )

    display_results([df1, df2, df3])

@st.fragment
def tab_ir_sa():
    col_control, _, col_info = st.columns((0.34, 0.02, 0.64))
    
    with col_info:
        # settings
        step_size, var_range, *opt_settings = settings_popover('I', DEFAULT_SETTING_VALUES)
        opt_method, tolerance, guess_bound, warm_start, cuttoff_outliers, plot_param = opt_settings
        txt = f'(warm starting*)' if warm_start else ''
        param_index = 'p' if plot_param=='Hot loop' else 'g'

        display_info(guess_bound, txt, plot_param)
    

    with col_control:
        # sliders
        init_eps_t = init_slider(r'$\varepsilon_{total}:$', 'I_e_t', 0.4, 4.0, 0.01)
        init_c_t = init_slider('$c_{total}:$', 'I_c_t', 0.1, 1.0, 0.01)
        init_q = init_slider(
            '$q_{0}:$', 'I_q0', 1.0, 100.0, 0.1, fmt="%.1f",
            help=fr'$q_{{0}} \times 10^{{-{POWER_OF_10:.0f}}}$'
        )
        init_t_s = init_slider('$t_{s}:$', 'I_t_s', 0.8, 1.0, 0.01)

        st.button(
            "Reset", on_click=lambda: reset_sliders(DEFAULT_SA_SLIDERS), 
            key='btn_sai', help='Reset Parameters to Defaults'
        )
        st_runtime_info = st.empty()


    I = np.arange(var_range[0], var_range[1]+step_size, step_size)
    initial_params = {
        'ir': [init_eps_t, init_c_t, init_q, init_t_s, MULTIPLIER],
        'ep': [init_eps_t, init_c_t, init_q, init_t_s, MULTIPLIER],
    }

    # Perform optimization
    start_time = time.time()
    with st.spinner("Calculating..."):
        opt_config = [('sa', 'I'), ('sa', 's')]
        results = {
            'ir': find_minimum_vectorized(
                objective_function_ir_ratio, I, opt_config[0],
                guess_bound, *initial_params['ir'],
                method=opt_method, tol=tolerance,
                warm_start=warm_start
                ),
            'ep': find_minimum_vectorized(
                objective_function_ep_rate, I, opt_config[1],
                guess_bound, *initial_params['ep'],
                method=opt_method, tol=tolerance,
                warm_start=warm_start
                ),
        }
        
        df1, df2 = results_to_df(results, I, 'I', fix=cuttoff_outliers)
        df2 = df2.reset_index().set_index('s')

    runtime_info(st_runtime_info, start_time, txt)


    calculate_gradient([df1, df2])

    plotting_sensitivity(
        [df1, df2.set_index('I')], 
        ['irrevers. ratio', 'entropy prod. rate'],                      # labels
        param_index,                                                    # parameter index (e.g. 'p' for hot loop)
        POWER_OF_10,
        theme_session
    )

    display_results([df1, df2])

@st.fragment
def sensitivity_analysis():
    init_session_state(DEFAULT_SETTING_VALUES)
    init_session_state(DEFAULT_SA_SLIDERS)

    st.info(r'Choose a variable/parameter to analyze its impact on the minimum power consumption $min(w)$')
    tab_e, tab_c, tab_q, tab_t, tab_ir = st.tabs(
        [
        r'$\varepsilon_{total}$', '$c_{total}$', 
        '&nbsp;&nbsp;&nbsp;&nbsp;$q_0$&nbsp;&nbsp;&nbsp;&nbsp;', 
        '&nbsp;&nbsp;&nbsp;&nbsp;$t_s$&nbsp;&nbsp;&nbsp;&nbsp;', 
        '&nbsp;&nbsp;&nbsp;$I\ \&\ s$&nbsp;&nbsp;&nbsp;', 
        ]
    )


    with tab_e:
        st.markdown(
            r'''
            Minimum power consumption $\min(w)$ as a function of $\varepsilon_{total}$, with parameters $c_{total}$, $q_0$, $t_0$, $I$, and $s$.  
            Parameters $I$ and $s$ are for the irreversibility ratio and entropy production rate, respectively.
            '''
        )
        st.write('')
        tab_e_total_sa()

    with tab_c:
        st.markdown(
            r'''
            Minimum power consumption $\min(w)$ as a function of $c_{total}$, with parameters $\varepsilon_{total}$, $q_0$, $t_0$, $I$, and $s$.  
            Parameters $I$ and $s$ are for the irreversibility ratio and entropy production rate, respectively.
            '''
        )
        st.write('')
        tab_c_total_sa()

    with tab_q:
        st.markdown(
            r'''
            Minimum power consumption $\min(w)$ as a function of $q_0$, with parameters $\varepsilon_{total}$, $c_{total}$, $t_0$, $I$, and $s$.  
            Parameters $I$ and $s$ are for the irreversibility ratio and entropy production rate, respectively.
            '''
        )
        st.write('')
        tab_q0_sa()

    with tab_t:
        st.markdown(
            r'''
            Minimum power consumption $\min(w)$ as a function of $t_s$, with parameters $\varepsilon_{total}$, $c_{total}$, $q_0$, $I$, and $s$.  
            Parameters $I$ and $s$ are for the irreversibility ratio and entropy production rate, respectively.
            '''
        )
        st.write('')
        tab_ts_sa()

    with tab_ir:
        st.markdown(
            r'''
            Minimum power consumption $\min(w)$ as a function of parameters $I$ and $s$, with parameters $\varepsilon_{total}$, $c_{total}$, $q_0$, and $t_s$.  
            Parameters $I$ and $s$ are for the irreversibility ratio and entropy production rate, respectively.
            '''
        )
        st.write('')
        tab_ir_sa()



#==============SENSITIVITY ANALYSIS==============
st.markdown('***')
st.markdown('## Sensitivity analysis')
st.markdown('Here you can perform a sensitivity analysis.')

if st.button("Analyze", type="primary", icon=":material/search_insights:", help='Run sensitivity analysis'):
    with st.spinner("Calculating..."):
        start = time.time()
        sensitivity_analysis()
        stop = time.time()

        st.toast(
            f'''
            Sensitivity analysis done!  
            Run time: {stop - start:.2f} s
            ''', 
            icon=':material/done_all:'
        )