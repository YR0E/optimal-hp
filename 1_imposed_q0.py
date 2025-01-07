import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from streamlit_theme import st_theme
from util.calc_imposed_q0 import find_minimum, objective_function, objective_function_ir_ratio, objective_function_ep_rate
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


#============MAIN==============
st.markdown('***')
st.markdown("## Minimum power consumption")
st.markdown('Text about...')


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
DEFAULT_VALUES_SA_E = {
    'e_t_sae': (1.0, 4.0),
    'c_t_sae': 0.7,
    'q0_sae': 10.0,
    't_s_sae': 0.9,
    'I_sae': 1.1,
    's_sae': 1.1
}

def init_session_state(default):
    for key, value in default.items():
        if key not in st.session_state:
            st.session_state[key] = value

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
    init_session_state(DEFAULT_VALUES_EPS)
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



#===========C_TOTAL================
with tab_c_total:
    init_session_state(DEFAULT_VALUES_C)
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


    with col_plot:
        plotting3D(results, initial_params, opt_var)

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



#=========SENSITIVITY ANALYSIS================
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
def find_minimum_of_(e_t, opt_var):
    # Dynamically construct initial_params for each `e_t`
    initial_params = (e_t, init_c_t, init_q, init_t_s, MULTIPLIER)
    return find_minimum(objective_function, initial_params, opt_var)

def find_minimum_ofir_(e_t, opt_var):
    # Dynamically construct initial_params for each `e_t`
    initial_params_ir = (e_t, init_c_t, init_q, init_t_s, init_I, MULTIPLIER)
    return find_minimum(objective_function_ir_ratio, initial_params_ir, opt_var)

def find_minimum_ofep_(e_t, opt_var):
    # Dynamically construct initial_params for each `e_t`
    initial_params_ep = (e_t, init_c_t, init_q, init_t_s, init_s, MULTIPLIER)
    return find_minimum(objective_function_ep_rate, initial_params_ep, opt_var)

# Vectorize the wrapper function
find_minimum_v = np.vectorize(find_minimum_of_)
find_minimum_ir_v = np.vectorize(find_minimum_ofir_)
find_minimum_ep_v = np.vectorize(find_minimum_ofep_)


with tab_e:
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

    e_total = np.arange(init_eps_t[0], init_eps_t[1]+0.1, 0.1)
    result, result_ir, result_ep = np.zeros((3, len(e_total)), dtype='object')
    minw, minw_ir, minw_ep = np.zeros((3, len(e_total)))
    res, res_ir, res_ep = np.zeros((3, len(e_total)))
    

    # Call the vectorized function
    with st.spinner("Calculating..."):
        opt_var = 'sae'
        result = find_minimum_v(e_total, opt_var)
        result_ir = find_minimum_ir_v(e_total, opt_var)
        result_ep = find_minimum_ep_v(e_total, opt_var)
    
    df1 = pd.DataFrame.from_records(
        [(e, r.fun * MULTIPLIER, *r.x) for e, r in zip(e_total, result)],
        columns=['ε_t', 'minw', 'ε*_g', 'ε*_p', 'ε*_ev', 'ε*_cd', 'c*_g', 'c*_p']
    ).set_index('ε_t')
    df2 = pd.DataFrame.from_records(
        [(e, r.fun * MULTIPLIER, *r.x) for e, r in zip(e_total, result_ir)],
        columns=['ε_t', 'minw', 'ε*_g', 'ε*_p', 'ε*_ev', 'ε*_cd', 'c*_g', 'c*_p']
    ).set_index('ε_t')
    df3 = pd.DataFrame.from_records(
        [(e, r.fun * MULTIPLIER, *r.x) for e, r in zip(e_total, result_ep)],
        columns=['ε_t', 'minw', 'ε*_g', 'ε*_p', 'ε*_ev', 'ε*_cd', 'c*_g', 'c*_p']
    ).set_index('ε_t')
    df1[(df1 < 0) & (df1 > 1000)] = None
    df2[(df2 < 0) & (df2 > 1000)] = None
    df3[(df3 < 0) & (df3 > 1000)] = None



    with col_plot:
        plotting_sensitivity(
            [df1, df2, df3], 
            ['reversibility', 'irreversibility ratio', 'entropy production rate'], 
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