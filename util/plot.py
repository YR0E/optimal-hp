import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.colors import DEFAULT_PLOTLY_COLORS
from util.calc_imposed_q0 import objective_function, objective_function_ir_ratio, objective_function_ep_rate 

# Set default layout settings for axes title fonts
DEFAULT_FONT = dict(family='STIX Two Math', size=14)

def plotting3D(res, initial_params, opt_var):
    """
    Plots a 3D surface chart using Plotly based on the optimization results.

    Parameters:
    - res: Dictionary containing optimization results for different configurations ('r', 'ir', 'ep').
    - initial_params: Dictionary containing initial parameters including 'r', 'ir', 'ep'.
    - opt_var: String specifying the optimization variable ('e' or 'c').

    This function creates a 3D plot with surfaces representing different objectives:
    - Reversibility
    - Irreversibility Ratio
    - Entropy Production Rate

    It also plots the constraint line and highlights the minimum points on the surfaces.
    The plot configuration supports high-resolution image export.
    """

    MULTIPLIER = initial_params['r'][-1]
    POWER_OF_10 = np.log10(MULTIPLIER)
  
    # Define variable-specific parameters and ranges
    var_map = {
        'e': {
            'init_value': initial_params['r'][2],   # e_total
            'var_name': 'ε',
            'values_range': [0.1, 1],
            'step_size': 0.025,
        },
        'c': {
            'init_value': initial_params['r'][2],   # c_total
            'var_name': 'c',
            'values_range': [0.05, 1],
            'step_size': 0.01,
        }
    }
   
    params = var_map[opt_var]
    init_val = params['init_value']
    var_name = params['var_name']
    step_size = params['step_size']
    val_range = params['values_range']
    
    # Create meshgrid for plotting
    x_axis = np.arange(val_range[0], val_range[1] + step_size, step_size)    # g
    y_axis = np.arange(val_range[0], val_range[1] + step_size, step_size)    # p
    X, Y = np.meshgrid(x_axis, y_axis)
    
    # objective function arguments/ constraint limit
    if opt_var=='e':
        args = [X, Y, X, Y]
        constraint_limit = init_val/2
    elif opt_var=='c':
        args = [X, Y]
        constraint_limit = init_val
    
    Z = objective_function(args, initial_params['r'], opt_var) * MULTIPLIER
    Z_ir = objective_function_ir_ratio(args, initial_params['ir'], opt_var) * MULTIPLIER
    Z_ep = objective_function_ep_rate(args, initial_params['ep'], opt_var) * MULTIPLIER


    # contraint line
    min_line_lim, max_line_lim = np.min([Z, Z_ir, Z_ep]), np.max([Z, Z_ir, Z_ep])
    threshold = val_range[0] + val_range[1]

    if constraint_limit <= threshold:
        X_line = np.linspace(val_range[0], constraint_limit - val_range[0], 11)
    else:
        X_line = np.linspace(constraint_limit - val_range[1], val_range[1], 11)

    X_line, Z_line = np.meshgrid(X_line, np.linspace(min_line_lim, max_line_lim, 11))
    Y_line = constraint_limit - X_line


    # minimum points
    min_points = [(res[k].x[0], res[k].x[1], res[k].fun * MULTIPLIER) for k in ['r', 'ir', 'ep']]
    x_min, y_min, z_min = zip(*min_points)

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
    common_surface_props = {
        'x': X, 
        'y': Y, 
        'opacity': 0.75, 
        'showlegend': True, 
        'showscale': False, 
        'contours': contours,
    }

    fig = go.Figure()
    fig.add_trace(go.Surface(
        z=Z,
        name='reversibility', 
        legendgroup='reversibility', 
        colorscale='Viridis', 
        **common_surface_props,
        hovertemplate="<b style='color:red;'>reversibility</b><br>" + 
            "x: %{x}<br>" +
            "y: %{y}<br>" +
            "z: %{z}<extra></extra>"
    ))
    fig.add_trace(go.Surface(
        z=Z_ir,  
        name='irreversibility ratio', 
        legendgroup='irreversibility',
        colorscale='RdBu_r', 
        **common_surface_props,
        hovertemplate="<b style='color:red;'>irreversibility ratio</b><br>" + 
            "x: %{x}<br>" +
            "y: %{y}<br>" +
            "z: %{z}<extra></extra>"
    ))
    fig.add_trace(go.Surface(
        z=Z_ep,  
        name='entropy production rate', 
        legendgroup='entropy production', 
        colorscale='rdylgn_r', 
        **common_surface_props,
        hovertemplate="<b style='color:red;'>entropy production rate</b><br>" + 
            "x: %{x}<br>" +
            "y: %{y}<br>" +
            "z: %{z}<extra></extra>"
    ))
    fig.add_trace(go.Surface(
        z=Z_line, x=X_line, y=Y_line, 
        name='constraint', legendgroup='constraint', 
        colorscale=[[0, 'red'], [1, 'red']], showlegend=True, showscale=False, opacity=0.1,
        hovertemplate="<b style='color:gray;'>constrain</b><br>" + 
            "x: %{x}<br>" +
            "y: %{y}<br>" +
            "z: %{z}<extra></extra>"
    ))
    fig.add_trace(go.Scatter3d(
        x=x_min, y=y_min, z=z_min,
        mode='markers',
        name='minimum', showlegend=True,
        marker=dict(size=5, color='red', symbol='circle'),
        hovertemplate="<b style='color:red;'>minimum</b><br>" + 
            "x: %{x}<br>" +
            "y: %{y}<br>" +
            "z: %{z}<extra></extra>"
    ))

    fig.update_layout(
        title=dict(
            text="Objective function surface, Constraint and Minimum",
            x=0.5,
            y=0.005,
            xanchor="center",
            yanchor="bottom",
            font=dict(family="Arial, sans-serif​", size=13, color="#abacb0")
        ),
        autosize=True,
        height=520,
        margin=dict(l=10, r=10, b=40, t=30),
        scene=dict(
            xaxis=dict(
                title=f'<i>{var_name}<sub>g</sub></i>',
                title_font=DEFAULT_FONT,
                range=[0, 1],
            ),
            yaxis=dict(
                title=f'<i>{var_name}<sub>p</sub></i>',
                title_font=DEFAULT_FONT,
                range=[0, 1],
            ),
            zaxis=dict(
                title=f'<i>w</i> · 10<sup>−{POWER_OF_10:.0f}</sup>',
                title_font=DEFAULT_FONT,
            ),
            # aspectratio=dict(x=1, y=1, z=1),
            aspectmode='cube',
            camera=dict(
                eye=dict(x=2, y=1., z=0.5)
            )
        ),
        legend=dict(
            yanchor="top",
            y=1,
            xanchor="center",
            x=0.5,
            orientation="h"
        ),
        
    )

    st.plotly_chart(fig, use_container_width=True, config=config)



def plotting_sensitivity(data, params, labels, power, theme_session):
    """
    Plots sensitivity analysis of optimization results and optima.

    Parameters:
    data (list of pd.DataFrame): DataFrames containing the results with columns 'minw', 'c*_g', and 'ε*_g'.
    labels (list of str): List of labels for the different data sets.
    power (float): Power of 10 to scale the y-axis.
    theme_session (str): Streamlit theme session name for consistent UI styling.
    """

    fig = make_subplots(
        rows=1, cols=3,
        horizontal_spacing=0.075,
    )

    color_cycle = DEFAULT_PLOTLY_COLORS[:3]  # Get as many colors for 3 labels
    dash_cycle = ['5px', 'solid', 'solid']
    if len(labels)==2:
        color_cycle = color_cycle[1:]
        dash_cycle = dash_cycle[1:]

    # Set the configuration for the Plotly chart, including the resolution settings
    config = {
        "toImageButtonOptions": {
            "format": "png",  # The format of the exported image (png, svg, etc.)
            "filename": "sensitivity_plot",  # Default filename
            # "height": 1080,  # Image height
            # "width": 1920,   # Image width
            "scale": 3       # Increase the resolution (scales up the image)
        },
        'modeBarButtonsToRemove': ['zoomIn', 'zoomOut'],
        'displaylogo': False
    }

    varname = data[0].index.name
    if varname[0] in ['c', 'ε']:    
        x_title = f'<i>{varname[0]}<sub>total</sub></i>'
    elif varname[0] in ['q', 't']:
        x_title = f'<i>{varname[0]}<sub>{varname[-1]}</sub></i>'
    else:
        x_title = f'<i>{varname}</i>'

    ytitle_index = params[1][-1]    # for example, g from e*_g
    ytitles = [
        f'<i>min(w)</i> · 10<sup>−{power:.0f}</sup>',
        f'<i>ε*<sub>{ytitle_index}</sub></i>',
        f'<i>c*<sub>{ytitle_index}</sub></i>',
    ]

    for df, label, color, dash in zip(data, labels, color_cycle, dash_cycle):
        fig.add_trace(go.Scatter(
            x=df.index, y=df[params[0]],
            mode='lines',
            line=dict(color=color, dash=dash),
            name=label, legendgroup=label,
            showlegend=True,
            ), 
            row=1, col=1
        )

    for df, label, color, dash in zip(data, labels, color_cycle, dash_cycle):
        fig.add_trace(go.Scatter(
            x=df.index, y=df[params[1]],
            mode='lines',
            line=dict(color=color, dash=dash),
            name=label, legendgroup=label, 
            showlegend=False,
            ), 
            row=1, col=2
        )
    
    for df, label, color, dash in zip(data, labels, color_cycle, dash_cycle):
        fig.add_trace(go.Scatter(
            x=df.index, y=df[params[2]],
            mode='lines',
            line=dict(color=color, dash=dash),
            name=label, legendgroup=label, 
            showlegend=False,
            ), 
            row=1, col=3
        )

    
    fig.update_layout(
        title=dict(
            text="Influence of the parameter on the optimal value and optimal solutions",
            x=0.5,
            y=0.01, 
            xanchor="center",
            yanchor="bottom",
            font=dict(family="Arial Light, sans-serif​", size=13, color="#84858B")
        ),

        autosize=True,
        # height=450,
        margin=dict(l=5, r=5, b=80, t=10),
        xaxis=dict(
            title=x_title,
            title_font=DEFAULT_FONT,
            title_standoff=18
        ),
        xaxis2=dict(
            title=x_title,
            title_font=DEFAULT_FONT,
            title_standoff=18,
        ),
        xaxis3=dict(
            title=x_title,
            title_font=DEFAULT_FONT,
            title_standoff=18,
        ),

        yaxis=dict(
            title=ytitles[0],
            title_font=DEFAULT_FONT,
            title_standoff=18
        ),
        yaxis2=dict(
            title=ytitles[1],
            title_font=DEFAULT_FONT,
            title_standoff=18
        ),
        yaxis3=dict(
            title=ytitles[2],
            title_font=DEFAULT_FONT,
            title_standoff=18
        ),
        legend=dict(
            yanchor="top", y=1.1,
            xanchor="center", x=0.5,
            orientation="h",
            
        ),
        hovermode='x',
    )

    st.plotly_chart(fig, use_container_width=True, config=config, theme=theme_session)