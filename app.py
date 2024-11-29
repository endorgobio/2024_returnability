# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 11:28:16 2024

@author: DELL
"""

import numpy as np
import pandas as pd
import dash
from dash import html
from dash import dcc
from dash import dash_table
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from utilities import create_instance, create_map, create_df_coord, create_df_OF, graph_costs, parameters
from optimiser_gurobi import create_model, get_vars_sol, get_obj_components
import plotly.graph_objects as go

tabs_styles = {
    'height': '44px',
    'align-items': 'center'
}

tab_label_style = {
    'color' : 'black'
}

activetab_label_style = {
    'color': '#079877',
    'fontWeight': 'bold'
}


# Read data
df_coord = pd.read_csv('https://docs.google.com/uc?export=download&id=1VYEnH735Tdgqe9cS4ccYV0OUxMqQpsQh') # coordenadas
df_dist = pd.read_csv('https://docs.google.com/uc?export=download&id=1Apbc_r3CWyWSVmxqWqbpaYEacbyf1wvV') # distancias
df_demand = pd.read_csv('https://docs.google.com/uc?export=download&id=1w0PMK36H4Aq39SAaJ8eXRU2vzHMjlWGe') # demandas escenario base
parameters['df_coord'] = df_coord
parameters['df_dist'] = df_dist
parameters['df_demand'] = df_demand



# Define the stylesheets
external_stylesheets = [dbc.themes.BOOTSTRAP,
    #'https://codepen.io/chriddyp/pen/bWLwgP.css'
    'https://fonts.googleapis.com/css2?family=Lato:wght@400;700&display=swap',
    #'https://fonts.googleapis.com/css2?family=Roboto&display=swap" rel="stylesheet'
]

# Creates the app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                title="retornabilidad",
                suppress_callback_exceptions=True)

# Create tabs
# TAB 2: Results

controls_model = dbc.Container([
    dbc.Form([
        dbc.Row([
            dbc.Label("Valor envase", width=4),
            dbc.Col(
                dbc.Input(id="valor_envase", type="number", min=0, max=3000, step=1, value=1000, placeholder="Enter email"),
                width=4)
            ]),
        dbc.Row([
            dbc.Label("valor depósito", width=4),
            dbc.Col(
                dbc.Input(id="deposito", type="number", min=0, max=300, step=1, value=100, placeholder="Enter email"),
                width=4)
            ]),
        dbc.Row([
            dbc.Col(md=2),
            dbc.Col(dbc.Button("Resolver", id="solving", className="mt-3", n_clicks=0), md=2)           
            ],
            align="right")
        ])   
    ]
    )

tab1_content = dbc.Container([
    dbc.Row([
        dbc.Col([dbc.Label("Ajuste de parámetros"),
                 controls_model,
                 
                 ],  md=4),
        dbc.Col(md=1),
        dbc.Col(
            dcc.Loading(
            dcc.Graph(id="map", 
                      figure = create_map(pd.DataFrame(columns=['type']))
                      )),
            md=7
        )],
        align="center",
        ),
    dbc.Row(
        id="toggle-row",
        children=[
            dbc.Col([
                html.H4("Ingresos y costos", className="text-center"),
                dcc.Graph(id="graph_utility")],
                # dash_table.DataTable(
                #     id='tableOF',
                #     columns=[{"name": col, "id": col} for col in df.columns],  # Initial columns
                #     data=df.to_dict('records'),  # Initial data
                #     style_table={'overflowX': 'auto'},  # Handle horizontal scrolling if needed
                #     style_cell={
                #         'textAlign': 'left',  # Align text to the left
                #         'padding': '10px'  # Add some padding
                #     },
                #     style_header={
                #         'backgroundColor': 'lightgrey',  # Header styling
                #         'fontWeight': 'bold'
                #     },
                # ),
            width=6),            
            dbc.Col(html.P("This is column 1 in a row."), width=6),
        ],
        style={"display": "None"}  # Initially not visible
    )],
    className="align-items-start")
tab2_content = html.Div("hola")
tab3_content = html.Div("hola")




# creates layout
app.layout = dbc.Container([
    dbc.Row(html.Img(src='assets/images/header1.png', style={'width': '100%'})),
    # Loading allows the spinner showing something is runing
    # dcc.Loading(
    #             # dcc.Store inside the app that stores the intermediate value
    #             children=[dcc.Store(id='data_solver'),
    #                       dcc.Store(id='data_solver_filtered')],
    #             id="loading",
    #             fullscreen=True,
    #             type='circle'
    #             ),
    dbc.Tabs(
        [
            dbc.Tab(label="La historia", tab_id="historia", label_style=tab_label_style, active_label_style=activetab_label_style),
            dbc.Tab(label="Instrucciones", tab_id="solucion", label_style=tab_label_style, active_label_style=activetab_label_style),
            dbc.Tab(label="Resultados", tab_id="detalles", label_style=tab_label_style, active_label_style=activetab_label_style),
        ],
        id="tabs",
        active_tab="historia",
        ),
    dbc.Row(id="tab-content", className="p-4"),],
    fluid=True
    )

# Render the tabs depending on the selection
@app.callback(
    Output("tab-content", "children"),
    Input("tabs", "active_tab"),
)
def render_tab_content(active_tab):
    """
    This callback takes the 'active_tab' property as input, as well as the
    stored graphs, and renders the tab content depending on what the value of
    'active_tab' is.
    """
    if active_tab == "historia":
        return tab1_content
    elif active_tab == "solucion":
        return tab2_content
    elif active_tab == "detalles":
        return tab3_content


    
# # need to run it in heroku
# server = app.server
app.run_server(debug=True)

# Solve the model or apply filter
@app.callback(Output('map', 'figure'),
              Output("toggle-row", "style"),
              Output('graph_utility', 'figure'),
              # Output("tableOF", "columns"),
              # Output("tableOF", "data"),
              Input('solving', 'n_clicks'),
              State('valor_envase', 'value'),
              State('deposito', 'value')
              )
def run_model_graph(click_resolver, valor_envase, deposito):
    
    if click_resolver > 0:
        # update parameter values
        parameters['enr'] = valor_envase
        parameters['dep'] = deposito
        # create and solve model
        instance = create_instance(parameters)
        model = create_model(instance)
        model.setParam('MIPGap', 0.05) # Set the MIP gap tolerance to 5% (0.05)
        model.optimize()
        # get solution
        var_sol = get_vars_sol(model)
        df_sol = create_df_coord(var_sol, df_coord)
        results_obj = get_obj_components(model)
        df_obj = create_df_OF(results_obj)
        # create map
        map_actors = create_map(df_sol)
        # create utility graph
        graph_utility = graph_costs(df_obj)   
        
        
        
        df_obj = df_obj[['Type', 'Name', '%']]    
        df_obj = df_obj.drop([1, 2, 3])
        #df_obj = df_obj.set_index('Category')
        updated_columns=[{"name": col, "id": col} for col in df_obj.columns]
        updated_data=df_obj.to_dict('records')  # Convert DataFrame to dictionary for DataTable
            
        
        return map_actors, {"display": "flex"}, graph_utility #updated_columns, updated_data
    else:
        raise PreventUpdate

# instance = create_instance(parameters)
# model = create_model(instance)
# model.setParam('MIPGap', 0.2) # Set the MIP gap tolerance to 5% (0.05)
# model.optimize()
# var_sol = get_vars_sol(model)
# results_obj = get_obj_components(model)
# df_obj = create_df_OF(results_obj)

# import dash
# from dash import dcc, html
# import plotly.graph_objects as go

# # Sample data
# categories = ['A', 'B', 'C', 'D']
# categories2 = ['E', 'F']
# percentages = [45, 80, 90, 65]  # Example percentages
# percentages2 = [45, 80]

    
# fig = graph_costs(df_obj)   


# # Dash app
# app = dash.Dash(__name__)

# app.layout = html.Div([
#     html.H1("Dash Plotly Dashboard"),
#     dcc.Graph(figure=fig)  # Embed the figure in the dashboard
# ])

# if __name__ == '__main__':
#     app.run_server(debug=True)
