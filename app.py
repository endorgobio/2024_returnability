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
from utilities import create_instance, create_map, create_df_coord, create_df_OF, graph_costs, create_df_util, graph_utilization, parameters
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

# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
# Creates the app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets,
                title="retornabilidad",
                suppress_callback_exceptions=True)

# Create tabs
# TAB 2: Results

# controls_model = dbc.Col([
#     # dbc.Form()   
    
#         dbc.Row([
#             dbc.Col(dbc.Label("Valor envase retornable"), width=3),
#             dbc.Col(
#                 dbc.Input(id="valor_envase", type="number", min=0, max=3000, step=1, value=1000, placeholder="Enter email"),
#                 width=2
                
#                 )
#             ]),
#         dbc.Row([
#             dbc.Col(dbc.Label("valor depósito"), width=3),
#             dbc.Col(
#                 dbc.Input(id="deposito", type="number", min=0, max=300, step=1, value=100, placeholder="Enter email"),
#                 width=2)
#             ]),
#         dbc.Row([
#             dbc.Col(dbc.Label("Costo clasificación"), width=3),
#             dbc.Col(
#                 dbc.Input(id="cost_class", type="number", min=0, max=300, step=1, value=100, placeholder="Enter email"),
#                 width=2)
#             ]),
#         dbc.Row([
#             dbc.Col(dbc.Label("Costo lavado"), width=3),
#             dbc.Col(
#                 dbc.Input(id="cost_wash", type="number", min=0, max=300, step=1, value=100, placeholder="Enter email"),
#                 width=2)
#             ]),
#         dbc.Row([
#             dbc.Col(dbc.Label("Costo transporte"), width=3),
#             dbc.Col(
#                 dbc.Input(id="cost_transp", type="number", min=0, max=300, step=1, value=100, placeholder="Enter email"),
#                 width=2)
#             ]),
#         dbc.Row([
#             dbc.Col(width=3),
#             dbc.Col(dbc.Button("Resolver", id="solving", className="mt-3", n_clicks=0), width=2)           
#             ],
#             align="right"
#             )
        
#     ]
#     )

tab1_content = dbc.Container([
       
    
    dbc.Row([
        dbc.Col(
            [dbc.Row(html.H4("Cargar archivos",  className="text-left"), className="mt-3 mb-3"),
            dbc.Row([dbc.Col(dbc.Label("Lista actores"), width=8),
                     dbc.Col(dcc.Upload(
                             id="upload-actors",
                             children=dbc.Button("Cargar"),
                             accept=".xlsx"), 
                         width=3)],
                    className="mt-1 mb-1"), 
            dbc.Row([dbc.Col(dbc.Label("Distancias"), width=8),
                     dbc.Col(dcc.Upload(
                             id="upload-distances",
                             children=dbc.Button("Cargar"),
                             accept=".xlsx"), 
                         width=3)],
                    className="mt-1 mb-1"), 
            dbc.Row(html.H4("Configuración", className="text-left"), className="mt-3 mb-3"),#dbc.Label("Ajuste de parámetros"),
                  #dbc.Row(controls_model)
            dbc.Row([
                      dbc.Col(dbc.Label("Valor envase retornable"), width=8),
                      dbc.Col(
                          dbc.Input(id="valor_envase", type="number", min=0, max=3000, step=1, value=parameters['enr'], placeholder="Enter email"),
                          width=3                    
                          )
                      ]),
            dbc.Row([
                      dbc.Col(dbc.Label("valor depósito"), width=8),
                      dbc.Col(
                          dbc.Input(id="deposito", type="number", min=0, max=300, step=1, value=parameters['dep'], placeholder="Enter email"),
                          width=3)
                      ]),
            dbc.Row([
                      dbc.Col(dbc.Label("Costo clasificación"), width=8),
                      dbc.Col(
                          dbc.Input(id="cost_class", type="number", min=0, max=300, step=1, value=parameters['qc'], placeholder="Enter email"),
                          width=3)
                      ]),
            dbc.Row([
                      dbc.Col(dbc.Label("Costo lavado"), width=8),
                      dbc.Col(
                          dbc.Input(id="cost_wash", type="number", min=0, max=300, step=1, value=parameters['ql'], placeholder="Enter email"),
                          width=3)
                      ]),
            dbc.Row([
                      dbc.Col(dbc.Label("Costo transporte"), width=8),
                      dbc.Col(
                          dbc.Input(id="cost_transp", type="number", min=0, max=300, step=1, value=parameters['qa'], placeholder="Enter email"),
                          width=3)
                      ]),
            dbc.Row([
                      dbc.Col(width=8),
                      dbc.Col(dbc.Button("Resolver", id="solving", className="mt-3", n_clicks=0), width=3)           
                      ],
                      align="right"
                      )],              
            width=5),
        dbc.Col(
            dcc.Loading(
            dcc.Graph(id="map", 
                      figure = create_map(pd.DataFrame(columns=['type']))
                      )
            ),
            width=7
        )],
        # align="center",
        ),
    dbc.Row(
        id="toggle-row",
        children=[
            dbc.Col([
                dbc.Row(html.H4("Ingresos y costos", className="text-center")),
                dbc.Row(dcc.Graph(id="graph_utility"))],
            width=5),,
            dbc.Col([
                dbc.Row(html.H4("Utilización de la capacidad", className="text-center")),
                dbc.Row(dcc.Graph(id="graph_utilization"))], 
                width=5),
        ],
        style={"display": "None"}  # Initially not visible
    )],
    className="align-items-start",
    fluid=True)
tab2_content = html.Div("hola")





# creates layout
app.layout = dbc.Container([
    dbc.Row(html.Img(src='assets/images/header1.png', style={'width': '100%'})),
    dbc.Tabs(
        [
            dbc.Tab(label="Tablero", tab_id="dashboard", label_style=tab_label_style, active_label_style=activetab_label_style),
            dbc.Tab(label="Instrucciones", tab_id="instructions", label_style=tab_label_style, active_label_style=activetab_label_style),
        ],
        id="tabs",
        active_tab="dashboard",
        ),
    dbc.Container(id="tab-content", fluid=True, className="inner-container d-flex justify-content-center align-items-center",),
    ],
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
    if active_tab == "dashboard":
        return tab1_content
    elif active_tab == "instructions":
        return tab2_content


    
# # need to run it in heroku
# server = app.server
app.run_server(debug=True)

# Solve the model or apply filter
@app.callback(Output('map', 'figure'),
              Output("toggle-row", "style"),
              Output('graph_utility', 'figure'),
              Output('graph_utilization', 'figure'),
              Input('solving', 'n_clicks'),
              State('valor_envase', 'value'),
              State('deposito', 'value'),
              State('cost_class', 'value'),
              State('cost_wash', 'value'),
              State('cost_transp', 'value')
              )
def run_model_graph(click_resolver, 
                    container_value, 
                    deposit, 
                    cost_class,
                    cost_wash,
                    cost_transp):
    
    if click_resolver > 0:
        # update parameter values
        parameters['enr'] = container_value
        parameters['dep'] = deposit
        parameters['qc'] = cost_class
        parameters['ql'] = cost_wash
        parameters['qa'] = cost_transp
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
        
        df_utiliz = create_df_util(var_sol, parameters)
        graph_utiliz= graph_utilization(df_utiliz)
        
        df_obj = df_obj[['Type', 'Name', '%']]    
        df_obj = df_obj.drop([1, 2, 3])
        #df_obj = df_obj.set_index('Category')
        # updated_columns=[{"name": col, "id": col} for col in df_obj.columns]
        # updated_data=df_obj.to_dict('records')  # Convert DataFrame to dictionary for DataTable
            
        
        return map_actors, {"display": "flex"}, graph_utility, graph_utiliz #updated_columns, updated_data
    else:
        raise PreventUpdate

