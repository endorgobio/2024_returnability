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
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
from utilities import create_instance, parameters, get_vars_sol
from optimiser_gurobi import create_model
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


# CLAIO
df_coord = pd.read_csv('https://docs.google.com/uc?export=download&id=1VYEnH735Tdgqe9cS4ccYV0OUxMqQpsQh') # coordenadas
df_dist = pd.read_csv('https://docs.google.com/uc?export=download&id=1Apbc_r3CWyWSVmxqWqbpaYEacbyf1wvV') # distancias
df_demand = pd.read_csv('https://docs.google.com/uc?export=download&id=1w0PMK36H4Aq39SAaJ8eXRU2vzHMjlWGe') # demandas escenario base




# Data for the bar chart
categories = ['Category A', 'Category B', 'Category C', 'Category D']
values = [15, 30, 45, 10]

# Create the bar chart
barplot = go.Figure(data=[
    go.Bar(x=categories, y=values, marker_color='lightblue')
])


def create_map(df):

    #actors = ['clasification', 'collection', 'producer', 'washing']
    map_actors = go.Figure()
    
    actors = list(df['type'].unique())
    if len(actors) == 0:
        map_actors.add_trace(go.Scattermapbox(
            lat=[],
            lon=[],
            mode='markers',
        ))
        
    for actor in actors:
        df_filter = df[df['type']==actor]
    
        # Add layer 1
        map_actors.add_trace(go.Scattermapbox(
            lat=df_filter['latitude'],
            lon=df_filter['longitude'],
            mode='markers',
            marker=go.scattermapbox.Marker(
                size=12,
                opacity=0.8
            ),
            text=df_filter['id'],
            textposition='top right',
            name=actor
        ))
    
    # Set map layout
    map_actors.update_layout(
        mapbox=dict(
            style="open-street-map",  # Map style
            center=dict(lat=6.261943611002649, lon=-75.58979925245441),  # Center of the map
            zoom=4  # Zoom level
        ), 
        autosize=True,
        hovermode='closest',
        showlegend=True,
        height=600
    )   


    return map_actors

# parameters = {
#     # Basic parameters
#     "n_acopios": 10,               # maximum 344
#     "n_centros": 5,                # maximum 5
#     "n_plantas": 3,                # maximum 3
#     "n_productores": 5,            # maximum 5
#     "n_envases": 3,
#     "n_periodos": 5,

#     # Technical parameters
#     "ccv": 337610,  #130*2597                  # Classification capacity of the valorization centers
#     "acv": 418117, # 161*2597                    # Storage capacity of the valorization centers
#     "lpl": 168805, # 65*2597                     # Washing capacity of the washing plants
#     "apl": 623280, #240*2597                    # Storage capacity of the washing plants
#     "ta": 1, # 0.95,                    # Approval rate in valorization centers
#     "tl": 1, # 0.90,                     # Approval rate in washing plants

#     # Cost parameters
#     "arr_cv": 5100000,            # Rental cost of valorization centers
#     "arr_pl": 7000000,            # Rental cost of washing plants
#     "renta_increm": 0.0069,        # Annual rent increase
#     "ade_cv": 20000000,           # Adaptation cost of valorization centers
#     "ade_pl": 45000000,           # Adaptation cost of washing plants
#     "adecua_increm": 0.0069,       # Annual adaptation cost increase
#     "qc": 140, # 363580/2597,                  # Classification and inspection cost
#     "qt": 0.81, # 2120/2597,                    # Crushing cost
#     "ql": 210, # 545370/2597,                  # Washing cost
#     "qb": 140, # 363580/2597,                  # Laboratory test cost
#     "qa": 140, # 363580/2597,                  # Transportation cost
#     "cinv": 12.20, # 31678/2597,                 # Inventory cost of valorization centers
#     "pinv": 11.20, # 29167/2597,                 # Inventory cost of washing plants

#     # Environmental parameters
#     "em": 0.0008736,               # CO2 emissions in kilometers
#     "el": 0.002597,                # CO2 emissions in the washing process
#     "et": 0.001096,                # CO2 emissions in the crushing process
#     "en": 820.65,                 # CO2 emissions in the production of new containers

#     # Contextual parameters
#     "wa": 0.01,                    # WACC
#     "recup_increm": 0, # 0.0025,        # Recovery rate increase
#     "enr": 1039.66, # 2700000,                # Price of returnable container
#     "tri": 200, # 300000,                 # Price of crushed container
#     "adem": 0.01,                  # Demand increase
#     "recup": 1, # 0.89,                 # Recovery rate
#     "envn": 1250, # 3246250,               # Price of new containers
#     "dep": 70, # 181790,                 # Deposit cost
#     "n_pack_prod": 2,              # maximum number of containers that use each producer
#     "dem_interval": [40000, 40001],     # interval in which the demand lies
#     "inflation": 0.05,     # infaltion


#     # Dataframes
#     'df_coord': df_coord, # coordenadas
#     'df_dist': df_dist, # distancias
#     'df_demand': df_demand, # demandas escenario base

#     # Optional = None
#     'type_distance' : 'distance_geo',
#     'initial_demand': None,
# }


parameters['df_coord'] = df_coord
parameters['df_dist'] = df_dist
parameters['df_demand'] = df_demand


# # Basic parameters
# parameters['n_acopios'] = 5
# parameters['n_centros'] = 5
# parameters['n_plantas'] = 3
# parameters['n_productores'] = 5
# parameters['n_envases'] = 3
# parameters['n_periodos'] = 120

# # Technical parameters
# parameters['ccv'] = 337610
# parameters['acv'] = 418117
# parameters['lpl'] = 168805
# parameters['apl'] = 623280
# parameters['ta'] = 0.95
# parameters['tl'] = 0.90

# # Cost parameters
# parameters['arr_cv'] = 5100000
# parameters['arr_pl'] = 7000000
# parameters['ade_cv'] = 20000000
# parameters['ade_pl'] = 45000000
# parameters['qc'] = 140
# parameters['qt'] = 0.81
# parameters['ql'] = 210
# parameters['qb'] = 140
# parameters['qa'] = 0.3
# parameters['cinv'] = 12.20
# parameters['pinv'] = 11.20
# parameters['em'] = 0.0008736
# parameters['el'] = 0.002597
# parameters['et'] = 0.001096
# parameters['en'] = 820.65

# # Contextual parameters
# parameters['wa'] = 0.01
# parameters['inflation'] = 0.05
# parameters['recup_increm'] = 0.2
# parameters['enr'] = 1039.66
# parameters['tri'] = 200
# parameters['adem'] = 0.02
# parameters['recup'] = 0.5
# parameters['envn'] = 1250
# parameters['dep'] = 70
# parameters['n_pack_prod'] = 2
# parameters['dem_interval'] = [40000, 40001]


# # instance = create_instance(parameters)
# # model = create_model(instance)
# # model.setParam('OutputFlag', 0)
# # model.setParam('MIPGap', 0.05)
# # # Optimize
# # model.optimize()




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
            dbc.Label("Valor envase", width=8),
            dbc.Col(
                dbc.Input(id="valor_envase", type="number", min=0, max=3000, step=1, value=1000, placeholder="Enter email"),
                width=4)
            ]),
        dbc.Row([
            dbc.Label("valor depósito", width=8),
            dbc.Col(
                dbc.Input(id="deposito", type="number", min=0, max=300, step=1, value=100, placeholder="Enter email"),
                width=4)
            ]),
        dbc.Row([
            dbc.Col(md=True),
            dbc.Col(dbc.Button("Resolver", id="solving", className="mt-3", n_clicks=0), md=2)           
            ])
        ])   
    ]
    )

tab1_content = dbc.Container([
    dbc.Row([
        dbc.Col([dbc.Label("Ajuste de parámetros"),
                 controls_model,
                 
                 ],  md=4),
        dbc.Col(
            dcc.Loading(
            dcc.Graph(id="map", figure = create_map(pd.DataFrame(columns=['type'])))),
            md=8
        )],
        align="center",
        ),
    dbc.Row(
        id="toggle-row",
        children=[
            dbc.Col(html.P("This is column 1 in a row."), width=6),
            dbc.Col(html.P("This is column 2 in a row."), width=6),
        ],
        style={"display": "none"}  # Initially visible
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
              Input('solving', 'n_clicks'),
              State('valor_envase', 'value'),
              State('deposito', 'value')
              )
def run_model_graph(click_resolver, valor_envase, deposito):
    
    if click_resolver < 1:
        raise PreventUpdate
        #return create_map(pd.DataFrame(columns=['type'])), {"None": "flex"}
    else:
        parameters['enr'] = valor_envase
        parameters['dep'] = deposito
        instance = create_instance(parameters)
        model = create_model(instance)
        model.setParam('MIPGap', 0.1) # Set the MIP gap tolerance to 5% (0.05)
        model.optimize()
        var_sol = get_vars_sol(model)
        # list of active actos
        active_act = []
        # active collection 
        df_q = var_sol['q']
        df_q = df_q[df_q['cantidad'] > 0.01]
        active_act.extend(list(df_q['acopio'].unique()))
        # active collection 
        df_y = var_sol['y']
        df_y = df_y[df_y['apertura'] > 0.01]
        active_act.extend(list(df_y['centro'].unique()))
        # active washing 
        df_w = var_sol['w']
        df_w = df_w[df_w['apertura'] > 0.01]
        active_act.extend(list(df_w['planta'].unique()))
        # active producer 
        df_u = var_sol['u']
        df_u = df_u[df_u['cantidad'] > 0.01]
        active_act.extend(list(df_u['productor'].unique()))
        
        df_sol =  df_coord[df_coord["id"].isin(active_act)]
        df_sol.reset_index(inplace=True)
        map_actors = create_map(df_sol)
        
        
        return map_actors, {"display": "flex"}

# instance = create_instance(parameters)
# model = create_model(instance)
# model.setParam('MIPGap', 0.05) # Set the MIP gap tolerance to 5% (0.05)
# model.optimize()
# var_sol = get_vars_sol(model)
# # list of active actos
# active_act = []
# # active collection 
# df_q = var_sol['q']
# df_q = df_q[df_q['cantidad'] > 0.01]
# active_act.extend(list(df_q['acopio'].unique()))
# # active collection 
# df_y = var_sol['y']
# df_y = df_y[df_y['apertura'] > 0.01]
# active_act.extend(list(df_y['centro'].unique()))
# # active washing 
# df_w = var_sol['w']
# df_w = df_w[df_w['apertura'] > 0.01]
# active_act.extend(list(df_w['planta'].unique()))
# # active producer 
# df_u = var_sol['u']
# df_u = df_u[df_u['cantidad'] > 0.01]
# active_act.extend(list(df_u['productor'].unique()))

# df_sol =  df_coord[df_coord["id"].isin(active_act)]
# df_sol.reset_index(inplace=True)
# map_actors = create_map(df_sol)
