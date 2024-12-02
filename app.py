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
            [dbc.Row(html.H4("Configuración", className="text-left")),#dbc.Label("Ajuste de parámetros"),
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
                width=5,                  
                ),
        dbc.Col(
            dcc.Loading(
            dcc.Graph(id="map", 
                      figure = create_map(pd.DataFrame(columns=['type']))
                      )
            ),
            width=7
        )],
        align="center",
        ),
    dbc.Row(
        id="toggle-row",
        children=[
            dbc.Col([
                dbc.Row(html.H4("Ingresos y costos", className="text-center")),
                dbc.Row(dcc.Graph(id="graph_utility"))],
            width=6),            
            dbc.Col([
                dbc.Row(html.H4("Utilización de la capacidad", className="text-center")),
                dbc.Row(dcc.Graph(id="graph_utilization"))], 
                width=6),
        ],
        style={"display": "None"}  # Initially not visible
    )],
    className="align-items-start",
    fluid=True)
tab2_content = html.Div("hola")
tab3_content = html.Div("hola")




# creates layout
app.layout = dbc.Container([
    dbc.Row(html.Img(src='assets/images/header1.png', style={'width': '100%'})),
    dbc.Tabs(
        [
            dbc.Tab(label="La historia", tab_id="historia", label_style=tab_label_style, active_label_style=activetab_label_style),
            dbc.Tab(label="Instrucciones", tab_id="solucion", label_style=tab_label_style, active_label_style=activetab_label_style),
            dbc.Tab(label="Resultados", tab_id="detalles", label_style=tab_label_style, active_label_style=activetab_label_style),
        ],
        id="tabs",
        active_tab="historia",
        ),
    dbc.Container(id="tab-content")],
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
        model.setParam('MIPGap', 0.1) # Set the MIP gap tolerance to 5% (0.05)
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

# # Use of the classification inventory capacity
# df_util = var_sol['ic'].groupby(['periodo']).agg(inv_class=("cantidad", "sum")).reset_index()
# df_util['inv_class'] = np.round(100*df_util['inv_class'] / (var_sol['y']['apertura'].sum()*parameters['acv']), 1)
# # Use of the washing inventory capacity
# df_temp = var_sol['ip'].groupby(['periodo']).agg(inv_wash=("cantidad", "sum")).reset_index()
# df_temp['inv_wash'] = np.round(100*df_temp['inv_wash'] / (var_sol['w']['apertura'].sum()*parameters['apl']), 1)
# df_util = pd.merge(df_util, df_temp, on="periodo", how="inner")
# # Use of the classification processing capacity
# df_temp = var_sol['r'].groupby(['periodo']).agg(cap_class=("cantidad", "sum")).reset_index()
# df_temp['cap_class'] = np.round(100*df_temp['cap_class'] / (var_sol['y']['apertura'].sum()*parameters['ccv']), 1)
# df_util = pd.merge(df_util, df_temp, on="periodo", how="inner")
# # Use of the classification processing capacity
# df_temp = var_sol['u'].groupby(['periodo']).agg(cap_wash=("cantidad", "sum")).reset_index()
# df_temp['cap_wash'] = np.round(100*df_temp['cap_wash'] / (var_sol['w']['apertura'].sum()*parameters['lpl']), 1)
# df_util = pd.merge(df_util, df_temp, on="periodo", how="inner")
# df_util["periodo"] = df_util["periodo"].astype(int)
# df_util = df_util.sort_values(by="periodo", ascending=True).reset_index(drop=True)

# df_util = create_df_util(var_sol, parameters)
# fig = graph_util(df_util)
    
# import plotly.graph_objects as go

# # Create the figure
# fig = go.Figure()

# # Add lines for each column
# fig.add_trace(go.Scatter(x=df_util["periodo"], y=df_util["inv_class"], mode='lines', name='inv. clasificación'))
# fig.add_trace(go.Scatter(x=df_util["periodo"], y=df_util["inv_wash"], mode='lines', name='inv. lavado'))
# fig.add_trace(go.Scatter(x=df_util["periodo"], y=df_util["cap_class"], mode='lines', name='prod. clasificación'))
# fig.add_trace(go.Scatter(x=df_util["periodo"], y=df_util["cap_wash"], mode='lines', name='prod. lavado'))

# # Update layout
# fig.update_layout(
#     title=" ",
#     xaxis_title="periodo",
#     yaxis_title="% de uso",
#     legend_title=" ",
#     template="plotly_white",
# )

# # Show the figure
# fig.show()


# # Dash app
# app = dash.Dash(__name__)

# app.layout = html.Div([
#     html.H1("Dash Plotly Dashboard"),
#     dcc.Graph(figure=fig)  # Embed the figure in the dashboard
# ])

# if __name__ == '__main__':
#     app.run_server(debug=True)


# import dash
# from dash import html
# import dash_bootstrap_components as dbc

# # Initialize the Dash app with Bootstrap theme
# app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])

# # Define the layout
# app.layout = dbc.Container(
#     [
#         # Main Row
#         dbc.Row(
#             [
#                 # First Column (width 5)
#                 dbc.Col(
#                     [
#                         # Row 1 (divided into two columns with width 3 and 2 respectively)
#                         dbc.Row(
#                             [
#                                 dbc.Col(html.Div("Col 1, Row 1, Sub-Col 1", className="border p-2"), width=8),
#                                 dbc.Col(html.Div("Col 1, Row 1, Sub-Col 2", className="border p-2"), width=4),
#                             ],
#                             className="mb-2",  # Margin between rows
#                         ),
#                         # Row 2
#                         dbc.Row(
#                             dbc.Col(html.Div("Col 1, Row 2", className="border p-2")),
#                             className="mb-2",
#                         ),
#                         # Row 3
#                         dbc.Row(
#                             dbc.Col(html.Div("Col 1, Row 3", className="border p-2")),
#                         ),
#                     ],
#                     width=5,  # Width of 5 for the first column
#                 ),
#                 # Second Column (width 7)
#                 dbc.Col(
#                     [
#                         # Row 1 (divided into two columns)
#                         dbc.Row(
#                             [
#                                 dbc.Col(html.Div("Col 2, Row 1, Sub-Col 1", className="border p-2")),
#                                 dbc.Col(html.Div("Col 2, Row 1, Sub-Col 2", className="border p-2")),
#                             ],
#                             className="mb-2",
#                         ),
#                         # Row 2
#                         dbc.Row(
#                             dbc.Col(html.Div("Col 2, Row 2", className="border p-2")),
#                         ),
#                     ],
#                     width=7,  # Width of 7 for the second column
#                 ),
#             ]
#         )
#     ],
#     fluid=True,  # Full-width container
# )

# if __name__ == "__main__":
#     app.run_server(debug=True)
