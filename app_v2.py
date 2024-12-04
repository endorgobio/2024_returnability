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


# Read data
df_coord = pd.read_csv('https://docs.google.com/uc?export=download&id=1VYEnH735Tdgqe9cS4ccYV0OUxMqQpsQh') # coordenadas
df_dist = pd.read_csv('https://docs.google.com/uc?export=download&id=1Apbc_r3CWyWSVmxqWqbpaYEacbyf1wvV') # distancias
df_demand = pd.read_csv('https://docs.google.com/uc?export=download&id=1w0PMK36H4Aq39SAaJ8eXRU2vzHMjlWGe') # demandas escenario base
parameters['df_coord'] = df_coord
parameters['df_dist'] = df_dist
parameters['df_demand'] = df_demand



app = dash.Dash(__name__, external_stylesheets=[dbc.themes.FLATLY],
                title="retornabilidad",
                suppress_callback_exceptions=True)
    
# # need to run it in heroku
# server = app.server
app.run_server(debug=True)

tab1_content = dbc.Container([
    dbc.Row([
        dbc.Col(
            [
                dbc.Row(html.H4("Cargar archivos", className="text-left"), className="mt-3 mb-3"),
                dbc.Row([
                    dbc.Col(dbc.Label("Lista actores"), width=8),
                    dbc.Col(dcc.Upload(
                        id="upload-actors",
                        children=dbc.Button("Cargar"),
                        accept=".xlsx"),
                        width=3)
                ], className="mt-1 mb-1"),
                dbc.Row([
                    dbc.Col(dbc.Label("Distancias"), width=8),
                    dbc.Col(dcc.Upload(
                        id="upload-distances",
                        children=dbc.Button("Cargar"),
                        accept=".xlsx"),
                        width=3)
                ], className="mt-1 mb-1"),
                dbc.Row(html.H4("Configuración", className="text-left"), className="mt-3 mb-3"),
                dbc.Row([
                    dbc.Col(dbc.Label("Valor envase retornable"), width=8),
                    dbc.Col(
                        dbc.Input(id="valor_envase", type="number", min=0, max=3000, step=1,
                                  value=parameters['enr'], placeholder="Enter value"),
                        width=3
                    )
                ]),
                dbc.Row([
                    dbc.Col(dbc.Label("Valor depósito"), width=8),
                    dbc.Col(
                        dbc.Input(id="deposito", type="number", min=0, max=300, step=1,
                                  value=parameters['dep'], placeholder="Enter value"),
                        width=3)
                ]),
                dbc.Row([
                    dbc.Col(dbc.Label("Costo clasificación"), width=8),
                    dbc.Col(
                        dbc.Input(id="cost_class", type="number", min=0, max=300, step=1,
                                  value=parameters['qc'], placeholder="Enter value"),
                        width=3)
                ]),
                dbc.Row([
                    dbc.Col(dbc.Label("Costo lavado"), width=8),
                    dbc.Col(
                        dbc.Input(id="cost_wash", type="number", min=0, max=300, step=1,
                                  value=parameters['ql'], placeholder="Enter value"),
                        width=3)
                ]),
                dbc.Row([
                    dbc.Col(dbc.Label("Costo transporte"), width=8),
                    dbc.Col(
                        dbc.Input(id="cost_transp", type="number", min=0, max=300, step=1,
                                  value=parameters['qa'], placeholder="Enter value"),
                        width=3)
                ]),
                dbc.Row([
                    dbc.Col(width=8),
                    dbc.Col(dbc.Button("Resolver", id="solving", className="mt-3", n_clicks=0), width=3)
                ], align="right")
            ],
            width=5
        ),
        dbc.Col(
            dcc.Loading(
                dcc.Graph(id="map",
                          figure=create_map(pd.DataFrame(columns=['type'])))
            ),
            width=7
        )
    ]),
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
        style={"display": "None"}
    )
], className="align-items-start", fluid=True)

tab2_content = html.Div("Hola")

# Layout setup
app.layout = dbc.Container([
    dbc.Row(html.Img(src='assets/images/header1.png', style={'width': '100%'})),
    dbc.Tabs(
        [
            dbc.Tab(tab1_content, label="Tablero", tab_id="dashboard"),
            dbc.Tab(tab2_content, label="Instrucciones", tab_id="instructions"),
        ],
        id="tabs",
        active_tab="dashboard"
    ),
    dbc.Container(id="tab-content")
], fluid=True)


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


if __name__ == "__main__":
    app.run_server(debug=True)


app.layout = dbc.Container(
    [
        dbc.Row(
            dbc.Col(
                dbc.Container(
                    [
                        html.Div("Inner container content", style={"background-color": "lightblue"}),
                    ],
                    className="inner-container d-flex justify-content-center align-items-center",
                ),
                className="outer-container",
            ),
        ),
    ],
    style={"height": "100vh", "background-color": "lightgray"},
    fluid=True,
)