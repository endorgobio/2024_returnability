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
from utilities import create_instance, create_map, create_df_coord, create_df_OF, parameters
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

# Sample DataFrame
df = pd.DataFrame({
    'Category': ['utilidad_total', '_ingreso_retornable', '_ingreso_triturado', '_egreso_uso', '_egreso_pruebas'],
    'Value': [2953282934.4319563, 33978091942.8, 1107863433.8, 3525569784.0, 4575469742.0]
})

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
            dash_table.DataTable(
                id='tableOF',
                columns=[{"name": col, "id": col} for col in df.columns],  # Initial columns
                data=df.to_dict('records'),  # Initial data
                style_table={'overflowX': 'auto'},  # Handle horizontal scrolling if needed
                style_cell={
                    'textAlign': 'left',  # Align text to the left
                    'padding': '10px'  # Add some padding
                },
                style_header={
                    'backgroundColor': 'lightgrey',  # Header styling
                    'fontWeight': 'bold'
                },
            ),
            #dbc.Col(html.P("This is column 2 in a row."), width=6),
            # dash_table.DataTable(
            #     id='tableOF',
            #     #columns=[],
            #     #data=None,  # Convert DataFrame to dictionary for DataTable
            #     style_table={'overflowX': 'auto'},  # Handle horizontal scrolling if needed
            #     style_cell={
            #         'textAlign': 'left',  # Align text to the left
            #         'padding': '10px'  # Add some padding
            #     },
            #     style_header={
            #         'backgroundColor': 'lightgrey',  # Header styling
            #         'fontWeight': 'bold'
            #     },
            # )
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
              Output("tableOF", "columns"),
              Output("tableOF", "data"),
              Input('solving', 'n_clicks'),
              State('valor_envase', 'value'),
              State('deposito', 'value')
              )
def run_model_graph(click_resolver, valor_envase, deposito):
    
    if click_resolver > 0:
        parameters['enr'] = valor_envase
        parameters['dep'] = deposito
        instance = create_instance(parameters)
        model = create_model(instance)
        model.setParam('MIPGap', 0.05) # Set the MIP gap tolerance to 5% (0.05)
        model.optimize()
        var_sol = get_vars_sol(model)
        df_sol = create_df_coord(var_sol, df_coord)
        map_actors = create_map(df_sol)
        
        # crete df_OF
        results_obj = get_obj_components(model)
        df_obj = create_df_OF(results_obj)
        #df_obj = df_obj.set_index('Category')
        updated_columns=[{"name": col, "id": col} for col in df_obj.columns]
        updated_data=df_obj.to_dict('records')  # Convert DataFrame to dictionary for DataTable
            
        
        return map_actors, {"display": "flex"}, updated_columns, updated_data
    else:
        raise PreventUpdate

# instance = create_instance(parameters)
# model = create_model(instance)
# model.setParam('MIPGap', 0.2) # Set the MIP gap tolerance to 5% (0.05)
# model.optimize()
# var_sol = get_vars_sol(model)
# results_obj = get_obj_components(model)
# df_obj = create_df_OF(results_obj)
# df_obj = df_obj.set_index('Category')
# updated_columns=[{"name": col, "id": col} for col in df_obj.columns]
# updated_data=df_obj.to_dict('records')  # Convert DataFrame to dictionary for DataTable


# # Create a new DataFrame when the button is clicked
# updated_df = pd.DataFrame({
#     'New Category': ['new_category1', 'new_category2', 'new_category3'],
#     'New Value': [100000, 200000, 300000],
#     'Extra Column': [1, 2, 3]
# })

# # Convert updated DataFrame to a list of dictionaries
# updated_data1 = updated_df.to_dict('records')

# # Define new columns
# updated_columns1 = [{"name": col, "id": col} for col in updated_df.columns]

# instance = create_instance(parameters)
# model = create_model(instance)
# model.setParam('MIPGap', 0.05) # Set the MIP gap tolerance to 5% (0.05)
# model.optimize()
# var_sol = get_vars_sol(model)
# list of active actos
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

# results_obj = get_obj_components(model)
# df_obj = pd.DataFrame(list(results_obj.items()), columns=["Category", "Value"])
# # Function to determine the type based on the Category column
# def type_OF(row):
#     if 'egreso' in row["Category"]:
#         return "egreso"
#     elif 'ingreso' in row["Category"]:
#         return "ingreso"
#     else:
#         return "other"  # Default case if neither 'egreso' nor 'ingreso' is found

# # Apply the function row-wise to create a new column 'type'
# df_obj['Type'] = df_obj.apply(type_OF, axis=1)

# # Group by 'Type' and calculate the sum of 'Value' for each group
# grouped_df = df_obj.groupby('Type')['Value'].sum().reset_index()

# # Separate the total sums for 'egreso' and 'ingreso'
# total_egreso = grouped_df[grouped_df['Type'] == 'egreso']['Value'].iloc[0]
# total_ingreso = grouped_df[grouped_df['Type'] == 'ingreso']['Value'].iloc[0]

# # Create a new column to store the divided values
# df_obj['Percentaje'] = df_obj.apply(
#     lambda row: np.round(100*row['Value'] / total_egreso, 1) if row['Type'] == 'egreso' else 
#                (np.round(100*row['Value'] / total_ingreso, 1) if row['Type'] == 'ingreso' else None), axis=1)

# df_obj = df_obj.sort_values(by=['Type', 'Percentaje'], ascending=[False, False])
# df_obj = create_df_OF(results_obj)

# # Sample DataFrame
# df = pd.DataFrame({
#     "Name": ["Alice", "Bob", "Charlie"],
#     "Age": [24, 27, 22],
#     "City": ["New York", "San Francisco", "Los Angeles"]
# })

# df = create_df_OF(results_obj)
# updated_columns=[{"name": col, "id": col} for col in df_obj.columns],
# updated_records=df_obj.to_dict('records'),  # Convert DataFrame to dictionary for DataTable

# # Create Dash app
# app = dash.Dash(__name__)

# app.layout = html.Div([
#     html.H1("DataFrame Display Example"),
    
#     # Display the DataFrame
#     dash_table.DataTable(
#         id='table',
#         columns=[{"name": col, "id": col} for col in df.columns],
#         data=df.to_dict('records'),  # Convert DataFrame to dictionary for DataTable
#         style_table={'overflowX': 'auto'},  # Handle horizontal scrolling if needed
#         style_cell={
#             'textAlign': 'left',  # Align text to the left
#             'padding': '10px'  # Add some padding
#         },
#         style_header={
#             'backgroundColor': 'lightgrey',  # Header styling
#             'fontWeight': 'bold'
#         },
#     )
# ])

# if __name__ == "__main__":
#     app.run_server(debug=True)

# import dash
# from dash import dcc, html, Input, Output
# import dash_table
# import pandas as pd

# # Sample DataFrame
# df = pd.DataFrame({
#     'Category': ['utilidad_total', '_ingreso_retornable', '_ingreso_triturado', '_egreso_uso', '_egreso_pruebas'],
#     'Value': [2953282934.4319563, 33978091942.8, 1107863433.8, 3525569784.0, 4575469742.0]
# })

# # Create a Dash app
# app = dash.Dash(__name__)

# # Define the layout of the app
# app.layout = html.Div([
#     # Table component
#     dash_table.DataTable(
#         id='table',
#         columns=[{"name": col, "id": col} for col in df.columns],  # Initial columns
#         data=df.to_dict('records'),  # Initial data
#         style_table={'overflowX': 'auto'},  # Handle horizontal scrolling if needed
#         style_cell={
#             'textAlign': 'left',  # Align text to the left
#             'padding': '10px'  # Add some padding
#         },
#         style_header={
#             'backgroundColor': 'lightgrey',  # Header styling
#             'fontWeight': 'bold'
#         },
#     ),
#     # Button to trigger table update
#     html.Button('Update Table', id='update-button', n_clicks=0)
# ])

# # Define callback to update the table data and columns
# @app.callback(
#     [Output('table', 'data'),        # Update table data
#      Output('table', 'columns')],     # Update table columns
#     Input('update-button', 'n_clicks')  # The input is the button click count
# )
# def update_table(n_clicks):
#     if n_clicks > 0:
#         # Create a new DataFrame when the button is clicked
#         updated_df = pd.DataFrame({
#             'New Category': ['new_category1', 'new_category2', 'new_category3'],
#             'New Value': [100000, 200000, 300000],
#             'Extra Column': [1, 2, 3]
#         })
        
#         # Convert updated DataFrame to a list of dictionaries
#         updated_data = updated_df.to_dict('records')
        
#         # Define new columns
#         updated_columns = [{"name": col, "id": col} for col in updated_df.columns]
        
#         # Return the updated DataFrame data and new columns
#         return updated_data, updated_columns
#     else:
#         # If button hasn't been clicked, return the original DataFrame and original columns
#         return df.to_dict('records'), [{"name": col, "id": col} for col in df.columns]

# # Run the Dash app
# if __name__ == '__main__':
#     app.run_server(debug=True)
