# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:45:27 2024

@author: DELL
"""
import pandas as pd
import re
import requests
import json
import numpy as np
import gurobipy as gp
from gurobipy import GRB




def get_vars_sol(model):
    """
    Extracts variable solutions from the optimization model and organizes them
    into dataframes based on variable groups.

    Args:
        model (Model): The optimization model containing variables to extract.

    Returns:
        dict: A dictionary where each key is a variable group (e.g., 'x', 'y', 'z'),
              and the corresponding value is a pandas DataFrame containing the
              variable indexes and their solution values.
              
              The column names for the DataFrame are predefined based on the
              variable group, such as ['centro', 'periodo', 'apertura'] for 'x'.
    """
    dict_df = {}

    # Get all variables from the model
    variables = model.getVars()

    # Iterate over variables and extract variable names, indexes, and values
    for var in variables:
        var_name = var.VarName
        # Extract the group and indexes from variable name
        var_group = var_name[:var_name.find('[')] if var_name.find('[') != -1 else var_name
        indexes = (var_name[var_name.find('[') + 1:var_name.find(']')]
                   if var_name.find('[') != -1 and var_name.find(']') != -1 else None)

        # Organize variables into dictionary by group
        if var_group in dict_df:
            dict_df[var_group].append(indexes.split(',') + [var.X])
        else:
            dict_df[var_group] = [indexes.split(',') + [var.X]]

    # Predefined column names for each variable group
    col_names = {
        'x': ['centro', 'periodo', 'uso'],
        'y': ['centro', 'periodo', 'apertura'],
        'z': ['planta', 'periodo', 'uso'],
        'w': ['planta', 'periodo', 'apertura'],
        'q': ['envase', 'acopio', 'centro', 'periodo', 'cantidad'],
        'r': ['envase', 'centro', 'planta', 'periodo', 'cantidad'],
        'u': ['envase', 'planta', 'productor', 'periodo', 'cantidad'],
        'ic': ['envase', 'centro', 'periodo', 'cantidad'],
        'ip': ['envase', 'planta', 'periodo', 'cantidad'],
        'er': ['envase', 'productor', 'periodo', 'cantidad']
    }

    # Convert the lists of variable values into DataFrames with proper columns
    dict_df = {key: pd.DataFrame(value, columns=col_names[key]) for key, value in dict_df.items()}

    return dict_df

def get_obj_components(model):
    """
    Extracts specific components of the objective function from the optimization model
    and calculates their values. The components represent various revenue and cost-related
    terms, which are aggregated into a dictionary.

    Args:
        model (Model): The optimization model containing the objective function components.

    Returns:
        dict: A dictionary containing the total utility ('utilidad_total') and individual 
              objective function components. Each component is represented by its name 
              and the corresponding value in the objective function.
    """
    # List of objective function components to extract
    components = [
        '_ingreso_retornable',
        '_ingreso_triturado',
        # '_egreso_envnuevo',
        '_egreso_adecuar',
        '_egreso_uso',
        '_egreso_transporte',
        '_egreso_compra',
        '_egreso_inspeccion',
        '_egreso_lavado',
        '_egreso_pruebas',
        '_egreso_trituracion',
        '_egreso_invcentros',
        '_egreso_invplantas',
        '_emisiones_transporte',
        '_emisiones_lavado',
        '_emisiones_trituracion',
        # '_emisiones_envnuevo'
    ]

    data_FO = {}

    # Get the total objective value (utility) from the model
    data_FO["utilidad_total"] = model.ObjVal

    # Iterate through each component and calculate its value
    for attr in components:
        expr = getattr(model, attr)
        # Ensure the attribute is a linear expression
        if isinstance(expr, gp.LinExpr):
            value = expr.getValue()
            data_FO[attr] = value  # Store the component's value
        else:
            data_FO[attr] = None  # Set to None if the component is not an expression

    return data_FO



def distancia_geo(punto1: tuple, punto2: tuple) -> float:
    """
    Calcular la distancia de conducción entre dos puntos usando la API de OSRM.

    Args:
        punto1 (tuple): Coordenadas del primer punto (latitud, longitud).
        punto2 (tuple): Coordenadas del segundo punto (latitud, longitud).

    Returns:
        float: Distancia en kilómetros entre los dos puntos, o None si hay un error.
    """
    url = 'http://router.project-osrm.org/route/v1/driving/'
    o1 = f"{punto1[1]},{punto1[0]}"  # Invertir a (longitud, latitud) para OSRM
    o2 = f"{punto2[1]},{punto2[0]}"
    ruta = f"{o1};{o2}"
    
    response = requests.get(url + ruta)

    if response.status_code == 200:
        data = json.loads(response.content)
        return data['routes'][0]['legs'][0]['distance'] / 1000  # Convertir a km
    else:
        return None
    


def create_model(instance,
                 model_integer = False # when True the model considers integer variables (False by default)
                 ):

  # read instance data
  n_acopios = instance['n_acopios']
  n_centros = instance['n_centros']
  n_plantas = instance['n_plantas']
  n_productores = instance['n_productores']
  n_envases = instance['n_envases']
  n_periodos = instance['n_periodos']
  inflation = instance['inflation']
  qc = instance['qc']
  qt = instance['qt']
  ql = instance['ql']
  qb = instance['qb']
  ta = instance['ta']
  tl = instance['tl']
  qa = instance['qa']
  wa = instance['wa']
  em = instance['em']
  el = instance['el']
  et = instance['et']
  en = instance['en']
  recup_increm = instance['recup_increm']
  ccv = instance['ccv']
  acv = instance['acv']
  lpl = instance['lpl']
  apl = instance['apl']
  # dimdist = instance['dimdist']
  arr_cv = instance['arr_cv']
  arr_pl = instance['arr_pl']
  ade_cv = instance['ade_cv']
  ade_pl = instance['ade_pl']
  dep = instance['dep']
  enr = instance['enr']
  tri = instance['tri']
  adem = instance['adem']
  # demanda = instance['demanda']
  recup = instance['recup']
  cinv = instance['cinv']
  pinv = instance['pinv']
  envn = instance['envn']


  # Assign sets and other generated variables from the dictionary
  acopios = instance['acopios']
  centros = instance['centros']
  plantas = instance['plantas']
  productores = instance['productores']
  envases = instance['envases']
  periodos = instance['periodos']

  # Assign parameters and calculations from the dictionary
  cc = instance['cc']
  ca = instance['ca']
  cp = instance['cp']
  cl = instance['cl']
  # coord_acopios = instance['coord_acopios']
  # coord_centros = instance['coord_centros']
  # coord_plantas = instance['coord_plantas']
  # coord_productores = instance['coord_productores']
  da = instance['da']
  dl = instance['dl']
  dp = instance['dp']
  rv = instance['rv']
  rl = instance['rl']
  av = instance['av']
  al = instance['al']
  qd = instance['qd']
  pv = instance['pv']
  pt = instance['pt']
  b = instance['b']
  # env_pdtor = instance['env_pdtor']
  dem = instance['dem']
  de = instance['de']
  gi = instance['gi']
  ge = instance['ge']
  iv = instance['iv']
  il = instance['il']
  ci = instance['ci']
  cv = instance['cv']
  pe = instance['pe']
  a = instance['a']


  model = gp.Model('CircularEconomy')


  # Define variables
  x = model.addVars(centros, periodos, vtype=GRB.BINARY, name="x")
  y = model.addVars(centros, periodos, vtype=GRB.BINARY, name="y")
  z = model.addVars(plantas, periodos, vtype=GRB.BINARY, name="z")
  w = model.addVars(plantas, periodos, vtype=GRB.BINARY, name="w")


  if model_integer:
    q = model.addVars(envases, acopios, centros, periodos, vtype=GRB.INTEGER, name="q")
    r = model.addVars(envases, centros, plantas, periodos, vtype=GRB.INTEGER, name="r")
    combinations_u = [(p,k,l,t) for p,l,t in de.keys() for k in plantas]
    u = model.addVars(combinations_u, vtype=GRB.INTEGER, name="u")
    ic = model.addVars(envases, centros, periodos, vtype=GRB.INTEGER, name="ic")
    ip = model.addVars(envases, plantas, periodos, vtype=GRB.INTEGER, name="ip")
    combinations_er = [(p,l,t) for p,l,t in de.keys()]
    # er = model.addVars(combinations_er, vtype=GRB.INTEGER, name="er")
  else:
    q = model.addVars(envases, acopios, centros, periodos, vtype=GRB.CONTINUOUS, name="q")
    r = model.addVars(envases, centros, plantas, periodos, vtype=GRB.CONTINUOUS, name="r")
    combinations_u = [(p,k,l,t) for p,l,t in de.keys() for k in plantas]
    u = model.addVars(combinations_u, vtype=GRB.CONTINUOUS, name="u")
    ic = model.addVars(envases, centros, periodos, vtype=GRB.CONTINUOUS, name="ic")
    ip = model.addVars(envases, plantas, periodos, vtype=GRB.CONTINUOUS, name="ip")
    combinations_er = [(p,l,t) for p,l,t in de.keys()]
    # er = model.addVars(combinations_er, vtype=GRB.CONTINUOUS, name="er")


  ## FUNCIÓN OBJETIVO
  # Componentes función objetivo
  model._ingreso_retornable = sum(u[p,k,l,t] * pv[p] * (1 + inflation)**((t-1)//12) for p in envases for k in plantas for l in productores for t in periodos if (p,k,l,t) in combinations_u)

  model._ingreso_triturado = (sum(r[p,j,k,t] * (1 - ta) / ta * pt[p] * (1 + inflation)**((t-1)//12) for p in envases for j in centros for k in plantas for t in periodos) +\
                              sum(u[p,k,l,t] * (1 - tl) / tl * pt[p] * (1 + inflation)**((t-1)//12) for p in envases for k in plantas for l in productores for t in
                              periodos if (p,k,l,t) in combinations_u))

  # model._egreso_envnuevo = sum(er[p,l,t] * pe[p] for p in envases for l in productores for t in periodos if (p,l,t) in combinations_er)

  model._egreso_adecuar = sum(y[j,t]*av[j,t] for j in centros for t in periodos) + sum(w[k,t]*al[k,t] for k in plantas for t in periodos)

  model._egreso_uso = sum(x[j,t]*rv[j,t] for j in centros for t in periodos) + sum(z[k,t]*rl[k,t] for k in plantas for t in periodos)

  model._egreso_transporte = sum(q[p,i,j,t]*qa*da[i,j] * (1 + inflation)**((t-1)//12) for p in envases for i in acopios for j in centros for t in periodos) +\
                      sum(r[p,j,k,t]*qa*dl[j,k] * (1 + inflation)**((t-1)//12) for p in envases for j in centros for k in plantas for t in periodos) +\
                      sum(u[p,k,l,t]*qa*dp[k,l] * (1 + inflation)**((t-1)//12) for p in envases for k in plantas for l in productores for t in periodos if (p,k,l,t) in combinations_u)

  model._egreso_compra = sum(q[p,i,j,t]*qd[p] * (1 + inflation)**((t-1)//12) for p in envases for i in acopios for j in centros for t in periodos) #depósito

  model._egreso_inspeccion = sum((r[p,j,k,t]/ta)*qc * (1 + inflation)**((t-1)//12) for p in envases for j in centros for k in plantas for t in periodos) +\
                              sum((u[p,k,l,t]/tl)*qc * (1 + inflation)**((t-1)//12) for p in envases for k in plantas for l in productores for t in periodos if (p,k,l,t) in combinations_u)

  model._egreso_lavado = sum((u[p,k,l,t]/tl)*ql * (1 + inflation)**((t-1)//12) for p in envases for k in plantas for l in productores for t in periodos if (p,k,l,t) in combinations_u)

  model._egreso_pruebas = sum(u[p,k,l,t]*qb * (1 + inflation)**((t-1)//12) for p in envases for k in plantas for l in productores for t in periodos if (p,k,l,t) in combinations_u)

  model._egreso_trituracion = (sum(r[p,j,k,t] * ((1 - ta)/ ta) * qt * (1 + inflation)**((t-1)//12) for p in envases for j in centros for k in plantas for t in periodos) +\
                               sum(u[p,k,l,t] * ((1 - tl)/ tl) * qt * (1 + inflation)**((t-1)//12) for p in envases for k in plantas for l in productores
                               for t in periodos if (p,k,l,t) in combinations_u))

  model._egreso_invcentros= sum(ic[p,j,t]*ci[j] * (1 + inflation)**((t-1)//12) for p in envases for j in centros for t in periodos)

  model._egreso_invplantas = sum(ip[p,k,t]*cv[k] * (1 + inflation)**((t-1)//12) for p in envases for k in plantas for t in periodos)

  model._emisiones_transporte = (sum(da[i,j]*q[p,i,j,t] for p in envases for i in acopios for j in centros for t in periodos) + \
                          sum(dl[j,k]*r[p,j,k,t] for p in envases for j in centros for k in plantas for t in periodos) + \
                          sum(dp[k,l]*u[p,k,l,t] for p in envases for k in plantas for l in productores for t in periodos if (p,k,l,t) in combinations_u))*em

  model._emisiones_lavado = sum((u[p,k,l,t]/tl)*el for p in envases for k in plantas for l in productores for t in periodos if (p,k,l,t) in combinations_u)

  model._emisiones_trituracion = (sum(r[p,j,k,t] * (1 - ta) / ta * et for p in envases for j in centros for k in plantas for t in periodos) + \
                                  sum(u[p,k,l,t] * (1 - tl) / tl * et for p in envases for k in plantas for l in productores
                                  for t in periodos if (p,k,l,t) in combinations_u))

  # model._emisiones_envnuevo = (sum(er[p,l,t]*en for p in envases for l in productores for t in periodos if (p,l,t) in combinations_er))

  # Agregar objetivo
  # funcion_objetivo = model._ingreso_retornable + model._ingreso_triturado - model._egreso_adecuar  - model._egreso_uso - model._egreso_envnuevo -\
  #             model._egreso_transporte - model._egreso_compra - model._egreso_inspeccion - model._egreso_lavado - model._egreso_pruebas -\
  #             model._egreso_trituracion - model._egreso_invcentros - model._egreso_invplantas

  funcion_objetivo = model._ingreso_retornable + model._ingreso_triturado - model._egreso_adecuar  - model._egreso_uso  -\
              model._egreso_transporte - model._egreso_compra - model._egreso_inspeccion - model._egreso_lavado - model._egreso_pruebas -\
              model._egreso_trituracion - model._egreso_invcentros - model._egreso_invplantas

  model.setObjective(funcion_objetivo,GRB.MAXIMIZE)

  # Restriccion 1: Capacidad de procesamiento centro de clasificación
  model.addConstrs((gp.quicksum(r[p,j,k,t] / ta for p in envases for k in plantas) <= cc[j] * x[j,t] for j in centros for t in periodos),
                  name='cap_proc_centros')

  # Restriccion 2: Capacidad de procesamiento plantas de lavado
  model.addConstrs((gp.quicksum(u[p,k,l,t] / tl for p in envases for l in productores if  (p,k,l,t) in combinations_u) <= cl[k]*z[k,t]
                   for k in plantas for t in periodos),name='cap_proc_plantas')

  # Restriccion 3: Cumplimiento de demanda
  # model.addConstrs((gp.quicksum(u[p,k,l,t] for k in plantas) + er[p,l,t] == de[p,l,t] for p in envases for l in productores
  #                  for t in periodos if (p,l,t) in er),name='demanda')
  model.addConstrs((gp.quicksum(u[p,k,l,t] for k in plantas) <= de[p,l,t] for p in envases for l in productores
                   for t in periodos if (p,l,t) in de),name='demanda')

  # Restriccion 4: No debe recogerse más de la generación
  model.addConstrs((gp.quicksum(q[p,i,j,t] for j in centros) <= ge[p,i,t] for p in envases for i in acopios for t in periodos),
                  name='no_recoger_mas_gen')

  ## Adecuación y apertura centros de clasificacion
  # Restriccion 5:
  model.addConstrs((x[j,t] >= y[j,tp] for j in centros for t in periodos for tp in periodos if t >=tp),
                  name='mantener_abierto_centro')

  # Restriccion 6:
  model.addConstrs((gp.quicksum(y[j,tp] for tp in range(1,t+1)) >= x[j,t] for j in centros for t in periodos),
                  name='usar_cuando_centro')

  # Restriccion 7
  model.addConstrs((gp.quicksum(y[j,t] for t in periodos) <= 1 for j in centros),
                  name='adecuar_centro')

  ## Adecuación y apertura plantas de lavado
  # Restriccion 8:
  model.addConstrs((z[k,t] >= w[k,tp] for k in plantas for t in periodos for tp in periodos if t >=tp),
                  name='mantener_abierta_planta')

  # Restriccion 9
  model.addConstrs((gp.quicksum(w[k,tp] for tp in range(1,t+1)) >= z[k,t] for k in plantas for t in periodos),
                  name='usar_cuando_planta')

  # Restriccion 10
  model.addConstrs((gp.quicksum(w[k,t] for t in periodos) <= 1 for k in plantas),
                  name='adecuar_planta')

  # Restriccion 11: Inventario en centros de clasificación
  model.addConstrs(
      ((ic[p,j,t]==ic[p,j,t-1] + gp.quicksum(q[p,i,j,t] for i in acopios)-gp.quicksum(r[p,j,k,t]/ta for k in plantas)) if t >1
      else (ic[p,j,t]==iv[p,j] + gp.quicksum(q[p,i,j,t] for i in acopios)-gp.quicksum(r[p,j,k,t]/ta for k in plantas))
      for p in envases for j in centros for t in periodos ),
      name='inv_centros')

  # Restricción 12: Capacidad de almacenamiento en centros de clasificación
  model.addConstrs((gp.quicksum(ic[p,j,t] for p in envases) <= ca[j]*x[j,t] for j in centros for t in periodos),
                  name='cap_alm_centros')

  # Restriccion 13: Inventario en plantas de lavado
  model.addConstrs(
      ((ip[p,k,t]==ip[p,k,t-1] + gp.quicksum(r[p,j,k,t] for j in centros)-gp.quicksum(u[p,k,l,t]/tl for l in productores if (p,k,l,t) in combinations_u)) if t >1
      else (ip[p,k,t]==il[p,k] + gp.quicksum(r[p,j,k,t] for j in centros)-gp.quicksum(u[p,k,l,t]/tl for l in productores if (p,k,l,t) in combinations_u))
      for p in envases for k in plantas for t in periodos ),
      name='inv_plantas')

  # Restricción 14: Capacidad de almacenamiento en plantas de lavado
  model.addConstrs((gp.quicksum(ip[p,k,t]  for p in envases) <= cp[k]*z[k,t] for k in plantas for t in periodos),
                  name='cap_alm_centros')

  return model


def get_vars_sol(model):
    """
    Extracts variable solutions from the optimization model and organizes them
    into dataframes based on variable groups.

    Args:
        model (Model): The optimization model containing variables to extract.

    Returns:
        dict: A dictionary where each key is a variable group (e.g., 'x', 'y', 'z'),
              and the corresponding value is a pandas DataFrame containing the
              variable indexes and their solution values.
              
              The column names for the DataFrame are predefined based on the
              variable group, such as ['centro', 'periodo', 'apertura'] for 'x'.
    """
    dict_df = {}

    # Get all variables from the model
    variables = model.getVars()

    # Iterate over variables and extract variable names, indexes, and values
    for var in variables:
        var_name = var.VarName
        # Extract the group and indexes from variable name
        var_group = var_name[:var_name.find('[')] if var_name.find('[') != -1 else var_name
        indexes = (var_name[var_name.find('[') + 1:var_name.find(']')]
                   if var_name.find('[') != -1 and var_name.find(']') != -1 else None)

        # Organize variables into dictionary by group
        if var_group in dict_df:
            dict_df[var_group].append(indexes.split(',') + [var.X])
        else:
            dict_df[var_group] = [indexes.split(',') + [var.X]]

    # Predefined column names for each variable group
    col_names = {
        'x': ['centro', 'periodo', 'uso'],
        'y': ['centro', 'periodo', 'apertura'],
        'z': ['planta', 'periodo', 'uso'],
        'w': ['planta', 'periodo', 'apertura'],
        'q': ['envase', 'acopio', 'centro', 'periodo', 'cantidad'],
        'r': ['envase', 'centro', 'planta', 'periodo', 'cantidad'],
        'u': ['envase', 'planta', 'productor', 'periodo', 'cantidad'],
        'ic': ['envase', 'centro', 'periodo', 'cantidad'],
        'ip': ['envase', 'planta', 'periodo', 'cantidad'],
        'er': ['envase', 'productor', 'periodo', 'cantidad']
    }

    # Convert the lists of variable values into DataFrames with proper columns
    dict_df = {key: pd.DataFrame(value, columns=col_names[key]) for key, value in dict_df.items()}

    return dict_df

def get_obj_components(model):
    """
    Extracts specific components of the objective function from the optimization model
    and calculates their values. The components represent various revenue and cost-related
    terms, which are aggregated into a dictionary.

    Args:
        model (Model): The optimization model containing the objective function components.

    Returns:
        dict: A dictionary containing the total utility ('utilidad_total') and individual 
              objective function components. Each component is represented by its name 
              and the corresponding value in the objective function.
    """
    # List of objective function components to extract
    components = [
        '_ingreso_retornable',
        '_ingreso_triturado',
        # '_egreso_envnuevo',
        '_egreso_adecuar',
        '_egreso_uso',
        '_egreso_transporte',
        '_egreso_compra',
        '_egreso_inspeccion',
        '_egreso_lavado',
        '_egreso_pruebas',
        '_egreso_trituracion',
        '_egreso_invcentros',
        '_egreso_invplantas',
        '_emisiones_transporte',
        '_emisiones_lavado',
        '_emisiones_trituracion',
        # '_emisiones_envnuevo'
    ]

    data_FO = {}

    # Get the total objective value (utility) from the model
    data_FO["utilidad_total"] = np.round(model.ObjVal, 1)

    # Iterate through each component and calculate its value
    for attr in components:
        expr = getattr(model, attr)
        # Ensure the attribute is a linear expression
        if isinstance(expr, gp.LinExpr):
            value = expr.getValue()
            data_FO[attr] = np.round(value, 1)  # Store the component's value
        else:
            data_FO[attr] = None  # Set to None if the component is not an expression
    
    return data_FO
