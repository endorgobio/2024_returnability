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


def select_nodes(df_coord, node_type, n):
    """
    Select and return a list of node IDs from a DataFrame based on type and sample size.
    The IDs are sorted such that numeric parts are sorted naturally (e.g., "A5" comes
    before "A50" and "A51").

    Parameters
    ----------
    df_coord : pd.DataFrame
        DataFrame containing node information, including 'type' and 'id' columns.
    node_type : str or int
        The type of node to filter by. Only nodes with this type will be selected.
    n : int
        The number of nodes to randomly sample from the filtered DataFrame.

    Returns
    -------
    list
        A sorted list of node IDs from the sample, with natural sorting applied.

    Examples
    --------
    >>> data = {'id': ['A5', 'A50', 'A51', 'A6'], 'type': ['A', 'A', 'A', 'A']}
    >>> df_coord = pd.DataFrame(data)
    >>> select_nodes(df_coord, 'A', 3)
    ['A5', 'A6', 'A50']
    """

    df = df_coord[df_coord['type'] == node_type]  # Filter by node type
    df = df.sample(n)                             # Sample 'n' nodes
    nodes = df['id'].tolist()                     # Convert 'id' column to list

    def custom_sort_key(s):
        """
        Custom sort key that extracts the numeric part of the string for natural sorting.

        Parameters
        ----------
        s : str
            The string to be sorted, containing both letters and numbers.

        Returns
        -------
        int
            The numeric part of the string for sorting.
        """
        return int(re.search(r'\d+', s).group())  # Extract and return numeric part of the string

    # Sort nodes based on numeric part of 'id'
    sorted_nodes = sorted(nodes, key=custom_sort_key)

    return sorted_nodes




def generate_demand(producers, packages, periodos, n_packings, dem_interval, 
                    demand_increment, initial_demand=None):
    """
    Generates initial demands and future demands for a set of producers, 
    packages, and periods with a demand increment.

    Args:
        producers (list): List of producers.
        packages (list): List of available package types.
        periodos (list): List of periods for which demands will be generated.
        n_packings (int): Number of packages selected by each producer.
        dem_interval (tuple): Demand range (minimum, maximum) for random generation.
        demand_increment (float): Annual demand increment
        initial_demand (dict, optional): Dictionary of initial demands, if provided. 
            If not provided, it will be generated randomly. Default is None.

    Returns:
        tuple: 
            - initial_demand (dict): Dictionary of generated or provided initial demands.
            - demands (dict): Dictionary of demands by package, producer, and period.
    """
    # define monthtly increment
    month_increment = (1 + demand_increment)**(1/12)-1
    # Create initial demands if none are provided
    if initial_demand is None:
        initial_demand = {}
        for producer in producers:
            chosen_packages = np.random.choice(packages, size=n_packings, replace=False)
            list_packages = []
            for p in chosen_packages:
                list_packages.append({p: np.random.randint(dem_interval[0], dem_interval[1])})
            initial_demand[producer] = list_packages

    demands = {}
    # Create demands for future periods according to the demand increment
    for producer, list_packages in initial_demand.items():
        for dict_pack in list_packages:
            for t in periodos:
                pack_id = list(dict_pack.keys())[0]
                demands[(pack_id, producer, t)] = int(dict_pack[pack_id] * (1 + month_increment) ** (t - 1))

    return initial_demand, demands

def read_dem_initial(df_demand):
    """
    Reads initial demand data from a DataFrame and organizes it by producer and packaging.

    Args:
        df_demand (pd.DataFrame): DataFrame containing demand data with columns for 
                                  'producer', 'packing', and 'demand'.

    Returns:
        dict: A dictionary where each producer is mapped to a list of dictionaries, 
              each containing a package and its corresponding demand.
    """
    demand_initial = {}

    for index, row in df_demand.iterrows():
        producer = row['producer']
        packing = row['packing']
        demand = row['demand']
        
        # Append the demand to the list for the producer
        if producer in demand_initial:
            demand_initial[producer].append({packing: demand})
        else:
            # Initialize the list for the producer if it doesn't exist
            demand_initial[producer] = [{packing: demand}]

    return demand_initial


def calculate_initialgeneration(initial_demand, packages):
    """
    Calculates the total initial generation of demand for each package type 
    based on the initial demand data.

    Args:
        initial_demand (dict): Dictionary where each producer is mapped to a list of 
                               dictionaries, each containing a package and its corresponding demand.
        packages (list): List of all possible package types.

    Returns:
        dict: A dictionary with each package as the key and the total initial demand 
              as the value. Packages with no demand are initialized to 0.
    """
    initial_generation = {}

    # Sum the demand for each package across all producers
    for producer, demands in initial_demand.items():
        for pack_demand in demands:
            for pack, demand in pack_demand.items():
                if pack in initial_generation:
                    initial_generation[pack] += demand
                else:
                    initial_generation[pack] = demand

    # Ensure all packages are included, even those with no demand
    for package in packages:
        if package not in initial_generation:
            initial_generation[package] = 0

    return initial_generation



def distribute_demand(n, total_demand):
    """
    Distributes a total demand into 'n' random parts, ensuring that the sum equals the total demand.

    Args:
        n (int): The number of parts to divide the demand into.
        total_demand (float): The total amount of demand to be distributed.

    Returns:
        np.ndarray: An array of 'n' random values that sum to the total demand.
    """
    # Generate n-1 random values in the interval (0, total_demand)
    cuts = np.sort(np.random.uniform(0, total_demand, n-1))

    # Add the boundaries 0 and total_demand to the cuts
    cuts = np.concatenate(([0], cuts, [total_demand]))

    # The differences between consecutive cuts give the random numbers
    random_numbers = np.diff(cuts)

    return random_numbers


def create_instance(parameters, seed=None):
    """Creates an instance for the optimization model using provided parameters.

    Args:
        parameters (dict): A dictionary containing various parameters for the instance, including:
            - df_coord: DataFrame containing node coordinates.
            - df_dist: DataFrame containing distances between nodes.
            - df_demand: DataFrame containing demand data.
            - n_acopios: Number of collection nodes.
            - n_centros: Number of classification centers.
            - n_plantas: Number of washing plants.
            - n_productores: Number of producers.
            - n_envases: Number of types of containers.
            - n_periodos: Number of periods for the model.
            - ccv: Classification capacity for each center.
            - acv: Storage capacity for each center.
            - lpl: Washing capacity for each plant.
            - apl: Storage capacity for each plant.
            - arr_cv: Rental cost for classification centers.
            - inflation: Anuual estimated inflation
            - ade_cv: Adaptation cost for classification centers.
            - ade_pl: Adaptation cost for washing plants.
            - dep: Deposit cost for containers.
            - enr: Price for returnable containers.
            - tri: Price for crushed containers.
            - adem: Annual demand increment rate.
            - initial_demand: Initial demand values.
            - recup: Initial recovery rate.
            - recup_increm: Annual Increment rate for recovery.
            - n_pack_prod: Number of packs produced.
            - dem_interval: Demand interval.
        seed: Random seed

    Returns:
        dict: A dictionary representing the optimization model instance with the following keys:
            - acopios: List of selected collection nodes.
            - centros: List of selected classification centers.
            - plantas: List of selected washing plants.
            - productores: List of selected producers.
            - envases: List of container types.
            - periodos: List of time periods.
            - cc: Classification capacities for centers.
            - ca: Storage capacities for centers.
            - cl: Washing capacities for plants.
            - cp: Storage capacities for plants.
            - rv: Rental costs for classification centers per period.
            - da: Distances from collection points to classification centers.
            - dl: Distances from classification centers to washing plants.
            - dp: Distances from washing plants to producers.
            - rl: Rental costs for washing plants per period.
            - av: Adaptation costs for classification centers per period.
            - al: Adaptation costs for washing plants per period.
            - qd: Deposit costs for containers.
            - pv: Prices for returnable containers.
            - pt: Prices for crushed containers.
            - b: Demand increment rates per period.
            - dem: Aggregate initial demand.
            - de: Periodic demand.
            - gi: Initial generation values.
            - ge: Generation per period.
            - iv: Initial inventory for classification centers.
            - il: Initial inventory for washing plants.
            - ci: Inventory costs for classification centers.
            - cv: Inventory costs for washing plants.
            - pe: Prices for new containers.
            - inflation: Annual inflation
    """
    
    # set random seed
    if seed != None:
        np.random.seed(seed)
    
    # Copy parameters into data for the instance
    instance = {k: v for k, v in parameters.items()}

    # Read dataframes
    df_coord = instance['df_coord']
    df_dist = instance['df_dist']
    df_demand = instance['df_demand']

    # Define sets
    acopios = select_nodes(df_coord, 'collection', instance['n_acopios'])
    centros = select_nodes(df_coord, 'clasification', instance['n_centros'])
    plantas = select_nodes(df_coord, 'washing', instance['n_plantas'])
    productores = select_nodes(df_coord, 'producer', instance['n_productores'])
    envases = ['E' + str(i) for i in range(instance['n_envases'])]
    periodos = [i + 1 for i in range(instance['n_periodos'])]

    instance['acopios'] = acopios
    instance['centros'] = centros
    instance['plantas'] = plantas
    instance['productores'] = productores
    instance['envases'] = envases
    instance['periodos'] = periodos

    instance['cc'] = {centro: instance['ccv'] for centro in centros}  # Classification capacity for centers
    instance['ca'] = {centro: instance['acv'] for centro in centros}  # Storage capacity for centers
    instance['cl'] = {planta: instance['lpl'] for planta in plantas}  # Washing capacity for plants
    instance['cp'] = {planta: instance['apl'] for planta in plantas}  # Storage capacity for plants
    

    # Rental cost for classification centers
    rv = {(c, 1): instance['arr_cv'] for c in centros}  
    for t in range(2, instance['n_periodos'] + 1):
       for c in centros:
           rv[(c, t)] = int(instance['arr_cv'] * (1 + instance['inflation'])**((t-1)//12)) # The price changes every year
    instance['rv'] = rv
    
    # Rental cost for washing plants
    rl = {(l, 1): instance['arr_pl'] for l in plantas}  
    for t in range(2, instance['n_periodos'] + 1):
        for l in plantas:
            rl[(l, t)] = int(instance['arr_pl'] * (1 + instance['inflation'])**((t-1)//12))
    instance['rl'] = rl
    
    # Adaptation cost for classification centers
    av = {(c, 1): instance['ade_cv'] for c in centros}  
    for t in range(2, instance['n_periodos'] + 1):
        for c in centros:
            av[(c, t)] = int(instance['ade_cv'] * (1 + instance['inflation'])**((t-1)//12))
    instance['av'] = av
    
    # Adaptation cost for washing plants
    al = {(l, 1): instance['ade_pl'] for l in plantas}  
    for t in range(2, instance['n_periodos'] + 1):
        for l in plantas:
            al[(l, t)] = int(instance['ade_pl'] * (1 + instance['inflation'])**((t-1)//12))
    instance['al'] = al
    
    # Distance from collection points to classification centers
    instance['da'] = {(a, c): df_dist[(df_dist['origin'] == a) & (df_dist['destination'] == c)][
        instance['type_distance']].item() for a in acopios for c in centros}
    
    # Distance from classification centers to washing plants
    instance['dl'] = {(c, l): df_dist[(df_dist['origin'] == c) & (df_dist['destination'] == l)][
        instance['type_distance']].item() for c in centros for l in plantas}
    
    # Distance from washing plants to producers
    instance['dp'] = {(l, p): df_dist[(df_dist['origin'] == l) & (df_dist['destination'] == p)][
        instance['type_distance']].item() for l in plantas for p in productores}

 



    instance['qd'] = {envase: instance['dep'] for envase in envases}  # Deposit cost
    instance['pv'] = {envase: instance['enr'] for envase in envases}  # Price for returnable containers
    instance['pt'] = {envase: instance['tri'] for envase in envases}  # Price for crushed containers
    instance['b'] = {t: instance['adem'] for t in range(1, instance['n_periodos'] + 1)}  # Demand increment rate

    # Generate demand and initial generation
    de_agg, de = generate_demand(productores, envases, periodos, instance['n_pack_prod'],
                                  instance['dem_interval'], instance['adem'], instance['initial_demand'])
    instance['dem'] = de_agg  # Initial demand
    instance['de'] = de  # Periodic demand
    gi = calculate_initialgeneration(de_agg, envases)  # Initial generation
    instance['gi'] = gi

    ge_agg = {}  # Aggregate generation
    for p in envases:
        for t in range(1, instance['n_periodos']):
            suma = 0
            for k in de.keys():
                if (k[0] == p and k[2] == t):
                    suma += de[k]
            ge_agg[(p, t + 1)] = suma
    
    month_incr_recup = (1+instance['recup_increm'])**(1/12) - 1
    a = {1: instance['recup']}
    for t in range(2, instance['n_periodos'] + 1):
        a[t] = min(1, a[t - 1] * (1 + month_incr_recup))
    instance['a'] = a  # Recovery rate

    ge = {}
    for p in envases:
        for t in range(1, instance['n_periodos'] + 1):
            if t > 1:
                dist = distribute_demand(instance['n_acopios'], ge_agg[(p, t)])
                for i in range(instance['n_acopios']):
                    ge[(p, acopios[i], t)] = dist[i] * a[t]
            else:
                dist = distribute_demand(instance['n_acopios'], gi[p])
                for i in range(instance['n_acopios']):
                    ge[(p, acopios[i], 1)] = dist[i] * a[t]
    instance['ge'] = ge  # Generation per period

    # Initial inventory and inventory costs
    instance['iv'] = {(e, c): 0 for e in envases for c in centros}  # Initial inventory for classification centers
    instance['il'] = {(e, l): 0 for e in envases for l in plantas}  # Initial inventory for washing plants
    instance['ci'] = {centro: instance['cinv'] for centro in centros}  # Inventory cost for centers
    instance['cv'] = {planta: instance['pinv'] for planta in plantas}  # Inventory cost for plants
    instance['pe'] = {envase: instance['envn'] for envase in envases}  # Price for new containers
    
    return instance


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
    
parameters = {
    # Basic parameters
    "n_acopios": 5,               # maximum 344
    "n_centros": 5,                # maximum 5
    "n_plantas": 3,                # maximum 3
    "n_productores": 5,            # maximum 5
    "n_envases": 3,
    "n_periodos": 120,

    # Technical parameters
    "ccv": 337610,  #130*2597                  # Classification capacity of the valorization centers
    "acv": 418117, # 161*2597                    # Storage capacity of the valorization centers
    "lpl": 168805, # 65*2597                     # Washing capacity of the washing plants
    "apl": 623280, #240*2597                    # Storage capacity of the washing plants
    "ta":  0.95,                    # Approval rate in valorization centers
    "tl":  0.90,                     # Approval rate in washing plants

    # Cost parameters
    "arr_cv": 5100000,            # Rental cost of valorization centers
    "arr_pl": 7000000,            # Rental cost of washing plants
    "ade_cv": 20000000,           # Adaptation cost of valorization centers
    "ade_pl": 45000000,           # Adaptation cost of washing plants
    "qc": 140, # 363580/2597,                  # Classification and inspection cost
    "qt": 0.81, # 2120/2597,                    # Crushing cost
    "ql": 210, # 545370/2597,                  # Washing cost
    "qb": 140, # 363580/2597,                  # Laboratory test cost
    "qa": 1, # 363580/2597,                  # Transportation cost x km
    "cinv": 12.20, # 31678/2597,                 # Inventory cost of valorization centers
    "pinv": 11.20, # 29167/2597,                 # Inventory cost of washing plants

    # Environmental parameters
    "em": 0.0008736,               # CO2 emissions in kilometers
    "el": 0.002597,                # CO2 emissions in the washing process
    "et": 0.001096,                # CO2 emissions in the crushing process
    "en": 820.65,                 # CO2 emissions in the production of new containers

    # Contextual parameters
    "wa": 0.01,                    # WACC
    "inflation": 0.05,          # Annual inflation
    "recup_increm": 0.05,        # Annual recovery rate increase
    "enr": 1039.66, # 2700000,                # Price of returnable container
    "tri": 200, # 300000,                 # Price of crushed container
    "adem": 0.02,                  # Annual Demand increase
    "recup": 0.8,                 # Recovery rate
    "envn": 1250, # 3246250,               # Price of new containers
    "dep": 70, # 181790,                 # Deposit cost
    "n_pack_prod": 2,              # maximum number of containers that use each producer
    "dem_interval": [30000, 30001],     # interval in which the demand lies




    # Optional = None
    'type_distance' : 'distance_geo',
    'initial_demand': None,
}



# CLAIO
df_coord = pd.read_csv('https://docs.google.com/uc?export=download&id=1VYEnH735Tdgqe9cS4ccYV0OUxMqQpsQh') # coordenadas
df_dist = pd.read_csv('https://docs.google.com/uc?export=download&id=1Apbc_r3CWyWSVmxqWqbpaYEacbyf1wvV') # distancias
df_demand = pd.read_csv('https://docs.google.com/uc?export=download&id=1w0PMK36H4Aq39SAaJ8eXRU2vzHMjlWGe') # demandas escenario base


parameters = {
    # Basic parameters
    "n_acopios": 10,               # maximum 344
    "n_centros": 5,                # maximum 5
    "n_plantas": 3,                # maximum 3
    "n_productores": 5,            # maximum 5
    "n_envases": 3,
    "n_periodos": 5,

    # Technical parameters
    "ccv": 337610,  #130*2597                  # Classification capacity of the valorization centers
    "acv": 418117, # 161*2597                    # Storage capacity of the valorization centers
    "lpl": 168805, # 65*2597                     # Washing capacity of the washing plants
    "apl": 623280, #240*2597                    # Storage capacity of the washing plants
    "ta": 1, # 0.95,                    # Approval rate in valorization centers
    "tl": 1, # 0.90,                     # Approval rate in washing plants

    # Cost parameters
    "arr_cv": 5100000,            # Rental cost of valorization centers
    "arr_pl": 7000000,            # Rental cost of washing plants
    "renta_increm": 0.0069,        # Annual rent increase
    "ade_cv": 20000000,           # Adaptation cost of valorization centers
    "ade_pl": 45000000,           # Adaptation cost of washing plants
    "adecua_increm": 0.0069,       # Annual adaptation cost increase
    "qc": 140, # 363580/2597,                  # Classification and inspection cost
    "qt": 0.81, # 2120/2597,                    # Crushing cost
    "ql": 210, # 545370/2597,                  # Washing cost
    "qb": 140, # 363580/2597,                  # Laboratory test cost
    "qa": 140, # 363580/2597,                  # Transportation cost
    "cinv": 12.20, # 31678/2597,                 # Inventory cost of valorization centers
    "pinv": 11.20, # 29167/2597,                 # Inventory cost of washing plants

    # Environmental parameters
    "em": 0.0008736,               # CO2 emissions in kilometers
    "el": 0.002597,                # CO2 emissions in the washing process
    "et": 0.001096,                # CO2 emissions in the crushing process
    "en": 820.65,                 # CO2 emissions in the production of new containers

    # Contextual parameters
    "wa": 0.01,                    # WACC
    "recup_increm": 0, # 0.0025,        # Recovery rate increase
    "enr": 1039.66, # 2700000,                # Price of returnable container
    "tri": 200, # 300000,                 # Price of crushed container
    "adem": 0.01,                  # Demand increase
    "recup": 1, # 0.89,                 # Recovery rate
    "envn": 1250, # 3246250,               # Price of new containers
    "dep": 70, # 181790,                 # Deposit cost
    "n_pack_prod": 2,              # maximum number of containers that use each producer
    "dem_interval": [1000, 2000],     # interval in which the demand lies
    "inflation": 0.05,     # infaltion


    # Dataframes
    'df_coord': df_coord, # coordenadas
    'df_dist': df_dist, # distancias
    'df_demand': df_demand, # demandas escenario base

    # Optional = None
    'type_distance' : 'distance_geo',
    'initial_demand': None,
}

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

# Basic parameters
parameters['n_acopios'] = 5
parameters['n_centros'] = 5
parameters['n_plantas'] = 3
parameters['n_productores'] = 5
parameters['n_envases'] = 3
parameters['n_periodos'] = 120

# Technical parameters
parameters['ccv'] = 337610
parameters['acv'] = 418117
parameters['lpl'] = 168805
parameters['apl'] = 623280
parameters['ta'] = 0.95
parameters['tl'] = 0.90

# Cost parameters
parameters['arr_cv'] = 5100000
parameters['arr_pl'] = 7000000
parameters['ade_cv'] = 20000000
parameters['ade_pl'] = 45000000
parameters['qc'] = 140
parameters['qt'] = 0.81
parameters['ql'] = 210
parameters['qb'] = 140
parameters['qa'] = 0.3
parameters['cinv'] = 12.20
parameters['pinv'] = 11.20
parameters['em'] = 0.0008736
parameters['el'] = 0.002597
parameters['et'] = 0.001096
parameters['en'] = 820.65

# Contextual parameters
parameters['wa'] = 0.01
parameters['inflation'] = 0.05
parameters['recup_increm'] = 0.2
parameters['enr'] = 1039.66
parameters['tri'] = 200
parameters['adem'] = 0.02
parameters['recup'] = 0.5
parameters['envn'] = 1250
parameters['dep'] = 70
parameters['n_pack_prod'] = 2
parameters['dem_interval'] = [40000, 40001]
instance = create_instance(parameters)
start = time.time()
model = create_model(instance)
end = time.time()
print(f'execution time {end-start}')

# Run optimization
model = create_model(instance)
print("model created")

# Optimize
model.optimize()
if model.status == GRB.OPTIMAL:
    print(f"Objective value: {model.ObjVal}")
else:
    print("Optimization was not successful")
