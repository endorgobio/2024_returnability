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

