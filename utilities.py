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
        demand_increment (float): Rate of demand increase for subsequent periods.
        initial_demand (dict, optional): Dictionary of initial demands, if provided. 
            If not provided, it will be generated randomly. Default is None.

    Returns:
        tuple: 
            - initial_demand (dict): Dictionary of generated or provided initial demands.
            - demands (dict): Dictionary of demands by package, producer, and period.
    """
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
                demands[(pack_id, producer, t)] = int(dict_pack[pack_id] * (1 + demand_increment) ** (t - 1))

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


def calculate_initial_generation(initial_demand, packages):
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

import numpy as np

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


def create_instance(
    # Basic parameters
    n_acopios,  # Maximum 344
    n_centros,  # Maximum 5
    n_plantas,  # Maximum 3
    n_productores,  # Maximum 5
    n_envases,
    n_periodos,

    # Technical parameters
    ccv,  # Classification capacity of valorization centers
    acv,  # Storage capacity of valorization centers
    lpl,  # Washing capacity of washing plants
    apl,  # Storage capacity of washing plants
    ta,  # Approval rate in valorization centers
    tl,  # Approval rate in washing plants

    # Cost parameters
    arr_cv,  # Rental cost of valorization centers
    arr_pl,  # Rental cost of washing plants
    renta_increm,  # Annual rent increase
    ade_cv,  # Adaptation cost of valorization centers
    ade_pl,  # Adaptation cost of washing plants
    adecua_increm,  # Annual adaptation cost increase
    qc,  # Classification and inspection cost
    qt,  # Crushing cost
    ql,  # Washing cost
    qb,  # Laboratory test cost
    qa,  # Transportation cost
    cinv,  # Inventory cost of valorization centers
    pinv,  # Inventory cost of washing plants

    # Environmental parameters
    em,  # CO2 emissions in kilometers
    el,  # CO2 emissions in the washing process
    et,  # CO2 emissions in the crushing process
    en,  # CO2 emissions in the production of new containers

    # Contextual parameters
    wa,  # WACC
    recup_increm,  # Recovery rate increase
    enr,  # Price of returnable container
    tri,  # Price of crushed container
    adem,  # Demand increase
    recup,  # Recovery rate
    envn,  # Price of new containers
    dep,  # Deposit cost
    n_pack_prod,  # Maximum number of containers used by each producer
    dem_interval,  # Interval in which the demand lies

    # Dataframe
    df_coord,
    df_dist,

    # Optional = None
    type_distance='distance_geo',
    initial_demand=None,
):
    """
    Creates an instance with various parameters including demand, inventory, 
    environmental, and technical capacities for a valorization system.

    Args:
        n_acopios (int): Number of collection points.
        n_centros (int): Number of valorization centers.
        n_plantas (int): Number of washing plants.
        n_productores (int): Number of producers.
        n_envases (int): Number of packages.
        n_periodos (int): Number of periods.

        Technical, cost, environmental, and contextual parameters define system capacities, emissions,
        rental costs, etc. Dataframes provide node coordinates and distances.

        df_coord (DataFrame): Node coordinates.
        df_dist (DataFrame): Distance information.
        type_distance (str): Type of distance measure used. Default is 'distance_geo'.
        initial_demand (dict, optional): Initial demand data.

    Returns:
        dict: A dictionary representing the entire instance configuration with the given parameters.
    """
    # Create an empty dictionary
    instance = {}

    # Store basic parameters
    instance['n_acopios'] = n_acopios
    instance['n_centros'] = n_centros
    instance['n_plantas'] = n_plantas
    instance['n_productores'] = n_productores
    instance['n_envases'] = n_envases
    instance['n_periodos'] = n_periodos

    # Store technical parameters
    instance['ccv'] = ccv
    instance['acv'] = acv
    instance['lpl'] = lpl
    instance['apl'] = apl
    instance['ta'] = ta
    instance['tl'] = tl

    # Store cost parameters
    instance['arr_cv'] = arr_cv
    instance['arr_pl'] = arr_pl
    instance['renta_increm'] = renta_increm
    instance['ade_cv'] = ade_cv
    instance['ade_pl'] = ade_pl
    instance['adecua_increm'] = adecua_increm
    instance['qc'] = qc
    instance['qt'] = qt
    instance['ql'] = ql
    instance['qb'] = qb
    instance['qa'] = qa
    instance['cinv'] = cinv
    instance['pinv'] = pinv

    # Store environmental parameters
    instance['em'] = em
    instance['el'] = el
    instance['et'] = et
    instance['en'] = en

    # Store contextual parameters
    instance['wa'] = wa
    instance['recup_increm'] = recup_increm
    instance['enr'] = enr
    instance['tri'] = tri
    instance['adem'] = adem
    instance['recup'] = recup
    instance['envn'] = envn
    instance['dep'] = dep

    # Define sets for collection points, valorization centers, plants, producers, packages, and periods
    acopios = select_nodes(df_coord, 'collection', n_acopios)
    centros = select_nodes(df_coord, 'clasification', n_centros)
    plantas = select_nodes(df_coord, 'washing', n_plantas)
    productores = select_nodes(df_coord, 'producer', n_productores)
    envases = ['E' + str(i) for i in range(n_envases)]
    periodos = [i + 1 for i in range(n_periodos)]

    instance['acopios'] = acopios
    instance['centros'] = centros
    instance['plantas'] = plantas
    instance['productores'] = productores
    instance['envases'] = envases
    instance['periodos'] = periodos

    # Classification capacity of the valorization centers
    instance['cc'] = {centro: ccv for centro in centros}
    
    # Storage capacity of the valorization centers
    instance['ca'] = {centro: acv for centro in centros}
    
    # Washing capacity of the washing plants
    instance['cl'] = {planta: lpl for planta in plantas}
    
    # Storage capacity of the washing plants
    instance['cp'] = {planta: apl for planta in plantas}
    
    # Rental cost of the valorization center for the first period
    rv = {(c, 1): arr_cv for c in centros}
    
    # Rental cost of valorization centers for subsequent periods,
    # applying the annual rent increase
    for t in range(2, n_periodos + 1):
        for c in centros:
            rv[(c, t)] = int(rv[(c, t - 1)] * (1 + renta_increm))
    
    instance['rv'] = rv
    
    # Distance from collection points to valorization centers
    instance['da'] = {
        (a, c): df_dist[
            (df_dist['origin'] == a) & (df_dist['destination'] == c)
        ][type_distance].item() 
        for a in acopios for c in centros
    }
    
    # Distance from valorization centers to washing plants
    instance['dl'] = {
        (c, l): df_dist[
            (df_dist['origin'] == c) & (df_dist['destination'] == l)
        ][type_distance].item() 
        for c in centros for l in plantas
    }
    
    # Distance from washing plants to producers
    instance['dp'] = {
        (l, p): df_dist[
            (df_dist['origin'] == l) & (df_dist['destination'] == p)
        ][type_distance].item() 
        for l in plantas for p in productores
    }
    
    # Rental cost of the washing plant for the first period
    rl = {(l, 1): arr_pl for l in plantas}
    
    # Rental cost of washing plants for subsequent periods,
    # applying the annual rent increase
    for t in range(2, n_periodos + 1):
        for l in plantas:
            rl[(l, t)] = int(rl[(l, t - 1)] * (1 + renta_increm))
    
    instance['rl'] = rl
    
    # Adaptation cost of the valorization centers for the first period
    av = {(c, 1): ade_cv for c in centros}
    
    # Adaptation cost of valorization centers for subsequent periods,
    # applying the annual adaptation cost increase
    for t in range(2, n_periodos + 1):
        for c in centros:
            av[(c, t)] = int(av[(c, t - 1)] * (1 + adecua_increm))
    
    instance['av'] = av
    
    # Adaptation cost of the washing plants for the first period
    al = {(l, 1): ade_pl for l in plantas}
    
    # Adaptation cost of washing plants for subsequent periods,
    # applying the annual adaptation cost increase
    for t in range(2, n_periodos + 1):
        for l in plantas:
            al[(l, t)] = int(al[(l, t - 1)] * (1 + adecua_increm))
    
    instance['al'] = al
    
    # Deposit cost per container
    instance['qd'] = {envase: dep for envase in envases}
    
    # Price of returnable containers
    instance['pv'] = {envase: enr for envase in envases}
    
    # Price of crushed containers
    instance['pt'] = {envase: tri for envase in envases}
    
    # Demand increase rate for each period
    instance['b'] = {t: adem for t in range(1, n_periodos + 1)}
    
    # Generate demand data
    de_agg, de = generate_demand(
        productores, envases, periodos, n_pack_prod, dem_interval, adem, initial_demand
    )
    
    # Initial aggregated demand
    instance['dem'] = de_agg
    
    # Periodic demand
    instance['de'] = de
    
    # Initial generation based on aggregated demand
    gi = calculate_initialgeneration(de_agg, envases)
    instance['gi'] = gi
    
    # Aggregate generation for future periods
    ge_agg = {}
    for p in envases:
        for t in range(1, n_periodos):
            suma = 0
            for k in de.keys():
                if k[0] == p and k[2] == t:
                    suma += de[k]
            ge_agg[(p, t + 1)] = suma
    
    # Recovery rate for the first period
    a = {1: recup}
    
    # Recovery rate for subsequent periods, applying the annual increase
    for t in range(2, n_periodos + 1):
        a[t] = a[t - 1] * (1 + recup_increm)
    
    instance['a'] = a
    
    # Generation per period for each container and collection point
    ge = {}
    for p in envases:
        for t in range(1, n_periodos + 1):
            if t > 1:
                dist = distribute_demand(n_acopios, ge_agg[(p, t)])
                for i in range(n_acopios):
                    ge[(p, acopios[i], t)] = dist[i] * a[t]
            else:
                dist = distribute_demand(n_acopios, gi[p])
                for i in range(n_acopios):
                    ge[(p, acopios[i], 1)] = dist[i] * a[t]
    
    instance['ge'] = ge
    
    # Initial inventory for valorization centers
    instance['iv'] = {(e, c): 0 for e in envases for c in centros}
    
    # Initial inventory for washing plants
    instance['il'] = {(e, l): 0 for e in envases for l in plantas}
    
    # Inventory cost for valorization centers
    instance['ci'] = {centro: cinv for centro in centros}
    
    # Inventory cost for washing plants
    instance['cv'] = {planta: pinv for planta in plantas}
    
    # Price of new containers
    instance['pe'] = {envase: envn for envase in envases}


    return instance


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
    
