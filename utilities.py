# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 16:45:27 2024

@author: DELL
"""
import pandas as pd
import re

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