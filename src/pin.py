from collections import defaultdict
from functools import partial
from itertools import combinations
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from biopandas.pdb import PandasPdb
from scipy.spatial import Delaunay
from scipy.spatial.distance import euclidean, pdist, rogerstanimoto, squareform
from sklearn.preprocessing import LabelBinarizer

# from .resi_atoms import (
#     AA_RING_ATOMS,
#     AROMATIC_RESIS,
#     BACKBONE_ATOMS,
#     BOND_TYPES,
#     CATION_PI_RESIS,
#     CATION_RESIS,
#     DISULFIDE_ATOMS,
#     DISULFIDE_RESIS,
#     HYDROPHOBIC_RESIS,
#     IONIC_RESIS,
#     ISOELECTRIC_POINTS_STD,
#     MOLECULAR_WEIGHTS_STD,
#     NEG_AA,
#     PI_RESIS,
#     POS_AA,
#     RESI_NAMES,
# )

def pdb2df(path):
    # Parse the PDB file into pandas DF object
    atomic_df = PandasPdb().read_pdb(str(path)).df['ATOM']
    atomic_df['node_id'] = atomic_df['chain_id'] + atomic_df['residue_number'].map(str) + atomic_df['residue_name']
    #print(atomic_df['chain_id'])
    return atomic_df

def compute_chain_pos_aa_mapping(pdb_df):
    # Compute the mapping: chain -> position -> amino acid
    chain_pos_aa = defaultdict(dict)
    for (chain, pos, aa), _ in pdb_df.groupby(
        ['chain_id', 'residue_number', 'residue_name']
    ):
        chain_pos_aa[chain][pos] = aa
    return chain_pos_aa

def read_pdb(path):
    pdb_df = pdb2df(path)
    chain_pos_aa = compute_chain_pos_aa_mapping(pdb_df=pdb_df)


def compute_distmat(pdb_df):
    # Compute pariwise euclidean distances between every atom

    eucl_dists = pdist(
        pdb_df[['x_coord', 'y_coord', 'z_coord']],
        metric='euclidean'
    )

    eucl_dists = pd.DataFrame(squareform(eucl_dists))
    eucl_dists.index = pdb_df.index
    eucl_dists.columns = pdb_df.index

    return eucl_dists

def node_coords(G, n):
    """
    Return the x, y, z coordinates of a node.
    This is a helper function. Simplifies the code.
    """
    x = G.nodes[n]["x_coord"]
    y = G.nodes[n]["y_coord"]
    z = G.nodes[n]["z_coord"]

    return x, y, z