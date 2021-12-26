import os
from pathlib import Path
import pandas as pd
import pytest
from pyprojroot import here

from pin import (
    compute_chain_pos_aa_mapping,
    compute_distmat,
    get_interacting_atoms,
    get_ring_atoms,
    get_ring_centroids,
    pdb2df,
    read_pdb
)

from resi_atoms import (
    AROMATIC_RESIS,
    BOND_TYPES,
    CATION_RESIS,
    HYDROPHOBIC_RESIS,
    NEG_AA,
    PI_RESIS,
    POS_AA,
    RESI_NAMES,
    SULPHUR_RESIS
)

data_path = Path(__file__).parent / "file.pdb"

def generate_network():
    return read_pdb(data_path)

def net():
    return generate_network()

def pdb_df():
    return pdb2df(data_path)

pdb_df()

