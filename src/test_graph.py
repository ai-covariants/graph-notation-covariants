import pandas as pd
import networkx as nx
from graph import (create_centrality_graph)
from pin import (read_pdb)
from pathlib import Path
import os

data_path = Path(__file__).parent / "file.pdb"
df = create_centrality_graph(read_pdb(data_path))
print(df.head())