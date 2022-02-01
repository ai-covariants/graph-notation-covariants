import pandas as pd
import networkx as nx
from centrality import (create_centrality_df_from_graph, print_top)
from pin import (read_pdb)
from pathlib import Path


data_path = Path(__file__).parent / "file2.pdb"
df = create_centrality_df_from_graph(read_pdb(data_path))
print(df.head())
print_top(df, number=10, smallest_also=True)