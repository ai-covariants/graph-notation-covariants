import pandas as pd
import networkx as nx
from collections import defaultdict
from pin import (pdb2df, read_pdb)

def create_centrality_graph(G):
    degreec = nx.degree_centrality(G)
    eigenvectorc = nx.eigenvector_centrality(G, tol = 1e-03)
    closenessc = nx.closeness_centrality(G)
    betweennessc = nx.betweenness_centrality(G)
    clusteringc = nx.clustering(G)
    dd = defaultdict(list)

    for d in (degreec, eigenvectorc, closenessc, betweennessc, clusteringc):
        for key, value in d.items():
            dd[key].append(value)

    df = pd.DataFrame.from_dict(dd, orient='index').rename(columns={
        0: 'Degree',
        1: 'Eigenvector',
        2: 'Closeness',
        3: "Betweenness",
        4: "Clustering"
    })
    return df


gr = read_pdb('file.pdb')
pdb_df = pdb2df('file.pdb')
df = create_centrality_graph(gr)
print(df.head())
