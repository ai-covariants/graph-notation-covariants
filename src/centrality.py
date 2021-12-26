import pandas as pd
import networkx as nx
from collections import defaultdict
from sklearn.preprocessing import MinMaxScaler


def create_centrality_df_from_graph(G):
    degreecol = nx.degree(G)
    degreec = nx.degree_centrality(G)
    eigenvectorc = nx.eigenvector_centrality(G, tol=1e-03)
    closenessc = nx.closeness_centrality(G)
    betweennessc = nx.betweenness_centrality(G)
    clusteringc = nx.clustering(G)
    dd = defaultdict(list)

    for d in (degreecol, degreec, eigenvectorc, closenessc, betweennessc, clusteringc):
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

def print_top(df, number = 5, to_scale = True, smallest_also = False, crop_range = None):
    if to_scale == True:
        scaler = MinMaxScaler()
        datascaled = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    else:
        datascaled = df.copy()

    if crop_range is not None:
        datascaled = datascaled.iloc[crop_range]
    for col in ['Degree', 'Cluster_Coeff', 'Closeness', 'Betweenness', 'Eigenvector', 'centrality', 'Eccentricity']:
        print(f'TOP {number} Values in {col}:')
        print(datascaled.nlargest(number, col))
        if smallest_also == True:
            print(f'BOTTOM {number} Values in {col}:')
            print(datascaled.nsmallest(number, col))
        print("----------------------------------------------")
