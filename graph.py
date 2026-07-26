import networkx as nx
import pandas as pd

COLD_START_SENTINEL = -1.0
MAX_GRAPH_DISTANCE = 10.0


def build_access_graph(df: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    normal = df[df["label"] == "normal"]
    for _, row in normal.iterrows():
        entity_node = f"e:{row['entity_id']}"
        resource_node = f"r:{row['resource_accessed']}"
        G.add_node(entity_node, kind="entity")
        G.add_node(resource_node, kind="resource")
        G.add_edge(entity_node, resource_node)
    return G


def graph_distance(entity_id: str, resource: str, G: nx.Graph) -> float:
    entity_node = f"e:{entity_id}"
    resource_node = f"r:{resource}"

    if entity_node not in G:
        return COLD_START_SENTINEL

    if resource_node not in G:
        return MAX_GRAPH_DISTANCE

    try:
        d = nx.shortest_path_length(G, entity_node, resource_node)
        return float(min(d, MAX_GRAPH_DISTANCE))
    except nx.NetworkXNoPath:
        return MAX_GRAPH_DISTANCE
