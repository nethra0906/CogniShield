import pandas as pd

from graph import build_access_graph, graph_distance, COLD_START_SENTINEL, MAX_GRAPH_DISTANCE


def _df(rows):
    return pd.DataFrame(rows)


def test_direct_historical_resource_is_distance_one():
    df = _df([
        {"entity_id": "e1", "resource_accessed": "r1", "label": "normal"},
    ])
    G = build_access_graph(df)
    assert graph_distance("e1", "r1", G) == 1.0


def test_unknown_entity_is_cold_start_sentinel():
    df = _df([{"entity_id": "e1", "resource_accessed": "r1", "label": "normal"}])
    G = build_access_graph(df)
    assert graph_distance("never_seen", "r1", G) == COLD_START_SENTINEL


def test_known_entity_unknown_resource_is_max_distance():
    df = _df([{"entity_id": "e1", "resource_accessed": "r1", "label": "normal"}])
    G = build_access_graph(df)
    assert graph_distance("e1", "never_seen_resource", G) == MAX_GRAPH_DISTANCE


def test_one_hop_lateral_resource_is_distance_three():
    # e1-r1, e2-r1, e2-r2: e1 can reach r2 via e2's cluster (distance 3).
    df = _df([
        {"entity_id": "e1", "resource_accessed": "r1", "label": "normal"},
        {"entity_id": "e2", "resource_accessed": "r1", "label": "normal"},
        {"entity_id": "e2", "resource_accessed": "r2", "label": "normal"},
    ])
    G = build_access_graph(df)
    assert graph_distance("e1", "r2", G) == 3.0


def test_non_normal_sessions_are_excluded_from_the_graph():
    df = _df([{"entity_id": "e1", "resource_accessed": "r1", "label": "brute_force"}])
    G = build_access_graph(df)
    assert graph_distance("e1", "r1", G) == COLD_START_SENTINEL
