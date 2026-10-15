"""Create and map the subset of the case-study network used by routes."""

from __future__ import annotations

import argparse
import copy
import pickle
from pathlib import Path

import networkx as nx

from Reassemble_Case_Study_Network import write_geojson, write_network_map


DEFAULT_ROUTE_GRAPHS = "case_study_route_graphs.pkl"
DEFAULT_OUTPUT_GRAPH = "case_study_routes_network.gpickle"
DEFAULT_OUTPUT_GEOJSON = "case_study_routes_network.geojson"
DEFAULT_OUTPUT_MAP = "case_study_routes_network_map.html"
DEFAULT_OUTPUT_MAP_NO_NODES = "case_study_routes_network_map_no_nodes.html"


def load_route_graphs(route_graph_file=DEFAULT_ROUTE_GRAPHS):
    """Load the per-train graphs written by Route_Construction.py."""
    with Path(route_graph_file).open("rb") as handle:
        route_graphs = pickle.load(handle)
    if not isinstance(route_graphs, dict):
        raise ValueError(
            f"{route_graph_file!r} must contain a dictionary of route graphs"
        )
    if not route_graphs:
        raise ValueError(f"{route_graph_file!r} contains no route graphs")
    return route_graphs


def build_used_routes_network(route_graphs):
    """Build a directed union retaining maxspeed on used track edges."""
    network = nx.MultiDiGraph()
    edge_route_ids = {}
    for train_id, route_graph in route_graphs.items():
        network.graph.update(copy.deepcopy(route_graph.graph))
        route_nodes = route_graph.graph.get("route_node_coordinates")
        assigned_segments = route_graph.graph.get("assigned_segments")
        if not route_nodes or not assigned_segments:
            raise ValueError(
                f"Route graph for train {train_id!r} lacks ordered traversal data"
            )

        for node, attributes in route_graph.nodes(data=True):
            network.add_node(node, **copy.deepcopy(attributes))

        for index, segment_id in enumerate(assigned_segments):
            if index + 1 >= len(route_nodes):
                break
            start = route_nodes[index]
            end = route_nodes[index + 1]
            segment_graph = nx.MultiGraph()
            segment_graph.add_nodes_from(
                node for node in route_graph if node in route_graph
            )
            for first, second, key, attributes in route_graph.edges(
                keys=True, data=True
            ):
                if attributes.get("segment_id") == segment_id:
                    segment_graph.add_edge(
                        first,
                        second,
                        key=key,
                        **copy.deepcopy(attributes),
                    )
            if start not in segment_graph or end not in segment_graph:
                continue
            try:
                path = nx.shortest_path(segment_graph, start, end)
            except nx.NetworkXNoPath:
                continue
            for first, second in zip(path, path[1:]):
                candidates = route_graph.get_edge_data(first, second) or {}
                for key, attributes in candidates.items():
                    if attributes.get("segment_id") == segment_id:
                        edge_identity = (first, second, key)
                        edge_route_ids.setdefault(edge_identity, set()).add(
                            train_id
                        )
                        if not network.has_edge(first, second, key):
                            edge_attributes = copy.deepcopy(attributes)
                            geometry = edge_attributes.get("geometry")
                            if geometry and tuple(geometry[0]) != tuple(first):
                                edge_attributes["geometry"] = list(reversed(geometry))
                            network.add_edge(
                                first,
                                second,
                                key=key,
                                **edge_attributes,
                            )
                        break

    for (first, second, key), train_ids in edge_route_ids.items():
        network[first][second][key]["route_train_ids"] = sorted(
            train_ids, key=str
        )
    network.graph["route_count"] = len(route_graphs)
    network.graph["used_by_routes_only"] = True
    return network


def write_used_routes_network(
    route_graph_file=DEFAULT_ROUTE_GRAPHS,
    output_graph=DEFAULT_OUTPUT_GRAPH,
    output_geojson=DEFAULT_OUTPUT_GEOJSON,
    output_map=DEFAULT_OUTPUT_MAP,
    output_map_no_nodes=DEFAULT_OUTPUT_MAP_NO_NODES,
):
    """Create all outputs for the union of tracks used by any route."""
    route_graphs = load_route_graphs(route_graph_file)
    network = build_used_routes_network(route_graphs)
    with Path(output_graph).open("wb") as handle:
        pickle.dump(network, handle, protocol=pickle.HIGHEST_PROTOCOL)
    write_geojson(network, output_geojson)
    write_network_map(network, output_map, color_by_speed=True)
    write_network_map(
        network,
        output_map_no_nodes,
        color_by_speed=True,
        show_nodes=False,
    )
    print(
        f"Used-route network: {network.number_of_nodes():,} nodes, "
        f"{network.number_of_edges():,} tracks from "
        f"{len(route_graphs):,} routes."
    )
    print(f"Graph: {output_graph}")
    print(f"GeoJSON: {output_geojson}")
    print(f"Map: {output_map}")
    print(f"Map without nodes: {output_map_no_nodes}")
    return network


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-graphs", default=DEFAULT_ROUTE_GRAPHS)
    parser.add_argument("--output-graph", default=DEFAULT_OUTPUT_GRAPH)
    parser.add_argument("--output-geojson", default=DEFAULT_OUTPUT_GEOJSON)
    parser.add_argument("--output-map", default=DEFAULT_OUTPUT_MAP)
    parser.add_argument(
        "--output-map-no-nodes",
        default=DEFAULT_OUTPUT_MAP_NO_NODES,
        help="HTML map showing only track edges.",
    )
    args = parser.parse_args()
    write_used_routes_network(
        route_graph_file=args.route_graphs,
        output_graph=args.output_graph,
        output_geojson=args.output_geojson,
        output_map=args.output_map,
        output_map_no_nodes=args.output_map_no_nodes,
    )


if __name__ == "__main__":
    main()