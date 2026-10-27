"""Construct shortest feasible train routes from required track edges."""

from __future__ import annotations

import csv
import copy
import heapq
import math
import pickle
import argparse
from pathlib import Path

import networkx as nx
from shapely.geometry import Point


def load_route_station_stops(csv_file="rail_routes.csv"):
    """Return station stops grouped by train ID from the timetable CSV."""
    with Path(csv_file).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"Route file {csv_file!r} is empty")

    station_stops = {}
    for column, header in enumerate(rows[0]):
        if not header.strip().isdigit():
            continue
        train_id = rows[0][column].strip()
        try:
            train_number = int(train_id)
        except ValueError as error:
            raise ValueError(
                f"Invalid train ID {train_id!r} in column {column + 1}"
            ) from error
        stops = []
        for row in rows[1:]:
            if column >= len(row):
                continue
            station = row[column].strip()
            if station and (not stops or stops[-1] != station):
                stops.append(station)
        station_stops[train_number] = stops
    if not station_stops:
        raise ValueError(f"No train station stops found in {csv_file!r}")
    return station_stops


def load_route_station_stops_files(*csv_files):
    """Merge station stops from multiple timetable CSV files."""
    if not csv_files:
        raise ValueError("At least one station-stop CSV file is required")
    station_stops = {}
    for csv_file in csv_files:
        for train_id, stops in load_route_station_stops(csv_file).items():
            if train_id in station_stops:
                raise ValueError(
                    f"Train ID {train_id} appears in multiple station-stop "
                    f"files"
                )
            station_stops[train_id] = stops
    return station_stops


def _parse_integer_list(value, *, field_name):
    """Parse a comma-separated list of positive integer IDs."""
    values = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        try:
            number = int(item)
        except ValueError as error:
            raise ValueError(
                f"{field_name} contains a non-integer value: {item!r}"
            ) from error
        if number < 1:
            raise ValueError(f"{field_name} contains invalid ID {number}")
        values.append(number)
    return values


def load_rail_routes(csv_file="required_edges.csv"):
    """Load route definitions from the column-oriented route CSV.

    The first row of each column contains one or more train IDs. Non-empty
    cells below it contain the edge IDs required by those trains.
    """
    with Path(csv_file).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"Route file {csv_file!r} is empty")

    width = max(len(row) for row in rows)
    routes = []
    for column in range(width):
        header = rows[0][column].strip() if column < len(rows[0]) else ""
        if header:
            try:
                train_ids = _parse_integer_list(
                    header, field_name=f"column {column + 1} header"
                )
            except ValueError:
                if column != 0 or header.casefold() != "edges":
                    raise
                train_ids = [column + 1]
        else:
            train_ids = [column + 1]
        edge_ids = []
        for row in rows[1:]:
            if column < len(row) and row[column].strip():
                edge_ids.extend(
                    _parse_integer_list(
                        row[column],
                        field_name=f"column {column + 1}",
                    )
                )
        if not edge_ids:
            raise ValueError(
                f"Route column {column + 1} has no required edges"
            )
        required_edges = tuple(dict.fromkeys(edge_ids))
        for train_id in train_ids:
            routes.append({
                "train_id": train_id,
                "route_column": column + 1,
                "required_edges": required_edges,
            })
    if not routes:
        raise ValueError("No routes were found in the CSV")
    return routes


def _edge_length(first, second, attributes):
    geometry = attributes.get("geometry")
    if not geometry:
        geometry = [first, second]
    total = 0.0
    for start, end in zip(geometry, geometry[1:]):
        latitude = math.radians((start[1] + end[1]) / 2)
        total += math.hypot(
            (end[0] - start[0]) * 111320 * math.cos(latitude),
            (end[1] - start[1]) * 110540,
        )
    return total


def _edge_lengths(network_graph):
    lengths = {}
    for first, second, key, attributes in network_graph.edges(
        keys=True, data=True
    ):
        edge_id = attributes.get("edge_id")
        if edge_id is None:
            raise ValueError(
                f"Network edge {(first, second, key)!r} has no edge_id"
            )
        lengths[str(edge_id)] = _edge_length(first, second, attributes)
    return lengths


def _edge_direction_from_node(attributes, node):
    """Return direction to the geometry point immediately beside ``node``."""
    coordinates = list(attributes.get("geometry", []))
    if len(coordinates) < 2:
        return None
    node_point = Point(node)
    if node_point.distance(Point(coordinates[0])) <= node_point.distance(
        Point(coordinates[-1])
    ):
        adjacent_point = coordinates[1]
    else:
        adjacent_point = coordinates[-2]
    vector = (
        adjacent_point[0] - node[0],
        adjacent_point[1] - node[1],
    )
    length = math.hypot(*vector)
    if length == 0:
        return None
    return vector[0] / length, vector[1] / length


def _angle_between_vectors(center, first, second):
    """Return the angle in radians between two links sharing center."""
    first_vector = (first[0] - center[0], first[1] - center[1])
    second_vector = (second[0] - center[0], second[1] - center[1])
    first_length = math.hypot(*first_vector)
    second_length = math.hypot(*second_vector)
    if first_length == 0 or second_length == 0:
        return math.pi
    cosine = (
        first_vector[0] * second_vector[0]
        + first_vector[1] * second_vector[1]
    ) / (first_length * second_length)
    return math.acos(max(-1.0, min(1.0, cosine)))


def _junction_angle(incoming_attributes, current, outgoing_attributes):
    """Return the angle between outward edge directions at a junction."""
    incoming = _edge_direction_from_node(incoming_attributes, current)
    outgoing = _edge_direction_from_node(outgoing_attributes, current)
    if incoming is None or outgoing is None:
        return 0.0
    return math.degrees(
        _angle_between_vectors((0, 0), incoming, outgoing)
    )


def _network_edges(network_graph):
    """Index network edges by their stable edge ID."""
    indexed = {}
    for first, second, key, attributes in network_graph.edges(
        keys=True, data=True
    ):
        edge_id = attributes.get("edge_id")
        if edge_id is None:
            raise ValueError(
                f"Network edge {(first, second, key)!r} has no edge_id"
            )
        indexed.setdefault(str(edge_id), []).append(
            (first, second, key, attributes)
        )
    return indexed


def _shortest_route_for_edges(network_graph, required_edges, minimum_junction_angle=90.0, heuristic_weight=2.0,):
    """Find a feasible route through the required network edges in order."""
    if heuristic_weight < 1.0:
        raise ValueError("heuristic_weight must be at least 1")
    edge_index = _network_edges(network_graph)
    edge_lengths = _edge_lengths(network_graph)
    required = tuple(str(edge_id) for edge_id in required_edges)
    if not required:
        raise ValueError("At least one required edge is needed")
    missing = set(required) - set(edge_index)
    if missing:
        raise ValueError(f"Required edges are absent: {sorted(missing)}")
    missing_lengths = set(required) - set(edge_lengths)
    if missing_lengths:
        raise ValueError(
            f"Required edges have no network geometry: "
            f"{sorted(missing_lengths)}"
        )

    edge_endpoints = {
        edge_id: {
            endpoint
            for first, second, _key, _attributes
            in edge_index[edge_id]
            for endpoint in (first, second)
        }
        for edge_id in required
    }

    def heuristic(current, next_index):
        """Lower-bound travel to the next required edge."""
        if next_index >= len(required):
            return 0.0
        return min(
            _edge_length(current, endpoint, {})
            for edge_id in required[next_index:]
            for endpoint in edge_endpoints[edge_id]
        )

    queue = []
    best = {}
    previous_states = {}
    initial_edges = {}
    counter = 0

    def reconstruct_route(state, distance):
        state_path = [state]
        while state_path[-1] in previous_states:
            state_path.append(previous_states[state_path[-1]][0])
        state_path.reverse()
        route_nodes = [state_path[0][0], state_path[0][1]]
        edge_ids = [initial_edges[state_path[0]]]
        for route_state in state_path[1:]:
            route_nodes.append(route_state[1])
            edge_ids.append(previous_states[route_state][1])
        nodes = [
            {
                "node_id": network_graph.nodes[node].get("node_id"),
                "station": network_graph.nodes[node].get("station"),
            }
            for node in route_nodes
        ]
        covered_count = state[3]
        return {
            "nodes": nodes,
            "node_ids": [item["node_id"] for item in nodes],
            "edge_ids": edge_ids,
            "length_meters": distance,
            "complete": covered_count == len(required),
            "covered_required_edges": list(required[:covered_count]),
        }

    for first, second, key, attributes in edge_index[required[0]]:
        for start, next_node in ((first, second), (second, first)):
            edge_id = str(attributes["edge_id"])
            edge_length = edge_lengths[edge_id]
            state = (start, next_node, key, 1)
            best[state] = edge_length
            initial_edges[state] = edge_id
            heapq.heappush(
                queue,
                (
                    edge_length + heuristic_weight * heuristic(next_node, 1),
                    counter,
                    edge_length,
                    state,
                ),
            )
            counter += 1

    best_partial_state = None
    best_partial_distance = math.inf
    while queue:
        _priority, _, distance, state = heapq.heappop(queue)
        if distance != best.get(state):
            continue
        previous, current, incoming_key, next_index = state
        if (
            best_partial_state is None
            or next_index > best_partial_state[3]
            or (
                next_index == best_partial_state[3]
                and distance < best_partial_distance
            )
        ):
            best_partial_state = state
            best_partial_distance = distance
        if next_index == len(required):
            return reconstruct_route(state, distance)
        for next_node in network_graph.neighbors(current):
            if next_node == previous:
                continue
            for key, attributes in network_graph[current][next_node].items():
                if network_graph.degree(current) > 2:
                    incoming_attributes = network_graph[previous][current][
                        incoming_key
                    ]
                    if _junction_angle(
                        incoming_attributes,
                        current,
                        attributes,
                    ) <= minimum_junction_angle:
                        continue
                edge_id = str(attributes["edge_id"])
                new_index = next_index
                if edge_id in required:
                    if (
                        next_index >= len(required)
                        or edge_id != required[next_index]
                    ):
                        continue
                    new_index += 1
                new_distance = distance + edge_lengths[edge_id]
                new_state = (current, next_node, key, new_index)
                if new_distance < best.get(new_state, math.inf):
                    best[new_state] = new_distance
                    previous_states[new_state] = (state, edge_id)
                    heapq.heappush(
                        queue,
                        (
                            new_distance + heuristic_weight * heuristic(
                                next_node, new_index
                            ),
                            counter,
                            new_distance,
                            new_state,
                        ),
                    )
                    counter += 1
    if best_partial_state is None:
        raise ValueError(
            f"No route can start with edge {required[0]}"
        )
    partial_route = reconstruct_route(
        best_partial_state,
        best_partial_distance,
    )
    partial_route["uncovered_required_edges"] = list(
        required[best_partial_state[3]:]
    )
    return partial_route


def assign_routes(routes, network_graph_file="case_study_network.gpickle", minimum_junction_angle=90.0, heuristic_weight=2.0,):
    """Assign the shortest feasible track route to every train definition.

    If all required edges cannot be covered in order, the best partial
    route is returned and displayed instead of aborting the remaining routes.
    """
    if not 0.0 <= minimum_junction_angle < 180.0:
        raise ValueError("minimum_junction_angle must be between 0 and 180")
    with Path(network_graph_file).open("rb") as handle:
        network_graph = pickle.load(handle)

    assignments = []
    for route in routes:
        required = tuple(route["required_edges"])
        assignment = _shortest_route_for_edges(
            network_graph,
            required,
            minimum_junction_angle,
            heuristic_weight,
        )
        if not assignment["complete"]:
            covered = assignment["covered_required_edges"]
            print(
                f"No feasible route covers all required edges for train "
                f"{route['train_id']}. Best partial route covers "
                f"{len(covered)}/{len(required)} required edges: "
                f"{assignment['edge_ids']}"
            )
            print(f"  Nodes: {assignment['nodes']}")
        assignments.append({**route, **assignment})
    return assignments


def build_route_graphs(assignments, network_graph_file):
    """Build directed, attribute-preserving graph for each assigned route."""
    with Path(network_graph_file).open("rb") as handle:
        network_graph = pickle.load(handle)

    route_graphs = {}
    for assignment in assignments:
        train_id = assignment["train_id"]
        route_graph = nx.MultiDiGraph()
        route_graph.graph.update(copy.deepcopy(network_graph.graph))
        route_graph.graph.update({
            "train_id": train_id,
            "route_column": assignment.get("route_column"),
            "required_edges": list(
                assignment.get("required_edges", ())
            ),
            "assigned_edges": list(assignment["edge_ids"]),
            "route_node_ids": list(assignment["node_ids"]),
            "length_meters": assignment.get("length_meters"),
        })

        nodes_by_id = {
            attributes["node_id"]: node
            for node, attributes in network_graph.nodes(data=True)
            if attributes.get("node_id") is not None
        }
        route_node_ids = list(assignment["node_ids"])
        for node_id in route_node_ids:
            node = nodes_by_id.get(node_id)
            if node is None:
                raise ValueError(
                    f"Route for train {train_id!r} references unknown "
                    f"node ID {node_id!r}"
                )
            route_graph.add_node(
                node,
                **copy.deepcopy(network_graph.nodes[node]),
            )

        for start_id, end_id, edge_id in zip(
            route_node_ids,
            route_node_ids[1:],
            assignment["edge_ids"],
        ):
            start = nodes_by_id[start_id]
            end = nodes_by_id[end_id]
            candidates = [
                (first, second, key, attributes)
                for first, second, key, attributes in network_graph.edges(
                    keys=True, data=True
                )
                if str(attributes.get("edge_id")) == str(edge_id)
                and {first, second} == {start, end}
            ]
            if not candidates:
                raise ValueError(
                    f"Route for train {train_id!r} cannot orient edge "
                    f"{edge_id!r} between node IDs {start_id!r} and "
                    f"{end_id!r}"
                )
            _first, _second, key, attributes = candidates[0]
            route_graph.add_edge(
                start,
                end,
                key=key,
                **copy.deepcopy(attributes),
            )
        route_graphs[train_id] = route_graph
    return route_graphs


def build_station_graph(route_graphs, station_stops_by_train=None):
    """Build a directed multigraph of station connections and train stops."""
    station_graph = nx.MultiDiGraph()
    station_graph.graph.update({
        "source_route_count": len(route_graphs),
        "station_graph": True,
    })
    connection_edges = {}
    station_stops_by_train = station_stops_by_train or {}
    stopping_trains_by_station = {}
    for train_id, stops in station_stops_by_train.items():
        for station in stops:
            station_key = str(station).strip().casefold()
            stopping_trains_by_station.setdefault(station_key, set()).add(
                train_id
            )

    for train_id, route_graph in route_graphs.items():
        node_by_id = {
            attributes["node_id"]: node
            for node, attributes in route_graph.nodes(data=True)
            if attributes.get("node_id") is not None
        }
        route_node_ids = route_graph.graph.get("route_node_ids", ())
        route_nodes = [node_by_id[node_id] for node_id in route_node_ids]
        assigned_edges = route_graph.graph.get("assigned_edges", ())
        station_events = []
        for node_id, node in zip(route_node_ids, route_nodes):
            station = route_graph.nodes[node].get("station")
            if station in (None, "None"):
                continue
            if station_events and station_events[-1][0] == station:
                continue
            station_events.append((station, node_id))
            station_graph.add_node(
                station,
                stopping_train_ids=sorted(
                    stopping_trains_by_station.get(
                        str(station).strip().casefold(),
                        set(),
                    ),
                    key=str,
                ),
            )

        for (origin, origin_id), (destination, destination_id) in zip(
            station_events, station_events[1:]
        ):
            if origin == destination:
                continue
            origin_index = route_node_ids.index(origin_id)
            destination_index = route_node_ids.index(destination_id)
            edge_ids = list(assigned_edges[origin_index:destination_index])
            connection_key = (origin, destination, tuple(edge_ids))
            edge_key = connection_edges.get(connection_key)
            if edge_key is None:
                edge_key = station_graph.add_edge(
                    origin,
                    destination,
                    train_id=train_id,
                    train_ids=[train_id],
                    route_node_ids=(origin_id, destination_id),
                    edge_ids=edge_ids,
                )
                connection_edges[connection_key] = edge_key
            else:
                station_graph[origin][destination][edge_key]["train_ids"].append(
                    train_id
                )

    return station_graph


def _ordered_route_nodes_and_edges(route_graph):
    """Return the traversed infrastructure nodes and edge IDs in order."""
    node_by_id = {
        attributes["node_id"]: node
        for node, attributes in route_graph.nodes(data=True)
        if attributes.get("node_id") is not None
    }
    route_node_ids = route_graph.graph.get("route_node_ids", ())
    route_nodes = [node_by_id[node_id] for node_id in route_node_ids]
    assigned_edges = route_graph.graph.get("assigned_edges", ())
    ordered_nodes = []
    ordered_edge_ids = []

    for edge_index, edge_id in enumerate(assigned_edges):
        if edge_index + 1 >= len(route_nodes):
            break
        start = route_nodes[edge_index]
        end = route_nodes[edge_index + 1]
        matching_edges = [
            (first, second, key)
            for first, second, key, attributes in route_graph.edges(
                keys=True, data=True
            )
            if str(attributes.get("edge_id")) == str(edge_id)
        ]
        if not matching_edges:
            continue
        if not any(
            {first, second} == {start, end}
            for first, second, _key in matching_edges
        ):
            continue
        if not ordered_nodes:
            ordered_nodes.append(start)
        elif ordered_nodes[-1] != start:
            ordered_nodes.append(start)
        ordered_nodes.append(end)
        ordered_edge_ids.append(edge_id)

    return ordered_nodes, ordered_edge_ids


def build_station_junction_graph(route_graphs, infrastructure_graph, station_stops_by_train=None,):
    """Build a multigraph of stations and traversed infrastructure junctions."""
    junction_graph = nx.MultiDiGraph()
    junction_graph.graph.update({
        "source_route_count": len(route_graphs),
        "station_junction_graph": True,
    })
    station_stops_by_train = station_stops_by_train or {}
    stopping_trains_by_station = {}
    for train_id, stops in station_stops_by_train.items():
        for station in stops:
            station_key = str(station).strip().casefold()
            stopping_trains_by_station.setdefault(station_key, set()).add(
                train_id
            )
    connection_edges = {}

    for node, attributes in infrastructure_graph.nodes(data=True):
        station = attributes.get("station")
        if station not in (None, "None"):
            station_graph_node = station
            junction_graph.add_node(
                station_graph_node,
                stopping_train_ids=sorted(
                    stopping_trains_by_station.get(
                        str(station).strip().casefold(),
                        set(),
                    ),
                    key=str,
                ),
            )
        elif infrastructure_graph.degree(node) > 2:
            junction_graph.add_node(node, passing_train_ids=[])

    for train_id, route_graph in route_graphs.items():
        ordered_nodes, ordered_edge_ids = _ordered_route_nodes_and_edges(
            route_graph
        )
        significant_events = []
        for node in ordered_nodes:
            station = route_graph.nodes[node].get("station")
            if station not in (None, "None"):
                event = station
                node_attributes = {
                    "stopping_train_ids": sorted(
                        stopping_trains_by_station.get(
                            str(station).strip().casefold(),
                            set(),
                        ),
                        key=str,
                    )
                }
            elif infrastructure_graph.degree(node) > 2:
                event = node
                node_attributes = {"passing_train_ids": []}
            else:
                continue

            if significant_events and significant_events[-1][0] == event:
                continue
            significant_events.append((
                event,
                route_graph.nodes[node]["node_id"],
            ))
            if event not in junction_graph:
                junction_graph.add_node(event, **node_attributes)
            elif "passing_train_ids" in node_attributes:
                junction_graph.nodes[event].setdefault(
                    "passing_train_ids", []
                )
            if "passing_train_ids" in node_attributes:
                passing_train_ids = junction_graph.nodes[event][
                    "passing_train_ids"
                ]
                if train_id not in passing_train_ids:
                    passing_train_ids.append(train_id)

        for (origin, origin_id), (destination, destination_id) in zip(
            significant_events, significant_events[1:]
        ):
            if origin == destination:
                continue
            ordered_ids = [
                route_graph.nodes[node]["node_id"]
                for node in ordered_nodes
            ]
            origin_index = ordered_ids.index(origin_id)
            destination_index = ordered_ids.index(destination_id)
            edge_ids = ordered_edge_ids[origin_index:destination_index]
            connection_key = (origin, destination, tuple(edge_ids))
            edge_key = connection_edges.get(connection_key)
            if edge_key is None:
                edge_key = junction_graph.add_edge(
                    origin,
                    destination,
                    train_id=train_id,
                    train_ids=[train_id],
                    route_node_ids=(origin_id, destination_id),
                    edge_ids=list(edge_ids),
                )
                connection_edges[connection_key] = edge_key
            else:
                junction_graph[origin][destination][edge_key][
                    "train_ids"
                ].append(train_id)

    return junction_graph


def write_station_graph(station_graph, output_file):
    """Write the station-level directed multigraph to a pickle file."""
    with Path(output_file).open("wb") as handle:
        pickle.dump(station_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)


def write_route_graphs(route_graphs, output_file="case_study_route_graphs.pkl"):
    """Write the per-train route graphs and their complete attributes."""
    with Path(output_file).open("wb") as handle:
        pickle.dump(route_graphs, handle, protocol=pickle.HIGHEST_PROTOCOL)


def assign_routes_from_csv(csv_file="required_edges.csv", network_graph_file="case_study_network.gpickle", minimum_junction_angle=90.0, route_graph_output="case_study_route_graphs.pkl", station_graph_output="case_study_station_graph.gpickle", station_junction_graph_output="case_study_station_junction_graph.gpickle", existing_station_stops_file="existing_rail_routes.csv", new_station_stops_file="new_rail_routes.csv", heuristic_weight=2.0,):
    """Load edge-based rail routes from CSV and assign feasible paths."""
    assignments = assign_routes(
        load_rail_routes(csv_file),
        network_graph_file=network_graph_file,
        minimum_junction_angle=minimum_junction_angle,
        heuristic_weight=heuristic_weight,
    )
    route_graphs = build_route_graphs(assignments, network_graph_file)
    write_route_graphs(route_graphs, route_graph_output)
    station_stops_by_train = load_route_station_stops_files(
        existing_station_stops_file,
        new_station_stops_file,
    )
    with Path(network_graph_file).open("rb") as handle:
        infrastructure_graph = pickle.load(handle)
    write_station_graph(
        build_station_graph(route_graphs, station_stops_by_train),
        station_graph_output,
    )
    write_station_graph(
        build_station_junction_graph(
            route_graphs,
            infrastructure_graph,
            station_stops_by_train,
        ),
        station_junction_graph_output,
    )
    return assignments


def main():
    parser = argparse.ArgumentParser(
        description="Assign shortest feasible routes to trains."
    )
    parser.add_argument("--routes", default="required_edges.csv")
    parser.add_argument(
        "--network-graph",
        default="case_study_network.gpickle",
    )
    parser.add_argument(
        "--output",
        default="case_study_route_graphs.pkl",
        help="Output pickle containing one graph per train.",
    )
    parser.add_argument(
        "--station-output",
        default="case_study_station_graph.gpickle",
        help="Output pickle containing the station-level directed multigraph.",
    )
    parser.add_argument(
        "--station-junction-output",
        default="case_study_station_junction_graph.gpickle",
        help=(
            "Output pickle containing the station-and-junction directed "
            "multigraph."
        ),
    )
    parser.add_argument(
        "--existing-station-stops",
        default="existing_rail_routes.csv",
        help="CSV containing station stops for existing trains.",
    )
    parser.add_argument(
        "--new-station-stops",
        default="new_rail_routes.csv",
        help="CSV containing station stops for new trains.",
    )
    parser.add_argument(
        "--minimum-junction-angle",
        type=float,
        default=90.0,
        help="Required angle at degree-greater-than-2 nodes.",
    )
    parser.add_argument(
        "--heuristic-weight",
        type=float,
        default=2.0,
        help=(
            "Weighted-A* heuristic multiplier; 1 is exact A*, higher values "
            "usually run faster but may not be strictly shortest (default: 2)."
        ),
    )
    args = parser.parse_args()

    routes = load_rail_routes(args.routes)
    print(f"Loaded {len(routes):,} train routes from {args.routes}.")
    assignments = assign_routes(
        routes,
        network_graph_file=args.network_graph,
        minimum_junction_angle=args.minimum_junction_angle,
        heuristic_weight=args.heuristic_weight,
    )
    route_graphs = build_route_graphs(assignments, args.network_graph)
    write_route_graphs(route_graphs, args.output)
    station_stops_by_train = load_route_station_stops_files(
        args.existing_station_stops,
        args.new_station_stops,
    )
    with Path(args.network_graph).open("rb") as handle:
        infrastructure_graph = pickle.load(handle)
    station_graph = build_station_graph(route_graphs, station_stops_by_train)
    write_station_graph(station_graph, args.station_output)
    station_junction_graph = build_station_junction_graph(
        route_graphs,
        infrastructure_graph,
        station_stops_by_train,
    )
    write_station_graph(
        station_junction_graph,
        args.station_junction_output,
    )
    print(
        f"Wrote {len(route_graphs):,} route graphs to {args.output}."
    )
    print(
        f"Wrote {station_graph.number_of_edges():,} station connections to "
        f"{args.station_output}."
    )
    print(
        f"Wrote {station_junction_graph.number_of_edges():,} station/junction "
        f"connections to {args.station_junction_output}."
    )


if __name__ == "__main__":
    main()
