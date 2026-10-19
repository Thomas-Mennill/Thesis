"""Construct shortest feasible train routes from required track segments."""

from __future__ import annotations

import csv
import copy
import heapq
import math
import pickle
import argparse
from pathlib import Path

import networkx as nx


def load_route_station_stops(csv_file="rail_routes.csv"):
    """Return station stops grouped by train ID from the timetable CSV."""
    with Path(csv_file).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"Route file {csv_file!r} is empty")

    station_stops = {}
    for column in range(1, len(rows[0]), 4):
        train_id = rows[0][column].strip()
        if not train_id or train_id.lower() == "time":
            continue
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


def load_rail_routes(csv_file="required_segments.csv"):
    """Load route definitions from the column-oriented route CSV.

    The first row of each column contains one or more train IDs. Non-empty
    cells below it contain the segment IDs required by those trains.
    """
    with Path(csv_file).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"Route file {csv_file!r} is empty")

    width = max(len(row) for row in rows)
    routes = []
    for column in range(width):
        header = rows[0][column].strip() if column < len(rows[0]) else ""
        if not header:
            continue
        train_ids = _parse_integer_list(
            header, field_name=f"column {column + 1} header"
        )
        segment_ids = []
        for row in rows[1:]:
            if column < len(row) and row[column].strip():
                segment_ids.extend(
                    _parse_integer_list(
                        row[column],
                        field_name=f"column {column + 1}",
                    )
                )
        if not segment_ids:
            raise ValueError(
                f"Route column {column + 1} has no required segments"
            )
        required_segments = tuple(dict.fromkeys(segment_ids))
        for train_id in train_ids:
            routes.append({
                "train_id": train_id,
                "route_column": column + 1,
                "required_segments": required_segments,
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


def _segment_lengths(network_graph):
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


def _junction_angle(first, current, next_node):
    """Return the angle between incoming and outgoing travel directions."""
    incoming = (first[0] - current[0], first[1] - current[1])
    outgoing = (next_node[0] - current[0], next_node[1] - current[1])
    incoming_length = math.hypot(*incoming)
    outgoing_length = math.hypot(*outgoing)
    if not incoming_length or not outgoing_length:
        return 0.0
    cosine = sum(a * b for a, b in zip(incoming, outgoing))
    cosine /= incoming_length * outgoing_length
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _segment_edges(segment_graph):
    """Index segment-graph edges by segment ID."""
    indexed = {}
    for first, second, key, attributes in segment_graph.edges(
        keys=True, data=True
    ):
        segment_id = attributes.get("segment_id")
        if segment_id is None:
            raise ValueError(
                f"Segment graph edge {(first, second, key)!r} has no segment_id"
            )
        indexed.setdefault(segment_id, []).append(
            (first, second, key, attributes)
        )
    return indexed


def _shortest_route_for_segments(
    segment_graph,
    network_graph,
    required_segments,
    minimum_junction_angle=90.0,
    heuristic_weight=2.0,
):
    """Find a feasible route with weighted A* search.

    The route starts at either endpoint of an edge carrying the first required
    segment. A turn through a node is allowed only when its angle is strictly
    greater than ``minimum_junction_angle``. A weight above one prioritizes
    routes that appear closer to the remaining required segments and can
    reduce runtime at the cost of strict shortest-path optimality.
    """
    if heuristic_weight < 1.0:
        raise ValueError("heuristic_weight must be at least 1")
    segment_edges = _segment_edges(segment_graph)
    segment_lengths = _segment_lengths(network_graph)
    missing = set(required_segments) - set(segment_edges)
    if missing:
        raise ValueError(f"Required segments are absent: {sorted(missing)}")
    missing_lengths = set(required_segments) - set(segment_lengths)
    if missing_lengths:
        raise ValueError(
            f"Required segments have no network geometry: "
            f"{sorted(missing_lengths)}"
        )

    required = tuple(required_segments)
    if not required:
        raise ValueError("At least one required segment is needed")
    segment_endpoints = {
        segment_id: {
            endpoint
            for first, second, _key, _attributes
            in segment_edges[segment_id]
            for endpoint in (first, second)
        }
        for segment_id in required
    }

    def heuristic(current, next_index):
        """Lower-bound travel to the next required segment."""
        if next_index >= len(required):
            return 0.0
        return min(
            _edge_length(current, endpoint, {})
            for segment_id in required[next_index:]
            for endpoint in segment_endpoints[segment_id]
        )

    queue = []
    best = {}
    previous_states = {}
    initial_segments = {}
    counter = 0

    def reconstruct_route(state, distance):
        state_path = [state]
        while state_path[-1] in previous_states:
            state_path.append(previous_states[state_path[-1]][0])
        state_path.reverse()
        route_nodes = [state_path[0][0], state_path[0][1]]
        segment_ids = [initial_segments[state_path[0]]]
        for route_state in state_path[1:]:
            route_nodes.append(route_state[1])
            segment_ids.append(previous_states[route_state][1])
        nodes = [
            {
                "node_id": network_graph.nodes[node].get("node_id"),
                "station": network_graph.nodes[node].get("station"),
            }
            for node in route_nodes
        ]
        covered_count = state[2]
        return {
            "nodes": nodes,
            "node_coordinates": route_nodes,
            "segment_ids": segment_ids,
            "length_meters": distance,
            "complete": covered_count == len(required),
            "covered_required_segments": list(required[:covered_count]),
        }

    for first, second, key, attributes in segment_edges[required_segments[0]]:
        for start, next_node in ((first, second), (second, first)):
            segment_id = attributes["segment_id"]
            edge_length = segment_lengths[segment_id]
            state = (start, next_node, 1)
            best[state] = edge_length
            initial_segments[state] = segment_id
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
        previous, current, next_index = state
        if (
            best_partial_state is None
            or next_index > best_partial_state[2]
            or (
                next_index == best_partial_state[2]
                and distance < best_partial_distance
            )
        ):
            best_partial_state = state
            best_partial_distance = distance
        if next_index == len(required):
            return reconstruct_route(state, distance)
        for next_node in segment_graph.neighbors(current):
            if next_node == previous:
                continue
            if segment_graph.degree(current) > 2 and _junction_angle(
                previous, current, next_node
            ) <= minimum_junction_angle:
                continue
            for key, attributes in segment_graph[current][next_node].items():
                segment_id = attributes["segment_id"]
                new_index = next_index
                if segment_id in required:
                    if segment_id != required[next_index]:
                        continue
                    new_index += 1
                new_distance = distance + segment_lengths[segment_id]
                new_state = (current, next_node, new_index)
                if new_distance < best.get(new_state, math.inf):
                    best[new_state] = new_distance
                    previous_states[new_state] = (state, segment_id)
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
            f"No route can start with segment {required[0]}"
        )
    partial_route = reconstruct_route(
        best_partial_state,
        best_partial_distance,
    )
    partial_route["uncovered_required_segments"] = list(
        required[best_partial_state[2]:]
    )
    return partial_route


def assign_routes(
    routes,
    segment_graph_file="case_study_segment_graph.pkl",
    network_graph_file="case_study_network.gpickle",
    minimum_junction_angle=90.0,
    heuristic_weight=2.0,
):
    """Assign the shortest feasible track route to every train definition.

    If all required segments cannot be covered in order, the best partial
    route is returned and displayed instead of aborting the remaining routes.
    """
    if not 0.0 <= minimum_junction_angle < 180.0:
        raise ValueError("minimum_junction_angle must be between 0 and 180")
    with Path(segment_graph_file).open("rb") as handle:
        segment_graph = pickle.load(handle)
    with Path(network_graph_file).open("rb") as handle:
        network_graph = pickle.load(handle)

    assignments = []
    for route in routes:
        required = tuple(route["required_segments"])
        assignment = _shortest_route_for_segments(
            segment_graph,
            network_graph,
            required,
            minimum_junction_angle,
            heuristic_weight,
        )
        if not assignment["complete"]:
            covered = assignment["covered_required_segments"]
            print(
                f"No feasible route covers all required segments for train "
                f"{route['train_id']}. Best partial route covers "
                f"{len(covered)}/{len(required)} required segments: "
                f"{assignment['segment_ids']}"
            )
            print(f"  Nodes: {assignment['nodes']}")
        assignments.append({**route, **assignment})
    return assignments


def build_route_graphs(assignments, network_graph_file):
    """Build an attribute-preserving network graph for each assigned route."""
    with Path(network_graph_file).open("rb") as handle:
        network_graph = pickle.load(handle)

    route_graphs = {}
    for assignment in assignments:
        train_id = assignment["train_id"]
        segment_ids = set(assignment["segment_ids"])
        route_graph = network_graph.__class__()
        route_graph.graph.update(copy.deepcopy(network_graph.graph))
        route_graph.graph.update({
            "train_id": train_id,
            "route_column": assignment.get("route_column"),
            "required_segments": list(
                assignment.get("required_segments", ())
            ),
            "assigned_segments": list(assignment["segment_ids"]),
            "route_node_coordinates": copy.deepcopy(
                assignment.get("node_coordinates", ())
            ),
            "length_meters": assignment.get("length_meters"),
        })

        for first, second, key, attributes in network_graph.edges(
            keys=True, data=True
        ):
            if attributes.get("segment_id") not in segment_ids:
                continue
            route_graph.add_node(first, **copy.deepcopy(network_graph.nodes[first]))
            route_graph.add_node(second, **copy.deepcopy(network_graph.nodes[second]))
            route_graph.add_edge(
                first,
                second,
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
        route_nodes = route_graph.graph.get("route_node_coordinates", ())
        assigned_segments = route_graph.graph.get("assigned_segments", ())
        station_events = []
        for node_index, node in enumerate(route_nodes):
            station = route_graph.nodes[node].get("station")
            if station in (None, "None"):
                continue
            if station_events and station_events[-1][0] == station:
                continue
            station_events.append((station, node_index))
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

        for (origin, origin_index), (destination, destination_index) in zip(
            station_events, station_events[1:]
        ):
            if origin == destination:
                continue
            segment_ids = list(
                assigned_segments[origin_index:destination_index]
            )
            connection_key = (origin, destination, tuple(segment_ids))
            edge_key = connection_edges.get(connection_key)
            if edge_key is None:
                edge_key = station_graph.add_edge(
                    origin,
                    destination,
                    train_id=train_id,
                    train_ids=[train_id],
                    route_node_indices=(origin_index, destination_index),
                    segment_ids=segment_ids,
                )
                connection_edges[connection_key] = edge_key
            else:
                station_graph[origin][destination][edge_key]["train_ids"].append(
                    train_id
                )

    return station_graph


def _ordered_route_nodes_and_segments(route_graph):
    """Return the traversed infrastructure nodes and segment IDs in order."""
    route_nodes = route_graph.graph.get("route_node_coordinates", ())
    assigned_segments = route_graph.graph.get("assigned_segments", ())
    ordered_nodes = []
    ordered_segment_ids = []

    for segment_index, segment_id in enumerate(assigned_segments):
        if segment_index + 1 >= len(route_nodes):
            break
        start = route_nodes[segment_index]
        end = route_nodes[segment_index + 1]
        segment_graph = nx.MultiGraph()
        for first, second, key, attributes in route_graph.edges(
            keys=True, data=True
        ):
            if attributes.get("segment_id") == segment_id:
                segment_graph.add_edge(first, second, key=key)
        if start not in segment_graph or end not in segment_graph:
            continue
        try:
            path = nx.shortest_path(segment_graph, start, end)
        except nx.NetworkXNoPath:
            continue
        if ordered_nodes and path[0] == ordered_nodes[-1]:
            path = path[1:]
        if not path:
            continue
        ordered_nodes.extend(path)
        ordered_segment_ids.extend([segment_id] * (len(path) - 1))

    return ordered_nodes, ordered_segment_ids


def build_station_junction_graph(
    route_graphs,
    infrastructure_graph,
    station_stops_by_train=None,
):
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
        ordered_nodes, ordered_segment_ids = _ordered_route_nodes_and_segments(
            route_graph
        )
        significant_events = []
        for node_index, node in enumerate(ordered_nodes):
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
            significant_events.append((event, node_index))
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

        for (origin, origin_index), (destination, destination_index) in zip(
            significant_events, significant_events[1:]
        ):
            if origin == destination:
                continue
            segment_ids = ordered_segment_ids[origin_index:destination_index]
            connection_key = (origin, destination, tuple(segment_ids))
            edge_key = connection_edges.get(connection_key)
            if edge_key is None:
                edge_key = junction_graph.add_edge(
                    origin,
                    destination,
                    train_id=train_id,
                    train_ids=[train_id],
                    route_node_indices=(origin_index, destination_index),
                    segment_ids=list(segment_ids),
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


def assign_routes_from_csv(
    csv_file="required_segments.csv",
    segment_graph_file="case_study_segment_graph.pkl",
    network_graph_file="case_study_network.gpickle",
    minimum_junction_angle=90.0,
    route_graph_output="case_study_route_graphs.pkl",
    station_graph_output="case_study_station_graph.gpickle",
    station_junction_graph_output="case_study_station_junction_graph.gpickle",
    station_stops_file="rail_routes.csv",
    heuristic_weight=2.0,
):
    """Load rail routes from CSV and return their shortest feasible paths."""
    assignments = assign_routes(
        load_rail_routes(csv_file),
        segment_graph_file=segment_graph_file,
        network_graph_file=network_graph_file,
        minimum_junction_angle=minimum_junction_angle,
        heuristic_weight=heuristic_weight,
    )
    route_graphs = build_route_graphs(assignments, network_graph_file)
    write_route_graphs(route_graphs, route_graph_output)
    station_stops_by_train = load_route_station_stops(station_stops_file)
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
    parser.add_argument("--routes", default="required_segments.csv")
    parser.add_argument(
        "--segment-graph",
        default="case_study_segment_graph.pkl",
    )
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
        "--station-stops",
        default="rail_routes.csv",
        help="Timetable CSV used to assign stopping trains to stations.",
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
        segment_graph_file=args.segment_graph,
        network_graph_file=args.network_graph,
        minimum_junction_angle=args.minimum_junction_angle,
        heuristic_weight=args.heuristic_weight,
    )
    route_graphs = build_route_graphs(assignments, args.network_graph)
    write_route_graphs(route_graphs, args.output)
    station_stops_by_train = load_route_station_stops(args.station_stops)
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