"""Construct shortest feasible train routes from required track segments."""

from __future__ import annotations

import csv
import copy
import heapq
import math
import pickle
import argparse
from pathlib import Path


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
        segment_id = attributes.get("segment_id")
        if segment_id is None:
            continue
        lengths[segment_id] = lengths.get(segment_id, 0.0) + _edge_length(
            first, second, attributes
        )
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
    print(
        f"Wrote {len(route_graphs):,} route graphs to {args.output}."
    )


if __name__ == "__main__":
    main()