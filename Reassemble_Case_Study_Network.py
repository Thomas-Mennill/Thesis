"""Reassemble selected GeoJSON graph fragments for the case study.

Each fragment contains Point features with ``feature_type=node`` and
LineString features with ``feature_type=track``.  Repeated station and
station-to-station files are deduplicated by geometry.  Parallel tracks are
retained in a NetworkX MultiGraph.  By default, the selected fragments are
listed in ``case_study_section_names.txt`` and located in
``between_station_sections`` and ``station_nodes``.  Node and track features
are annotated with ``has_platform`` when they are within a configurable
distance of railway platforms.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
import pickle
import re
import tempfile
from collections import defaultdict
from pathlib import Path

import folium
import networkx as nx
from shapely.geometry import LineString, Point, shape
from shapely.ops import transform
from shapely.strtree import STRtree


DEFAULT_INPUT_FOLDER = "Network_For_Case_Study"
DEFAULT_SECTION_NAMES_FILE = "case_study_section_names.txt"
DEFAULT_BETWEEN_STATION_FOLDER = "between_station_sections"
DEFAULT_STATION_NODES_FOLDER = "station_nodes"
DEFAULT_OUTPUT_GRAPH = "case_study_network.gpickle"
DEFAULT_OUTPUT_GEOJSON = "case_study_network.geojson"
DEFAULT_OUTPUT_MAP = "case_study_network_map.html"
DEFAULT_CENTERLINE_GEOJSON = "parallel_track_centrelines.geojson"
DEFAULT_CENTERLINE_MAX_ANGLE_DEGREES = 5.0
DEFAULT_SEGMENT_GEOJSON = "case_study_segments.geojson"
DEFAULT_SEGMENT_GRAPH = "case_study_segment_graph.pkl"
DEFAULT_SEGMENT_OUTPUT_FOLDER = "Segments"
DEFAULT_SWITCH_GEOJSON = "NL_Switches.geojson"
DEFAULT_PLATFORM_SOURCE = "third_try_filtered_data.geojson"
DEFAULT_PLATFORM_FALLBACK_SOURCE = "NL_Railway_Data_Signals.geojson"
DEFAULT_PLATFORM_DISTANCE_METERS = 3.0
DEFAULT_ROOT_STATIONS = (
    "Arnhem Centraal",
    "Utrecht Centraal",
    "Amsterdam Centraal",
    "Amsterdam Zuid"
)
COORDINATE_DIGITS = 7
REMOVED_ATTRIBUTES = frozenset({
    "wikidata", "wifi:operator", "wheelchair", "wheelchair:description",
    "air_conditioning", "wikimedia_commons", "wikipedia", "addr:city",
    "addr:country", "addr:housenumber", "addr:postcode", "addr:street",
    "brand", "bridge:wikipedia", "bridge:structure", "bridge:material",
    "bridge:movable", "bridge:name", "bridge:start_date", "building",
    "building:material", "check_date:wheelchair", "check_date",
    "access:conditional", "alt_name", "alt_name:-1999-05-30", "attraction",
    "comment", "construction", "covered", "construction:railway:etcs",
    "cutting", "cutting:name", "departures_board", "description",
    "description:ProRail", "description:nl", "designation", "detail",
    "disused:railway", "drinking_water",
    "drinking_water:refill:self_service", "ele", "embankment",
    "embankment:name", "entrance", "fixme", "gauge", "height", "iata",
    "image", "internet_access", "internet_access:fee",
    "internet_access:operator", "internet_access:ssid", "layer", "level",
    "loc_name", "local_name", "material", "name:ar", "name:ckb", "name:de",
    "name:el", "name:es", "name:etymology:wikidata", "name:fa", "name:fr",
    "name:fy", "name:ja", "name:ko", "name:nl", "name:pnb", "name:ru",
    "name:ur", "name:zh", "name:zh-Hant", "network", "network:short",
    "network:subway", "network:train", "network:wikidata",
    "network:wikipedia", "not:building", "not:network:wikidata", "note:nl",
    "note:start_date", "note", "note:NL", "opening_date", "opening_hours",
    "operator", "operator:short", "operator:subway", "operator:train",
    "operator:website", "operator:wikidata", "outdoor_seating", "old_name",
    "old_name:-1997", "old_name:-1999-05-30", "old_name:-2000-05-29",
    "old_name:-2015-11-19", "old_name:-2019-12-15", "old_name:1863-2015",
    "old_name:1866-1989", "old_name:1866-2015", "old_name:1867-2016",
    "old_name:1868-1999", "old_name:1881-1921", "old_name:1885-2015",
    "old_name:1921-1955", "old_name:1969-2010", "old_name:1971-2015",
    "old_name:1985-2006", "proposed:bridge", "proposed:electrified",
    "proposed:frequency", "proposed:voltage", "ruins", "source",
    "source:ProRail", "source:date", "source:height", "source:height:date",
    "source:name", "spotpass", "start_date", "subway",
    "supervised", "toilets", "toilets:wheelchair", "tunnel", "tunnel:name",
    "tunnel:wikipedia", "usage", "voltage", "website", "workrules", "ref",
    "ref:DB_Netz_AG", "ref:FR:sncf:resarail", "ref:FR:uic8", "ref:ProRail",
    "ref:ProRailID", "ref:ProRailSpoortak", "ref:bag", "ref:crs",
})


def coordinate_key(coordinates):
    """Return a stable node key for GeoJSON longitude/latitude coordinates."""
    return tuple(round(float(value), COORDINATE_DIGITS) for value in coordinates)


def geometry_key(coordinates):
    """Treat a track and its reversed representation as the same geometry."""
    forward = tuple(coordinate_key(coordinate) for coordinate in coordinates)
    reverse = tuple(reversed(forward))
    return min(forward, reverse)


def track_tangent_and_right_normal(coordinates):
    """Return the unit tangent and rightward normal for a track.

    The tangent follows the track from its first coordinate to its last
    coordinate. Coordinates are interpreted as ``[x, y]`` pairs, and the
    rightward normal is ``[t_y, -t_x]``.
    """
    if len(coordinates) < 2:
        raise ValueError("A track must contain at least two coordinates")

    first = coordinates[0]
    last = coordinates[-1]
    delta_x = float(last[0]) - float(first[0])
    delta_y = float(last[1]) - float(first[1])
    length = math.hypot(delta_x, delta_y)
    if not length:
        raise ValueError("A track must have distinct endpoints")

    tangent = [delta_x / length, delta_y / length]
    right_normal = [tangent[1], -tangent[0]]
    return tangent, right_normal


def _undirected_track_angle_degrees(first_coordinates, second_coordinates):
    """Return the smaller angle between two track orientations."""
    first_tangent, _ = track_tangent_and_right_normal(first_coordinates)
    second_tangent, _ = track_tangent_and_right_normal(second_coordinates)
    return _undirected_tangent_angle_degrees(first_tangent, second_tangent)


def _undirected_tangent_angle_degrees(first_tangent, second_tangent):
    """Return the smaller angle between two undirected tangent vectors."""
    cosine = abs(sum(
        first * second
        for first, second in zip(first_tangent, second_tangent)
    ))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def _line_local_tangent(line, point):
    """Return the tangent of the line segment nearest to a point."""
    coordinates = list(line.coords)
    nearest_segment = min(
        zip(coordinates, coordinates[1:]),
        key=lambda segment: Point(point).distance(
            LineString(segment)
        ),
    )
    tangent, _ = track_tangent_and_right_normal(nearest_segment)
    return tangent


def read_fragments(input_folder, input_files=None):
    """Yield ``(source_name, feature)`` pairs from selected GeoJSON files."""
    folder = Path(input_folder)
    files = sorted(input_files or folder.glob("*.geojson"))
    if not files:
        raise FileNotFoundError(f"No GeoJSON files found in {folder}")

    for path in files:
        with path.open(encoding="utf-8") as handle:
            collection = json.load(handle)
        for feature in collection.get("features", []):
            yield path.name, feature


def restore_source_attributes(properties):
    """Restore retained source attributes stored by the splitter."""
    source_attributes = properties.pop("source_attributes", None)
    if source_attributes is None:
        return properties
    if isinstance(source_attributes, str):
        try:
            source_attributes = json.loads(source_attributes)
        except json.JSONDecodeError as error:
            raise ValueError(
                "Invalid source_attributes JSON in a section fragment"
            ) from error
    if not isinstance(source_attributes, dict):
        raise ValueError("source_attributes must contain a JSON object")
    for key, value in source_attributes.items():
        if key in REMOVED_ATTRIBUTES or key in properties:
            continue
        if isinstance(key, str) and key.startswith("source_"):
            source_key = key.removeprefix("source_")
            if source_key in REMOVED_ATTRIBUTES:
                continue
        if key not in properties:
            properties[key] = value
    return properties


def find_case_study_section_files(
    section_names_file=DEFAULT_SECTION_NAMES_FILE,
    between_station_folder=DEFAULT_BETWEEN_STATION_FOLDER,
    station_nodes_folder=DEFAULT_STATION_NODES_FOLDER,
):
    """Find the listed section files in the station and inter-station folders."""
    names_path = Path(section_names_file)
    if not names_path.is_file():
        raise FileNotFoundError(f"Section names file not found: {names_path}")

    requested_names = []
    for line in names_path.read_text(encoding="utf-8").splitlines():
        name = line.strip().strip('"').strip("'")
        if not name:
            continue
        requested_names.append(Path(name.replace("\\", "/")).name)
    requested_names = list(dict.fromkeys(requested_names))
    if not requested_names:
        raise ValueError(f"No section names found in {names_path}")

    candidates = {}
    for folder_name in (between_station_folder, station_nodes_folder):
        folder = Path(folder_name)
        if not folder.is_dir():
            raise FileNotFoundError(f"Section folder not found: {folder}")
        for path in folder.rglob("*.geojson"):
            candidates.setdefault(path.name, []).append(path)

    missing = [
        name for name in requested_names
        if name not in candidates
    ]
    if missing:
        raise FileNotFoundError(
            "The following listed sections were not found in "
            f"{between_station_folder} or {station_nodes_folder}: "
            + ", ".join(missing)
        )

    selected_files = []
    for name in requested_names:
        matches = candidates[name]
        selected_files.append(matches[0])
        if len(matches) > 1:
            print(
                f"[Reassemble] Using {matches[0]} for {name}; "
                f"also found {len(matches) - 1} duplicate location(s)."
            )
    print(
        f"[Reassemble] Selected {len(selected_files):,} section files from "
        f"{names_path}."
    )
    return selected_files


def reassemble_network(
    input_folder=DEFAULT_INPUT_FOLDER,
    section_names_file=None,
    between_station_folder=DEFAULT_BETWEEN_STATION_FOLDER,
    station_nodes_folder=DEFAULT_STATION_NODES_FOLDER,
):
    """Build a deduplicated MultiGraph from selected case-study fragments."""
    input_path = Path(input_folder)
    input_files = (
        find_case_study_section_files(
            section_names_file,
            between_station_folder,
            station_nodes_folder,
        )
        if section_names_file
        else sorted(input_path.glob("*.geojson"))
    )
    if not input_files:
        source_description = (
            f"{section_names_file}" if section_names_file else str(input_path)
        )
        raise FileNotFoundError(f"No GeoJSON files found for {source_description}")
    print(
        f"[Reassemble] Reading {len(input_files):,} GeoJSON fragments "
        f"from {input_path}..."
    )

    graph = nx.MultiGraph()
    node_sources = defaultdict(set)
    edge_sources = defaultdict(set)
    edge_features = {}
    skipped = defaultdict(int)

    processed_sources = 0
    current_source = None
    for source_name, feature in read_fragments(input_folder, input_files):
        if source_name != current_source:
            current_source = source_name
            processed_sources += 1
            print(
                f"[Reassemble] Fragment {processed_sources:,}/"
                f"{len(input_files):,}: {source_name}"
            )
        properties = restore_source_attributes({
            key: value for key, value in (feature.get("properties") or {}).items()
            if key not in REMOVED_ATTRIBUTES
        })
        geometry = feature.get("geometry") or {}
        feature_type = properties.get("feature_type")
        coordinates = geometry.get("coordinates")

        if feature_type == "node" and geometry.get("type") == "Point":
            node = coordinate_key(coordinates)
            graph.add_node(node)
            node_sources[node].add(source_name)
            for key, value in properties.items():
                if key not in graph.nodes[node] or graph.nodes[node][key] in (None, "None"):
                    graph.nodes[node][key] = value
            continue

        if feature_type == "track" and geometry.get("type") == "LineString":
            if not isinstance(coordinates, list) or len(coordinates) < 2:
                skipped["tracks with fewer than two coordinates"] += 1
                continue
            start = coordinate_key(coordinates[0])
            end = coordinate_key(coordinates[-1])
            graph.add_node(start)
            graph.add_node(end)
            key = geometry_key(coordinates)
            edge_sources[key].add(source_name)
            if key not in edge_features:
                edge_features[key] = {
                    "start": start,
                    "end": end,
                    "coordinates": coordinates,
                    "properties": properties,
                }
            continue

        skipped[feature_type or geometry.get("type", "unknown")] += 1

    for node, sources in node_sources.items():
        graph.nodes[node]["source_files"] = sorted(sources)

    for key, feature in edge_features.items():
        graph.add_edge(
            feature["start"],
            feature["end"],
            geometry=feature["coordinates"],
            source_files=sorted(edge_sources[key]),
            **feature["properties"],
        )

    remove_attributes(graph)
    print(
        f"[Reassemble] Complete: {graph.number_of_nodes():,} nodes, "
        f"{graph.number_of_edges():,} unique tracks."
    )

    graph.graph.update(
        input_folder=str(input_folder),
        coordinate_digits=COORDINATE_DIGITS,
        fragment_count=len(set(
            source for sources in node_sources.values() for source in sources
        ) | {
            source for sources in edge_sources.values() for source in sources
        }),
        skipped_features=dict(skipped),
    )
    assign_network_ids(graph)
    return graph


def remove_attributes(graph):
    """Remove source attributes that are not part of the reassembled network."""
    removed_count = 0
    for _, attributes in graph.nodes(data=True):
        for attribute in list(attributes):
            source_attribute = (
                attribute.removeprefix("source_")
                if isinstance(attribute, str)
                else attribute
            )
            if (
                attribute not in REMOVED_ATTRIBUTES
                and source_attribute not in REMOVED_ATTRIBUTES
                and attribute != "source_attributes"
            ):
                continue
            if attribute in attributes:
                attributes.pop(attribute)
                removed_count += 1
    for _, _, attributes in graph.edges(data=True):
        for attribute in list(attributes):
            source_attribute = (
                attribute.removeprefix("source_")
                if isinstance(attribute, str)
                else attribute
            )
            if (
                attribute not in REMOVED_ATTRIBUTES
                and source_attribute not in REMOVED_ATTRIBUTES
                and attribute != "source_attributes"
            ):
                continue
            if attribute in attributes:
                attributes.pop(attribute)
                removed_count += 1
    return removed_count


def angle_between(center, first, second):
    """Return the smaller angle in degrees between two links at a node."""
    first_vector = (first[0] - center[0], first[1] - center[1])
    second_vector = (second[0] - center[0], second[1] - center[1])
    first_length = math.hypot(*first_vector)
    second_length = math.hypot(*second_vector)
    if not first_length or not second_length:
        return 0.0
    cosine = sum(a * b for a, b in zip(first_vector, second_vector))
    cosine /= first_length * second_length
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def reachable_nodes_from_station(graph, station, minimum_angle):
    """Return nodes reachable from one station using allowed turns."""
    roots = {
        node for node, attributes in graph.nodes(data=True)
        if attributes.get("station") == station
    }
    if not roots:
        raise ValueError(f"No graph nodes are labelled {station!r}")

    reachable = set(roots)
    states = []
    for root in roots:
        states.extend((root, neighbor) for neighbor in graph.neighbors(root))
        reachable.update(graph.neighbors(root))

    visited = set(states)
    while states:
        previous, current = states.pop()
        for neighbor in graph.neighbors(current):
            if neighbor == previous:
                continue
            if angle_between(current, previous, neighbor) <= minimum_angle:
                continue
            reachable.add(neighbor)
            state = (current, neighbor)
            if state not in visited:
                visited.add(state)
                states.append(state)
    return reachable


def prune_to_stations(graph, stations=DEFAULT_ROOT_STATIONS, minimum_angle=90.0):
    """Keep nodes reachable from every station using allowed turns.

    The previous node is part of the traversal state because an outgoing link
    can only be judged relative to the link by which the current node was
    reached.  The first link from a station node is unrestricted.
    """
    stations = tuple(stations)
    reachable_sets = [
        reachable_nodes_from_station(graph, station, minimum_angle)
        for station in stations
    ]
    retained = set.intersection(*reachable_sets)

    pruned = graph.subgraph(retained).copy()
    pruned.graph.update(graph.graph)
    pruned.graph.update(
        reachability_stations=list(stations),
        reachability_minimum_angle_degrees=minimum_angle,
        retained_nodes=len(retained),
        removed_nodes=graph.number_of_nodes() - len(retained),
    )
    return pruned


def prune_network_to_stations(
    graph, stations, minimum_angle=90.0
):
    """Keep nodes that can reach at least one requested station.

    Reachability follows the same minimum-turn-angle rule as the existing
    traversal, while the station list controls the complete pruning scope.
    """
    stations = tuple(dict.fromkeys(stations))
    if not stations:
        raise ValueError("At least one station is required for pruning")

    reachable_sets = [
        reachable_nodes_from_station(graph, station, minimum_angle)
        for station in stations
    ]
    retained = set.union(*reachable_sets)
    pruned = graph.subgraph(retained).copy()
    pruned.graph.update(graph.graph)
    pruned.graph.update(
        reachability_stations=list(stations),
        reachability_minimum_angle_degrees=minimum_angle,
        retained_nodes=pruned.number_of_nodes(),
        removed_nodes=graph.number_of_nodes() - pruned.number_of_nodes(),
    )
    return pruned


def delete_nodes_near_origin_without_target_path(
    graph,
    origin_station,
    target_station,
    maximum_distance_meters,
    minimum_angle=90.0,
):
    """Delete near-origin nodes without a path to the target station.

    Distance is measured from the node to the nearest node labelled
    ``origin_station``.  Paths use the same turn-angle restriction as the
    station pruning helpers.
    """
    if maximum_distance_meters < 0:
        raise ValueError("maximum_distance_meters must not be negative")

    origin_nodes = {
        node for node, attributes in graph.nodes(data=True)
        if attributes.get("station") == origin_station
    }
    if not origin_nodes:
        raise ValueError(f"No graph nodes are labelled {origin_station!r}")

    destination_nodes = {
        node for node, attributes in graph.nodes(data=True)
        if attributes.get("station") == target_station
    }
    if not destination_nodes:
        raise ValueError(f"No graph nodes are labelled {target_station!r}")

    reachable_from_destination = reachable_nodes_from_station(
        graph,
        target_station,
        minimum_angle,
    )

    nodes_to_delete = {
        node for node in graph
        if (
            min(
                _distance_meters(node, origin_node)
                for origin_node in origin_nodes
            ) <= maximum_distance_meters
            and node not in reachable_from_destination
        )
    }
    pruned = graph.copy()
    pruned.remove_nodes_from(nodes_to_delete)
    pruned.graph.update(
        graph.graph,
        local_origin_station=origin_station,
        local_target_station=target_station,
        local_origin_radius_meters=maximum_distance_meters,
        local_reachability_minimum_angle_degrees=minimum_angle,
        local_nodes_removed=len(nodes_to_delete),
    )
    return pruned


def delete_nodes_near_utrecht_without_amsterdam_path(
    graph,
    maximum_distance_meters=5000.0,
    origin_station="Utrecht Centraal",
    destination_stations=("Amsterdam Zuid", "Amsterdam Centraal"),
):
    """Delete near-Utrecht nodes without a path to either Amsterdam station."""
    if maximum_distance_meters < 0:
        raise ValueError("maximum_distance_meters must not be negative")

    destination_stations = tuple(dict.fromkeys(destination_stations))
    if not destination_stations:
        raise ValueError("At least one destination station is required")

    origin_nodes = {
        node for node, attributes in graph.nodes(data=True)
        if attributes.get("station") == origin_station
    }
    if not origin_nodes:
        raise ValueError(f"No graph nodes are labelled {origin_station!r}")

    target_nodes = {
        node for node, attributes in graph.nodes(data=True)
        if attributes.get("station") in destination_stations
    }
    if not target_nodes:
        raise ValueError(
            "No graph nodes are labelled with a requested destination station"
        )

    reachable_from_target = set()
    for component in nx.connected_components(graph):
        if component & target_nodes:
            reachable_from_target.update(component)

    nodes_to_delete = {
        node for node in graph
        if (
            min(
                _distance_meters(node, origin_node)
                for origin_node in origin_nodes
            ) <= maximum_distance_meters
            and node not in reachable_from_target
        )
    }
    pruned = graph.copy()
    pruned.remove_nodes_from(nodes_to_delete)
    pruned.graph.update(
        graph.graph,
        local_origin_station=origin_station,
        local_destination_stations=list(destination_stations),
        local_origin_radius_meters=maximum_distance_meters,
        local_nodes_removed=len(nodes_to_delete),
    )
    return pruned


def delete_disconnected_edges(graph):
    """Delete edges in components that contain no station-labelled node."""
    station_nodes = {
        node for node, attributes in graph.nodes(data=True)
        if _station_name(attributes) is not None
    }
    connected_nodes = set()
    for component in nx.connected_components(graph):
        if component & station_nodes:
            connected_nodes.update(component)

    edges_to_delete = [
        (first, second, key)
        for first, second, key in graph.edges(keys=True)
        if first not in connected_nodes or second not in connected_nodes
    ]
    cleaned = graph.copy()
    cleaned.remove_edges_from(edges_to_delete)
    cleaned.graph.update(
        graph.graph,
        disconnected_edges_removed=len(edges_to_delete),
    )
    return cleaned


def delete_non_station_degree_one_nodes(graph):
    """Delete non-station leaf nodes until no such nodes remain."""
    cleaned = graph.copy()
    removed_nodes = 0
    while True:
        leaf_nodes = [
            node for node, attributes in cleaned.nodes(data=True)
            if cleaned.degree(node) == 1
            and _station_name(attributes) is None
        ]
        if not leaf_nodes:
            break
        cleaned.remove_nodes_from(leaf_nodes)
        removed_nodes += len(leaf_nodes)

    cleaned.graph.update(
        graph.graph,
        non_station_degree_one_nodes_removed=removed_nodes,
    )
    return cleaned


def _point_line_distance(point, first, second):
    """Return the perpendicular distance from a point to an infinite line."""
    line_x = second[0] - first[0]
    line_y = second[1] - first[1]
    length = math.hypot(line_x, line_y)
    if not length:
        return math.inf
    return abs(
        line_y * point[0] - line_x * point[1]
        + second[0] * first[1] - second[1] * first[0]
    ) / length


def _branch_lookahead(graph, center, first, limit=3):
    """Collect up to ``limit`` nodes along a branch from a junction."""
    points = []
    previous, current = center, first
    while current is not None and len(points) < limit:
        points.append(current)
        if graph.degree(current) >= 3:
            break
        next_nodes = [node for node in graph.neighbors(current) if node != previous]
        if not next_nodes:
            break
        previous, current = current, next_nodes[0]
    return points


def _average_junction_angle(graph, previous, current, candidate):
    """Average angles using up to three nodes behind and ahead of a junction."""
    behind = _branch_lookahead(graph, current, previous)
    ahead = _branch_lookahead(graph, current, candidate)
    angles = [
        angle_between(current, first, second)
        for first in behind
        for second in ahead
    ]
    return sum(angles) / len(angles) if angles else 0.0


def _angle_reachable_nodes(graph, start, minimum_angle=120.0):
    """Return nodes reachable from start using only allowed path turns."""
    reachable = {start}
    states = [(None, start)]
    visited = set(states)
    while states:
        previous, current = states.pop()
        for neighbor in graph.neighbors(current):
            if previous is not None and (
                angle_between(current, previous, neighbor) <= minimum_angle
            ):
                continue
            state = (current, neighbor)
            if state in visited:
                continue
            visited.add(state)
            reachable.add(neighbor)
            states.append(state)
    return reachable
    """Relabel outside-station reversible tails with their sole boundary direction."""
    changed = True
    while changed:
        changed = False
        for node, attributes in graph.nodes(data=True):
            if attributes.get("direction") != "reversible":
                continue
            if attributes.get("station") not in (None, "None"):
                continue
            boundary = {
                graph.nodes[neighbor].get("direction")
                for neighbor in graph.neighbors(node)
                if graph.nodes[neighbor].get("direction") in {"Up", "Down"}
            }
            if len(boundary) == 1:
                attributes["direction"] = next(iter(boundary))
                changed = True
    return graph
    geometry = graph.graph.get("direction_seed_geometry", {}).get(origin, {})
    center = geometry.get("platform_center")
    neighbors = list(graph.neighbors(seed))
    if center is None or len(neighbors) <= 1:
        return neighbors[0] if neighbors else None
    return max(
        neighbors,
        key=lambda neighbor: math.hypot(
            neighbor[0] - center[0], neighbor[1] - center[1]
        ),
    )

def label_terminal_unlabelled_branches(graph):
    """Extend labels only along branches that do not reconnect to a label."""
    labelled = {
        node for node, attributes in graph.nodes(data=True)
        if attributes.get("direction")
    }
    visited = set()
    for source in list(labelled):
        direction = graph.nodes[source]["direction"]
        for first in graph.neighbors(source):
            if first in labelled or (source, first) in visited:
                continue
            states = [(source, first, [])]
            while states:
                previous, current, path = states.pop()
                state = (previous, current)
                if state in visited or current in labelled:
                    continue
                visited.add(state)
                path = [*path, current]
                candidates = [
                    neighbor for neighbor in graph.neighbors(current)
                    if neighbor != previous and neighbor not in labelled
                ]
                if not candidates:
                    for node in path:
                        graph.nodes[node]["direction"] = direction
                        labelled.add(node)
                    continue
                states.extend(
                    (current, neighbor, path) for neighbor in candidates
                )

def label_unlabelled_terminal_subgraphs(graph):
    """Label degree-below-three unknown components ending at a buffer stop."""
    unknown = {
        node for node, attributes in graph.nodes(data=True)
        if not attributes.get("direction")
    }
    unknown_graph = graph.subgraph(unknown)
    resolved = 0
    for component in nx.connected_components(unknown_graph):
        if any(graph.degree(node) >= 3 for node in component):
            continue
        boundary_directions = {
            graph.nodes[neighbor].get("direction")
            for node in component
            for neighbor in graph.neighbors(node)
            if neighbor not in component
            and graph.nodes[neighbor].get("direction")
        }
        has_degree_one_end = any(graph.degree(node) == 1 for node in component)
        if len(boundary_directions) != 1 or not has_degree_one_end:
            continue
        direction = next(iter(boundary_directions))
        for node in component:
            graph.nodes[node]["direction"] = direction
        resolved += 1
    graph.graph["terminal_unknown_subgraphs_resolved"] = resolved
    return graph

def _distance_meters(first, second):
    """Return approximate WGS84 distance in metres between two coordinates."""
    latitude = math.radians((first[1] + second[1]) / 2)
    longitude_scale = 111320 * math.cos(latitude)
    latitude_scale = 110540
    return math.hypot(
        (first[0] - second[0]) * longitude_scale,
        (first[1] - second[1]) * latitude_scale,
    )


def annotate_has_platform(
    graph,
    platform_source=DEFAULT_PLATFORM_SOURCE,
    platform_distance_meters=DEFAULT_PLATFORM_DISTANCE_METERS,
):
    """Annotate nodes and edges that fall within the platform distance."""
    if platform_distance_meters < 0:
        raise ValueError("platform_distance_meters must not be negative")

    source_path = Path(platform_source)
    if not source_path.is_file():
        fallback_path = Path(DEFAULT_PLATFORM_FALLBACK_SOURCE)
        if (
            platform_source == DEFAULT_PLATFORM_SOURCE
            and fallback_path.is_file()
        ):
            source_path = fallback_path
        else:
            raise FileNotFoundError(
                f"Platform source file not found: {source_path}"
            )

    with source_path.open(encoding="utf-8") as handle:
        collection = json.load(handle)
    platform_geometries = []
    for feature in collection.get("features", []):
        properties = feature.get("properties") or {}
        if properties.get("railway") != "platform":
            continue
        geometry = feature.get("geometry")
        if geometry is None:
            continue
        platform_geometry = shape(geometry)
        if platform_geometry.is_empty:
            continue
        platform_geometries.append(platform_geometry)
    if not platform_geometries:
        raise ValueError(
            f"No railway=platform geometries found in {source_path}"
        )

    if graph.number_of_nodes() > 0:
        mean_latitude = sum(node[1] for node in graph.nodes) / graph.number_of_nodes()
    else:
        mean_latitude = 52.0
    latitude_radians = math.radians(mean_latitude)
    longitude_scale = 111320 * math.cos(latitude_radians)
    latitude_scale = 110540

    def to_metric(x, y, z=None):
        x_metric = x * longitude_scale
        y_metric = y * latitude_scale
        if z is None:
            return x_metric, y_metric
        return x_metric, y_metric, z

    platform_geometries_metric = [
        transform(to_metric, platform_geometry)
        for platform_geometry in platform_geometries
    ]
    platform_tree = STRtree(platform_geometries_metric)

    def has_nearby_platform(geometry_metric):
        candidate_matches = platform_tree.query(
            geometry_metric.buffer(platform_distance_meters)
        )
        for candidate in candidate_matches:
            if hasattr(candidate, "geom_type"):
                candidate_geometry = candidate
            else:
                candidate_geometry = platform_geometries_metric[int(candidate)]
            if (
                geometry_metric.distance(candidate_geometry)
                <= platform_distance_meters
            ):
                return True
        return False

    nodes_with_platform = 0
    for node, attributes in graph.nodes(data=True):
        node_geometry_metric = Point(
            node[0] * longitude_scale,
            node[1] * latitude_scale,
        )
        has_platform = has_nearby_platform(node_geometry_metric)
        attributes["has_platform"] = has_platform
        nodes_with_platform += int(has_platform)

    edges_with_platform = 0
    for first, second, key, attributes in graph.edges(keys=True, data=True):
        coordinates = attributes.get("geometry") or [first, second]
        if len(coordinates) < 2:
            coordinates = [first, second]
        edge_geometry_metric = transform(to_metric, LineString(coordinates))
        has_platform = has_nearby_platform(edge_geometry_metric)
        attributes["has_platform"] = has_platform
        edges_with_platform += int(has_platform)

    graph.graph.update(
        has_platform_source=str(source_path),
        has_platform_distance_meters=platform_distance_meters,
        has_platform_nodes=nodes_with_platform,
        has_platform_edges=edges_with_platform,
    )
    return graph


def assign_network_ids(graph):
    """Assign deterministic 1-based IDs to all nodes and edges in a graph."""
    for node_id, node in enumerate(graph.nodes, start=1):
        graph.nodes[node]["node_id"] = node_id

    for edge_id, (first, second, key) in enumerate(
        graph.edges(keys=True), start=1
    ):
        graph[first][second][key]["edge_id"] = edge_id

    graph.graph["node_count"] = graph.number_of_nodes()
    graph.graph["edge_count"] = graph.number_of_edges()
    return graph


def write_graph(graph, output_graph):
    remove_attributes(graph)
    with Path(output_graph).open("wb") as handle:
        pickle.dump(graph, handle, protocol=pickle.HIGHEST_PROTOCOL)


def _write_json(collection, output_path):
    """Write JSON through a temporary file before replacing the destination."""
    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        file_descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{destination.stem}.",
            suffix=destination.suffix,
            dir=destination.parent,
        )
        temporary_path = Path(temporary_name)
        with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
            json.dump(collection, handle)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def write_geojson(graph, output_geojson):
    remove_attributes(graph)
    features = []
    for node, attributes in graph.nodes(data=True):
        properties = {
            key: value for key, value in attributes.items()
            if key != "geometry"
        }
        properties["feature_type"] = "node"
        features.append({
            "type": "Feature",
            "properties": properties,
            "geometry": {"type": "Point", "coordinates": list(node)},
        })

    for start, end, attributes in graph.edges(data=True):
        properties = {
            key: value for key, value in attributes.items()
            if key != "geometry"
        }
        properties["feature_type"] = "track"
        features.append({
            "type": "Feature",
            "properties": properties,
            "geometry": {
                "type": "LineString",
                "coordinates": attributes["geometry"],
            },
        })

    collection = {
        "type": "FeatureCollection",
        "name": "case_study_network",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }
    _write_json(collection, output_geojson)


def write_parallel_track_centrelines(
    graph,
    output_geojson=DEFAULT_CENTERLINE_GEOJSON,
    maximum_distance_meters=150.0,
    advance_distance_meters=10.0,
    maximum_angle_degrees=2.0,
):
    """Write centreline samples from normal intersections between stations."""
    if not 0.0 <= maximum_angle_degrees <= 90.0:
        raise ValueError("maximum_angle_degrees must be between 0 and 90")

    area_edges = defaultdict(list)
    for first, second, attributes in graph.edges(data=True):
        area = attributes.get("nearest_stations")
        station = attributes.get("station")
        if (
            station not in (None, "None")
            or not isinstance(area, (list, tuple))
            or len(area) < 2
            or not attributes.get("geometry")
        ):
            continue
        area_edges[tuple(sorted(area))].append(
            (first, second, attributes["geometry"], attributes)
        )

    def ordered_area_path(edges):
        adjacency = defaultdict(list)
        for edge_number, (first, second, coordinates, attributes) in enumerate(edges):
            adjacency[first].append(
                (second, coordinates, edge_number)
            )
            adjacency[second].append(
                (first, list(reversed(coordinates)), edge_number)
            )
        start = next(
            (node for node in adjacency
             if _station_name(graph.nodes[node]) is not None),
            None,
        )
        if start is None:
            return None
        coordinates = []
        previous = None
        current = start
        used_edges = set()
        while True:
            neighbours = [
                item for item in adjacency[current] if item[0] != previous
            ]
            if not neighbours:
                break
            unused_neighbours = [
                item for item in neighbours if item[2] not in used_edges
            ]
            if not unused_neighbours:
                break
            if previous is not None and graph.degree(current) > 2:
                next_node, edge_coordinates, edge_number = max(
                    unused_neighbours,
                    key=lambda item: (
                        angle_between(current, previous, item[0]),
                        repr(item[0]),
                        repr(item[2]),
                    ),
                )
            else:
                next_node, edge_coordinates, edge_number = unused_neighbours[0]
            if next_node is None:
                break
            used_edges.add(edge_number)
            if not coordinates:
                coordinates.extend(edge_coordinates)
            else:
                coordinates.extend(edge_coordinates[1:])
            previous, current = current, next_node
            if (
                current != start
                and _station_name(graph.nodes[current]) is not None
            ):
                break
        return coordinates, used_edges

    def sample_polyline(coordinates, distance_meters):
        lengths = [
            _distance_meters(first, second)
            for first, second in zip(coordinates, coordinates[1:])
        ]
        remaining = distance_meters
        for index, length in enumerate(lengths):
            if remaining <= length:
                fraction = remaining / length if length else 0.0
                first, second = coordinates[index], coordinates[index + 1]
                point = [
                    first[coordinate]
                    + fraction * (second[coordinate] - first[coordinate])
                    for coordinate in (0, 1)
                ]
                tangent, normal = track_tangent_and_right_normal(
                    [first, second]
                )
                return point, tangent, normal
            remaining -= length
        return None, None, None

    def normal_intersections(
        point, normal, reference_tangent, other_lines
    ):
        latitude_scale = 110540
        longitude_scale = 111320 * math.cos(math.radians(point[1]))
        delta = [
            maximum_distance_meters * normal[0] / longitude_scale,
            maximum_distance_meters * normal[1] / latitude_scale,
        ]
        normal_line = LineString([
            (point[0] - delta[0], point[1] - delta[1]),
            (point[0] + delta[0], point[1] + delta[1]),
        ])
        intersections = []
        for line in other_lines:
            intersection = normal_line.intersection(line)
            if intersection.is_empty:
                continue
            if intersection.geom_type == "Point":
                if (
                    _undirected_tangent_angle_degrees(
                        reference_tangent,
                        _line_local_tangent(line, intersection),
                    ) <= maximum_angle_degrees
                ):
                    intersections.append([intersection.x, intersection.y])
            elif intersection.geom_type == "MultiPoint":
                for intersection_point in intersection.geoms:
                    if (
                        _undirected_tangent_angle_degrees(
                            reference_tangent,
                            _line_local_tangent(line, intersection_point),
                        ) <= maximum_angle_degrees
                    ):
                        intersections.append(
                            [intersection_point.x, intersection_point.y]
                        )
        return intersections

    centreline_graph = nx.MultiGraph()
    features = []
    group_number = 0
    for area, edges in area_edges.items():
        ordered = ordered_area_path(edges)
        if not ordered:
            continue
        reference, reference_edge_numbers = ordered
        other_lines = [
            LineString(edge[2])
            for edge_number, edge in enumerate(edges)
            if edge_number not in reference_edge_numbers
        ]
        total_length = sum(
            _distance_meters(first, second)
            for first, second in zip(reference, reference[1:])
        )
        points = []
        distance = 0.0
        while distance <= total_length:
            reference_point, reference_tangent, normal = sample_polyline(
                reference, distance
            )
            intersections = normal_intersections(
                reference_point,
                normal,
                reference_tangent,
                other_lines,
            )
            intersections.append(reference_point)

            def coordinate_median(coordinate):
                values = sorted(
                    point[coordinate] for point in intersections
                )
                middle = len(values) // 2
                if len(values) % 2:
                    return values[middle]
                return (values[middle - 1] + values[middle]) / 2

            median = [
                coordinate_median(coordinate)
                for coordinate in (0, 1)
            ]
            node = tuple(
                round(value, COORDINATE_DIGITS) for value in median
            )
            centreline_graph.add_node(
                node, between_stations=list(area)
            )
            if points:
                centreline_graph.add_edge(
                    points[-1],
                    node,
                    group_id=group_number,
                    geometry=[list(points[-1]), list(node)],
                )
            points.append(node)
            distance += advance_distance_meters
        if len(points) > 1:
            group_number += 1
            features.append({
                "type": "Feature",
                "properties": {
                    "feature_type": "parallel_track_centreline",
                    "group_id": group_number,
                    "track_count": len(edges),
                    "between_stations": list(area),
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [list(point) for point in points],
                },
            })

    graph.graph["centreline_graph"] = centreline_graph
    print(
        f"[Centrelines] Complete: {centreline_graph.number_of_nodes():,} "
        f"nodes, {centreline_graph.number_of_edges():,} edges."
    )

    collection = _segment_collection("parallel_track_centrelines", features)
    _write_json(collection, output_geojson)
    return features


def _map_property_value(value):
    """Convert graph attributes into readable, JSON-safe tooltip values."""
    if isinstance(value, (dict, list, tuple, set)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value)


def _speed_limit_kmh(attributes):
    """Return maxspeed, falling back to the other speed-limit attributes."""
    values = []
    speed_keys = (
        ("maxspeed",)
        if attributes.get("maxspeed") is not None
        else ("speed_limit", "speed_limits")
    )
    for key in speed_keys:
        value = attributes.get(key)
        if value is None:
            continue
        candidates = value if isinstance(value, (list, tuple, set)) else [value]
        for candidate in candidates:
            text = str(candidate).lower().replace(",", ".")
            match = re.search(r"-?\d+(?:\.\d+)?", text)
            if not match:
                continue
            speed = float(match.group())
            if "mph" in text:
                speed *= 1.609344
            elif "m/s" in text or "ms-1" in text:
                speed *= 3.6
            if math.isfinite(speed) and speed >= 0:
                values.append(speed)
    return min(values) if values else None


def _speed_color(speed_kmh, minimum, maximum):
    """Map a speed limit to a blue-green-orange-red gradient."""
    if speed_kmh is None:
        return "#000000"
    if maximum <= minimum:
        position = 0.5
    else:
        position = (speed_kmh - minimum) / (maximum - minimum)
    position = max(0.0, min(1.0, position))
    stops = (
        (0.0, (0, 102, 255)),
        (1 / 3, (0, 180, 80)),
        (2 / 3, (255, 165, 0)),
        (1.0, (220, 0, 0)),
    )
    for (first_position, first_color), (second_position, second_color) in zip(
        stops, stops[1:]
    ):
        if position <= second_position:
            fraction = (
                position - first_position
            ) / (second_position - first_position)
            color = [
                round(
                    first_component
                    + fraction * (second_component - first_component)
                )
                for first_component, second_component in zip(
                    first_color, second_color
                )
            ]
            return "#" + "".join(f"{component:02x}" for component in color)
    return "#dc0000"


def _segment_colors(graph):
    """Assign one of four colors to each pairwise-adjacent segment."""
    segment_adjacency = nx.Graph()
    segments_at_node = defaultdict(set)
    for node in graph:
        for _, _, _, attributes in graph.edges(node, keys=True, data=True):
            segment_id = attributes.get("segment_id")
            if segment_id is not None:
                segments_at_node[node].add(segment_id)

    for segment_ids in segments_at_node.values():
        segment_adjacency.add_nodes_from(segment_ids)
        for first, second in itertools.combinations(sorted(segment_ids), 2):
            segment_adjacency.add_edge(first, second)

    colors = nx.coloring.greedy_color(
        segment_adjacency,
        strategy="saturation_largest_first",
    )
    if not colors or max(colors.values()) < 4:
        return colors

    def four_color_component(component):
        """Find a four-coloring using DSATUR ordering and backtracking."""
        component_graph = segment_adjacency.subgraph(component)
        assigned = {}

        def assign_next():
            if len(assigned) == len(component_graph):
                return True

            unassigned = [
                node for node in component_graph
                if node not in assigned
            ]
            node = max(
                unassigned,
                key=lambda candidate: (
                    len({
                        assigned[neighbor]
                        for neighbor in component_graph[candidate]
                        if neighbor in assigned
                    }),
                    component_graph.degree(candidate),
                    -candidate,
                ),
            )
            used_colors = {
                assigned[neighbor]
                for neighbor in component_graph[node]
                if neighbor in assigned
            }
            for color in range(4):
                if color in used_colors:
                    continue
                assigned[node] = color
                if assign_next():
                    return True
                del assigned[node]
            return False

        if not assign_next():
            return None
        return assigned

    colors = {}
    for component in nx.connected_components(segment_adjacency):
        component_colors = four_color_component(component)
        if component_colors is None:
            raise ValueError(
                "The segment adjacency graph requires more than four colors; "
                "the four-color map assumption does not hold for this network."
            )
        colors.update(component_colors)
    return colors


def write_network_map(
    graph,
    output_html=DEFAULT_OUTPUT_MAP,
    centerline_features=None,
    color_by_speed=False,
    show_nodes=True,
):
    """Write a map showing tracks and optional nodes with speed-limit colors."""
    if not graph:
        raise ValueError("Cannot create a map for an empty graph")

    remove_attributes(graph)
    incident_edge_ids = {node: [] for node in graph.nodes}
    edge_records = []
    for start, end, key, attributes in graph.edges(
        keys=True, data=True
    ):
        edge_id = attributes.get("edge_id")
        incident_edge_ids[start].append(edge_id)
        incident_edge_ids[end].append(edge_id)
        edge_records.append((start, end, key, attributes))

    node_features = []
    if show_nodes:
        for node, attributes in graph.nodes(data=True):
            speed_limit = _speed_limit_kmh(attributes)
            properties = {
                "node_id": _map_property_value(attributes.get("node_id")),
                "segment_ids": _map_property_value(
                    attributes.get("segment_ids", [])
                ),
                "speed_limit_kmh": _map_property_value(speed_limit),
            }
            properties["connected_edge_ids"] = _map_property_value(
                sorted(
                    edge_id for edge_id in incident_edge_ids[node]
                    if edge_id is not None
                )
            )
            properties["feature_type"] = "node"
            node_features.append({
                "type": "Feature",
                "properties": properties,
                "geometry": {"type": "Point", "coordinates": list(node)},
                "speed_limit_kmh": speed_limit,
            })

    segment_colors = _segment_colors(graph)
    color_palette = ["red", "blue", "green", "yellow"]
    speed_values = [
        speed for _, _, _, attributes in edge_records
        for speed in [_speed_limit_kmh(attributes)]
        if speed is not None
    ]
    speed_values.extend(
        feature["speed_limit_kmh"]
        for feature in node_features
        if feature["speed_limit_kmh"] is not None
    )
    speed_minimum = min(speed_values) if speed_values else 0.0
    speed_maximum = max(speed_values) if speed_values else 0.0
    edge_features = []
    for start, end, _key, attributes in edge_records:
        speed_limit = _speed_limit_kmh(attributes)
        properties = {
            "edge_id": _map_property_value(attributes.get("edge_id")),
            "segment_id": _map_property_value(attributes.get("segment_id")),
        }
        properties.update({
            "feature_type": "track",
            "start_node_id": _map_property_value(
                graph.nodes[start].get("node_id")
            ),
            "end_node_id": _map_property_value(
                graph.nodes[end].get("node_id")
            ),
            "speed_limit_kmh": _map_property_value(speed_limit),
        })
        for speed_key in ("maxspeed", "speed_limit", "speed_limits"):
            if speed_key in attributes:
                properties[speed_key] = _map_property_value(
                    attributes[speed_key]
                )
        segment_id = attributes.get("segment_id")
        if segment_id is not None:
            properties["segment_color"] = color_palette[
                segment_colors[segment_id]
            ]
        edge_features.append({
            "type": "Feature",
            "properties": properties,
            "geometry": {
                "type": "LineString",
                "coordinates": attributes["geometry"],
            },
            "speed_limit_kmh": speed_limit,
        })

    def add_tooltip_layer(
        parent, features, name, color, weight, display_fields,
        property_color=False
    ):
        if not features:
            return
        fields = [
            field for field in display_fields
            if any(field in feature["properties"] for feature in features)
        ]
        for feature in features:
            for field in fields:
                feature["properties"].setdefault(field, "")
            if color_by_speed:
                feature["properties"]["speed_color"] = _speed_color(
                    feature.get("speed_limit_kmh"),
                    speed_minimum,
                    speed_maximum,
                )
        feature_groups = {name: features}
        if color_by_speed and name == "Nodes":
            feature_groups = {}
            for feature in features:
                feature_color = feature["properties"]["speed_color"]
                feature_groups.setdefault(feature_color, []).append(feature)
        for group_name, group_features in feature_groups.items():
            group_color = (
                group_name if name == "Nodes" and color_by_speed else color
            )
            folium.GeoJson(
                {"type": "FeatureCollection", "features": group_features},
                name=name,
                style_function=(
                    lambda feature, layer_color=group_color,
                    layer_weight=weight: {
                        "color": (
                            feature["properties"].get(
                                "speed_color", layer_color
                            )
                            if color_by_speed
                            else (
                                feature["properties"].get(
                                    "segment_color", layer_color
                                )
                                if property_color else layer_color
                            )
                        ),
                        "weight": layer_weight,
                        "fillColor": (
                            feature["properties"].get(
                                "speed_color", layer_color
                            )
                            if color_by_speed
                            else (
                                feature["properties"].get(
                                    "segment_color", layer_color
                                )
                                if property_color else layer_color
                            )
                        ),
                        "fillOpacity": 0.9,
                    }
                ),
                marker=folium.CircleMarker(
                    radius=4,
                    color=group_color,
                    fill=True,
                    fill_color=group_color,
                    fill_opacity=0.9,
                ),
                tooltip=folium.GeoJsonTooltip(
                    fields=fields,
                    aliases=[
                        field.replace("_", " ").title() for field in fields
                    ],
                    localize=True,
                    sticky=False,
                    labels=True,
                ),
            ).add_to(parent)

    coordinates = [list(node) for node in graph.nodes]
    center = [
        sum(coordinate[1] for coordinate in coordinates) / len(coordinates),
        sum(coordinate[0] for coordinate in coordinates) / len(coordinates),
    ]
    network_map = folium.Map(
        location=center,
        zoom_start=8,
        tiles="OpenStreetMap",
        control_scale=True,
    )
    add_tooltip_layer(
        network_map,
        edge_features,
        "Tracks",
        "#1565c0",
        3,
        [
            "edge_id", "segment_id", "start_node_id", "end_node_id",
            "maxspeed", "speed_limit", "speed_limits",
            "speed_limit_kmh",
        ],
        property_color=not color_by_speed,
    )
    if show_nodes:
        add_tooltip_layer(
            network_map,
            node_features,
            "Nodes",
            "#d32f2f",
            1,
            [
                "node_id", "segment_ids", "connected_edge_ids",
                "speed_limit_kmh",
            ],
            property_color=False,
        )
    if centerline_features:
        folium.GeoJson(
            {"type": "FeatureCollection", "features": centerline_features},
            name="Parallel track centrelines",
            style_function=lambda _feature: {
                "color": "purple",
                "weight": 5,
                "opacity": 0.9,
            },
        ).add_to(network_map)
    bounds = [
        [min(coordinate[1] for coordinate in coordinates),
         min(coordinate[0] for coordinate in coordinates)],
        [max(coordinate[1] for coordinate in coordinates),
         max(coordinate[0] for coordinate in coordinates)],
    ]
    network_map.fit_bounds(bounds)
    folium.LayerControl().add_to(network_map)
    network_map.save(output_html)


def _station_name(attributes):
    station = attributes.get("station")
    return None if station in (None, "None") else station


def _edge_id(first, second, key):
    """Return an orientation-independent identity for an undirected edge."""
    return frozenset((first, second)), key


def _walk_segment(graph, start, first_edge, boundary_nodes, assigned_edges):
    """Walk a maximal edge path from one boundary until the next boundary."""
    nodes = [start]
    edges = []
    previous = None
    current = start
    edge = first_edge

    while edge is not None:
        first, second, key = edge
        edge_identity = _edge_id(first, second, key)
        if edge_identity in assigned_edges:
            break
        assigned_edges.add(edge_identity)
        edges.append(edge)
        next_node = second if first == current else first
        nodes.append(next_node)
        if next_node in boundary_nodes or next_node == start:
            break

        previous, current = current, next_node
        candidates = [
            (current, neighbor, next_key)
            for neighbor, edge_data in graph[current].items()
            for next_key in edge_data
            if _edge_id(current, neighbor, next_key) != edge_identity
        ]
        edge = candidates[0] if len(candidates) == 1 else None

    return nodes, edges


def write_origin_geojsons(graph, output_folder="origin_direction_geojson"):
    """Write one full-network GeoJSON with directions for each origin."""
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    for stale_file in output_path.glob("*.geojson"):
        stale_file.unlink()
    origins = graph.graph.get("direction_origin_seeds", {})

    for origin in origins:
        features = []
        for node, attributes in graph.nodes(data=True):
            properties = {
                key: value for key, value in attributes.items()
                if key not in {"direction", "temporary_directions"}
            }
            labels = attributes.get("temporary_directions", {}).get(origin, [])
            if node in origins[origin] and origins[origin][node] == "reversible":
                direction = "reversible"
            elif len(labels) == 1:
                direction = labels[0]
            elif len(labels) > 1:
                direction = "reversible"
            else:
                direction = None
            if direction:
                properties["direction"] = direction
            properties["origin_station"] = origin
            properties["feature_type"] = "node"
            features.append({
                "type": "Feature",
                "properties": properties,
                "geometry": {"type": "Point", "coordinates": list(node)},
            })

        for start, end, attributes in graph.edges(data=True):
            properties = {
                key: value for key, value in attributes.items()
                if key != "geometry"
            }
            direction = attributes.get("origin_directions", {}).get(origin)
            if direction:
                properties["direction"] = direction
            properties.pop("origin_directions", None)
            properties["origin_station"] = origin
            properties["feature_type"] = "track"
            features.append({
                "type": "Feature",
                "properties": properties,
                "geometry": {
                    "type": "LineString",
                    "coordinates": attributes["geometry"],
                },
            })

        collection = {
            "type": "FeatureCollection",
            "name": f"case_study_network_{origin}",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
            },
            "features": features,
        }
        filename = origin.replace(" ", "_") + ".geojson"
        (output_path / filename).write_text(
            json.dumps(collection), encoding="utf-8"
        )


def _geometry_length_meters(coordinates):
    return sum(
        _distance_meters(first, second)
        for first, second in zip(coordinates, coordinates[1:])
    )


def _edge_weight(first, second, attributes):
    return min(
        _geometry_length_meters(data.get("geometry", [first, second]))
        for data in attributes.values()
    )


def _station_distances(graph, station_nodes):
    """Calculate each station's network distances once for all segments."""
    edge_length_cache = {}

    def cached_edge_weight(first, second, attributes):
        edge_pair = frozenset((first, second))
        if edge_pair not in edge_length_cache:
            edge_length_cache[edge_pair] = _edge_weight(
                first, second, attributes
            )
        return edge_length_cache[edge_pair]

    return {
        station: nx.multi_source_dijkstra_path_length(
            graph, roots, weight=cached_edge_weight
        )
        for station, roots in station_nodes.items()
    }


def _segment_nearest_stations(nodes, station_distances):
    distances = {
        station: min(
            (lengths[node] for node in nodes if node in lengths),
            default=math.inf,
        )
        for station, lengths in station_distances.items()
    }
    distances = {
        station: distance
        for station, distance in distances.items()
        if math.isfinite(distance)
    }
    if not distances:
        raise ValueError("No configured station can reach this segment")
    nearest = sorted(
        distances, key=lambda station: (distances[station], station)
    )[:2]
    return sorted(nearest)


def _safe_filename_part(value):
    """Return a station name suitable for use in an output filename."""
    part = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value)).strip("_")
    return part or "unknown"


def _segment_collection(name, features):
    return {
        "type": "FeatureCollection",
        "name": name,
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }


def split_into_segments(graph, segment_geojson=DEFAULT_SEGMENT_GEOJSON,
                        segment_graph_output=DEFAULT_SEGMENT_GRAPH,
                        segment_output_folder=DEFAULT_SEGMENT_OUTPUT_FOLDER):
    """Export maximal edge-disjoint segments and their boundary graph."""
    remove_attributes(graph)
    print(
        f"[Segments] Splitting {graph.number_of_nodes():,} nodes and "
        f"{graph.number_of_edges():,} tracks..."
    )
    boundary_nodes = {
        node for node in graph if graph.degree(node) != 2
    }
    for node in graph:
        station = _station_name(graph.nodes[node])
        if station is not None and any(
            _station_name(graph.nodes[neighbor]) != station
            for neighbor in graph.neighbors(node)
        ):
            boundary_nodes.add(node)
    print(f"[Segments] Identified {len(boundary_nodes):,} boundary nodes.")

    assigned_edges = set()
    segments = []
    for start in sorted(boundary_nodes, key=repr):
        for neighbor, edge_data in graph[start].items():
            for key in sorted(edge_data, key=repr):
                edge = (start, neighbor, key)
                if _edge_id(*edge) in assigned_edges:
                    continue
                nodes, edges = _walk_segment(
                    graph, start, edge, boundary_nodes, assigned_edges
                )
                if edges:
                    segments.append((nodes, edges))

    # A degree-two closed component has no natural border; use one stable node
    # as both ends so it is still represented in the segment graph.
    for start, end, key in graph.edges(keys=True):
        if _edge_id(start, end, key) in assigned_edges:
            continue
        nodes, edges = _walk_segment(
            graph, start, (start, end, key), {start}, assigned_edges
        )
        if edges:
            segments.append((nodes, edges))

    print(f"[Segments] Discovered {len(segments):,} segments.")
    segment_features = []
    segment_features_by_id = {}
    segment_graph = nx.MultiGraph()
    station_nodes = defaultdict(set)
    for node, attributes in graph.nodes(data=True):
        station = _station_name(attributes)
        if station is not None:
            station_nodes[station].add(node)

    for boundary in boundary_nodes:
        segment_graph.add_node(
            boundary,
            **{
                name: value for name, value in graph.nodes[boundary].items()
                if name != "geometry"
            }
        )

    non_station_segments = [
        (nodes, edges) for nodes, edges in segments
        if not (
            _station_name(graph.nodes[nodes[0]]) is not None
            and all(
                _station_name(graph.nodes[node]) is not None
                and _station_name(graph.nodes[node])
                == _station_name(graph.nodes[nodes[0]])
                for node in dict.fromkeys(nodes)
            )
        )
    ]
    station_distances = {}
    if non_station_segments:
        print("[Segments] Calculating station distances once for all segments...")
        station_distances = _station_distances(graph, station_nodes)

    next_segment_id = 1
    non_station_segment_count = 0
    progress_interval = max(1, len(segments) // 10)
    for segment_number, (nodes, edges) in enumerate(segments, start=1):
        unique_nodes = list(dict.fromkeys(nodes))
        node_stations = [
            _station_name(graph.nodes[node]) for node in unique_nodes
        ]
        station = node_stations[0] if node_stations else None
        is_station_segment = (
            station is not None
            and all(node_station == station for node_station in node_stations)
        )
        nearest = (
            [station]
            if is_station_segment
            else _segment_nearest_stations(unique_nodes, station_distances)
        )
        segment_id = next_segment_id
        next_segment_id += 1
        if not is_station_segment:
            non_station_segment_count += 1

        for node in unique_nodes:
            segment_ids = graph.nodes[node].setdefault("segment_ids", set())
            segment_ids.add(segment_id)
        for first, second, key in edges:
            graph[first][second][key].update(
                segment_id=segment_id,
                nearest_stations=nearest,
            )

        common_properties: dict[str, object] = {
            "nearest_stations": nearest,
            "segment_id": segment_id,
        }
        segment_features_by_id[segment_id] = []

        for node in unique_nodes:
            properties = {
                name: value for name, value in graph.nodes[node].items()
                if name not in {"geometry", "segment_ids"}
            }
            properties.update(common_properties)
            properties["feature_type"] = "node"
            feature = {
                "type": "Feature",
                "properties": properties,
                "geometry": {"type": "Point", "coordinates": list(node)},
            }
            segment_features.append(feature)
            segment_features_by_id[segment_id].append(feature)

        for first, second, key in edges:
            attributes = graph[first][second][key]
            properties = {
                name: value for name, value in attributes.items()
                if name != "geometry"
            }
            properties.update(common_properties)
            properties["feature_type"] = "track"
            feature = {
                "type": "Feature",
                "properties": properties,
                "geometry": {
                    "type": "LineString",
                    "coordinates": attributes["geometry"],
                },
            }
            segment_features.append(feature)
            segment_features_by_id[segment_id].append(feature)

        start_boundary = nodes[0]
        end_boundary = nodes[-1]
        for boundary in (start_boundary, end_boundary):
            if boundary not in segment_graph:
                segment_graph.add_node(
                    boundary, **{
                        name: value for name, value in graph.nodes[boundary].items()
                        if name != "geometry"
                    }
                )
        edge_properties = {
            "nearest_stations": nearest,
            "travel_time": None,
        }
        edge_properties["segment_id"] = segment_id
        segment_graph.add_edge(
            start_boundary, end_boundary, **edge_properties
        )
        if (
            segment_number == 1
            or segment_number % progress_interval == 0
            or segment_number == len(segments)
        ):
            print(
                f"[Segments] Processed {segment_number:,}/{len(segments):,} "
                f"segments."
            )

    for node in graph:
        graph.nodes[node]["segment_ids"] = sorted(
            graph.nodes[node].get("segment_ids", set())
        )

    collection = _segment_collection("case_study_segments", segment_features)
    Path(segment_geojson).write_text(
        json.dumps(collection), encoding="utf-8"
    )
    output_path = Path(segment_output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    for stale_file in output_path.glob("segment_*.geojson"):
        stale_file.unlink()
    for segment_id, features in segment_features_by_id.items():
        nearest = features[0]["properties"]["nearest_stations"]
        station_label = "_and_".join(
            _safe_filename_part(station) for station in nearest
        )
        filename = f"segment_{segment_id}_{station_label}.geojson"
        segment_collection = _segment_collection(
            f"case_study_segment_{segment_id}", features
        )
        (output_path / filename).write_text(
            json.dumps(segment_collection), encoding="utf-8"
        )
    with Path(segment_graph_output).open("wb") as handle:
        pickle.dump(segment_graph, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"[Segments] Exported {len(segments):,} segments to "
        f"{segment_geojson} and {segment_output_folder}."
    )
    graph.graph["segment_count"] = len(segments)
    graph.graph["non_station_segment_count"] = non_station_segment_count
    return graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-folder", default=DEFAULT_INPUT_FOLDER)
    parser.add_argument(
        "--section-names-file",
        default=DEFAULT_SECTION_NAMES_FILE,
        help="Text file listing the section GeoJSON files to reassemble.",
    )
    parser.add_argument(
        "--between-station-folder",
        default=DEFAULT_BETWEEN_STATION_FOLDER,
        help="Folder containing station-to-station section GeoJSON files.",
    )
    parser.add_argument(
        "--station-nodes-folder",
        default=DEFAULT_STATION_NODES_FOLDER,
        help="Folder containing station-node GeoJSON files.",
    )
    parser.add_argument("--output-graph", default=DEFAULT_OUTPUT_GRAPH)
    parser.add_argument("--output-geojson", default=DEFAULT_OUTPUT_GEOJSON)
    parser.add_argument(
        "--centerline-geojson",
        default=DEFAULT_CENTERLINE_GEOJSON,
        help="GeoJSON containing centerlines of locally parallel tracks.",
    )
    parser.add_argument(
        "--centerline-max-angle-degrees",
        type=float,
        default=DEFAULT_CENTERLINE_MAX_ANGLE_DEGREES,
        help=(
            "Maximum angle between tracks included in a centreline "
            f"(default: {DEFAULT_CENTERLINE_MAX_ANGLE_DEGREES:g})."
        ),
    )
    parser.add_argument(
        "--output-map",
        default=DEFAULT_OUTPUT_MAP,
        help="Interactive HTML map of the network.",
    )
    parser.add_argument(
        "--segment-geojson",
        default=DEFAULT_SEGMENT_GEOJSON,
        help="GeoJSON containing all recursively irreducible segments.",
    )
    parser.add_argument(
        "--segment-graph-output",
        default=DEFAULT_SEGMENT_GRAPH,
        help="Pickle path for the boundary-node segment graph.",
    )
    parser.add_argument(
        "--segment-output-folder",
        default=DEFAULT_SEGMENT_OUTPUT_FOLDER,
        help="Folder for one GeoJSON file per segment.",
    )
    parser.add_argument(
        "--platform-source",
        default=DEFAULT_PLATFORM_SOURCE,
        help=(
            "GeoJSON source containing railway=platform features used for "
            "has_platform annotation."
        ),
    )
    parser.add_argument(
        "--platform-distance-meters",
        type=float,
        default=DEFAULT_PLATFORM_DISTANCE_METERS,
        help=(
            "Distance threshold for has_platform (default: "
            f"{DEFAULT_PLATFORM_DISTANCE_METERS:g} m)."
        ),
    )
    parser.add_argument("--stations", nargs="+", default=list(DEFAULT_ROOT_STATIONS))
    args = parser.parse_args()

    graph = reassemble_network(
        input_folder=args.input_folder,
        section_names_file=args.section_names_file,
        between_station_folder=args.between_station_folder,
        station_nodes_folder=args.station_nodes_folder,
    )
    graph = prune_network_to_stations(graph, args.stations)
    graph = delete_nodes_near_origin_without_target_path(
        graph,
        origin_station="Utrecht Centraal",
        target_station="Amsterdam Centraal",
        maximum_distance_meters=5000.0,
    )
    graph = delete_nodes_near_origin_without_target_path(
        graph,
        origin_station="Amsterdam Centraal",
        target_station="Utrecht Centraal",
        maximum_distance_meters=5000.0,
    )
    graph = delete_disconnected_edges(graph)
    graph = delete_non_station_degree_one_nodes(graph)
    graph = annotate_has_platform(
        graph,
        platform_source=args.platform_source,
        platform_distance_meters=args.platform_distance_meters,
    )
    assign_network_ids(graph)
    # Direction calculation is temporarily suspended. Keep the implementation
    # above available so it can be re-enabled without restoring code.
    graph = split_into_segments(
        graph,
        args.segment_geojson,
        args.segment_graph_output,
        args.segment_output_folder,
    )
    write_graph(graph, args.output_graph)
    write_geojson(graph, args.output_geojson)
    centerline_features = write_parallel_track_centrelines(
        graph,
        args.centerline_geojson,
        maximum_angle_degrees=args.centerline_max_angle_degrees,
    )
    write_network_map(graph, args.output_map, centerline_features)
    print(f"Fragments: {graph.graph['fragment_count']:,}")
    print(f"Nodes: {graph.number_of_nodes():,}")
    print(f"Tracks: {graph.number_of_edges():,}")
    print(f"Segments: {graph.graph['segment_count']:,}")
    print(f"Graph: {args.output_graph}")
    print(f"GeoJSON: {args.output_geojson}")
    print(f"Parallel track centrelines: {args.centerline_geojson}")
    print(f"Map: {args.output_map}")
    print(f"Segments GeoJSON: {args.segment_geojson}")
    print(f"Segment files: {args.segment_output_folder}")
    print(f"Segment graph: {args.segment_graph_output}")
    if graph.graph["skipped_features"]:
        print(f"Skipped: {graph.graph['skipped_features']}")


if __name__ == "__main__":
    main()