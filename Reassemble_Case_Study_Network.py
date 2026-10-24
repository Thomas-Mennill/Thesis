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


DEFAULT_SECTION_NAMES_FILE = "case_study_section_names.txt"
DEFAULT_BETWEEN_STATION_FOLDER = "between_station_sections"
DEFAULT_STATION_NODES_FOLDER = "station_nodes"
DEFAULT_OUTPUT_GRAPH = "case_study_network.gpickle"
DEFAULT_OUTPUT_GEOJSON = "case_study_network.geojson"
DEFAULT_OUTPUT_MAP = "case_study_network_map.html"
DEFAULT_SWITCH_GEOJSON = "NL_Switches.geojson"
DEFAULT_PLATFORM_SOURCE = "third_try_filtered_data.geojson"
DEFAULT_PLATFORM_FALLBACK_SOURCE = "NL_Railway_Data_Signals.geojson"
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


def find_case_study_section_files(section_names_file=DEFAULT_SECTION_NAMES_FILE, between_station_folder=DEFAULT_BETWEEN_STATION_FOLDER, station_nodes_folder=DEFAULT_STATION_NODES_FOLDER,):
    """Find listed fragments in the redefined graph output folders."""
    names_path = Path(section_names_file)
    if not names_path.is_file():
        raise FileNotFoundError(f"Section names file not found: {names_path}")

    requested_names = []
    for line in names_path.read_text(encoding="utf-8").splitlines():
        name = line.lstrip("\ufeff").strip().strip('"').strip("'")
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
            if not path.is_file():
                continue
            candidates.setdefault(path.name, []).append(path)

    selected_files = []
    missing = []
    for name in requested_names:
        matches = candidates.get(name)
        if matches:
            selected_files.append((name, matches))
            continue

        missing.append(name)

    if missing:
        raise FileNotFoundError(
            "The following listed sections were not found in "
            f"{between_station_folder} or {station_nodes_folder}: "
            + ", ".join(missing)
        )

    def track_count(path):
        with path.open(encoding="utf-8") as handle:
            collection = json.load(handle)
        return sum(
            (feature.get("properties") or {}).get("feature_type") == "track"
            for feature in collection.get("features", [])
        )

    resolved_files = []
    for name, matches in selected_files:
        if name == "Duivendrecht.geojson":
            selected = max(matches, key=track_count)
        else:
            selected = matches[0]
        resolved_files.append(selected)
        if len(matches) > 1:
            print(
                f"[Reassemble] Using {selected} for {name}; "
                f"also found {len(matches) - 1} duplicate location(s)."
            )
        elif len(Path(selected).stem.split("__")) > 2:
            print(
                f"[Reassemble] Using composite redefined fragment "
                f"{selected} for {name}."
            )
    print(
        f"[Reassemble] Selected {len(resolved_files):,} section files from "
        f"{names_path}."
    )
    return resolved_files


def reassemble_network(section_names_file=None, between_station_folder=DEFAULT_BETWEEN_STATION_FOLDER, station_nodes_folder=DEFAULT_STATION_NODES_FOLDER,):
    """Build a deduplicated MultiGraph from selected case-study fragments."""
    input_files = (
        find_case_study_section_files(
            section_names_file,
            between_station_folder,
            station_nodes_folder,
        )
        if section_names_file
        else sorted(
            Path(between_station_folder).glob("*.geojson")
        ) + sorted(Path(station_nodes_folder).glob("*.geojson"))
    )
    if not input_files:
        source_description = (
            f"{section_names_file}"
            if section_names_file
            else f"{between_station_folder} and {station_nodes_folder}"
        )
        raise FileNotFoundError(f"No GeoJSON files found for {source_description}")
    print(
        f"[Reassemble] Reading {len(input_files):,} GeoJSON fragments "
        f"from the redefined section folders..."
    )

    graph = nx.MultiGraph()
    node_sources = defaultdict(set)
    edge_sources = defaultdict(set)
    edge_features = {}
    skipped = defaultdict(int)

    processed_sources = 0
    current_source = None
    for source_name, feature in read_fragments(Path("."), input_files):
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
        input_folders=[
            str(between_station_folder),
            str(station_nodes_folder),
        ],
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




def write_network_map(graph, output_html=DEFAULT_OUTPUT_MAP, color_by_speed=False, show_nodes=True,):
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
        edge_features.append({
            "type": "Feature",
            "properties": properties,
            "geometry": {
                "type": "LineString",
                "coordinates": attributes["geometry"],
            },
            "speed_limit_kmh": speed_limit,
        })

    def add_tooltip_layer(parent, features, name, color, weight, display_fields, property_color=False):
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
                            else layer_color
                        ),
                        "weight": layer_weight,
                        "fillColor": (
                            feature["properties"].get(
                                "speed_color", layer_color
                            )
                            if color_by_speed
                            else layer_color
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
            "edge_id", "start_node_id", "end_node_id",
            "maxspeed", "speed_limit", "speed_limits",
            "speed_limit_kmh",
        ],
        property_color=False,
    )
    if show_nodes:
        add_tooltip_layer(
            network_map,
            node_features,
            "Nodes",
            "#d32f2f",
            1,
            [
                "node_id", "connected_edge_ids",
                "speed_limit_kmh",
            ],
            property_color=False,
        )
    bounds = [
        [min(coordinate[1] for coordinate in coordinates),
         min(coordinate[0] for coordinate in coordinates)],
        [max(coordinate[1] for coordinate in coordinates),
         max(coordinate[0] for coordinate in coordinates)],
    ]
    network_map.fit_bounds(bounds)
    folium.LayerControl().add_to(network_map)
    network_map.save(output_html)








def main():
    parser = argparse.ArgumentParser(description=__doc__)
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
        "--output-map",
        default=DEFAULT_OUTPUT_MAP,
        help="Interactive HTML map of the network.",
    )
    parser.add_argument(
        "--platform-source",
        default=DEFAULT_PLATFORM_SOURCE,
        help=(
            "GeoJSON source containing railway=platform features used for "
            "has_platform annotation."
        ),
    )
    args = parser.parse_args()

    graph = reassemble_network(
        section_names_file=args.section_names_file,
        between_station_folder=args.between_station_folder,
        station_nodes_folder=args.station_nodes_folder,
    )
    assign_network_ids(graph)
    write_graph(graph, args.output_graph)
    write_geojson(graph, args.output_geojson)
    write_network_map(graph, args.output_map)
    print(f"Fragments: {graph.graph['fragment_count']:,}")
    print(f"Nodes: {graph.number_of_nodes():,}")
    print(f"Tracks: {graph.number_of_edges():,}")
    print(f"Graph: {args.output_graph}")
    print(f"GeoJSON: {args.output_geojson}")
    print(f"Map: {args.output_map}")
    if graph.graph["skipped_features"]:
        print(f"Skipped: {graph.graph['skipped_features']}")


if __name__ == "__main__":
    main()
