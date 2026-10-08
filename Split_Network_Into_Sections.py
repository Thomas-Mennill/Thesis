import gc
import json
import math
import os
import pickle
import re
import gpxpy
import pandas as pd
import networkx as nx
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
import contextily as ctx
from pyproj import Transformer
from shapely.geometry import LineString, MultiLineString, Point
from shapely.ops import transform, linemerge


"""
# Read Geojson
nl_railways_gdf = gpd.read_file("nl_railways.geojson")

print(nl_railways_gdf.keys())
#removing unnecessary data to save on RAM and processing time
nl_railways_gdf =nl_railways_gdf.drop(columns=["wikidata","wifi:operator","wheelchair","wheelchair:description",
                                                "air_conditioning","wikimedia_commons","wikipedia","addr:city",
                                                "addr:country","addr:housenumber","addr:postcode","addr:street",
                                                "brand", "bridge:wikipedia","bridge:structure","bridge:material",
                                                "bridge:movable","bridge:name","bridge:start_date","building",
                                                "building:material","check_date:wheelchair","check_date","access:conditional",
                                                "alt_name","alt_name:-1999-05-30","attraction","comment","construction",
                                                "covered","construction:railway:etcs","covered","cutting","cutting:name",
                                                "departures_board","description","description:ProRail","description:nl",
                                                "designation","detail","disused:railway","drinking_water",
                                                "drinking_water:refill:self_service","ele","embankment","embankment:name","entrance",
                                                "fixme", "gauge","height","iata","image","internet_access","internet_access:fee",
                                                "internet_access:operator","internet_access:ssid","layer","level",
                                                "loc_name","local_name","material","name:ar","name:ckb","name:de","name:el",
                                                "name:es","name:etymology:wikidata","name:fa","name:fr","name:fy","name:ja",
                                                "name:ko","name:nl","name:pnb","name:ru","name:ur","name:zh",
                                                "name:zh-Hant","network","network:short","network:subway","network:train",
                                                "network:wikidata","network:wikipedia","not:building","not:network:wikidata",
                                                "note:nl","note:start_date","note","note:NL","opening_date","opening_hours",
                                                "operator","operator:short","operator:subway","operator:train","operator:website",
                                                "operator:wikidata","outdoor_seating","old_name","old_name:-1997",
                                                "old_name:-1999-05-30","old_name:-2000-05-29","old_name:-2015-11-19",
                                                "old_name:-2019-12-15","old_name:1863-2015","old_name:1866-1989","old_name:1866-2015",
                                                "old_name:1867-2016","old_name:1868-1999","old_name:1881-1921","old_name:1885-2015",
                                                "old_name:1921-1955","old_name:1969-2010","old_name:1971-2015","old_name:1985-2006",
                                                "proposed:bridge","proposed:electrified","proposed:frequency","proposed:voltage",
                                                "ruins","source","source:ProRail","source:date","source:height","source:height:date",
                                                "source:maxspeed","source:name","spotpass","start_date","subway","supervised",
                                                "toilets","toilets:wheelchair","tunnel","tunnel:name","tunnel:wikipedia",
                                                "usage","voltage","website","workrules","ref","ref:DB_Netz_AG",
                                                "ref:FR:sncf:resarail","ref:FR:uic8","ref:ProRail","ref:ProRailID",
                                                "ref:ProRailSpoortak","ref:bag","ref:crs" 
                                                ])
print(nl_railways_gdf.columns)

with open("nl_railways_columns.txt", "w", encoding="utf-8") as file:
    file.write("\n".join(nl_railways_gdf.columns))

nl_railways_gdf.to_file("nl_railways_trimmed.geojson", driver="GeoJSON")
"""

def sanitize_station_name(value):
    cleaned = re.sub(r"[^A-Za-z0-9]+", "_", str(value).strip())
    return cleaned.strip("_") or "unknown"


def json_safe(value):
    """Convert source attributes into values that can be written to GeoJSON."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if value is pd.NA:
        return None
    item_method = getattr(value, "item", None)
    if callable(item_method):
        return json_safe(item_method())
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "__geo_interface__"):
        return None
    return value


def geojson_attribute(value):
    """Keep scalars readable and encode nested source attributes as JSON."""
    value = json_safe(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return value


def aggregate_attribute_values(records):
    """Collect distinct values from raw records without losing attributes."""
    values = {}
    for record in records:
        for key, value in record.items():
            if key in {"geometry", "source_row"}:
                continue
            value = json_safe(value)
            if value is None:
                continue
            values.setdefault(key, [])
            if value not in values[key]:
                values[key].append(value)
    return {
        key: geojson_attribute(items[0] if len(items) == 1 else items)
        for key, items in values.items()
    }


def flatten_line_geometries(geometries):
    """Convert a mixed list of LineString/MultiLineString objects into a flat list of LineStrings."""
    flat_geometries = []
    for geometry in geometries:
        if geometry is None or getattr(geometry, "is_empty", False):
            continue

        if geometry.geom_type == "LineString":
            flat_geometries.append(geometry)
        elif geometry.geom_type == "MultiLineString":
            flat_geometries.extend(geometry.geoms)
        elif geometry.geom_type == "GeometryCollection":
            flat_geometries.extend(
                [geom for geom in geometry.geoms if geom.geom_type in {"LineString", "MultiLineString"}]
            )

    return flat_geometries


def write_pair_geojson(station_a, station_b, geometry, output_dir, written_paths):
    """Persist a single station pair to disk and track the file path."""
    output_path = os.path.join(
        output_dir,
        f"{sanitize_station_name(station_a)}__{sanitize_station_name(station_b)}.geojson",
    )

    if os.path.exists(output_path):
        print(f"Skipping existing file: {output_path}")
        return output_path

    properties = {
        "station_a": station_a,
        "station_b": station_b,
    }
    pair_gdf = gpd.GeoDataFrame(
        {key: [value] for key, value in properties.items()} | {
            "geometry": [geometry],
        },
        geometry="geometry",
        crs="EPSG:4326",
    )

    pair_gdf.to_file(output_path, driver="GeoJSON")
    written_paths.append(output_path)
    return output_path


def flush_station_pairs_buffer(station_pairs, output_dir, written_paths):
    """Write buffered station-pair geometries to disk and release memory."""
    if not station_pairs:
        return 0

    flushed_count = 0
    for (station_a, station_b), geometries in sorted(station_pairs.items()):
        flat_geometries = flatten_line_geometries(geometries)
        if not flat_geometries:
            continue

        merged_geometry = linemerge(flat_geometries)
        if merged_geometry is None or getattr(merged_geometry, "is_empty", False):
            continue

        write_pair_geojson(station_a, station_b, merged_geometry, output_dir, written_paths)
        flushed_count += 1

    station_pairs.clear()
    gc.collect()
    return flushed_count


def get_station_name(station_row):
    """Return the best available station name from the OSM-like data row."""
    for key in ("name", "official_name", "uic_name", "@id"):
        value = station_row.get(key)
        if value not in (None, ""):
            return str(value)
    return "station"


def haversine_km(point_a, point_b):
    """Return straight-line distance in kilometres between two lon/lat coordinate pairs."""
    lon1, lat1 = map(math.radians, point_a)
    lon2, lat2 = map(math.radians, point_b)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * 6371 * math.asin(math.sqrt(a))


def build_raw_railway_graph(railways):
    """Create the first graph directly from the raw rail geometry: nodes are coordinate points and edges join adjacent points."""
    graph = nx.Graph()

    for _, row in railways.iterrows():
        geometry = row.geometry
        if geometry is None or geometry.is_empty:
            continue

        if geometry.geom_type == "MultiLineString":
            candidate_geoms = geometry.geoms
        elif geometry.geom_type == "LineString":
            candidate_geoms = [geometry]
        else:
            continue

        for geom in candidate_geoms:
            if geom.geom_type != "LineString":
                continue

            coords = list(geom.coords)
            for start, end in zip(coords, coords[1:]):
                start_pt = tuple(round(v, 6) for v in start)
                end_pt = tuple(round(v, 6) for v in end)
                seg = LineString([start, end])

                graph.add_node(start_pt, x=start_pt[0], y=start_pt[1])
                graph.add_node(end_pt, x=end_pt[0], y=end_pt[1])
                for key, value in row.items():
                    if key != "geometry":
                        graph.nodes[start_pt].setdefault(
                            key, json_safe(value)
                        )
                        graph.nodes[end_pt].setdefault(
                            key, json_safe(value)
                        )

                if not graph.has_edge(start_pt, end_pt):
                    edge_attributes = {
                        key: json_safe(value)
                        for key, value in row.items()
                        if key != "geometry"
                    }
                    edge_attributes.update(
                        geometry=seg,
                        length=seg.length,
                        source_row=row.to_dict(),
                    )
                    graph.add_edge(
                        start_pt,
                        end_pt,
                        **edge_attributes,
                    )

    return graph


def filter_railway_data(railways, signal_distance_m=10):
    """Remove signals away from rail and platforms belonging to non-rail transport."""
    if railways.crs is None:
        raise ValueError("Railway data must have a CRS before filtering features")

    metric_data = railways.to_crs("EPSG:28992")
    rail_geometries = metric_data.loc[metric_data["railway"].eq("rail"), "geometry"]
    if rail_geometries.empty:
        raise ValueError("No railway tracks found in the railway data")

    track_union = rail_geometries.union_all()
    is_signal = metric_data["railway"].eq("signal")
    signal_distances = metric_data.loc[is_signal, "geometry"].distance(track_union)
    remove_signal_ids = signal_distances[signal_distances > signal_distance_m].index

    is_platform = metric_data["railway"].eq("platform")
    non_rail_platform = (
        metric_data["tram"].fillna("").astype(str).str.lower().eq("yes")
        | metric_data["subway"].fillna("").astype(str).str.lower().eq("yes")
        | metric_data["light_rail"].fillna("").astype(str).str.lower().eq("yes")
    )
    remove_platform_ids = metric_data.index[is_platform & non_rail_platform]

    filtered = railways.drop(index=remove_signal_ids.union(remove_platform_ids)).copy()
    print(
        f"[Progress] Filtered railway data: removed {len(remove_signal_ids)} distant signals "
        f"and {len(remove_platform_ids)} tram/metro platforms."
    )
    return filtered


def build_platform_station_node_lookup(raw_graph, stations, platforms):
    """Map each station to raw track nodes nearest to its passenger platforms."""
    raw_nodes = gpd.GeoDataFrame(
        {"raw_node": list(raw_graph.nodes)},
        geometry=[Point(node) for node in raw_graph.nodes],
        crs="EPSG:4326",
    ).to_crs("EPSG:28992")
    metric_stations = stations.to_crs("EPSG:28992")
    metric_platforms = platforms.to_crs("EPSG:28992")

    station_rows = metric_stations[["geometry", "name", "official_name", "uic_name", "@id"]].copy()
    station_rows["station_name"] = station_rows.apply(get_station_name, axis=1)
    platform_station = gpd.sjoin_nearest(
        metric_platforms[["geometry"]],
        station_rows[["geometry", "station_name"]],
        how="left",
        max_distance=1000,
        distance_col="station_distance_m",
    )
    platform_nodes = gpd.sjoin_nearest(
        platform_station[["geometry", "station_name"]].dropna(subset=["station_name"]),
        raw_nodes,
        how="left",
        max_distance=100,
        distance_col="track_distance_m",
    )

    station_node_lookup = {}
    for station_name, group in platform_nodes.groupby("station_name"):
        nodes = list(dict.fromkeys(group["raw_node"].dropna().tolist()))
        if nodes:
            station_node_lookup[station_name] = nodes

    return station_node_lookup


def nearest_station_for_node(node, station_node_lookup):
    """Return the station name whose raw-graph station node is nearest to the supplied raw graph node."""
    return min(
        station_node_lookup,
        key=lambda station_name: min(
            (node[0] - station_node[0]) ** 2 + (node[1] - station_node[1]) ** 2
            for station_node in station_node_lookup[station_name]
        ),
    )


def display_station_graph_on_osm(station_graph, station_node_lookup, output_path=None, show=True):
    """Display the station graph over OpenStreetMap and optionally save the figure."""
    station_positions = {
        station_name: (
            sum(node[0] for node in raw_nodes) / len(raw_nodes),
            sum(node[1] for node in raw_nodes) / len(raw_nodes),
        )
        for station_name, raw_nodes in station_node_lookup.items()
        if raw_nodes
    }
    station_points = gpd.GeoDataFrame(
        {"station": list(station_positions)},
        geometry=[Point(position) for position in station_positions.values()],
        crs="EPSG:4326",
    ).to_crs("EPSG:28992")
    projected_positions = {
        row.station: (row.geometry.x, row.geometry.y)
        for row in station_points.itertuples()
    }

    edge_rows = [
        {"station_a": station_a, "station_b": station_b, "geometry": data["geometry"]}
        for station_a, station_b, data in station_graph.edges(data=True)
        if data.get("geometry") is not None and not data["geometry"].is_empty
    ]
    edge_geometries = gpd.GeoDataFrame(edge_rows, geometry="geometry", crs="EPSG:4326").to_crs("EPSG:28992")

    figure, axes = plt.subplots(figsize=(16, 14))
    edge_geometries.plot(ax=axes, color="#1769aa", linewidth=1.1, alpha=0.85, zorder=2)
    nx.draw_networkx_nodes(
        station_graph,
        projected_positions,
        ax=axes,
        node_size=16,
        node_color="#d94841",
        edgecolors="white",
        linewidths=0.3,
        alpha=0.95,
    )
    ctx.add_basemap(
        axes,
        source=ctx.providers.OpenStreetMap.Mapnik,
        crs="EPSG:28992",
        alpha=0.65,
    )
    axes.set_title("Railway station network")
    axes.set_axis_off()
    figure.tight_layout()

    if output_path:
        figure.savefig(output_path, dpi=220, bbox_inches="tight")
        print(f"[Progress] Saved OSM station graph display to {output_path}")
    if show:
        plt.show(block=True)
    else:
        plt.close(figure)


def build_station_connection_graph_and_write_geojsons(
    railways_path="NL_Railway_Data_Signals.geojson",
    output_dir="station_edge_geojsons",
    display_graph=True,
):
    """Build a raw railway graph and derive a station graph by traversing each station's raw branches until the next station is reached.

    This produces one edge per adjacent station pair, with the geometry and intermediate raw-node data stored as edge attributes so the full railway path can later be reconstructed as a route.
    """
    print(f"[Progress] Loading railway data from {railways_path}...")
    railways = gpd.read_file(railways_path)
    if railways.empty:
        raise ValueError(f"No data found in {railways_path}")

    railways = filter_railway_data(railways)
    railways = railways[~railways["name"].astype(str).str.contains("Nienoord", case=False, na=False)].copy()
    railways = railways[~railways["official_name"].astype(str).str.contains("Nienoord", case=False, na=False)].copy()

    stations = railways[railways.get("railway") == "station"].copy()
    stations = stations[stations.geometry.notnull()].copy()
    platforms = railways[railways.get("railway") == "platform"].copy()
    if stations.empty:
        raise ValueError("No station geometries found in the rail network")

    print(f"[Progress] Building the raw railway graph from all node/edge data...")
    raw_graph = build_raw_railway_graph(railways)
    print(f"[Progress] Raw graph ready: {raw_graph.number_of_nodes()} nodes and {raw_graph.number_of_edges()} edges.")

    station_node_lookup = build_platform_station_node_lookup(raw_graph, stations, platforms)
    station_names = list(station_node_lookup)

    raw_node_station_lookup = {
        node: nearest_station_for_node(node, station_node_lookup)
        for node in raw_graph.nodes
    }

    station_graph = nx.Graph()
    station_graph.add_nodes_from(station_names)
    print(f"[Progress] Station graph initialized with {station_graph.number_of_nodes()} nodes.")

    os.makedirs(output_dir, exist_ok=True)
    canonical_station_edges = set()

    for station_name, start_raw_nodes in station_node_lookup.items():
        print(f"[Progress] Starting station traversal from {station_name}.")
        for start_raw_node in start_raw_nodes:
            for neighbor in list(raw_graph.neighbors(start_raw_node)):
                branch_queue = [(neighbor, [start_raw_node, neighbor], [tuple(sorted((start_raw_node, neighbor)))], {start_raw_node, neighbor})]

                while branch_queue:
                    current_node, branch_nodes, branch_edges, visited = branch_queue.pop(0)

                    nearest_station = raw_node_station_lookup[current_node]
                    if current_node != start_raw_node and nearest_station != station_name:
                        next_station = nearest_station
                        pair = tuple(sorted((station_name, next_station)))
                        if pair not in canonical_station_edges:
                            canonical_station_edges.add(pair)

                            segment_geometries = []
                            node_records = []
                            for node in branch_nodes:
                                node_records.append({
                                    "node": list(node),
                                    "attributes": json_safe(
                                        dict(raw_graph.nodes[node])
                                    ),
                                })

                            edge_records = []
                            raw_edge_records = []
                            for u, v in [tuple(sorted(edge)) for edge in branch_edges]:
                                if raw_graph.has_edge(u, v):
                                    edge_data = raw_graph[u][v]
                                    edge_geom = edge_data.get("geometry")
                                    if edge_geom is not None and not edge_geom.is_empty:
                                        segment_geometries.append(edge_geom)
                                    edge_records.append({
                                        "edge": [list(u), list(v)],
                                        "attributes": json_safe({
                                            key: value for key, value
                                            in edge_data.items()
                                            if key != "geometry"
                                        }),
                                    })
                                    raw_edge_records.append(edge_data)

                            if segment_geometries:
                                merged_geometry = linemerge(MultiLineString(segment_geometries))
                                if merged_geometry is not None and not getattr(merged_geometry, "is_empty", False):
                                    station_graph.add_edge(
                                        station_name,
                                        next_station,
                                        geometry=merged_geometry,
                                        nodes=node_records,
                                        edges=edge_records,
                                        **aggregate_attribute_values(
                                            raw_edge_records
                                        ),
                                        track_count=len(segment_geometries),
                                        total_length=merged_geometry.length,
                                        source_station=station_name,
                                        target_station=next_station,
                                    )
                                    print(f"[Progress] Added adjacent-station edge: {station_name} -> {next_station}")
                        break

                    for next_neighbor in raw_graph.neighbors(current_node):
                        if next_neighbor in visited:
                            continue
                        branch_queue.append((
                            next_neighbor,
                            branch_nodes + [next_neighbor],
                            branch_edges + [tuple(sorted((current_node, next_neighbor)))],
                            visited | {next_neighbor},
                        ))

    print(f"[Progress] Finished station graph construction; exporting adjacent-station edges...")
    written_edge_names = set()
    for station_a, station_b, data in station_graph.edges(data=True):
        geometry = data.get("geometry")
        if geometry is None or getattr(geometry, "is_empty", False):
            continue

        ordered_a, ordered_b = sorted([station_a, station_b], key=lambda name: str(name))
        canonical_pair = (ordered_a, ordered_b)
        if canonical_pair in written_edge_names:
            continue
        written_edge_names.add(canonical_pair)

        output_path = os.path.join(
            output_dir,
            f"{sanitize_station_name(ordered_a)}__{sanitize_station_name(ordered_b)}.geojson",
        )

        properties = {
            key: geojson_attribute(value)
            for key, value in data.items()
            if key != "geometry"
        }
        properties.update({
            "station_a": ordered_a,
            "station_b": ordered_b,
            "source_station": ordered_a,
            "target_station": ordered_b,
            "total_length": float(data.get("total_length", geometry.length)),
        })
        edge_gdf = gpd.GeoDataFrame(
            {key: [value] for key, value in properties.items()} | {
                "geometry": [geometry],
            },
            geometry="geometry",
            crs="EPSG:4326",
        )
        edge_gdf.to_file(output_path, driver="GeoJSON")
        print(f"[Progress] Wrote adjacent-station edge file: {output_path}")

    print(f"[Progress] Created station graph with {len(written_edge_names)} adjacent-station edges in {output_dir}")
    print(f"[Summary] Station graph nodes: {station_graph.number_of_nodes()} | edges: {station_graph.number_of_edges()}")

    graph_path = os.path.join(output_dir, "station_graph.pkl")
    with open(graph_path, "wb") as graph_file:
        pickle.dump(station_graph, graph_file)
    print(f"[Progress] Saved station graph to {graph_path}")

    if display_graph:
        display_station_graph_on_osm(
            station_graph,
            station_node_lookup,
            output_path=os.path.join(output_dir, "station_graph_osm.png"),
        )

    return station_graph


if __name__ == "__main__":
    build_station_connection_graph_and_write_geojsons(output_dir="station_edge_geojsons")
