import json
import math
import os
import re

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, Point
from shapely.strtree import STRtree


INPUT_FILE = "NL_Railway_Data_Signals.geojson"
OUTPUT_FILTERED_DATA = "third_try_filtered_data.geojson"
OUTPUT_NON_TRACK_DATA = "third_try_non_track_data.geojson"
OUTPUT_COMPONENT_FOLDER = "between_station_sections"
OUTPUT_STATION_FOLDER = "station_nodes"
PROJECTED_CRS = "EPSG:28992"
TRACK_PLATFORM_DISTANCE_M = 250
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
	"old_name:1868-1999", "old_name:1881-1921", "old_name:1885-1921",
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


def is_yes(value):
	return str(value).strip().lower() == "yes"


def feature_name(row):
	for column in ("name", "official_name", "uic_name", "@id"):
		value = row.get(column)
		if value is not None and str(value).strip() and str(value).lower() != "nan":
			return str(value).strip()
	return "None"


def line_parts(geometry):
	if geometry is None or geometry.is_empty:
		return []
	if geometry.geom_type == "LineString":
		return [geometry]
	if geometry.geom_type == "MultiLineString":
		return list(geometry.geoms)
	return []


def serializable_attributes(row):
	attributes = {}
	for column, value in row.items():
		if column == "geometry" or column in REMOVED_ATTRIBUTES:
			continue
		if value is None:
			attributes[column] = None
			continue
		try:
			if value != value:
				attributes[column] = None
				continue
		except (TypeError, ValueError):
			pass
		if hasattr(value, "item"):
			try:
				value = value.item()
			except Exception:
				value = str(value)
		attributes[column] = value
	return attributes


def filter_input_features(data):
	"""Remove signals over 10 m from rail and non-heavy-rail platforms."""
	print(f"[Progress] Filtering {len(data):,} input features...")
	metric_data = data.to_crs(PROJECTED_CRS)
	tracks = metric_data[metric_data["railway"].eq("rail")]
	if tracks.empty:
		raise ValueError("The input contains no railway=rail features")

	track_union = tracks.geometry.union_all()
	signals = metric_data[metric_data["railway"].eq("signal")]
	signal_distance = signals.geometry.distance(track_union)
	distant_signal_indexes = signal_distance[signal_distance > TRACK_PLATFORM_DISTANCE_M].index

	platforms = metric_data[metric_data["railway"].eq("platform")]
	non_rail_platforms = (
		platforms["tram"].map(is_yes)
		| platforms["subway"].map(is_yes)
		| platforms["light_rail"].map(is_yes)
	)
	non_rail_platform_indexes = platforms.index[non_rail_platforms]

	filtered = data.drop(
		index=distant_signal_indexes.union(non_rail_platform_indexes)
	).copy()
	filtered = filtered.drop(
		columns=[
			column for column in filtered.columns
			if column in REMOVED_ATTRIBUTES
		],
		errors="ignore",
	)
	print(
		f"Filtered {len(distant_signal_indexes)} distant signals and "
		f"{len(non_rail_platform_indexes)} tram/metro platforms."
	)
	return filtered.to_crs(PROJECTED_CRS)


def station_platform_names(data):
	"""Assign each retained platform to its nearest named train station."""
	print("[Progress] Associating railway platforms with stations...")
	stations = data[data["railway"].eq("station")].copy()
	stations = stations[stations["train"].map(is_yes)]
	platforms = data[data["railway"].eq("platform")].copy()
	print(
		f"[Progress] Found {len(stations):,} train stations and "
		f"{len(platforms):,} platforms."
	)
	if stations.empty or platforms.empty:
		print("[Progress] No stations or platforms available for association.")
		return platforms, {}

	station_points = stations.copy()
	station_points["station_name"] = station_points.apply(feature_name, axis=1)
	joined = gpd.sjoin_nearest(
		platforms[["geometry"]],
		station_points[["geometry", "station_name"]],
		how="left",
		distance_col="station_distance_m",
	)
	print(
		f"[Progress] Associated {joined['station_name'].notna().sum():,}/"
		f"{len(joined):,} platforms with named stations."
	)
	return platforms, joined["station_name"].to_dict()


def add_track_edges(graph, data):
	tracks = data[data["railway"].eq("rail")]
	print(
		f"[Progress] Preparing {len(tracks):,} track features..."
	)
	for track_number, (index, row) in enumerate(tracks.iterrows(), start=1):
		if track_number == 1 or track_number % 1000 == 0 or track_number == len(tracks):
			print(f"[Progress] Building track graph: {track_number:,}/{len(tracks):,} tracks")
		for line in line_parts(row.geometry):
			coordinates = list(line.coords)
			for start, end in zip(coordinates, coordinates[1:]):
				start_node = (round(start[0], 3), round(start[1], 3))
				end_node = (round(end[0], 3), round(end[1], 3))
				if start_node == end_node:
					continue
				segment = LineString([start, end])
				graph.add_node(start_node, station="None")
				graph.add_node(end_node, station="None")
				graph.add_edge(
					start_node,
					end_node,
					geometry=list(segment.coords),
					length_m=segment.length,
					source_index=int(index),
					attributes=serializable_attributes(row),
				)


def station_nodes_from_platforms(
	graph, data, platform_station_names_by_index
):
	"""Assign graph nodes to nearby station-associated platforms."""
	print(
		f"[Progress] Assigning station membership to "
		f"{graph.number_of_nodes():,} graph nodes..."
	)
	platforms = data[data["railway"].eq("platform")]
	platform_geometries = []
	platform_names = []
	for index, platform in platforms.iterrows():
		station_name = platform_station_names_by_index.get(index)
		if station_name and station_name != "None":
			platform_geometries.append(platform.geometry)
			platform_names.append(station_name)
	platform_tree = STRtree(platform_geometries) if platform_geometries else None
	print(
		f"[Progress] Prepared {len(platform_geometries):,} station-associated "
		"platform geometries."
	)

	station_nodes = {}
	for node_number, node in enumerate(graph.nodes, start=1):
		if node_number == 1 or node_number % 10000 == 0 or node_number == graph.number_of_nodes():
			print(
				f"[Progress] Assigning station membership: "
				f"{node_number:,}/{graph.number_of_nodes():,}"
			)
		station_names = set()
		if platform_tree is not None:
			node_point = Point(node)
			nearby_platform_indexes = platform_tree.query(
				node_point,
				predicate="dwithin",
				distance=TRACK_PLATFORM_DISTANCE_M,
			)
			station_names.update(
				platform_names[index] for index in nearby_platform_indexes
			)
		station_nodes[node] = station_names
	print(f"[Progress] Assigned station membership to {len(station_nodes):,} nodes.")
	return station_nodes


def assign_station_edges(graph, station_nodes):
	"""Assign edges whose endpoints belong to one station."""
	for first, second, edge_data in graph.edges(data=True):
		common_stations = (
			station_nodes.get(first, set())
			& station_nodes.get(second, set())
		)
		edge_data["station"] = (
			next(iter(common_stations))
			if len(common_stations) == 1
			else "None"
		)


def connected_station_names(graph, nodes, station_nodes):
	"""Return all station labels attached to a calculated node set."""
	station_names = set()
	for node in nodes:
		station_names.update(station_nodes.get(node, set()))
		for _, _, edge_data in graph.edges(node, data=True):
			station_name = edge_data.get("station", "None")
			if station_name != "None":
				station_names.add(station_name)
	return tuple(sorted(station_names))


def trim_overdegree_station_ends(graph, station_nodes):
	"""Move station boundaries away from station-end nodes with degree greater than two."""
	station_names = sorted({
		station_name
		for memberships in station_nodes.values()
		for station_name in memberships
	})
	print(
		f"[Progress] Trimming over-degree ends for "
		f"{len(station_names):,} stations..."
	)
	for station_number, station_name in enumerate(station_names, start=1):
		print(
			f"[Progress] Trimming station ends: "
			f"{station_number:,}/{len(station_names):,} ({station_name})"
		)
		queue = [
		node for node, memberships in station_nodes.items()
		if station_name in memberships
		and any(not station_nodes[neighbor] for neighbor in graph.neighbors(node))
		and graph.degree(node) > 2
		]
		checked = set()
		while queue:
			node = queue.pop(0)
			if node in checked or station_name not in station_nodes[node]:
				continue
			checked.add(node)
			if graph.degree(node) <= 2:
				continue

			station_nodes[node].remove(station_name)
			for neighbor in graph.neighbors(node):
				if (
					station_name in station_nodes[neighbor]
					and graph.degree(neighbor) > 2
				):
					queue.append(neighbor)
	print("[Progress] Finished trimming over-degree station ends.")


def angle_between_vectors(center, first, second):
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


def next_path_nodes(graph, current_node, previous_node):
	"""Return forward neighbors, using the closest-angle pair at a junction."""
	neighbors = [neighbor for neighbor in graph.neighbors(current_node) if neighbor != previous_node]
	if len(neighbors) <= 2:
		return neighbors

	angle_pairs = [
		(
			angle_between_vectors(current_node, first, second),
			first,
			second,
		)
		for index, first in enumerate(neighbors)
		for second in neighbors[index + 1:]
	]
	_, first, second = min(angle_pairs, key=lambda item: item[0])
	return [first, second]


def append_known_unexplored(known_unexplored, seen_known_unexplored, state):
	if state not in seen_known_unexplored:
		seen_known_unexplored.add(state)
		known_unexplored.append(state)
	return known_unexplored


def safe_filename(value):
	"""Return a station name that is safe to use as a filename."""
	part = re.sub(r'[<>:"/\\|?*]+', "_", str(value))
	part = "_".join(part.split()).strip(" .")
	return part or "unknown"


def geojson_attribute_properties(attributes):
	"""Convert retained source attributes into a compact JSON property."""
	return {
		"source_attributes": json.dumps(
			attributes, ensure_ascii=False, default=str
		)
	}


def write_station_outputs(graph, station_nodes, output_folder=OUTPUT_STATION_FOLDER):
	"""Write one GeoJSON containing the nodes and edges assigned to each station."""
	os.makedirs(output_folder, exist_ok=True)
	for filename in os.listdir(output_folder):
		if filename.endswith(".geojson"):
			os.remove(os.path.join(output_folder, filename))
	station_names = sorted({
		station_name
		for memberships in station_nodes.values()
		for station_name in memberships
	})
	non_station_graph = nx.MultiGraph()
	for first, second, edge_key, edge_data in graph.edges(
		keys=True, data=True
	):
		if edge_data.get("station", "None") == "None":
			non_station_graph.add_edge(
				first, second, key=edge_key, **edge_data
			)
	station_only_components = {}
	for component in nx.connected_components(non_station_graph):
		component_stations = connected_station_names(
			graph, component, station_nodes
		)
		if len(component_stations) == 1:
			station_only_components.setdefault(
				component_stations[0], []
			).append(component)

	for station_number, station_name in enumerate(station_names, start=1):
		print(
			f"[Progress] Writing station GeoJSON: "
			f"{station_number:,}/{len(station_names):,} ({station_name})"
		)
		nodes = {
			node for node, memberships in station_nodes.items()
			if station_name in memberships
		}
		station_graph_nodes = set(nodes)
		extra_components = station_only_components.get(station_name, [])
		extra_nodes = set().union(*extra_components) if extra_components else set()
		nodes.update(extra_nodes)
		records = [
			{
				"feature_type": "node",
				"station": (
					station_name if node not in extra_nodes else "None"
				),
				"geometry": Point(node),
				**geojson_attribute_properties(
					graph.nodes[node].get("attributes", {})
				),
				"source_index": graph.nodes[node].get("source_index"),
			}
			for node in nodes
		]
		seen_edges = set()
		for node in nodes:
			for neighbor, edge_keys in graph[node].items():
				if (
					node not in station_graph_nodes
					or neighbor not in station_graph_nodes
				):
					continue
				for edge_key, edge_data in edge_keys.items():
					identifier = (node, neighbor, edge_key)
					reverse_identifier = (neighbor, node, edge_key)
					if identifier in seen_edges or reverse_identifier in seen_edges:
						continue
					seen_edges.add(identifier)
					geometry = edge_data.get("geometry")
					if geometry:
						edge_attributes = dict(edge_data.get("attributes", {}))
						records.append({
							"feature_type": "track",
							"station": station_name,
							"geometry": LineString(geometry),
							**geojson_attribute_properties(edge_attributes),
							"source_index": edge_data.get("source_index"),
							"length_m": edge_data.get("length_m"),
						})
		for component in extra_components:
			for first, second, edge_key in non_station_graph.edges(
				component, keys=True
			):
				edge_data = non_station_graph[first][second][edge_key]
				geometry = edge_data.get("geometry")
				if not geometry:
					continue
				records.append({
					"feature_type": "track",
					"station": "None",
					"geometry": LineString(geometry),
					**geojson_attribute_properties(
						edge_data.get("attributes", {})
					),
					"source_index": edge_data.get("source_index"),
					"length_m": edge_data.get("length_m"),
				})
		output_path = os.path.join(
			output_folder,
			f"{safe_filename(station_name)}.geojson",
		)
		gpd.GeoDataFrame(
			records,
			geometry="geometry",
			crs=PROJECTED_CRS,
		).to_crs("EPSG:4326").to_file(output_path, driver="GeoJSON")
	print(
		f"[Progress] Wrote {len(station_names):,} station GeoJSON files "
		f"to {output_folder}."
	)


def write_between_station_outputs(
	graph, station_nodes, output_folder=OUTPUT_COMPONENT_FOLDER
):
	"""Write non-station edge components after station membership is finalized."""
	os.makedirs(output_folder, exist_ok=True)
	for filename in os.listdir(output_folder):
		if filename.endswith(".geojson"):
			os.remove(os.path.join(output_folder, filename))

	non_station_graph = nx.MultiGraph()
	for first, second, edge_key, edge_data in graph.edges(
		keys=True, data=True
	):
		if edge_data.get("station", "None") == "None":
			non_station_graph.add_edge(
				first, second, key=edge_key, **edge_data
			)

	features_by_name = {}
	for component in nx.connected_components(non_station_graph):
		if not component:
			continue
		station_names = connected_station_names(
			graph, component, station_nodes
		)
		if len(station_names) < 2:
			continue
		name = "__".join(safe_filename(station) for station in station_names)
		features = features_by_name.setdefault(name, [])
		for node in component:
			features.append({
				"feature_type": "node",
				"station": "None",
				"geometry": Point(node),
				**geojson_attribute_properties(
					graph.nodes[node].get("attributes", {})
				),
				"source_index": graph.nodes[node].get("source_index"),
			})
		for first, second, edge_key in non_station_graph.edges(
			component, keys=True
		):
			edge_data = non_station_graph[first][second][edge_key]
			geometry = edge_data.get("geometry")
			if not geometry:
				continue
			features.append({
				"feature_type": "track",
				"station": "None",
				"geometry": LineString(geometry),
				**geojson_attribute_properties(
					edge_data.get("attributes", {})
				),
				"source_index": edge_data.get("source_index"),
				"length_m": edge_data.get("length_m"),
			})

	written = 0
	for name, features in features_by_name.items():
		output_path = os.path.join(output_folder, f"{name}.geojson")
		gpd.GeoDataFrame(
			features,
			geometry="geometry",
			crs=PROJECTED_CRS,
		).to_crs("EPSG:4326").to_file(output_path, driver="GeoJSON")
		written += 1
	print(
		f"[Progress] Wrote {written:,} between-station GeoJSON files "
		f"to {output_folder}."
	)


def build_graph(input_file=INPUT_FILE):
	"""Run the complete filtering, graph-building, and GeoJSON export pipeline."""
	print(f"[Progress] Reading input data from {input_file}...")
	data = gpd.read_file(input_file)
	if data.empty:
		raise ValueError(f"No features found in {input_file}")
	if data.crs is None:
		raise ValueError("Input GeoJSON has no CRS")
	print(f"[Progress] Loaded {len(data):,} input features.")

	filtered_data = filter_input_features(data)
	filtered_data.to_crs("EPSG:4326").to_file(
		OUTPUT_FILTERED_DATA,
		driver="GeoJSON",
	)
	print(f"[Progress] Wrote filtered data to {OUTPUT_FILTERED_DATA}.")
	non_track_data = filtered_data[
		~filtered_data["railway"].eq("rail")
	].copy()
	non_track_data.to_crs("EPSG:4326").to_file(
		OUTPUT_NON_TRACK_DATA,
		driver="GeoJSON",
	)
	print(
		f"[Progress] Wrote {len(non_track_data):,} non-track features to "
		f"{OUTPUT_NON_TRACK_DATA}."
	)

	graph = nx.MultiGraph()
	add_track_edges(graph, filtered_data)
	print(
		f"[Progress] Built graph with {graph.number_of_nodes():,} nodes and "
		f"{graph.number_of_edges():,} edges."
	)

	_, platform_station_names_by_index = station_platform_names(filtered_data)
	station_nodes = station_nodes_from_platforms(
		graph, filtered_data, platform_station_names_by_index
	)
	assign_station_edges(graph, station_nodes)
	for node, memberships in station_nodes.items():
		graph.nodes[node]["station"] = ", ".join(sorted(memberships)) or "None"
	write_between_station_outputs(graph, station_nodes)
	write_station_outputs(graph, station_nodes)
	print("[Progress] Processing complete.")
	return graph


if __name__ == "__main__":
	build_graph()
