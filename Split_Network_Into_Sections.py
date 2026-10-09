import json
import math
import os

import geopandas as gpd
import networkx as nx
from shapely.geometry import LineString, MultiLineString, Point
from shapely.strtree import STRtree


INPUT_FILE = "NL_Railway_Data_Signals.geojson"
OUTPUT_FILTERED_DATA = "third_try_filtered_data.geojson"
OUTPUT_COMPONENT_FOLDER = "third_try_station_components"
OUTPUT_STATION_FOLDER = "third_try_station_nodes"
PROJECTED_CRS = "EPSG:28992"
TRACK_PLATFORM_DISTANCE_M = 250.0
TRIM_OVERDEGREE_STATION_ENDS = True


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
		if column == "geometry":
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

	filtered = data.drop(index=distant_signal_indexes.union(non_rail_platform_indexes)).copy()
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


def add_point_nodes(graph, data):
	point_features = data[data.geometry.geom_type.eq("Point")]
	print(f"[Progress] Adding {len(point_features):,} point nodes...")
	for point_number, (index, row) in enumerate(point_features.iterrows(), start=1):
		if point_number == 1 or point_number % 1000 == 0 or point_number == len(point_features):
			print(
				f"[Progress] Adding point nodes: "
				f"{point_number:,}/{len(point_features):,}"
			)
		point = row.geometry
		node = (round(point.x, 3), round(point.y, 3))
		graph.add_node(
			node,
			station="None",
			geometry=[point.x, point.y],
			feature_type=row.get("railway", "None"),
			attributes=serializable_attributes(row),
			source_index=int(index),
		)


def add_track_edges(graph, data, platform_station_names_by_index):
	tracks = data[data["railway"].eq("rail")]
	platforms = data[data["railway"].eq("platform")]
	print(
		f"[Progress] Preparing {len(platforms):,} platforms for "
		f"{len(tracks):,} track features..."
	)
	platform_geometries = []
	platform_names = []
	for platform_number, (index, platform) in enumerate(platforms.iterrows(), start=1):
		station_name = platform_station_names_by_index.get(index)
		if station_name and station_name != "None":
			platform_geometries.append(platform.geometry)
			platform_names.append(station_name)
	platform_tree = STRtree(platform_geometries) if platform_geometries else None
	print(
		f"[Progress] Prepared {len(platform_geometries):,} station-associated "
		"platform geometries."
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
				station = "None"
				if platform_tree is not None:
					nearest_platform_index = platform_tree.nearest(segment)
					platform_geometry = platform_geometries[nearest_platform_index]
					if segment.distance(platform_geometry) <= TRACK_PLATFORM_DISTANCE_M:
						station = platform_names[nearest_platform_index]
				graph.add_edge(
					start_node,
					end_node,
					station=station,
					geometry=list(segment.coords),
					length_m=segment.length,
					source_index=int(index),
					attributes=serializable_attributes(row),
				)


def station_nodes_from_edges(graph):
	"""Return station membership for graph nodes based on labeled railway edges."""
	print(
		f"[Progress] Deriving station membership for "
		f"{graph.number_of_nodes():,} graph nodes..."
	)
	station_nodes = {}
	for node_number, node in enumerate(graph.nodes, start=1):
		if node_number == 1 or node_number % 10000 == 0 or node_number == graph.number_of_nodes():
			print(
				f"[Progress] Deriving station membership: "
				f"{node_number:,}/{graph.number_of_nodes():,}"
			)
		station_names = set()
		for _, _, edge_data in graph.edges(node, data=True):
			station_name = edge_data.get("station", "None")
			if station_name != "None":
				station_names.add(station_name)
		station_nodes[node] = station_names
	print(f"[Progress] Derived station membership for {len(station_nodes):,} nodes.")
	return station_nodes


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
	return "_".join(str(value).split())


def geojson_attribute_properties(attributes):
	"""Convert retained source attributes into safe GeoJSON properties."""
	properties = {
		"source_attributes": json.dumps(attributes, ensure_ascii=False, default=str)
	}
	for key, value in attributes.items():
		if isinstance(value, (dict, list, tuple, set)):
			value = json.dumps(value, ensure_ascii=False, default=str)
		properties[f"source_{key}"] = value
	return properties


def write_station_outputs(graph, station_nodes, output_folder=OUTPUT_STATION_FOLDER):
	"""Write one GeoJSON containing the nodes and edges assigned to each station."""
	os.makedirs(output_folder, exist_ok=True)
	station_names = sorted({
		station_name
		for memberships in station_nodes.values()
		for station_name in memberships
	})
	for station_number, station_name in enumerate(station_names, start=1):
		print(
			f"[Progress] Writing station GeoJSON: "
			f"{station_number:,}/{len(station_names):,} ({station_name})"
		)
		nodes = {
			node for node, memberships in station_nodes.items()
			if station_name in memberships
		}
		records = [
			{
				"feature_type": "node",
				"station": station_name,
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
				if neighbor not in nodes:
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

	_, platform_station_names_by_index = station_platform_names(filtered_data)
	graph = nx.MultiGraph()
	add_point_nodes(graph, filtered_data)
	add_track_edges(graph, filtered_data, platform_station_names_by_index)
	print(
		f"[Progress] Built graph with {graph.number_of_nodes():,} nodes and "
		f"{graph.number_of_edges():,} edges."
	)

	station_nodes = station_nodes_from_edges(graph)
	if TRIM_OVERDEGREE_STATION_ENDS:
		trim_overdegree_station_ends(graph, station_nodes)
	for node, memberships in station_nodes.items():
		graph.nodes[node]["station"] = ", ".join(sorted(memberships)) or "None"
	write_station_outputs(graph, station_nodes)
	print("[Progress] Processing complete.")
	return graph


if __name__ == "__main__":
	build_graph()
