import json
import math
import os
import re
import tempfile
from collections import defaultdict, deque

import geopandas as gpd
import folium
import networkx as nx
from shapely.geometry import LineString, Point, Polygon
from shapely.strtree import STRtree


INPUT_FILE = "NL_Railway_Data_Signals.geojson"
OUTPUT_FILTERED_DATA = "third_try_filtered_data.geojson"
OUTPUT_NON_TRACK_DATA = "third_try_non_track_data.geojson"
OUTPUT_COMPONENT_FOLDER = "between_station_sections"
OUTPUT_STATION_FOLDER = "station_nodes"
OUTPUT_NETWORK_MAP = "redefined_network_map.html"
PROJECTED_CRS = "EPSG:28992"
TRACK_PLATFORM_DISTANCE_M = 250
PLATFORM_TRACK_DISTANCE_M = 50
STATION_SEARCH_DISTANCE_M = 500
PLATFORM_END_DISTANCE_M = 4
PLATFORM_PARALLEL_TOLERANCE_DEGREES = 30
STATION_BOUNDARY_POINT_DISTANCE_M = 50
MAX_STATION_PAIR_DISTANCE_M = 50_000
MAX_STATION_COMPONENT_DIAMETER_M = 2_000
PLATFORM_PARALLEL_DOT_THRESHOLD = math.cos(math.radians(
	PLATFORM_PARALLEL_TOLERANCE_DEGREES
))
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


def remove_platform_closest_to_station(data, station_name):
	"""Remove the nearest platform to a specifically named station."""
	station_mask = (
		data["railway"].eq("station")
		& data.apply(feature_name, axis=1).eq(station_name)
	)
	station_features = data[station_mask]
	platforms = data[data["railway"].eq("platform")]
	if station_features.empty or platforms.empty:
		print(
			f"[Progress] No platform removed for {station_name!r}; "
			"the station or platforms were not found."
		)
		return data

	metric_data = data.to_crs(PROJECTED_CRS)
	station_geometry = metric_data.loc[
		station_features.index, "geometry"
	].iloc[0]
	distances = metric_data.loc[
		platforms.index, "geometry"
	].distance(station_geometry)
	platform_index = distances.idxmin()
	filtered = data.drop(index=platform_index).copy()
	print(
		f"[Progress] Removed platform {platform_index!r}, closest to "
		f"{station_name!r} ({distances.loc[platform_index]:.1f} m)."
	)
	return filtered


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
	filtered = remove_platform_closest_to_station(
		filtered,
		"Strandvliet",
	)
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


def add_graph_node(graph, node, **attributes):
	"""Add a graph node while preserving a monotonic, stable node ID."""
	if node not in graph:
		next_node_id = graph.graph.setdefault("_next_node_id", 1)
		attributes["node_id"] = next_node_id
		graph.graph["_next_node_id"] = next_node_id + 1
		graph.add_node(node, **attributes)
		return
	graph.nodes[node].update({
		key: value for key, value in attributes.items()
		if key != "node_id"
	})


def add_graph_edge(graph, first, second, **attributes):
	"""Add an edge while preserving a monotonic, stable edge ID."""
	if "edge_id" not in attributes:
		next_edge_id = graph.graph.setdefault("_next_edge_id", 1)
		attributes["edge_id"] = next_edge_id
		graph.graph["_next_edge_id"] = next_edge_id + 1
	else:
		graph.graph["_next_edge_id"] = max(
			graph.graph.get("_next_edge_id", 1),
			attributes["edge_id"] + 1,
		)
	return graph.add_edge(first, second, **attributes)


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
				segment = LineString([start_node, end_node])
				add_graph_node(graph, start_node, station="None")
				add_graph_node(graph, end_node, station="None")
				add_graph_edge(
					graph,
					start_node,
					end_node,
					geometry=list(segment.coords),
					length_m=segment.length,
					source_index=int(index),
					attributes=serializable_attributes(row),
				)


def platform_endpoints(geometry):
	"""Return the two endpoints of a platform geometry."""
	if geometry is None or geometry.is_empty:
		return []
	if geometry.geom_type == "MultiLineString":
		parts = [part for part in geometry.geoms if not part.is_empty]
		if not parts:
			return []
		geometry = max(parts, key=lambda part: part.length)
	if geometry.geom_type == "LineString":
		coordinates = list(geometry.coords)
		return [Point(coordinates[0]), Point(coordinates[-1])]
	if isinstance(geometry, Polygon):
		rectangle = geometry.minimum_rotated_rectangle
		coordinates = list(rectangle.exterior.coords)[:-1]
		edges = [
			(
				Point(coordinates[index]),
				Point(coordinates[(index + 1) % len(coordinates)]),
			)
			for index in range(len(coordinates))
		]
		shortest_edge = min(edges, key=lambda edge: edge[0].distance(edge[1]))
		opposite_edge = (
			edges[(edges.index(shortest_edge) + 2) % len(edges)]
		)
		shortest_edge_line = LineString([
			shortest_edge[0].coords[0],
			shortest_edge[1].coords[0],
		])
		opposite_edge_line = LineString([
			opposite_edge[0].coords[0],
			opposite_edge[1].coords[0],
		])
		return [
			shortest_edge_line.interpolate(0.5, normalized=True),
			opposite_edge_line.interpolate(0.5, normalized=True),
		]
	return []


def geometry_direction(geometry):
	"""Return a unit direction vector for a linear geometry."""
	if geometry is None or geometry.is_empty:
		return None
	if geometry.geom_type == "Polygon":
		rectangle = geometry.minimum_rotated_rectangle
		coordinates = list(rectangle.exterior.coords)[:-1]
		if len(coordinates) != 4:
			return None
		edges = [
			LineString([
				coordinates[index],
				coordinates[(index + 1) % len(coordinates)],
			])
			for index in range(len(coordinates))
		]
		geometry = max(edges, key=lambda edge: edge.length)
	if geometry.geom_type == "MultiLineString":
		parts = [part for part in geometry.geoms if not part.is_empty]
		if not parts:
			return None
		geometry = max(parts, key=lambda part: part.length)
	if geometry.geom_type != "LineString":
		return None
	coordinates = list(geometry.coords)
	if len(coordinates) < 2:
		return None
	first = coordinates[0]
	last = coordinates[-1]
	length = math.hypot(last[0] - first[0], last[1] - first[1])
	if length == 0:
		return None
	return ((last[0] - first[0]) / length, (last[1] - first[1]) / length)


def directions_are_parallel(first, second):
	"""Return whether two directions differ by no more than the tolerance."""
	if first is None or second is None:
		return False
	dot_product = abs(first[0] * second[0] + first[1] * second[1])
	return dot_product >= PLATFORM_PARALLEL_DOT_THRESHOLD


def split_edges_at_platform_ends(
	graph, data, platform_station_names_by_index
):
	"""Split tracks where nearest-platform distance crosses four metres."""
	stations = data[data["railway"].eq("station")]
	platforms = data[data["railway"].eq("platform")]
	station_geometries = []
	station_names = []
	for index, station in stations.iterrows():
		if is_yes(station.get("train")):
			station_geometries.append(station.geometry)
			station_names.append(feature_name(station))
	platform_geometries = list(platforms.geometry)
	platform_ids = [str(index) for index in platforms.index]
	platform_stations = [
		platform_station_names_by_index.get(index)
		for index in platforms.index
	]
	platform_directions = [
		geometry_direction(geometry)
		for geometry in platform_geometries
	]
	if not platform_geometries or not station_geometries:
		return

	station_tree = STRtree(station_geometries)
	platform_tree = STRtree(platform_geometries)

	def nearest_metadata(point, track_direction):
		station_index = station_tree.nearest(point)
		station_distance = point.distance(station_geometries[station_index])
		platform_index = platform_tree.nearest(point)
		platform_distance = point.distance(
			platform_geometries[platform_index]
		)
		if station_distance > STATION_SEARCH_DISTANCE_M:
			return None
		platform_direction = platform_directions[platform_index]
		if not directions_are_parallel(track_direction, platform_direction):
			return None
		return {
			"station": station_names[station_index],
			"station_distance": station_distance,
			"platform_id": platform_ids[platform_index],
			"platform_station": platform_stations[platform_index],
			"platform_distance": platform_distance,
			"platform_direction": platform_direction,
		}

	edge_records = []
	for first, second, edge_key, edge_data in graph.edges(
		keys=True, data=True
	):
		geometry = edge_data.get("geometry")
		if geometry:
			edge_records.append((
				first,
				second,
				edge_key,
				edge_data,
				LineString(geometry),
			))
	if not edge_records:
		return

	splits = {}
	for first, second, edge_key, edge_data, geometry in edge_records:
		length = geometry.length
		track_direction = geometry_direction(geometry)
		sample_count = max(2, int(math.ceil(length / 25)) + 1)
		samples = []
		for sample_number in range(sample_count):
			fraction = sample_number / (sample_count - 1)
			point = geometry.interpolate(fraction, normalized=True)
			metadata = nearest_metadata(point, track_direction)
			samples.append((fraction * length, metadata))

		for (first_position, first_metadata), (
			second_position, second_metadata
		) in zip(samples, samples[1:]):
			# A missing metadata value means the point is outside the
			# station search area, not that it is beyond a platform end.
			if first_metadata is None or second_metadata is None:
				continue
			first_inside = (
				first_metadata["platform_distance"] < PLATFORM_END_DISTANCE_M
			)
			second_inside = (
				second_metadata["platform_distance"] < PLATFORM_END_DISTANCE_M
			)
			if first_inside == second_inside:
				continue

			low_position = first_position
			high_position = second_position
			low_inside = first_inside
			for _ in range(20):
				middle_position = (low_position + high_position) / 2
				middle_point = geometry.interpolate(middle_position)
				middle_metadata = nearest_metadata(
					middle_point,
					track_direction,
				)
				middle_inside = (
					middle_metadata is not None
					and middle_metadata["platform_distance"]
					< PLATFORM_END_DISTANCE_M
				)
				if middle_inside == low_inside:
					low_position = middle_position
				else:
					high_position = middle_position
			position = (low_position + high_position) / 2
			point = geometry.interpolate(position)
			metadata = nearest_metadata(point, track_direction)
			if metadata is None:
				continue
			node = (round(point.x, 3), round(point.y, 3))
			splits.setdefault((first, second, edge_key), []).append(
				(position, node, metadata)
			)

		for node, point in ((first, Point(first)), (second, Point(second))):
			metadata = nearest_metadata(point, track_direction)
			if metadata is not None:
				graph.nodes[node].update({
					"nearest_station": metadata["station"],
					"nearest_station_distance_m": metadata["station_distance"],
					"nearest_platform": metadata["platform_id"],
					"nearest_platform_distance_m": metadata["platform_distance"],
				})

	for edge_identifier, cut_positions in splits.items():
		first, second, edge_key = edge_identifier
		if not graph.has_edge(first, second, edge_key):
			continue
		edge_data = graph[first][second][edge_key]
		geometry = LineString(edge_data["geometry"])
		cuts = sorted(cut_positions, key=lambda item: item[0])
		points = (
			[(0.0, first, None)]
			+ cuts
			+ [(geometry.length, second, None)]
		)
		graph.remove_edge(first, second, edge_key)
		for (start_position, start_node, _), (end_position, end_node, _) in zip(
			points, points[1:]
		):
			if start_node not in graph:
				add_graph_node(graph, start_node)
			if end_node not in graph:
				add_graph_node(graph, end_node)
			segment_coordinates = list(LineString([
				geometry.interpolate(start_position),
				geometry.interpolate(end_position),
			]).coords)
			segment_coordinates[0] = start_node
			segment_coordinates[-1] = end_node
			segment = LineString(segment_coordinates)
			segment_data = dict(edge_data)
			segment_data.pop("edge_id", None)
			segment_data["geometry"] = list(segment.coords)
			segment_data["length_m"] = segment.length
			add_graph_edge(graph, start_node, end_node, **segment_data)
		for position, node, metadata in cuts:
			graph.nodes[node].setdefault("retained_for", set()).add(
				"platform_end"
			)
			graph.nodes[node].update({
				"nearest_station": metadata["station"],
				"nearest_station_distance_m": metadata["station_distance"],
				"nearest_platform": metadata["platform_id"],
				"nearest_platform_distance_m": metadata["platform_distance"],
			})
			graph.nodes[node].setdefault("platform_stations", set()).add(
				metadata["platform_station"] or metadata["station"]
			)
			graph.nodes[node].setdefault("platform_ids", set()).add(
				metadata["platform_id"]
			)


def add_station_end_boundaries(graph, data, platform_station_names_by_index):
	"""Add transverse station-end boundaries for tracks without nearby platforms."""
	platforms = data[data["railway"].eq("platform")]
	station_platforms = {}
	for index, platform in platforms.iterrows():
		station_name = platform_station_names_by_index.get(index)
		if station_name and station_name != "None":
			station_platforms.setdefault(station_name, []).append(platform.geometry)

	platform_end_nodes_by_station = {}
	for node, attributes in graph.nodes(data=True):
		for station_name in attributes.get("platform_stations", set()):
			platform_end_nodes_by_station.setdefault(
				station_name, []
			).append(node)
	edge_records = []
	for first, second, edge_key, edge_data in graph.edges(
		keys=True, data=True
	):
		geometry = LineString(edge_data.get("geometry", []))
		if geometry.is_empty:
			continue
		edge_records.append((
			first,
			second,
			edge_key,
			edge_data,
			geometry,
			geometry_direction(geometry),
		))
	if not edge_records:
		return
	edge_tree = STRtree([record[4] for record in edge_records])

	for station_name, platform_geometries in station_platforms.items():
		platform_direction = next(
			(
				direction
				for direction in (
					geometry_direction(geometry)
					for geometry in platform_geometries
				)
				if direction is not None
			),
			None,
		)
		if platform_direction is None:
			continue

		platform_end_nodes = platform_end_nodes_by_station.get(
			station_name, []
		)
		if len(platform_end_nodes) < 2:
			continue

		origin = Point(platform_end_nodes[0])
		along_values = {
			node: (
				(node[0] - origin.x) * platform_direction[0]
				+ (node[1] - origin.y) * platform_direction[1]
			)
			for node in platform_end_nodes
		}
		ordered_nodes = sorted(platform_end_nodes, key=along_values.get)
		gaps = [
			(
				along_values[ordered_nodes[index + 1]]
				- along_values[ordered_nodes[index]],
				index,
			)
			for index in range(len(ordered_nodes) - 1)
		]
		if not gaps:
			continue
		_, split_index = max(gaps)
		end_groups = (
			ordered_nodes[:split_index + 1],
			ordered_nodes[split_index + 1:],
		)
		boundary_lines = []
		for group in end_groups:
			if len(group) < 2:
				continue
			first, second = max(
				(
					(first, second)
					for index, first in enumerate(group)
					for second in group[index + 1:]
				),
				key=lambda pair: Point(pair[0]).distance(Point(pair[1])),
			)
			first_x, first_y = first
			second_x, second_y = second
			dx = second_x - first_x
			dy = second_y - first_y
			line_length = math.hypot(dx, dy)
			if line_length == 0:
				continue
			extension_x = (
				STATION_BOUNDARY_POINT_DISTANCE_M * dx / line_length
			)
			extension_y = (
				STATION_BOUNDARY_POINT_DISTANCE_M * dy / line_length
			)
			boundary_lines.append(LineString([
				(first_x - extension_x, first_y - extension_y),
				(second_x + extension_x, second_y + extension_y),
			]))

		nearby_end_tree = STRtree([Point(node) for node in platform_end_nodes])
		cuts = {}
		for boundary_line in boundary_lines:
			candidate_indexes = edge_tree.query(
				boundary_line.buffer(STATION_BOUNDARY_POINT_DISTANCE_M)
			)
			for candidate_index in candidate_indexes:
				first, second, edge_key, edge_data, geometry, direction = edge_records[
					candidate_index
				]
				if not directions_are_parallel(direction, platform_direction):
					continue
				platform_edge_nodes = nearby_end_tree.query(
					geometry,
					predicate="dwithin",
					distance=0.5,
				)
				if len(platform_edge_nodes) > 0:
					continue
				intersection = geometry.intersection(boundary_line)
				if intersection.is_empty:
					continue
				intersection_points = (
					list(intersection.geoms)
					if intersection.geom_type == "MultiPoint"
					else [intersection]
				)
				for point in intersection_points:
					if point.geom_type != "Point":
						continue
					nearby = nearby_end_tree.query(
						point,
						predicate="dwithin",
						distance=STATION_BOUNDARY_POINT_DISTANCE_M,
					)
					if len(nearby) == 0:
						continue
					position = geometry.project(point)
					if position <= 0 or position >= geometry.length:
						continue
					node = (round(point.x, 3), round(point.y, 3))
					cuts.setdefault((first, second, edge_key), []).append(
						(position, node)
					)

		for edge_identifier, edge_cuts in cuts.items():
			first, second, edge_key = edge_identifier
			if not graph.has_edge(first, second, edge_key):
				continue
			edge_data = graph[first][second][edge_key]
			geometry = LineString(edge_data["geometry"])
			unique_cuts = {
				(position, node) for position, node in edge_cuts
			}
			ordered_cuts = sorted(unique_cuts)
			points = (
				[(0.0, first)]
				+ ordered_cuts
				+ [(geometry.length, second)]
			)
			graph.remove_edge(first, second, edge_key)
			for (start_position, start_node), (end_position, end_node) in zip(
				points, points[1:]
			):
				add_graph_node(graph, start_node)
				add_graph_node(graph, end_node)
				segment_coordinates = list(LineString([
					geometry.interpolate(start_position),
					geometry.interpolate(end_position),
				]).coords)
				segment_coordinates[0] = start_node
				segment_coordinates[-1] = end_node
				segment = LineString(segment_coordinates)
				segment_data = dict(edge_data)
				segment_data.pop("edge_id", None)
				segment_data["geometry"] = list(segment.coords)
				segment_data["length_m"] = segment.length
				add_graph_edge(graph, start_node, end_node, **segment_data)
			for _, node in ordered_cuts:
				graph.nodes[node].setdefault("retained_for", set()).add(
					"station_boundary"
				)
				graph.nodes[node].setdefault("platform_stations", set()).add(
					station_name
				)


def add_station_end_boundaries_by_platform_end_axis(
	graph,
	data,
	platform_station_names_by_index,
):
	"""Add station boundaries using the platform-end geometry as the axis.

	This alternative avoids choosing one arbitrary platform direction and does
	not reject candidate tracks merely because they are not parallel to that
	platform.  For each station, the dominant axis is estimated from all
	platform-end nodes, the largest axial gap separates the two station ends,
	and each end is represented by a line through its furthest-apart ends.
	"""
	platforms = data[data["railway"].eq("platform")]
	station_names_by_platform = {
		index: platform_station_names_by_index.get(index)
		for index in platforms.index
	}
	platform_stations = {}
	for index, platform in platforms.iterrows():
		station_name = station_names_by_platform.get(index)
		if station_name and station_name != "None":
			platform_stations.setdefault(station_name, []).append(
				platform.geometry
			)

	platform_end_nodes_by_station = {}
	for node, attributes in graph.nodes(data=True):
		for station_name in attributes.get("platform_stations", set()):
			platform_end_nodes_by_station.setdefault(
				station_name, []
			).append(node)

	edge_records = []
	for first, second, edge_key, edge_data in graph.edges(
		keys=True, data=True
	):
		geometry = LineString(edge_data.get("geometry", []))
		if geometry.is_empty:
			continue
		edge_records.append((first, second, edge_key, edge_data, geometry))
	if not edge_records:
		return
	edge_tree = STRtree([record[4] for record in edge_records])

	for station_name in platform_stations:
		platform_end_nodes = list(dict.fromkeys(
			platform_end_nodes_by_station.get(station_name, [])
		))
		if len(platform_end_nodes) < 4:
			continue

		platform_directions = [
			geometry_direction(geometry)
			for geometry in platform_stations.get(station_name, [])
		]
		platform_directions = [
			direction for direction in platform_directions
			if direction is not None
		]
		if not platform_directions:
			continue
		reference_direction = platform_directions[0]
		aligned_directions = []
		for direction in platform_directions:
			if (
				reference_direction[0] * direction[0]
				+ reference_direction[1] * direction[1]
				< 0
			):
				direction = (-direction[0], -direction[1])
			aligned_directions.append(direction)
		axis_vector = (
			sum(direction[0] for direction in aligned_directions),
			sum(direction[1] for direction in aligned_directions),
		)
		axis_length = math.hypot(*axis_vector)
		if axis_length == 0:
			continue
		axis = (
			axis_vector[0] / axis_length,
			axis_vector[1] / axis_length,
		)
		origin = Point(platform_end_nodes[0])
		along_values = {
			node: (
				(node[0] - origin.x) * axis[0]
				+ (node[1] - origin.y) * axis[1]
			)
			for node in platform_end_nodes
		}
		ordered_nodes = sorted(platform_end_nodes, key=along_values.get)
		gaps = [
			(
				along_values[ordered_nodes[index + 1]]
				- along_values[ordered_nodes[index]],
				index,
			)
			for index in range(len(ordered_nodes) - 1)
		]
		if not gaps:
			continue
		_, split_index = max(gaps)
		end_groups = (
			ordered_nodes[:split_index + 1],
			ordered_nodes[split_index + 1:],
		)

		boundary_lines = []
		for group in end_groups:
			if len(group) < 2:
				continue
			first, second = max(
				(
					(first, second)
					for index, first in enumerate(group)
					for second in group[index + 1:]
				),
				key=lambda pair: Point(pair[0]).distance(Point(pair[1])),
			)
			dx = second[0] - first[0]
			dy = second[1] - first[1]
			line_length = math.hypot(dx, dy)
			if line_length == 0:
				continue
			extension = STATION_BOUNDARY_POINT_DISTANCE_M
			boundary_lines.append(LineString([
				(
					first[0] - extension * dx / line_length,
					first[1] - extension * dy / line_length,
				),
				(
					second[0] + extension * dx / line_length,
					second[1] + extension * dy / line_length,
				),
			]))

		nearby_end_tree = STRtree([Point(node) for node in platform_end_nodes])
		cuts = {}
		for boundary_line in boundary_lines:
			for candidate_index in edge_tree.query(
				boundary_line.buffer(STATION_BOUNDARY_POINT_DISTANCE_M)
			):
				first, second, edge_key, edge_data, geometry = edge_records[
					candidate_index
				]
				if len(nearby_end_tree.query(
					geometry,
					predicate="dwithin",
					distance=0.5,
				)) > 0:
					continue
				intersection = geometry.intersection(boundary_line)
				if intersection.is_empty:
					continue
				intersection_points = (
					list(intersection.geoms)
					if intersection.geom_type == "MultiPoint"
					else [intersection]
				)
				for point in intersection_points:
					if point.geom_type != "Point":
						continue
					if len(nearby_end_tree.query(
						point,
						predicate="dwithin",
						distance=STATION_BOUNDARY_POINT_DISTANCE_M,
					)) == 0:
						continue
					position = geometry.project(point)
					if position <= 0 or position >= geometry.length:
						continue
					node = (round(point.x, 3), round(point.y, 3))
					cuts.setdefault((first, second, edge_key), []).append(
						(position, node)
					)

		for edge_identifier, edge_cuts in cuts.items():
			first, second, edge_key = edge_identifier
			if not graph.has_edge(first, second, edge_key):
				continue
			edge_data = graph[first][second][edge_key]
			geometry = LineString(edge_data["geometry"])
			ordered_cuts = sorted(set(edge_cuts))
			points = (
				[(0.0, first)]
				+ ordered_cuts
				+ [(geometry.length, second)]
			)
			graph.remove_edge(first, second, edge_key)
			for (start_position, start_node), (end_position, end_node) in zip(
				points, points[1:]
			):
				add_graph_node(graph, start_node)
				add_graph_node(graph, end_node)
				segment_coordinates = list(LineString([
					geometry.interpolate(start_position),
					geometry.interpolate(end_position),
				]).coords)
				segment_coordinates[0] = start_node
				segment_coordinates[-1] = end_node
				segment = LineString(segment_coordinates)
				segment_data = dict(edge_data)
				segment_data.pop("edge_id", None)
				segment_data["geometry"] = list(segment.coords)
				segment_data["length_m"] = segment.length
				add_graph_edge(graph, start_node, end_node, **segment_data)
			for _, node in ordered_cuts:
				graph.nodes[node].setdefault(
					"retained_for", set()
				).add("station_boundary")
				graph.nodes[node].setdefault(
					"platform_stations", set()
				).add(station_name)


def classify_junction_locations(graph):
	"""Mark switches, crossings, and end-of-line locations."""
	for node in graph.nodes:
		degree = graph.degree(node)
		if degree <= 1:
			graph.nodes[node].setdefault(
				"retained_for", set()
			).add("end_of_line")
			continue
		if degree >= 3:
			graph.nodes[node].setdefault("retained_for", set()).add("switch")
		if degree < 4:
			continue
		neighbors = list(graph.neighbors(node))
		straight_pairs = 0
		for index, first in enumerate(neighbors):
			for second in neighbors[index + 1:]:
				angle = angle_between_vectors(node, first, second)
				if angle >= math.radians(150):
					straight_pairs += 1
		if straight_pairs >= 2:
			graph.nodes[node].setdefault(
				"retained_for", set()
			).add("rail_rail_crossing")


def remove_redundant_station_boundaries(graph):
	"""Remove boundaries bypassed by a bounded same-station route."""
	boundary_nodes = [
		node for node, attributes in graph.nodes(data=True)
		if "station_boundary" in attributes.get("retained_for", set())
	]
	to_remove = set()

	for boundary_node in boundary_nodes:
		if boundary_node not in graph:
			continue
		boundary_stations = graph.nodes[boundary_node].get(
			"platform_stations", set()
		)
		queue = deque()
		for neighbor, edge_keys in graph[boundary_node].items():
			for edge_key, edge_data in edge_keys.items():
				edge_length = edge_data.get("length_m")
				if edge_length is None:
					edge_length = edge_data["geometry"].length
				if edge_length <= MAX_STATION_PAIR_DISTANCE_M:
					queue.append((
						boundary_node,
						neighbor,
						edge_key,
						edge_length,
					))
		visited = {}
		while queue:
			(
				previous_node,
				node,
				incoming_edge_key,
				distance,
			) = queue.popleft()
			state = (previous_node, node, incoming_edge_key)
			if distance > MAX_STATION_PAIR_DISTANCE_M:
				continue
			previous_distance = visited.get(state, math.inf)
			if distance >= previous_distance:
				continue
			visited[state] = distance
			attributes = graph.nodes[node]
			node_stations = attributes.get("platform_stations", set())
			retained_for = attributes.get("retained_for", set())
			is_platform_end = "platform_end" in retained_for
			is_station_boundary = "station_boundary" in retained_for
			if (
				is_platform_end
				and boundary_stations
				and boundary_stations & node_stations
			):
				to_remove.add(boundary_node)
				break
			if (
				(is_platform_end or is_station_boundary)
				and node_stations - boundary_stations
			):
				continue
			incoming_edge = graph[previous_node][node][incoming_edge_key]
			incoming_direction = edge_direction_from_node(
				incoming_edge,
				node,
			)
			for neighbor, edge_keys in graph[node].items():
				if neighbor == previous_node:
					continue
				for edge_key, edge_data in edge_keys.items():
					outgoing_direction = edge_direction_from_node(
						edge_data,
						node,
					)
					if (
						incoming_direction is None
						or outgoing_direction is None
						or angle_between_vectors(
							(0, 0),
							incoming_direction,
							outgoing_direction,
						) <= math.radians(90)
					):
						continue
					edge_length = edge_data.get("length_m")
					if edge_length is None:
						edge_length = edge_data["geometry"].length
					next_distance = distance + edge_length
					if next_distance > MAX_STATION_PAIR_DISTANCE_M:
						continue
					next_state = (node, neighbor, edge_key)
					queue.append((
						node,
						neighbor,
						edge_key,
						next_distance,
					))

	print(
		f"[Progress] Identified {len(to_remove):,} redundant station-boundary "
		f"nodes (searches limited to "
		f"{MAX_STATION_PAIR_DISTANCE_M / 1000:g} km)."
	)
	for node in to_remove:
		unretain_graph_node(
			graph,
			node,
			"station_boundary",
		)


def remove_unreachable_platform_ends(graph):
	"""Remove platform ends that cannot lead to another station."""
	platform_end_nodes = [
		node for node, attributes in graph.nodes(data=True)
		if "platform_end" in attributes.get("retained_for", set())
	]
	to_remove = set()

	for start_node in platform_end_nodes:
		if start_node not in graph:
			continue
		start_stations = graph.nodes[start_node].get(
			"platform_stations", set()
		)
		if not start_stations:
			to_remove.add(start_node)
			continue

		visited = {start_node}
		queue = deque([start_node])
		reachable_other_station = False
		while queue and not reachable_other_station:
			node = queue.popleft()
			for neighbor in graph.neighbors(node):
				if neighbor in visited:
					continue
				attributes = graph.nodes[neighbor]
				neighbor_stations = attributes.get(
					"platform_stations", set()
				)
				if neighbor_stations - start_stations:
					reachable_other_station = True
					break
				if (
					neighbor != start_node
					and "platform_end" in attributes.get(
						"retained_for", set()
					)
					and neighbor_stations & start_stations
				):
					continue
				visited.add(neighbor)
				queue.append(neighbor)

		if not reachable_other_station:
			to_remove.add(start_node)

	for node in to_remove:
		unretain_graph_node(
			graph,
			node,
			"platform_end",
		)
	print(
		f"Identified {len(to_remove):,} platform-end nodes with no route to "
		"another station."
	)


def unretain_graph_node(graph, node, cause):
	"""Convert a no-longer-retained node into ordinary edge geometry."""
	if node not in graph:
		return
	attributes = graph.nodes[node]
	retained_for = attributes.get("retained_for", set())
	retained_for.discard(cause)
	if retained_for:
		return
	attributes.pop("platform_stations", None)
	attributes.pop("platform_ids", None)
	attributes.pop("nearest_station", None)
	attributes.pop("nearest_station_distance_m", None)
	attributes.pop("nearest_platform", None)
	attributes.pop("nearest_platform_distance_m", None)
	contract_graph_node(graph, node)


def contract_graph_node(graph, node):
	"""Replace a degree-two node with one geometry-preserving edge."""
	if node not in graph or graph.degree(node) != 2:
		return False
	neighbors = list(graph.neighbors(node))
	if len(neighbors) != 2 or neighbors[0] == neighbors[1]:
		return False
	incident_edges = list(graph.edges(node, keys=True, data=True))
	if len(incident_edges) != 2:
		return False

	first, second = neighbors
	edge_by_neighbor = {
		(other if start == node else start): attributes
		for start, other, _key, attributes in incident_edges
	}
	first_edge = edge_by_neighbor.get(first)
	second_edge = edge_by_neighbor.get(second)
	if first_edge is None or second_edge is None:
		return False
	first_coordinates = list(first_edge.get("geometry", []))
	second_coordinates = list(second_edge.get("geometry", []))
	if not first_coordinates or not second_coordinates:
		return False
	if Point(first_coordinates[0]).distance(Point(first)) > Point(
		first_coordinates[-1]
	).distance(Point(first)):
		first_coordinates.reverse()
	if Point(second_coordinates[0]).distance(Point(node)) > Point(
		second_coordinates[-1]
	).distance(Point(node)):
		second_coordinates.reverse()
	first_coordinates[0] = first
	first_coordinates[-1] = node
	second_coordinates[0] = node
	second_coordinates[-1] = second
	combined_geometry = first_coordinates + second_coordinates[1:]
	combined_geometry[0] = first
	combined_geometry[-1] = second
	merged_data = dict(first_edge)
	merged_data.pop("edge_id", None)
	merged_data["geometry"] = combined_geometry
	merged_data["length_m"] = LineString(combined_geometry).length
	graph.remove_node(node)
	add_graph_edge(graph, first, second, **merged_data)
	return True


def reduce_graph_to_locations(graph):
	"""Contract non-location degree-two nodes while retaining connectivity."""
	queue = deque(
		node for node, attributes in graph.nodes(data=True)
		if not attributes.get("retained_for")
	)
	queued = set(queue)
	while queue:
		node = queue.popleft()
		queued.discard(node)
		if node not in graph or graph.nodes[node].get("retained_for"):
			continue

		degree = graph.degree(node)
		neighbors = list(graph.neighbors(node))
		if degree <= 1:
			affected = list(graph.neighbors(node))
			graph.remove_node(node)
			for neighbor in affected:
				if neighbor not in queued:
					queue.append(neighbor)
					queued.add(neighbor)
			continue
		if degree != 2 or len(neighbors) != 2 or neighbors[0] == neighbors[1]:
			continue

		first, second = neighbors
		incident_edges = list(graph.edges(node, keys=True, data=True))
		if len(incident_edges) != 2:
			continue

		first_edge = incident_edges[0][3]
		edge_by_neighbor = {
			(other if start == node else start): attributes
			for start, other, _key, attributes in incident_edges
		}
		first_edge = edge_by_neighbor.get(first)
		second_edge = edge_by_neighbor.get(second)
		if first_edge is None or second_edge is None:
			continue
		first_coordinates = list(first_edge.get("geometry", []))
		second_coordinates = list(second_edge.get("geometry", []))
		if not first_coordinates or not second_coordinates:
			continue
		if Point(first_coordinates[0]).distance(Point(first)) > Point(
			first_coordinates[-1]
		).distance(Point(first)):
			first_coordinates.reverse()
		if Point(second_coordinates[0]).distance(Point(node)) > Point(
			second_coordinates[-1]
		).distance(Point(node)):
			second_coordinates.reverse()
		first_coordinates[0] = first
		first_coordinates[-1] = node
		second_coordinates[0] = node
		second_coordinates[-1] = second
		combined_geometry = first_coordinates + second_coordinates[1:]
		combined_geometry[0] = first
		combined_geometry[-1] = second
		merged_data = dict(first_edge)
		merged_data.pop("edge_id", None)
		merged_data["geometry"] = combined_geometry
		merged_data["length_m"] = LineString(combined_geometry).length
		graph.remove_node(node)
		add_graph_edge(graph, first, second, **merged_data)
		for neighbor in (first, second):
			if neighbor not in queued:
				queue.append(neighbor)
				queued.add(neighbor)


def station_nodes_from_retained_platforms(graph):
	"""Assign anchors and unambiguous nearby non-anchor nodes to stations."""
	print(
		f"[Progress] Assigning station membership to "
		f"{graph.number_of_nodes():,} graph nodes..."
	)
	anchor_nodes = {
		node for node in graph
		if graph.nodes[node].get("retained_for", set()) & {
			"platform_end", "station_boundary"
		}
	}
	station_nodes = {node: set() for node in graph}
	for node in anchor_nodes:
		station_nodes[node].update(
			graph.nodes[node].get("platform_stations", set())
		)

	non_anchor_nodes = set(graph) - anchor_nodes
	non_anchor_graph = graph.subgraph(non_anchor_nodes)
	anchor_stations = {
		node: station_nodes[node]
		for node in anchor_nodes
		if station_nodes[node]
	}
	for component in nx.connected_components(non_anchor_graph):
		boundary_anchors = {
			neighbor
			for node in component
			for neighbor in graph.neighbors(node)
			if neighbor in anchor_stations
		}
		reachable_stations = {
			station
			for anchor in boundary_anchors
			for station in anchor_stations[anchor]
		}
		if len(reachable_stations) != 1:
			continue
		station = next(iter(reachable_stations))
		for node in component:
			if any(
				math.hypot(
					node[0] - anchor[0],
					node[1] - anchor[1],
				) <= 500
				for anchor in boundary_anchors
			):
				station_nodes[node].add(station)
	print(f"[Progress] Assigned station membership to {len(station_nodes):,} nodes.")
	return station_nodes


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


def assign_station_edges(
	graph,
	station_nodes,
	data=None,
	platform_station_names_by_index=None,
):
	"""Assign station edges and annotate edges running beside one platform."""
	platforms_by_station = defaultdict(list)
	platform_numbers = {}
	if data is not None and platform_station_names_by_index is not None:
		for index, platform in data[data["railway"].eq("platform")].iterrows():
			station_name = platform_station_names_by_index.get(index)
			if not station_name or station_name == "None":
				continue
			number = (
				platform.get("local_ref")
				or platform.get("ref")
				or platform.get("name")
			)
			if number is None or str(number).strip().lower() == "nan":
				number = str(len(platforms_by_station[station_name]) + 1)
			platform_numbers[str(index)] = str(number).strip()
			platforms_by_station[station_name].append((
				platform.geometry,
				str(number).strip(),
			))

	for first, second, edge_data in graph.edges(data=True):
		common_stations = (
			station_nodes.get(first, set())
			& station_nodes.get(second, set())
		)
		edge_data["station"] = next(iter(common_stations), "None")
		edge_data.pop("platform_number", None)
		if len(common_stations) != 1:
			continue
		station_name = next(iter(common_stations))
		first_attributes = graph.nodes[first]
		second_attributes = graph.nodes[second]
		if (
			first_attributes.get("nearest_platform_distance_m", math.inf)
			< PLATFORM_END_DISTANCE_M
			and second_attributes.get(
				"nearest_platform_distance_m", math.inf
			) < PLATFORM_END_DISTANCE_M
		):
			platform_id = first_attributes.get("nearest_platform")
			if platform_id == second_attributes.get("nearest_platform"):
				number = platform_numbers.get(str(platform_id))
				if number is not None:
					edge_data["platform_number"] = number


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


def edge_direction_from_node(edge_data, node):
	"""Return direction to the geometry point immediately beside ``node``."""
	coordinates = list(edge_data.get("geometry", []))
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


def valid_station_pair_edges(graph, station_nodes, first_station, second_station):
	"""Return edges that belong to at least one angle-valid station path."""
	allowed_stations = {first_station, second_station}
	directed_edges = []
	for first, second, edge_key, edge_data in graph.edges(
		keys=True, data=True
	):
		if first == second:
			continue
		directed_edges.extend([
			(first, second, edge_key, edge_data),
			(second, first, edge_key, edge_data),
		])

	start_states = set()
	target_states = set()
	transitions = {}
	for previous, current, edge_key, edge_data in directed_edges:
		current_stations = station_nodes.get(current, set())
		if current_stations - allowed_stations:
			continue
		state = (previous, current, edge_key)
		if first_station in station_nodes.get(previous, set()):
			start_states.add(state)
		if second_station in current_stations:
			target_states.add(state)
		if current_stations & allowed_stations:
			transitions[state] = []
			continue

		outgoing = []
		incoming_direction = edge_direction_from_node(edge_data, current)
		for neighbor, edge_keys in graph[current].items():
			if neighbor == previous:
				continue
			if station_nodes.get(neighbor, set()) - allowed_stations:
				continue
			for next_key, next_data in edge_keys.items():
				outgoing_direction = edge_direction_from_node(
					next_data, current
				)
				if incoming_direction is None or outgoing_direction is None:
					continue
				if angle_between_vectors(
					(0, 0),
					incoming_direction,
					outgoing_direction,
				) <= math.radians(90):
					continue
				outgoing.append((current, neighbor, next_key))
		transitions[state] = outgoing

	reachable = set(start_states)
	queue = deque(start_states)
	while queue:
		state = queue.popleft()
		for next_state in transitions.get(state, []):
			if next_state not in reachable:
				reachable.add(next_state)
				queue.append(next_state)

	reverse_transitions = {}
	for state, next_states in transitions.items():
		for next_state in next_states:
			reverse_transitions.setdefault(next_state, []).append(state)
	can_reach_target = set(target_states)
	queue = deque(target_states)
	while queue:
		state = queue.popleft()
		for previous_state in reverse_transitions.get(state, []):
			if previous_state not in can_reach_target:
				can_reach_target.add(previous_state)
				queue.append(previous_state)

	valid_states = reachable & can_reach_target
	return {
		(state[0], state[1], state[2])
		for state in valid_states
	}


def build_station_sections(graph, station_nodes):
	"""Build angle-valid paths between different assigned stations."""
	anchors = {
		node for node in graph
		if graph.nodes[node].get("retained_for", set()) & {
			"platform_end", "station_boundary"
		}
	}
	station_anchor_nodes = defaultdict(set)
	for node in anchors:
		for station in station_nodes.get(node, set()):
			station_anchor_nodes[station].add(node)
	station_names = sorted(station_anchor_nodes)
	allowed_pairs = {
		tuple(sorted((first_station, second_station)))
		for index, first_station in enumerate(station_names)
		for second_station in station_names[index + 1:]
		if station_pair_is_within_distance(
			station_anchor_nodes[first_station],
			station_anchor_nodes[second_station],
			MAX_STATION_PAIR_DISTANCE_M,
		)
	}

	sections = defaultdict(set)
	transition_cache = {}
	terminal_cache = {}
	error_cache = set()

	def expand_state(state, start_stations):
		if state in transition_cache:
			return transition_cache[state], terminal_cache[state]
		previous, current, edge_key = state
		incoming = edge_direction_from_node(
			graph[previous][current][edge_key],
			current,
		)
		if incoming is None:
			transition_cache[state] = ()
			terminal_cache[state] = ()
			return (), ()
		next_states = []
		terminal_states = []
		for neighbor, edge_keys in graph[current].items():
			for next_key, next_data in edge_keys.items():
				next_state = (current, neighbor, next_key)
				outgoing = edge_direction_from_node(
					next_data,
					current,
				)
				if outgoing is None:
					continue
				angle = angle_between_vectors(
					(0, 0), incoming, outgoing
				)
				if angle <= math.radians(90):
					continue
				neighbor_stations = station_nodes.get(neighbor, set())
				if neighbor in anchors:
					for target_station in (
						neighbor_stations
					):
						terminal_states.append((
							next_state, target_station
						))
					continue
				if neighbor_stations:
					error_cache.add((
						state, next_state, tuple(sorted(neighbor_stations))
					))
					continue
				next_states.append(next_state)
		transition_cache[state] = tuple(next_states)
		terminal_cache[state] = tuple(terminal_states)
		return transition_cache[state], terminal_cache[state]

	for first, second, edge_key in graph.edges(keys=True):
		if first not in anchors or second not in anchors:
			continue
		first_stations = station_nodes.get(first, set())
		second_stations = station_nodes.get(second, set())
		for first_station in first_stations:
			for second_station in second_stations:
				pair = tuple(sorted((first_station, second_station)))
				if first_station != second_station and pair in allowed_pairs:
					sections[pair].add((first, second, edge_key))
	for start in sorted(anchors):
		start_stations = station_nodes.get(start, set())
		if not start_stations:
			continue
		if not any(
			tuple(sorted((start_station, target_station)))
			in allowed_pairs
			for start_station in start_stations
			for target_station in station_names
			if target_station != start_station
		):
			continue
		start_states = {
			(start, neighbor, edge_key)
			for neighbor, edge_keys in graph[start].items()
			if not station_nodes.get(neighbor)
			for edge_key in edge_keys
		}
		reachable = set(start_states)
		queue = deque(start_states)
		terminal_states = []
		transitions = defaultdict(list)
		while queue:
			state = queue.popleft()
			next_states, state_terminals = expand_state(
				state, start_stations
			)
			for next_state, target_station in state_terminals:
				if target_station in start_stations:
					continue
				pairs = {
					tuple(sorted((start_station, target_station)))
					for start_station in start_stations
				}
				if pairs & allowed_pairs:
					terminal_states.append((
						state, next_state, target_station
					))
			transitions[state].extend(next_states)
			for next_state in next_states:
				if next_state not in reachable:
					reachable.add(next_state)
					queue.append(next_state)

		if error_cache:
			continue
		reverse = defaultdict(list)
		for state, next_states in transitions.items():
			for next_state in next_states:
				reverse[next_state].append(state)
		reachable_reverse = {
			state: [
				previous for previous in previous_states
				if previous in reachable
			]
			for state, previous_states in reverse.items()
		}
		for state, next_state, target_station in terminal_states:
			valid = {state, next_state}
			queue = deque([state])
			while queue:
				current = queue.popleft()
				for previous in reachable_reverse.get(current, []):
					if previous not in valid:
						valid.add(previous)
						queue.append(previous)
			for start_station in start_stations:
				if start_station == target_station:
					continue
				pair = tuple(sorted((start_station, target_station)))
				if pair not in allowed_pairs:
					continue
				for valid_state in valid:
					sections[pair].add((
						valid_state[0],
						valid_state[1],
						valid_state[2],
					))
	if error_cache:
		raise RuntimeError(
			"Station-assigned node encountered inside station path: "
			f"{len(error_cache)} path transitions."
		)
	return dict(sections)


def station_pair_is_within_distance(
	first_nodes, second_nodes, maximum_distance
):
	"""Return whether two station node sets are within a distance limit."""
	maximum_distance_squared = maximum_distance ** 2
	for first in first_nodes:
		for second in second_nodes:
			delta_x = first[0] - second[0]
			delta_y = first[1] - second[1]
			if delta_x * delta_x + delta_y * delta_y <= (
				maximum_distance_squared
			):
				return True
	return False


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


def clear_output_directory(output_folder):
	"""Delete old files in an output directory, reporting locked files."""
	os.makedirs(output_folder, exist_ok=True)
	for filename in os.listdir(output_folder):
		path = os.path.join(output_folder, filename)
		if not os.path.isfile(path):
			continue
		try:
			os.remove(path)
		except PermissionError:
			print(
				f"[Progress] Could not delete locked output file {path!r}; "
				"new output will use a fallback filename."
			)


def write_geojson_with_fallback(data, output_file):
	"""Write GeoJSON atomically, using a fallback if the target is locked."""
	output_path = os.path.abspath(output_file)
	output_directory = os.path.dirname(output_path) or os.curdir
	file_descriptor, temporary_file = tempfile.mkstemp(
		prefix=".geojson_",
		suffix=".tmp",
		dir=output_directory,
	)
	os.close(file_descriptor)
	try:
		data.to_file(temporary_file, driver="GeoJSON")
		try:
			os.replace(temporary_file, output_path)
			return output_file
		except PermissionError:
			fallback_path = os.path.splitext(output_path)[0] + "_new.geojson"
			fallback_number = 2
			while os.path.exists(fallback_path):
				fallback_path = (
					f"{os.path.splitext(output_path)[0]}_new_"
					f"{fallback_number}.geojson"
				)
				fallback_number += 1
			os.replace(temporary_file, fallback_path)
			print(
				f"Could not replace locked file {output_file!r}; "
				f"wrote the new output to {fallback_path!r}."
			)
			return fallback_path
	finally:
		if os.path.exists(temporary_file):
			os.remove(temporary_file)


def write_redefined_network_map(
	graph, output_file=OUTPUT_NETWORK_MAP
):
	"""Write the redefined network with retained nodes prominently marked."""
	if not graph.nodes:
		raise ValueError("Cannot create a map for an empty graph")

	edge_records = [
		{
			"geometry": LineString(edge_attributes["geometry"]),
			"edge_id": edge_attributes["edge_id"],
			"source_node_id": graph.nodes[first]["node_id"],
			"target_node_id": graph.nodes[second]["node_id"],
		}
		for first, second, _edge_key, edge_attributes in graph.edges(
			keys=True, data=True
		)
		if edge_attributes.get("geometry")
	]
	edge_data = gpd.GeoDataFrame(
		edge_records,
		geometry="geometry",
		crs=PROJECTED_CRS,
	).to_crs("EPSG:4326")

	node_records = []
	for node, attributes in graph.nodes(data=True):
		retained_for = sorted(attributes.get("retained_for", set()))
		node_records.append({
			"node": node,
			"node_id": attributes["node_id"],
			"geometry": Point(node),
			"retained_for": retained_for,
			"station": attributes.get("station", "None"),
			"nearest_station": attributes.get("nearest_station"),
			"nearest_station_distance_m": attributes.get(
				"nearest_station_distance_m"
			),
			"nearest_platform": attributes.get("nearest_platform"),
			"nearest_platform_distance_m": attributes.get(
				"nearest_platform_distance_m"
			),
		})
	node_data = gpd.GeoDataFrame(
		node_records,
		geometry="geometry",
		crs=PROJECTED_CRS,
	).to_crs("EPSG:4326")

	centroid = node_data.geometry.unary_union.centroid
	network_map = folium.Map(
		location=[centroid.y, centroid.x],
		zoom_start=8,
		tiles="OpenStreetMap",
	)
	folium.GeoJson(
		edge_data.to_json(),
		name="Redefined rail network",
		style_function=lambda feature: {
			"color": "#222222",
			"weight": 3,
			"opacity": 0.9,
		},
		tooltip=folium.GeoJsonTooltip(
			fields=["edge_id", "source_node_id", "target_node_id"],
			aliases=["Edge ID", "From node ID", "To node ID"],
		),
	).add_to(network_map)

	cause_colors = {
		"platform_end": "#e31a1c",
		"station_boundary": "#33a02c",
		"end_of_line": "#1f78b4",
		"switch": "#ff7f00",
		"rail_rail_crossing": "#6a3d9a",
	}
	for _, row in node_data.iterrows():
		causes = row["retained_for"]
		color = next(
			(cause_colors[cause] for cause in causes if cause in cause_colors),
			"#808080",
		)
		popup = (
			f"Node ID: {row['node_id']}<br>"
			f"Coordinates: {row['node']}<br>"
			f"Retained for: {', '.join(causes)}<br>"
			f"Station: {row['station']}<br>"
			f"Nearest station: {row['nearest_station']} "
			f"({row['nearest_station_distance_m']} m)<br>"
			f"Nearest platform: {row['nearest_platform']} "
			f"({row['nearest_platform_distance_m']} m)"
		)
		folium.CircleMarker(
			location=[row.geometry.y, row.geometry.x],
			radius=7 if causes else 3,
			color="#000000",
			weight=1 if causes else 0,
			fill=True,
			fill_color=color,
			fill_opacity=0.9 if causes else 0.65,
			tooltip=f"Node ID {row['node_id']}",
			popup=folium.Popup(popup, max_width=350),
		).add_to(network_map)

	folium.LayerControl().add_to(network_map)
	bounds = node_data.geometry.total_bounds
	network_map.fit_bounds([
		[bounds[1], bounds[0]],
		[bounds[3], bounds[2]],
	])
	network_map.save(output_file)
	return output_file


def write_station_outputs(graph, station_nodes, output_folder=OUTPUT_STATION_FOLDER):
	"""Write one GeoJSON containing the nodes and edges assigned to each station."""
	clear_output_directory(output_folder)
	station_names = sorted({
		station_name
		for memberships in station_nodes.values()
		for station_name in memberships
	})
	nodes_by_station = {station_name: set() for station_name in station_names}
	for node, memberships in station_nodes.items():
		if len(memberships) != 1:
			continue
		station_name = next(iter(memberships))
		nodes_by_station[station_name].add(node)
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
		nodes = nodes_by_station[station_name]
		station_graph_nodes = set(nodes)
		for first, second, edge_key, edge_data in graph.edges(
			keys=True, data=True
		):
			if edge_data.get("station") == station_name and (
				first in station_graph_nodes or second in station_graph_nodes
			):
				nodes.update((first, second))
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
				"retained_for": ", ".join(sorted(
					graph.nodes[node].get("retained_for", set())
				)),
				"nearest_station": graph.nodes[node].get("nearest_station"),
				"nearest_station_distance_m": graph.nodes[node].get(
					"nearest_station_distance_m"
				),
				"nearest_platform": graph.nodes[node].get("nearest_platform"),
				"nearest_platform_distance_m": graph.nodes[node].get(
					"nearest_platform_distance_m"
				),
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
					if edge_data.get("station") != station_name:
						continue
					seen_edges.add(identifier)
					geometry = edge_data.get("geometry")
					if geometry:
						edge_attributes = dict(edge_data.get("attributes", {}))
						records.append({
							"feature_type": "track",
							"station": station_name,
							**({
								"platform_number": edge_data[
									"platform_number"
								]
							} if edge_data.get("platform_number") is not None
							else {}),
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
					**({
						"platform_number": edge_data["platform_number"]
					} if edge_data.get("platform_number") is not None else {}),
					"geometry": LineString(geometry),
					**geojson_attribute_properties(
						edge_data.get("attributes", {})
					),
					"source_index": edge_data.get("source_index"),
					"length_m": edge_data.get("length_m"),
				})
		track_endpoints = {
			tuple(coordinate)
			for record in records
			if record["feature_type"] == "track"
			for coordinate in (
				list(record["geometry"].coords)[0],
				list(record["geometry"].coords)[-1],
			)
		}
		records = [
			record for record in records
			if record["feature_type"] == "track"
			or any(
				tuple(coordinate) in track_endpoints
				for coordinate in (
					record["geometry"].coords[0],
					record["geometry"].coords[-1],
				)
			)
		]
		if not records:
			print(
				f"[Progress] Skipping empty station GeoJSON for "
				f"{station_name}."
			)
			continue
		output_path = os.path.join(
			output_folder,
			f"{safe_filename(station_name)}.geojson",
		)
		write_geojson_with_fallback(
			gpd.GeoDataFrame(
			records,
			geometry="geometry",
			crs=PROJECTED_CRS,
			).to_crs("EPSG:4326"),
			output_path,
		)
	print(
		f"[Progress] Wrote {len(station_names):,} station GeoJSON files "
		f"to {output_folder}."
	)


def write_between_station_outputs(
	graph, station_nodes, output_folder=OUTPUT_COMPONENT_FOLDER
):
	"""Write only angle-valid paths between each pair of stations."""
	clear_output_directory(output_folder)

	station_names = sorted({
		station_name
		for memberships in station_nodes.values()
		for station_name in memberships
	})
	sections_by_pair = build_station_sections(graph, station_nodes)
	written = 0

	for (first_station, second_station), valid_edges in sections_by_pair.items():
		unique_edges = {
			(
				min(first, second),
				max(first, second),
				edge_key,
			)
			for first, second, edge_key in valid_edges
		}
		section_nodes = {
			node
			for first, second, edge_key in unique_edges
			for node in (first, second)
		}
		unique_edges = {
			(first, second, edge_key)
			for first, second, edge_key in unique_edges
			if (
				graph[first][second][edge_key].get("station", "None")
				== "None"
				or graph[first][second][edge_key].get("station", "None")
				in {first_station, second_station}
			)
		}
		section_nodes = {
			node
			for first, second, edge_key in unique_edges
			for node in (first, second)
		}

		features = []
		for node in section_nodes:
			node_stations = station_nodes.get(node, set())
			station_label = next(
				(
					station for station in (first_station, second_station)
					if station in node_stations
				),
				"None",
			)
			features.append({
				"feature_type": "node",
				"station": station_label,
				"geometry": Point(node),
				"retained_for": ", ".join(sorted(
					graph.nodes[node].get("retained_for", set())
				)),
				"nearest_station": graph.nodes[node].get("nearest_station"),
				"nearest_station_distance_m": graph.nodes[node].get(
					"nearest_station_distance_m"
				),
				"nearest_platform": graph.nodes[node].get("nearest_platform"),
				"nearest_platform_distance_m": graph.nodes[node].get(
					"nearest_platform_distance_m"
				),
				**geojson_attribute_properties(
					graph.nodes[node].get("attributes", {})
				),
				"source_index": graph.nodes[node].get("source_index"),
			})
		for first, second, edge_key in unique_edges:
			edge_data = graph[first][second][edge_key]
			geometry = edge_data.get("geometry")
			if not geometry:
				continue
			features.append({
				"feature_type": "track",
				"station": edge_data.get("station", "None"),
				**({
					"platform_number": edge_data["platform_number"]
				} if edge_data.get("platform_number") is not None else {}),
				"geometry": LineString(geometry),
				**geojson_attribute_properties(
					edge_data.get("attributes", {})
				),
				"source_index": edge_data.get("source_index"),
				"length_m": edge_data.get("length_m"),
			})

		if not features:
			continue
		name = "__".join((
			safe_filename(first_station),
			safe_filename(second_station),
		))
		output_path = os.path.join(output_folder, f"{name}.geojson")
		write_geojson_with_fallback(
			gpd.GeoDataFrame(
			features,
			geometry="geometry",
			crs=PROJECTED_CRS,
			).to_crs("EPSG:4326"),
			output_path,
		)
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
	_, platform_station_names_by_index = station_platform_names(filtered_data)
	write_geojson_with_fallback(
		filtered_data.to_crs("EPSG:4326"),
		OUTPUT_FILTERED_DATA,
	)
	print(f"[Progress] Wrote filtered data to {OUTPUT_FILTERED_DATA}.")
	non_track_data = filtered_data[
		~filtered_data["railway"].eq("rail")
	].copy()
	write_geojson_with_fallback(
		non_track_data.to_crs("EPSG:4326"),
		OUTPUT_NON_TRACK_DATA,
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

	# Redefine the graph immediately after loading the filtered rail network.
	split_edges_at_platform_ends(
		graph,
		filtered_data,
		platform_station_names_by_index,
	)
	add_station_end_boundaries_by_platform_end_axis(
		graph,
		filtered_data,
		platform_station_names_by_index,
	)
	classify_junction_locations(graph)
	remove_redundant_station_boundaries(graph)
	remove_unreachable_platform_ends(graph)
	reduce_graph_to_locations(graph)
	print(
		f"[Progress] Reduced graph to {graph.number_of_nodes():,} location "
		f"nodes and {graph.number_of_edges():,} edges."
	)
	station_nodes = station_nodes_from_retained_platforms(graph)
	assign_station_edges(
		graph,
		station_nodes,
		filtered_data,
		platform_station_names_by_index,
	)
	for node, memberships in station_nodes.items():
		graph.nodes[node]["station"] = ", ".join(sorted(memberships)) or "None"
	write_redefined_network_map(graph)
	write_between_station_outputs(graph, station_nodes)
	write_station_outputs(graph, station_nodes)
	print("[Progress] Processing complete.")
	return graph


if __name__ == "__main__":
	build_graph()
