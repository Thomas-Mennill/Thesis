"""Diagnose whether the current experimental graph can join two Amsterdam stations."""

from collections import deque
import colorsys
import json
import math
from pathlib import Path

import geopandas as gpd
import folium
from shapely.geometry import LineString, Point, box

import Split_Network_Into_Sections_Experimental as splitter


FIRST_STATION = "Amsterdam Centraal"
SECOND_STATION = "Amsterdam Muiderpoort"
MAX_DISTANCE_M = splitter.MAX_STATION_PAIR_DISTANCE_M
TEST_OUTPUT_FOLDER = Path("test_amsterdam_muiderpoort_routes")
TEST_MAP_OUTPUT = TEST_OUTPUT_FOLDER / (
	"all_amsterdam_centraal_muiderpoort_routes.html"
)
STUCK_MAP_OUTPUT = TEST_OUTPUT_FOLDER / (
	"unconnected_anchor_paths.html"
)
DIAGNOSTIC_EDGE_IDS = {285853, 297481, 258470}
DIAGNOSTIC_JUNCTION = (122835.084, 487748.461)
DIAGNOSTIC_NODE_TOLERANCE_M = 0.01
DIAGNOSTIC_CROP_MARGIN_M = 10_000
ENDPOINT_TANGENT_DISTANCE_M = 10


def edge_length(edge_data):
	"""Return an edge length for the graph's list-based geometries."""
	length = edge_data.get("length_m")
	if length is not None:
		return length
	return LineString(edge_data["geometry"]).length


def endpoint_direction(edge_data, node):
	"""Return a smoothed local tangent sampled from the edge endpoint."""
	coordinates = list(edge_data.get("geometry", []))
	if len(coordinates) < 2:
		return None
	node_point = Point(node)
	start_distance = node_point.distance(Point(coordinates[0]))
	end_distance = node_point.distance(Point(coordinates[-1]))
	if end_distance < start_distance:
		coordinates.reverse()
	geometry = LineString(coordinates)
	sample_distance = min(
		ENDPOINT_TANGENT_DISTANCE_M,
		geometry.length,
	)
	if sample_distance == 0:
		return None
	sample_point = geometry.interpolate(sample_distance)
	vector = (
		sample_point.x - node[0],
		sample_point.y - node[1],
	)
	length = math.hypot(*vector)
	if length == 0:
		return None
	return vector[0] / length, vector[1] / length


def is_diagnostic_edge(edge_data, first, second):
	"""Identify the original edges or their cropped junction neighborhood."""
	if edge_data.get("edge_id") in DIAGNOSTIC_EDGE_IDS:
		return True
	return any(
		math.hypot(
			node[0] - DIAGNOSTIC_JUNCTION[0],
			node[1] - DIAGNOSTIC_JUNCTION[1],
		) <= DIAGNOSTIC_NODE_TOLERANCE_M
		for node in (first, second)
	)


def build_diagnostic_graph():
	data = gpd.read_file(splitter.INPUT_FILE)
	filtered = splitter.filter_input_features(data)
	_, platform_names = splitter.station_platform_names(filtered)
	target_platform_indexes = [
		index for index, station_name in platform_names.items()
		if station_name in {FIRST_STATION, SECOND_STATION}
	]
	if not target_platform_indexes:
		raise ValueError(
			"Could not find target station platforms for diagnostic crop"
		)
	target_bounds = filtered.loc[
		target_platform_indexes, "geometry"
	].total_bounds
	min_x, min_y, max_x, max_y = target_bounds
	crop = box(
		min_x - DIAGNOSTIC_CROP_MARGIN_M,
		min_y - DIAGNOSTIC_CROP_MARGIN_M,
		max_x + DIAGNOSTIC_CROP_MARGIN_M,
		max_y + DIAGNOSTIC_CROP_MARGIN_M,
	)
	filtered = filtered[filtered.geometry.intersects(crop)].copy()
	print(
		f"[Progress] Cropped diagnostic data to {len(filtered):,} features "
		f"within {DIAGNOSTIC_CROP_MARGIN_M / 1000:g} km of target stations."
	)

	graph = splitter.nx.MultiGraph()
	splitter.add_track_edges(graph, filtered)
	splitter.split_edges_at_platform_ends(graph, filtered, platform_names)
	splitter.add_station_end_boundaries_by_platform_end_axis(
		graph, filtered, platform_names
	)
	splitter.classify_junction_locations(graph)
	splitter.remove_redundant_station_boundaries(graph)
	splitter.remove_unreachable_platform_ends(graph)
	splitter.reduce_graph_to_locations(graph)
	station_nodes = splitter.station_nodes_from_retained_platforms(graph)
	splitter.assign_station_edges(
		graph, station_nodes, filtered, platform_names
	)
	return graph, station_nodes


def route_diagnostics(graph, station_nodes):
	anchors = {
		node for node in graph
		if graph.nodes[node].get("retained_for", set()) & {
			"platform_end", "station_boundary"
		}
	}
	first_anchors = {
		node for node in anchors
		if FIRST_STATION in station_nodes.get(node, set())
	}
	second_anchors = {
		node for node in anchors
		if SECOND_STATION in station_nodes.get(node, set())
	}
	queue = deque()
	visited = {}
	paths = set()
	for start in first_anchors:
		for neighbor, edge_keys in graph[start].items():
			for key, data in edge_keys.items():
				distance = edge_length(data)
				if distance <= MAX_DISTANCE_M:
					state = (start, neighbor, key)
					queue.append((state, distance, (state,), frozenset({state})))
					visited[state] = min(
						distance, visited.get(state, float("inf"))
					)

	rejected_station = 0
	rejected_angle = 0
	while queue:
		state, distance, path, path_states = queue.popleft()
		previous, current, key = state
		if current in second_anchors:
			paths.add(path)
			continue
		incoming = endpoint_direction(
			graph[previous][current][key],
			current,
		)
		if incoming is None:
			continue
		for neighbor, edge_keys in graph[current].items():
			if neighbor == previous:
				continue
			for next_key, edge_data in edge_keys.items():
				neighbor_stations = station_nodes.get(neighbor, set())
				if is_diagnostic_edge(edge_data, current, neighbor):
					print(
						"[Diagnostic] edge "
						f"{edge_data['edge_id']}: current={current}, "
						f"neighbor={neighbor}, "
						f"neighbor stations={sorted(neighbor_stations)}"
					)
				if neighbor_stations and neighbor not in second_anchors:
					rejected_station += 1
					continue
				outgoing = endpoint_direction(edge_data, current)
				if outgoing is None:
					continue
				angle = splitter.angle_between_vectors(
					(0, 0), incoming, outgoing
				)
				if angle <= splitter.math.radians(90):
					rejected_angle += 1
					continue
				next_distance = distance + edge_length(edge_data)
				if next_distance > MAX_DISTANCE_M:
					continue
				next_state = (current, neighbor, next_key)
				if next_state in path_states:
					continue
				visited[next_state] = min(
					next_distance, visited.get(next_state, float("inf"))
				)
				queue.append((
					next_state,
					next_distance,
					path + (next_state,),
					path_states | {next_state},
				))

	print(f"Graph: {graph.number_of_nodes():,} nodes, "
	      f"{graph.number_of_edges():,} edges")
	print(f"{FIRST_STATION} possible anchors: {len(first_anchors)}")
	print(f"{SECOND_STATION} possible anchors: {len(second_anchors)}")
	print(f"Visited directed states: {len(visited):,}")
	print(f"Rejected station-assigned transitions: {rejected_station:,}")
	print(f"Rejected angle transitions: {rejected_angle:,}")
	print(f"Possible paths found: {len(paths):,}")
	used_first_anchors = {path[0][0] for path in paths}
	used_second_anchors = {path[-1][1] for path in paths}
	report_unrouted_anchors(
		graph,
		FIRST_STATION,
		first_anchors - used_first_anchors,
		used_first_anchors,
	)
	report_unrouted_anchors(
		graph,
		SECOND_STATION,
		second_anchors - used_second_anchors,
		used_second_anchors,
	)
	unrouted = (
		(FIRST_STATION, first_anchors - used_first_anchors),
		(SECOND_STATION, second_anchors - used_second_anchors),
	)
	stuck_paths = trace_unconnected_anchor_paths(
		graph,
		station_nodes,
		anchors,
		unrouted,
	)
	print(f"Unconnected anchor paths found: {len(stuck_paths):,}")
	for reason, count in sorted(
		count_path_reasons(stuck_paths).items()
	):
		print(f"  {reason}: {count:,}")
	splitter.clear_output_directory(str(TEST_OUTPUT_FOLDER))
	write_unconnected_map(graph, stuck_paths)
	write_combined_map(
		graph,
		paths,
		first_anchors,
		second_anchors,
	)
	if not paths:
		print("RESULT: no permitted route found.")
		return

	for path_number, path in enumerate(sorted(paths), start=1):
		write_path_output(graph, path, path_number)
		print(
			f"Path {path_number}: {len(path)} edges, "
			f"{path[0][0]} -> {path[-1][1]}"
		)


def report_unrouted_anchors(graph, station_name, unrouted, routed):
	"""Report route coverage and graph degree for each unrouted anchor."""
	print(
		f"{station_name} anchors with a route: {len(routed):,}; "
		f"without a route: {len(unrouted):,}"
	)
	for node in sorted(unrouted):
		print(
			f"  Unrouted {station_name} anchor: {node}; "
			f"degree={graph.degree(node)}"
		)


def trace_unconnected_anchor_paths(graph, station_nodes, anchors, unrouted):
	"""Trace every branch from unrouted anchors until it cannot continue."""
	stuck_paths = set()
	for station_name, starts in unrouted:
		for start in starts:
			queue = deque()
			for neighbor, edge_keys in graph[start].items():
				for key, edge_data in edge_keys.items():
					state = (start, neighbor, key)
					queue.append((
						state,
						edge_length(edge_data),
						(state,),
						frozenset({state}),
					))
			while queue:
				state, distance, path, path_states = queue.popleft()
				previous, current, key = state
				if current in anchors and current != start:
					continue
				candidates = []
				connected_anchor = False
				blocked_station = False
				blocked_angle = False
				blocked_distance = False
				blocked_cycle = False
				incoming = endpoint_direction(
					graph[previous][current][key],
					current,
				)
				if incoming is None:
					stuck_paths.add((station_name, "missing_direction", path))
					continue
				for neighbor, edge_keys in graph[current].items():
					if neighbor == previous:
						continue
					for next_key, edge_data in edge_keys.items():
						if neighbor in anchors:
							connected_anchor = True
							continue
						if is_diagnostic_edge(edge_data, current, neighbor):
							print(
								"[Diagnostic] trace edge "
								f"{edge_data['edge_id']}: current={current}, "
								f"neighbor={neighbor}, "
								f"neighbor stations="
								f"{sorted(station_nodes.get(neighbor, set()))}"
							)
						if station_nodes.get(neighbor):
							blocked_station = True
							continue
						outgoing = endpoint_direction(edge_data, current)
						if outgoing is None:
							if is_diagnostic_edge(edge_data, current, neighbor):
								print(
									"[Diagnostic] trace edge "
									f"{edge_data['edge_id']}: "
									"rejected=missing_direction"
								)
							continue
						angle_degrees = math.degrees(
							splitter.angle_between_vectors(
								(0, 0),
								incoming,
								outgoing,
							)
						)
						if is_diagnostic_edge(edge_data, current, neighbor):
							print(
								"[Diagnostic] trace edge "
								f"{graph[previous][current][key]['edge_id']} -> "
								f"{edge_data['edge_id']}: outward_ray_angle="
								f"{angle_degrees:.6f} degrees, "
								f"distance={distance + edge_length(edge_data):.3f}, "
								f"cycle={((current, neighbor, next_key) in path_states)}"
							)
						if angle_degrees <= 90:
							if is_diagnostic_edge(edge_data, current, neighbor):
								print(
									"[Diagnostic] trace edge "
									f"{edge_data['edge_id']}: "
									"rejected=angle_rule"
								)
							blocked_angle = True
							continue
						next_distance = distance + edge_length(edge_data)
						if next_distance > MAX_DISTANCE_M:
							blocked_distance = True
							continue
						next_state = (current, neighbor, next_key)
						if next_state in path_states:
							blocked_cycle = True
							continue
						candidates.append((
							next_state,
							next_distance,
							path + (next_state,),
							path_states | {next_state},
						))
				if candidates:
					queue.extend(candidates)
					continue
				if connected_anchor:
					continue
				reason = "dead_end"
				if blocked_station:
					reason = "station_assigned_node"
				elif blocked_angle:
					reason = "angle_rule"
				elif blocked_distance:
					reason = "distance_limit"
				elif blocked_cycle:
					reason = "cycle"
				stuck_paths.add((station_name, reason, path))
	return tuple(stuck_paths)


def count_path_reasons(stuck_paths):
	"""Count terminal reasons for unconnected diagnostic paths."""
	counts = {}
	for _, reason, _ in stuck_paths:
		counts[reason] = counts.get(reason, 0) + 1
	return counts


def route_color(path_number, path_count):
	"""Return a distinct deterministic color for a route."""
	hue = (path_number - 1) / max(path_count, 1)
	red, green, blue = colorsys.hsv_to_rgb(hue, 0.85, 0.9)
	return "#{:02x}{:02x}{:02x}".format(
		round(red * 255),
		round(green * 255),
		round(blue * 255),
	)


def write_combined_map(graph, paths, first_anchors, second_anchors):
	"""Write one interactive map containing every discovered route."""
	route_features = []
	for path_number, path in enumerate(sorted(paths), start=1):
		color = route_color(path_number, len(paths))
		for first, second, key in path:
			edge_data = graph[first][second][key]
			geometry = gpd.GeoSeries(
				[LineString(edge_data["geometry"])],
				crs=splitter.PROJECTED_CRS,
			).to_crs("EPSG:4326").iloc[0]
			route_features.append({
				"type": "Feature",
				"properties": {
					"path_number": path_number,
					"color": color,
				},
				"geometry": geometry.__geo_interface__,
			})

	anchor_features = []
	for station, anchors in (
		(FIRST_STATION, first_anchors),
		(SECOND_STATION, second_anchors),
	):
		for node in anchors:
			point = gpd.GeoSeries(
				[Point(node)], crs=splitter.PROJECTED_CRS
			).to_crs("EPSG:4326").iloc[0]
			anchor_features.append((station, point))

	route_data = {
		"type": "FeatureCollection",
		"features": route_features,
	}
	all_points = [
		point for _, point in anchor_features
	]
	center = (
		sum(point.y for point in all_points) / len(all_points),
		sum(point.x for point in all_points) / len(all_points),
	)
	network_map = folium.Map(location=center, zoom_start=13)
	folium.GeoJson(
		json.dumps(route_data),
		name="All possible routes",
		style_function=lambda feature: {
			"color": feature["properties"]["color"],
			"weight": 2,
			"opacity": 0.45,
		},
		tooltip=(
			folium.GeoJsonTooltip(
				fields=["path_number"],
				aliases=["Path"],
			)
			if route_features
			else None
		),
	).add_to(network_map)
	for station, point in anchor_features:
		color = "#d73027" if station == FIRST_STATION else "#4575b4"
		folium.CircleMarker(
			location=(point.y, point.x),
			radius=5,
			color=color,
			fill=True,
			fill_color=color,
			fill_opacity=1,
			tooltip=station,
		).add_to(network_map)
	folium.LayerControl().add_to(network_map)
	network_map.save(TEST_MAP_OUTPUT)
	print(f"Combined route map: {TEST_MAP_OUTPUT}")


def write_unconnected_map(graph, stuck_paths):
	"""Write a map of branches that failed to connect two anchors."""
	if not stuck_paths:
		print("No unconnected anchor paths to map.")
		return
	colors = {
		"dead_end": "#d73027",
		"station_assigned_node": "#fc8d59",
		"angle_rule": "#fee090",
		"distance_limit": "#91bfdb",
		"cycle": "#4575b4",
		"missing_direction": "#542788",
	}
	features = []
	for path_number, (station, reason, path) in enumerate(
		sorted(stuck_paths), start=1
	):
		for first, second, key in path:
			edge_data = graph[first][second][key]
			geometry = gpd.GeoSeries(
				[LineString(edge_data["geometry"])],
				crs=splitter.PROJECTED_CRS,
			).to_crs("EPSG:4326").iloc[0]
			features.append({
				"type": "Feature",
				"properties": {
					"path_number": path_number,
					"start_station": station,
					"reason": reason,
				},
				"geometry": geometry.__geo_interface__,
			})
	start_nodes = {
		path[2][0][0] for path in stuck_paths
	}
	points = [Point(node) for node in start_nodes]
	projected_points = gpd.GeoSeries(
		points, crs=splitter.PROJECTED_CRS
	).to_crs("EPSG:4326")
	route_data = {
		"type": "FeatureCollection",
		"features": features,
	}
	center = (
		projected_points.y.mean(),
		projected_points.x.mean(),
	)
	network_map = folium.Map(location=center, zoom_start=13)
	folium.GeoJson(
		json.dumps(route_data),
		name="Unconnected anchor paths",
		style_function=lambda feature: {
			"color": colors.get(
				feature["properties"]["reason"], "#000000"
			),
			"weight": 3,
			"opacity": 0.6,
		},
		tooltip=folium.GeoJsonTooltip(
			fields=["path_number", "start_station", "reason"],
			aliases=["Path", "Start station", "Why it stopped"],
		),
	).add_to(network_map)
	network_map.save(STUCK_MAP_OUTPUT)
	print(f"Unconnected anchor map: {STUCK_MAP_OUTPUT}")


def write_path_output(graph, path, path_number):
	"""Write one diagnostic path to the dedicated test output folder."""
	features = []
	nodes = {path[0][0]} | {state[1] for state in path}
	for node in nodes:
		features.append({
			"feature_type": "node",
			"path_number": path_number,
			"geometry": Point(node),
		})
	for first, second, key in path:
		edge_data = graph[first][second][key]
		features.append({
			"feature_type": "track",
			"path_number": path_number,
			"geometry": LineString(edge_data["geometry"]),
		})
	output = gpd.GeoDataFrame(
		features, geometry="geometry", crs=splitter.PROJECTED_CRS
	).to_crs("EPSG:4326")
	filename = (
		f"path_{path_number:04d}_"
		f"{splitter.safe_filename(FIRST_STATION)}_to_"
		f"{splitter.safe_filename(SECOND_STATION)}.geojson"
	)
	splitter.write_geojson_with_fallback(
		output, str(TEST_OUTPUT_FOLDER / filename)
	)


if __name__ == "__main__":
	if not Path(splitter.INPUT_FILE).is_file():
		raise FileNotFoundError(splitter.INPUT_FILE)
	route_diagnostics(*build_diagnostic_graph())
