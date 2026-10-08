"""Split an OpenRailwayMap railway GeoJSON into track segments.

The input is expected to contain ``railway=rail`` line features and may
contain ``railway=switch``, ``railway=level_crossing`` and buffer-stop point
features.  The output contains one feature for every track segment and a
separate switch feature collection describing the segments connected at each
switch.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import networkx as nx
from shapely.geometry import LineString, Point
from shapely.ops import unary_union


PROJECTED_CRS = "EPSG:28992"
DEFAULT_NODE_TOLERANCE_M = 0.5
DEFAULT_SWITCH_TOLERANCE_M = 20.0


def _line_parts(geometry):
	if geometry is None or geometry.is_empty:
		return []
	if geometry.geom_type == "LineString":
		return [geometry]
	if geometry.geom_type == "MultiLineString":
		return list(geometry.geoms)
	return []


def _point_features(data, railway_value):
	return [
		row.geometry
		for _, row in data.iterrows()
		if row.geometry is not None
		and not row.geometry.is_empty
		and row.get("railway") == railway_value
		and row.geometry.geom_type == "Point"
	]


def _node_key(point, tolerance):
	return (
		round(point[0] / tolerance) * tolerance,
		round(point[1] / tolerance) * tolerance,
	)


def _angle_between(center, first, second):
	first_vector = (first[0] - center[0], first[1] - center[1])
	second_vector = (second[0] - center[0], second[1] - center[1])
	first_length = math.hypot(*first_vector)
	second_length = math.hypot(*second_vector)
	if not first_length or not second_length:
		return math.pi
	cosine = sum(a * b for a, b in zip(first_vector, second_vector))
	cosine /= first_length * second_length
	return math.acos(max(-1.0, min(1.0, cosine)))


def _closest_node(graph, point):
	return min(graph.nodes, key=lambda node: point.distance(Point(node)))


def build_railway_graph(data, node_tolerance=DEFAULT_NODE_TOLERANCE_M):
	"""Return a noded graph and the source switch/crossing point features."""
	tracks = data[data["railway"].eq("rail")]
	if tracks.empty:
		raise ValueError("The input contains no railway=rail line features")

	lines = [line for geometry in tracks.geometry for line in _line_parts(geometry)]
	noded = unary_union(lines)
	graph = nx.Graph()
	for line in _line_parts(noded):
		coordinates = list(line.coords)
		for start, end in zip(coordinates, coordinates[1:]):
			first = _node_key(start, node_tolerance)
			second = _node_key(end, node_tolerance)
			if first == second:
				continue
			graph.add_edge(first, second)

	switches = _point_features(data, "switch")
	level_crossings = _point_features(data, "level_crossing")
	return graph, switches, level_crossings


def _mark_switches(graph, switch_points, tolerance):
	switches = set()
	for point in switch_points:
		node = _closest_node(graph, point)
		if point.distance(Point(node)) <= tolerance:
			switches.add(node)
	return switches


def _mark_points(graph, points, tolerance):
	marked = set()
	for point in points:
		node = _closest_node(graph, point)
		if point.distance(Point(node)) <= tolerance:
			marked.add(node)
	return marked


def _continuation(graph, previous, current, candidates):
	"""Select the candidate closest to an incoming 180-degree continuation."""
	if previous is None or len(candidates) == 1:
		return candidates[0] if candidates else None
	return min(
		candidates,
		key=lambda candidate: abs(
			math.pi - _angle_between(current, previous, candidate)
		),
	)


def _walk_segment(graph, start, first, switches, level_crossings,
				  assigned_edges, claimed_nodes):
	nodes = []
	if start not in switches:
		nodes.append(start)
		if start not in level_crossings:
			claimed_nodes.add(start)
	previous, current = start, first
	while True:
		assigned_edges.add(frozenset((previous, current)))
		if (current not in switches
				and current in claimed_nodes
				and current not in level_crossings):
			break
		nodes.append(current)
		if current not in switches and current not in level_crossings:
			claimed_nodes.add(current)
		if current in switches or graph.degree(current) == 1:
			break
		candidates = [node for node in graph.neighbors(current) if node != previous]
		if not candidates:
			break
		next_node = _continuation(graph, previous, current, candidates)
		edge = frozenset((current, next_node))
		if edge in assigned_edges:
			break
		previous, current = current, next_node
	return nodes


def split_railway_graph(graph, switch_points, level_crossing_points,
						switch_tolerance=DEFAULT_SWITCH_TOLERANCE_M):
	"""Return numbered segments and switch metadata from a railway graph.

	Switch nodes are excluded from segments.  Level crossings are ordinary
	graph nodes, so a crossing may occur in multiple segment node lists.
	"""
	switches = _mark_switches(graph, switch_points, switch_tolerance)
	level_crossings = _mark_points(graph, level_crossing_points, switch_tolerance)
	assigned_edges = set()
	claimed_nodes = set()
	paths = []

	starts = [
		(switch, neighbor)
		for switch in switches
		for neighbor in graph.neighbors(switch)
	]
	starts.extend(
		(node, neighbor)
		for node in graph
		if graph.degree(node) == 1
		for neighbor in graph.neighbors(node)
	)

	for start, first in starts:
		if frozenset((start, first)) not in assigned_edges:
			paths.append(_walk_segment(
				graph, start, first, switches, level_crossings,
				assigned_edges, claimed_nodes
			))

	for first, second in graph.edges:
		if frozenset((first, second)) not in assigned_edges:
			paths.append(_walk_segment(
				graph, first, second, switches, level_crossings,
				assigned_edges, claimed_nodes
			))

	segments = []
	for number, path in enumerate(paths, start=1):
		segment_nodes = [node for node in path if node not in switches]
		if len(segment_nodes) >= 2:
			segments.append({"number": number, "nodes": segment_nodes})

	switch_records = []
	for switch in sorted(switches):
		switch_neighbors = set(graph.neighbors(switch))
		incident = {
			segment["number"]: segment
			for segment in segments
			if any(node in switch_neighbors for node in segment["nodes"])
		}
		connected_pairs = []
		incident_items = list(incident.items())
		for index, (first_number, first_segment) in enumerate(incident_items):
			first_node = next(
				node for node in first_segment["nodes"] if node in switch_neighbors
			)
			for second_number, second_segment in incident_items[index + 1:]:
				second_node = next(
					node for node in second_segment["nodes"] if node in switch_neighbors
				)
				if _angle_between(switch, first_node, second_node) < math.pi / 2:
					connected_pairs.append([first_number, second_number])
		connected = []
		for pair in connected_pairs:
			connected.extend(pair)
		switch_records.append({
			"node": switch,
			"segments": sorted(set(connected)),
			"connections": connected_pairs,
		})

	return segments, switch_records


def _geojson_features(segments, switches):
	segment_features = [
		{
			"type": "Feature",
			"properties": {"segment": segment["number"]},
			"geometry": LineString(segment["nodes"]).__geo_interface__,
		}
		for segment in segments
	]
	switch_features = [
		{
			"type": "Feature",
			"properties": {"switch": number, "segments": switch["segments"]},
			"geometry": Point(switch["node"]).__geo_interface__,
		}
		for number, switch in enumerate(switches, start=1)
	]
	return segment_features, switch_features


def split_file(input_file, output_segments, output_switches,
			   switch_tolerance=DEFAULT_SWITCH_TOLERANCE_M):
	import geopandas as gpd

	data = gpd.read_file(input_file).to_crs(PROJECTED_CRS)
	graph, switch_points, level_crossing_points = build_railway_graph(data)
	segments, switches = split_railway_graph(
		graph, switch_points, level_crossing_points, switch_tolerance
	)
	segment_features, switch_features = _geojson_features(segments, switches)
	for path, features in ((output_segments, segment_features),
						   (output_switches, switch_features)):
		output = gpd.GeoDataFrame.from_features(features, crs=PROJECTED_CRS)
		output.to_crs("EPSG:4326").to_file(path, driver="GeoJSON")
	return segments, switches


def main():
	parser = argparse.ArgumentParser(description=__doc__)
	parser.add_argument("input_file", default="railway_network_with_switches.geojson",
					nargs="?")
	parser.add_argument("--output-segments", default="railway_segments.geojson")
	parser.add_argument("--output-switches", default="railway_switches.geojson")
	parser.add_argument("--switch-tolerance", type=float,
						default=DEFAULT_SWITCH_TOLERANCE_M)
	args = parser.parse_args()
	split_file(args.input_file, args.output_segments, args.output_switches,
			   args.switch_tolerance)


if __name__ == "__main__":
	main()
