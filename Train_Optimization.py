import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import pickle
import csv
import re
from pathlib import Path


def load_train_schedule(csv_file):
	"""Load a three-column-per-train schedule into a tidy DataFrame.

	Each train occupies three columns: event, station, and time. The train ID
	is in the first row of the station column; subsequent rows contain arrival
	and departure events. The returned DataFrame has one row per event with
	``train_id``, ``event``, ``station``, and ``time`` columns.
	"""
	schedule = pd.read_csv(
		csv_file,
		header=None,
		dtype=str,
		keep_default_na=False,
	)

	if schedule.shape[1] % 3 != 0:
		raise ValueError("The schedule CSV must contain groups of three columns")

	stops = []
	for event_column in range(0, schedule.shape[1], 3):
		station_column = event_column + 1
		time_column = event_column + 2
		train_id = schedule.iat[0, station_column].strip()
		if not train_id:
			raise ValueError(
				f"Missing train ID in row 1, column {station_column + 1}"
			)

		for row in range(1, schedule.shape[0]):
			event = schedule.iat[row, event_column].strip()
			station = schedule.iat[row, station_column].strip()
			stop_time = schedule.iat[row, time_column].strip()
			if not event and not station and not stop_time:
				continue
			if not event or not station or not stop_time:
				raise ValueError(
					f"Event, station, and time must all be present at row "
					f"{row + 1}, columns {event_column + 1}-{time_column + 1}"
				)
			stops.append(
				{
					"train_id": train_id,
					"event": event,
					"station": station,
					"time": stop_time,
				}
			)

	return pd.DataFrame(
		stops,
		columns=["train_id", "event", "station", "time"],
	)

def load_rolling_stock_data(csv_file="rolling_stock_data.csv"):
	"""Load rolling-stock data into the global ``U`` DataFrame.

	The CSV stores rolling-stock types in columns and their properties in
	rows, so the first column is used as the DataFrame index.
	"""
	global U

	U = pd.read_csv(csv_file, index_col=0, encoding="utf-8-sig")
	return U


def load_braking_times(csv_file="braking_times_by_node.csv"):
	"""Load braking times indexed by route-node ID and route ID."""
	braking_times = {}
	with Path(csv_file).open(
		encoding="utf-8-sig",
		newline="",
	) as braking_file:
		reader = csv.reader(braking_file)
		header = next(reader, None)
		if not header or len(header) < 2:
			raise ValueError(
				f"Braking-time file {csv_file!r} has no route columns"
			)
		route_ids = []
		for column in header[1:]:
			label = column.strip()
			if not label.casefold().startswith("route "):
				raise ValueError(
					f"Invalid braking-time route column {column!r}"
				)
			route_ids.append(label[6:].strip())
		for row_number, row in enumerate(reader, start=2):
			if not row or not row[0].strip():
				continue
			node_label = row[0].strip()
			if not node_label.casefold().startswith("node "):
				raise ValueError(
					f"Invalid braking-time node label {node_label!r} "
					f"on row {row_number}"
				)
			try:
				node_id = int(node_label[5:].strip())
			except ValueError as error:
				raise ValueError(
					f"Invalid braking-time node ID {node_label!r} "
					f"on row {row_number}"
				) from error
			if node_id in braking_times:
				raise ValueError(
					f"Duplicate braking-time node ID {node_id}"
				)
			values = {}
			for column, route_id in enumerate(route_ids, start=1):
				if column >= len(row) or not row[column].strip():
					continue
				try:
					values[route_id] = float(row[column])
				except ValueError as error:
					raise ValueError(
						f"Invalid braking time for node {node_id}, "
						f"route {route_id!r}"
					) from error
			braking_times[node_id] = values
	return braking_times


def calculate_braking_times(train_type, rolling_stock_data=None):
	"""Return full-stop braking times sampled at 0.1 m/s.

	The returned value at index ``i`` is the time in seconds required to stop
	from ``i * 0.1`` m/s. Maximum speed is read in km/h and converted
	to m/s. 	The braking acceleration from ``rolling_stock_data`` is assumed to be a
	positive constant deceleration.
	"""
	if rolling_stock_data is None:
		rolling_stock_data = U
	if not isinstance(rolling_stock_data, pd.DataFrame):
		raise TypeError("rolling_stock_data must be a pandas DataFrame")
	if train_type not in rolling_stock_data.columns:
		raise KeyError(
			f"Train type {train_type!r} is not present in rolling-stock data"
		)

	property_rows = {
		str(index).strip().casefold().replace("_", " "): index
		for index in rolling_stock_data.index
	}
	required_properties = {
		"max speed (km/h)": "maximum speed",
		"brake acceleration (m/s^2)": "brake acceleration",
	}
	missing_properties = [
		label for key, label in required_properties.items()
		if key not in property_rows
	]
	if missing_properties:
		raise KeyError(
			"Rolling-stock data is missing required properties: "
			+ ", ".join(sorted(missing_properties))
		)

	try:
		max_speed_kmh = float(
			rolling_stock_data.at[property_rows["max speed (km/h)"], train_type]
		)
		brake_acceleration = float(
			rolling_stock_data.at[
				property_rows["brake acceleration (m/s^2)"], train_type
			]
		)
	except (TypeError, ValueError) as error:
		raise ValueError(
			f"Train type {train_type!r} has invalid speed or braking data"
		) from error
	if not np.isfinite(max_speed_kmh) or max_speed_kmh < 0:
		raise ValueError("Maximum speed must be finite and non-negative")
	if not np.isfinite(brake_acceleration) or brake_acceleration <= 0:
		raise ValueError("Brake acceleration must be finite and positive")

	max_speed = max_speed_kmh / 3.6
	speeds = np.arange(
		0,
		np.ceil(max_speed / 0.1) * 0.1 + 0.05,
		0.1,
	)
	return speeds / brake_acceleration


def calculate_startup_travel_time(train_type, rolling_stock_data=None, time_step=0.01,):
	"""Return the time for a train to travel one train length from rest.

	Conventional trains use the Davis Equation
	``A + Bv + Cv²`` for resistance. Maglev trains use the traditional
	aerodynamic equation ``0.5 * rho * Cd * area * v²``. Combined types such
	as ``"VIRM-6+VIRM-6"`` are resolved by combining their component columns.
	"""
	if rolling_stock_data is None:
		rolling_stock_data = U
	if not isinstance(rolling_stock_data, pd.DataFrame):
		raise TypeError("rolling_stock_data must be a pandas DataFrame")
	if not np.isfinite(time_step) or time_step <= 0:
		raise ValueError("time_step must be finite and positive")

	normalized_rows = {
		str(index).strip().casefold().replace("_", " "): index
		for index in rolling_stock_data.index
	}
	components = [
		component.strip()
		for component in str(train_type).split("+")
		if component.strip()
	]
	if not components:
		raise ValueError("train_type cannot be empty")
	columns = {}
	for component in components:
		matches = [
			column for column in rolling_stock_data.columns
			if str(column).strip().casefold() == component.casefold()
		]
		if not matches:
			raise KeyError(
				f"Train type {component!r} is not present in rolling-stock data"
			)
		columns[component] = matches[0]

	def values(row_name):
		row = normalized_rows.get(row_name)
		if row is None:
			return []
		result = []
		for component in components:
			value = rolling_stock_data.at[row, columns[component]]
			if pd.isna(value) or not str(value).strip():
				continue
			result.append(float(value))
		return result

	mass_values = values("mass (kg)")
	length_values = values("length (m)")
	power_values = values("power (w)")
	tractive_force_values = values("max tractive force (n)")
	max_speed_values = values("max speed (km/h)")
	if not mass_values or not length_values or not power_values:
		raise ValueError(
			f"Train type {train_type!r} lacks mass, length, or power data"
		)
	if not tractive_force_values or not max_speed_values:
		raise ValueError(
			f"Train type {train_type!r} lacks force or maximum-speed data"
		)

	mass = sum(mass_values)
	length = sum(length_values)
	power = sum(power_values)
	max_tractive_force = sum(tractive_force_values)
	max_speed = min(max_speed_values) / 3.6
	is_maglev = all(
		component.casefold() == "maglev"
		for component in components
	)
	if is_maglev:
		drag_values = values("drag coefficient")
		width_values = values("width (m)")
		height_values = values("height (m)")
		if not drag_values or not width_values or not height_values:
			raise ValueError(
				f"Maglev type {train_type!r} lacks aerodynamic data"
			)
		air_density = 1.204
		drag_coefficient = max(drag_values)
		frontal_area = max(width_values) * max(height_values)

		def resistance_at(speed):
			return (
				0.5
				* air_density
				* drag_coefficient
				* frontal_area
				* speed**2
			)
	else:
		quadratic_values = values("quadratic resistance (ns^2/m^2)")
		linear_values = values("linear resistance (ns/m)")
		constant_values = values("constant resistance (n)")
		if not (
			quadratic_values
			and linear_values
			and constant_values
		):
			raise ValueError(
				f"Conventional type {train_type!r} lacks Davis Equation data"
			)
		quadratic_resistance = sum(quadratic_values)
		linear_resistance = sum(linear_values)
		constant_resistance = sum(constant_values)

		def resistance_at(speed):
			return (
				constant_resistance
				+ linear_resistance * speed
				+ quadratic_resistance * speed**2
			)

	distance = 0.0
	speed = 0.0
	elapsed = 0.0
	while distance < length:
		resistance = resistance_at(speed)
		available_force = min(
			max_tractive_force,
			power / max(speed, 1e-6),
		)
		acceleration = max((available_force - resistance) / mass, 0.0)
		if speed >= max_speed and acceleration <= 0:
			acceleration = 0.0
		new_speed = min(speed + acceleration * time_step, max_speed)
		distance += (speed + new_speed) * 0.5 * time_step
		speed = new_speed
		elapsed += time_step
		if elapsed > 3600:
			raise RuntimeError(
				f"Startup travel-time calculation did not converge for "
				f"{train_type!r}"
			)
	return elapsed


def assign_train_events(schedule):
	"""Assign schedule rows to the departure and arrival event sets.

	Each entry in the returned and module-level event lists is a
	``(train_id, station, time, event_number)`` tuple. Train and event numbers
	start at one for each train route. The event number is encoded as a four-digit
	string: two digits for the train number followed by two digits for the
	event number within that route. The input must contain the columns created by
	:func:`load_train_schedule`.
	"""
	if not {"train_id", "event", "station", "time"}.issubset(schedule.columns):
		raise ValueError(
			"The schedule must contain train_id, event, station, and time columns"
		)

	departure_events = []
	arrival_events = []
	route_event_numbers = {}
	for row in schedule.itertuples(index=False):
		event = str(row.event).strip().lower()
		try:
			train_number = int(str(row.train_id).strip())
		except ValueError as error:
			raise ValueError(
				f"Train ID {row.train_id!r} must be a number from 1 to 99"
			) from error
		if not 1 <= train_number <= 99:
			raise ValueError(
				f"Train ID {row.train_id!r} must be a number from 1 to 99"
			)

		event_number = route_event_numbers.get(train_number, 1)
		if event_number > 99:
			raise ValueError(
				f"Train {train_number} has more than 100 events; "
				"the event number only supports two digits"
			)
		event_id = f"{train_number:02d}{event_number:02d}"
		if event in {"departure", "depart"}:
			entry = (row.train_id, row.station, row.time, event_id)
			departure_events.append(entry)
		elif event in {"arrival", "arrive"}:
			entry = (row.train_id, row.station, row.time, event_id)
			arrival_events.append(entry)
		else:
			raise ValueError(
				f"Unknown event {row.event!r}; expected 'Arrive' or 'Depart'"
			)
		route_event_numbers[train_number] = event_number + 1

	global V_departure, V_arrival
	V_departure = departure_events
	V_arrival = arrival_events
	return V_departure, V_arrival

def assign_event_edges(schedule):
	"""Create dwell, travel, and headway edge sets from a train schedule.

	Each edge is a tuple containing two event tuples. Dwell and travel edges
	connect consecutive events on the same train route. Headway edges connect a
	departure with every later arrival at the same station. Restricted headway
	edges retain the smallest time difference for each departure, including ties.
	"""
	departure_events, arrival_events = assign_train_events(schedule)
	all_events = departure_events + arrival_events

	def event_time(event):
		time_text = str(event[2]).strip()
		if time_text.count(":") == 1:
			time_text = f"{time_text}:00"
		return pd.to_timedelta(time_text)

	events_by_train = {}
	for event in all_events:
		events_by_train.setdefault(event[0], []).append(event)

	dwell_edges = []
	travel_edges = []
	for train_events in events_by_train.values():
		train_events.sort(key=lambda event: int(event[3]) % 100)
		for first, second in zip(train_events, train_events[1:]):
			if first in arrival_events and second in departure_events:
				dwell_edges.append((first, second))
			elif first in departure_events and second in arrival_events:
				travel_edges.append((first, second))

	headway_edges = []
	for departure in departure_events:
		for arrival in arrival_events:
			if (
				departure[1] == arrival[1]
				and event_time(arrival) >= event_time(departure)
			):
				headway_edges.append((departure, arrival))

	restricted_headway_edges = []
	for departure in departure_events:
		candidates = [
			edge for edge in headway_edges if edge[0] == departure
		]
		if candidates:
			minimum_difference = min(
				event_time(arrival) - event_time(departure)
				for _, arrival in candidates
			)
			restricted_headway_edges.extend(
				edge
				for edge in candidates
				if event_time(edge[1]) - event_time(edge[0])
				== minimum_difference
			)

	global E_dwell, E_travel, E_headway, E_headway_restricted
	E_dwell = dwell_edges
	E_travel = travel_edges
	E_headway = headway_edges
	E_headway_restricted = restricted_headway_edges
	return E_dwell, E_travel, E_headway, E_headway_restricted

def load_track_segments(pickle_file):
	"""Load track-segment attributes and connectivity from a pickled DiGraph.

	Node attributes must contain a positive integer ``id``. Track IDs are
	one-based, while NumPy indices in ``L`` are zero-based.
	"""
	with open(pickle_file, "rb") as graph_file:
		graph = pickle.load(graph_file)

	track_segments = {}
	for node, attributes in graph.nodes(data=True):
		if "id" not in attributes:
			raise ValueError(f"Node {node!r} has no 'id' attribute")

		track_id = attributes["id"]
		if not isinstance(track_id, (int, np.integer)) or track_id < 1:
			raise ValueError(
				f"Node {node!r} has invalid track id {track_id!r}; "
				"IDs must be positive integers"
			)
		if track_id in track_segments:
			raise ValueError(f"Duplicate track id {track_id}")
		track_segments[track_id] = attributes

	if not track_segments:
		return track_segments, np.zeros((0, 0), dtype=int)

	highest_track_id = max(track_segments)
	L = np.zeros((highest_track_id, highest_track_id), dtype=int)
	node_ids = {node: attributes["id"] for node, attributes in graph.nodes(data=True)}

	for source, target in graph.edges:
		source_id = node_ids[source]
		target_id = node_ids[target]
		L[source_id - 1, target_id - 1] = 1

	diagonal = np.diag(L)
	if np.any(diagonal == 1):
		self_connected_ids = np.flatnonzero(diagonal == 1) + 1
		raise ValueError(
			"L contains self-connectivity for track ID(s): "
			f"{self_connected_ids.tolist()}"
		)

	return track_segments, L


def load_and_populate_route_events(route_graph_file="case_study_route_graphs.pkl", existing_schedule_file="existing_rail_routes.csv", additional_schedule_file="additional_rail_routes.csv", route_data_file="route_data.csv", braking_time_file="braking_times_by_node.csv",):
	"""Load route graphs and schedules and populate all route event sets.

	The returned ``V_*`` values contain event IDs, while ``Events`` contains
	the corresponding :class:`event` objects. ``new_rail_routes.csv`` is used
	automatically when the requested additional-schedule filename is absent.
	Graph nodes are expected to expose ``node_id``, ``retained_for``, and
	``platform_stations`` attributes where applicable.
	"""
	global Events, V_initial, V_additional, N_initial, N_additional
	global I, pi, s, t, q, l, tw_min, tw_max
	global running_time_max, running_time_min, dwell_time_max, dwell_time_min
	global headway_time_min, p, rho
	global E_dwell, E_running, E_travel, E_headway
	global train_priorities, startup_travel_times, braking_times
	global V_split, V_merge, V_crossing, V_run_through
	global V_departure, V_arrival, V_rt

	with Path(route_graph_file).open("rb") as graph_file:
		route_graphs = pickle.load(graph_file)
	if not isinstance(route_graphs, dict) or not route_graphs:
		raise ValueError("The route graph pickle must contain route graphs")

	def schedule_path(filename, fallback=None):
		path = Path(filename)
		if not path.exists() and fallback is not None:
			path = Path(fallback)
		if not path.exists():
			raise FileNotFoundError(path)
		return path

	existing_path = schedule_path(existing_schedule_file)
	additional_path = schedule_path(
		additional_schedule_file,
		"new_rail_routes.csv",
	)
	schedules = {
		"existing": load_train_schedule(existing_path),
		"additional": load_train_schedule(additional_path),
	}
	schedule_rows = pd.concat(schedules.values(), ignore_index=True)
	with Path(route_data_file).open(
		encoding="utf-8-sig",
		newline="",
	) as route_data_handle:
		route_data_rows = list(csv.reader(route_data_handle))
	priority_row = next(
		(
			row for row in route_data_rows
			if row and row[0].strip().casefold() == "priority"
		),
		None,
	)
	if priority_row is None:
		raise ValueError(
			f"Route data file {route_data_file!r} has no Priority row"
		)
	train_priorities = {}
	for column, train_id in enumerate(route_data_rows[0][1:], start=1):
		if not train_id.strip():
			continue
		if column >= len(priority_row):
			raise ValueError(
				f"Route data file {route_data_file!r} has no priority for "
				f"train {train_id!r}"
			)
		try:
			train_priorities[str(train_id).strip()] = float(
				priority_row[column]
			)
		except ValueError as error:
			raise ValueError(
				f"Invalid priority for train {train_id!r}"
			) from error
	train_type_row = next(
		(
			row for row in route_data_rows
			if row and row[0].strip().casefold() == "train type"
		),
		None,
	)
	if train_type_row is None:
		raise ValueError(
			f"Route data file {route_data_file!r} has no Train Type row"
		)
	route_train_types = {}
	for column, train_id in enumerate(route_data_rows[0][1:], start=1):
		if not train_id.strip():
			continue
		if column >= len(train_type_row):
			raise ValueError(
				f"Route data file {route_data_file!r} has no train type for "
				f"train {train_id!r}"
			)
		route_train_types[str(train_id).strip()] = train_type_row[column].strip()
	rolling_stock_data = load_rolling_stock_data()
	braking_times = load_braking_times(braking_time_file)
	startup_travel_times = {
		train_type: calculate_startup_travel_time(
			train_type,
			rolling_stock_data,
		)
		for train_type in set(route_train_types.values())
	}
	initial_routes = {
		str(value).strip() for value in schedules["existing"]["train_id"]
	}
	route_ids = {
		str(route_id).strip(): route_id for route_id in route_graphs
	}

	def route_graph(train_id):
		key = str(train_id).strip()
		if key not in route_ids:
			raise KeyError(f"No route graph found for train {train_id!r}")
		return route_graphs[route_ids[key]]

	def key(value):
		return " ".join(str(value).split()).casefold()

	def node_id(graph, node):
		return graph.nodes[node].get("node_id", node)

	def markers(graph, node):
		attributes = graph.nodes[node]
		retained = set(attributes.get("retained_for", ()))
		return retained & {"station_boundary", "platform_edge", "platform_end"}

	def station_names(graph, node):
		attributes = graph.nodes[node]
		names = set(attributes.get("platform_stations", ()))
		station = attributes.get("station")
		if station not in (None, "None", ""):
			names.add(station)
		return {key(name) for name in names}

	def ordered_edges(graph):
		route_nodes = list(graph.graph.get("route_node_coordinates", ()))
		assigned = list(graph.graph.get("assigned_edges", ()))
		records = []
		used = set()
		for edge_id in assigned:
			candidates = [
				(first, second, data)
				for first, second, data in graph.edges(data=True)
				if str(data.get("edge_id")) == str(edge_id)
				and id(data) not in used
			]
			if not candidates:
				continue
			if records:
				previous = records[-1]["destination"]
				candidates.sort(
					key=lambda item: item[0] != previous
				)
			first, second, data = candidates[0]
			used.add(id(data))
			records.append({
				"edge_id": data.get("edge_id", edge_id),
				"origin": first,
				"destination": second,
				"speed_limit": data.get(
					"speed_limit_kmh",
					data.get("speed_limit", data.get("maxspeed")),
				),
			})
		if records:
			return records
		for first, second, data in graph.edges(data=True):
			records.append({
				"edge_id": data.get("edge_id"),
				"origin": first,
				"destination": second,
				"speed_limit": data.get(
					"speed_limit_kmh",
					data.get("speed_limit", data.get("maxspeed")),
				),
			})
		return records

	route_data = {}
	for train_id in route_graphs:
		graph = route_graph(train_id)
		nodes = []
		for edge in ordered_edges(graph):
			if not nodes:
				nodes.append(edge["origin"])
			nodes.append(edge["destination"])
		if not nodes:
			nodes = list(graph.graph.get("route_node_coordinates", ()))
		distance = {node: index for index, node in enumerate(nodes)}
		route_rows = schedule_rows[
			schedule_rows["train_id"].astype(str).str.strip()
			== str(train_id).strip()
		]
		known_times = [
			pd.to_timedelta(row.time)
			for row in route_rows.itertuples()
		]
		route_data[str(train_id).strip()] = {
			"graph": graph,
			"nodes": nodes,
			"distance": distance,
			"edges": ordered_edges(graph),
			"rows": route_rows,
			"start_time": min(known_times) if known_times else None,
			"end_time": max(known_times) if known_times else None,
		}

	def speed_limit_mps(value, route_id, node):
		if value is None:
			raise ValueError(
				f"No speed limit found for route {route_id!r}, node {node!r}"
			)
		text = str(value).strip().lower().replace(",", ".")
		match = re.search(r"-?\d+(?:\.\d+)?", text)
		if match is None:
			raise ValueError(
				f"Invalid speed limit {value!r} for route {route_id!r}, "
				f"node {node!r}"
			)
		speed = float(match.group())
		if "mph" in text:
			speed *= 1.609344
		return speed / 3.6

	def node_metadata(route_id, route):
		graph = route["graph"]
		metadata = {}
		for index, node in enumerate(route["nodes"]):
			incoming = next(
				(edge for edge in route["edges"]
				 if edge["destination"] == node),
				None,
			)
			outgoing = next(
				(edge for edge in route["edges"]
				 if edge["origin"] == node),
				None,
			)
			attributes = graph.nodes[node]
			speed_value = (
				incoming["speed_limit"]
				if incoming and incoming["speed_limit"] is not None
				else attributes.get(
					"speed_limit_kmh",
					attributes.get(
						"speed_limit",
						attributes.get("maxspeed"),
					),
				)
			)
			incoming_speed_limit = speed_limit_mps(
				speed_value,
				route_id,
				node,
			)
			metadata[node] = {
				"ordinal": index + 1,
				"incoming_speed_limit": incoming_speed_limit,
				"speed_limit": incoming_speed_limit,
				"station": bool(station_names(graph, node)),
				"switch": "switch" in set(
					attributes.get("retained_for", ())
				),
				"crossing": "rail_rail_crossing" in set(
					attributes.get("retained_for", ())
				),
				"incoming_edge": (
					incoming["edge_id"] if incoming else None
				),
				"outgoing_edge": (
					outgoing["edge_id"] if outgoing else None
				),
			}
		for index, node in enumerate(route["nodes"]):
			metadata[node]["speed_change"] = (
				index > 0
				and not np.isclose(
					metadata[node]["speed_limit"],
					metadata[route["nodes"][index - 1]]["speed_limit"],
				)
			)
		return metadata

	for route_id, route in route_data.items():
		route["node_metadata"] = node_metadata(route_id, route)

	def vehicle_length(train_type):
		components = [
			component.strip()
			for component in str(train_type).split("+")
			if component.strip()
		]
		row = next(
			(
				index for index in rolling_stock_data.index
				if str(index).strip().casefold() == "length (m)"
			),
			None,
		)
		if row is None:
			raise ValueError("Rolling-stock data has no Length (m) row")
		length = 0.0
		for component in components:
			matches = [
				column for column in rolling_stock_data.columns
				if str(column).strip().casefold() == component.casefold()
			]
			if not matches:
				raise KeyError(
					f"Train type {component!r} is not in rolling-stock data"
				)
			length += float(rolling_stock_data.at[row, matches[0]])
		return length

	vehicle_lengths = {
		route_id: vehicle_length(route_train_types[route_id])
		for route_id in route_data
	}

	def vehicle_category(train_type):
		return "maglev" if str(train_type).strip().casefold() == "maglev" else "rail"

	def headway_value(first_event, second_event):
		first_route = route_data[str(first_event.route_id).strip()]
		second_route_id = str(second_event.route_id).strip()
		first_node = first_route["node_metadata"][first_event.node]
		second_vehicle = second_route_id
		braking_time = braking_times.get(first_node["ordinal"], {}).get(
			second_vehicle
		)
		if braking_time is None:
			raise KeyError(
				f"No braking time for node {first_node['ordinal']} and "
				f"route {second_vehicle!r}"
			)
		base = (
			braking_time
			+ sight_time
			+ first_node["incoming_speed_limit"]
			* vehicle_lengths[str(first_event.route_id).strip()]
		)
		if first_node["station"]:
			return base + startup_travel_times[
				route_train_types[str(first_event.route_id).strip()]
			]
		if first_node["speed_change"] or first_node["crossing"]:
			return base
		if first_node["switch"]:
			second_route = route_data[second_route_id]
			second_node = second_route["node_metadata"][second_event.node]
			different_edges = (
				first_node["incoming_edge"] != second_node["incoming_edge"]
				or first_node["outgoing_edge"] != second_node["outgoing_edge"]
			)
			if different_edges:
				first_type = vehicle_category(
					route_train_types[str(first_event.route_id).strip()]
				)
				second_type = vehicle_category(
					route_train_types[second_route_id]
				)
				return base + switching_time[
					f"{first_type}_{second_type}"
				]
		return None

	def time_at(route, node):
		if route["start_time"] is None or route["end_time"] is None:
			return None
		total = max(len(route["nodes"]) - 1, 1)
		fraction = route["distance"].get(node, 0) / total
		elapsed = route["end_time"] - route["start_time"]
		return route["start_time"] + elapsed * fraction

	def add_event(event_type, route_id, node, initial_time, related_route_id=None):
		event_id = len(Events)
		is_initial = (
			str(route_id).strip() in initial_routes
			or (
				related_route_id is not None
				and str(related_route_id).strip() in initial_routes
			)
		)
		initial_seconds = (
			initial_time.total_seconds()
			if initial_time is not None
			else None
		)
		item = event(
			event_id,
			is_initial,
			int(route_id),
			route_id,
			node_id(route_data[str(route_id).strip()]["graph"], node),
			event_type,
			initial_time=initial_seconds,
			route_id=route_id,
			related_route_id=related_route_id,
			node=node,
		)
		Events.append(item)
		return event_id

	Events = []
	interaction_sets = {
		"split": [],
		"merge": [],
		"crossing": [],
	}
	route_keys = list(route_data)
	for index, first_key in enumerate(route_keys):
		first = route_data[first_key]
		first_edges = first["edges"]
		for second_key in route_keys[index + 1:]:
			second = route_data[second_key]
			second_edges = second["edges"]
			first_by_node = {
				edge["destination"]: edge for edge in first_edges
			}
			second_by_node = {
				edge["destination"]: edge for edge in second_edges
			}
			for shared_node in set(first_by_node) & set(second_by_node):
				if shared_node not in first["graph"]:
					continue
				retained = set(
					first["graph"].nodes[shared_node].get("retained_for", ())
				)
				if not retained & {"switch", "rail_rail_crossing"}:
					continue
				first_in = first_by_node[shared_node]
				second_in = second_by_node[shared_node]
				first_out = next(
					(edge for edge in first_edges
					 if edge["origin"] == shared_node),
					None,
				)
				second_out = next(
					(edge for edge in second_edges
					 if edge["origin"] == shared_node),
					None,
				)
				if first_out is None or second_out is None:
					continue
				same_in = str(first_in["edge_id"]) == str(second_in["edge_id"])
				same_out = str(first_out["edge_id"]) == str(second_out["edge_id"])
				if same_in and not same_out:
					event_type = "split"
				elif not same_in and same_out:
					event_type = "merge"
				elif not same_in and not same_out:
					event_type = "crossing"
				else:
					continue
				first_time = time_at(first, shared_node)
				second_time = time_at(second, shared_node)
				if first_time is not None and second_time is not None:
					if abs((first_time - second_time).total_seconds()) > 1800:
						continue
				interaction_sets[event_type].append(add_event(
					event_type,
					first_key,
					shared_node,
					first_time,
					second_key,
				))

	V_split = interaction_sets["split"]
	V_merge = interaction_sets["merge"]
	V_crossing = interaction_sets["crossing"]
	V_run_through = []
	V_departure = []
	V_arrival = []
	station_activity_events = []
	for route_key, route in route_data.items():
		graph = route["graph"]
		station_nodes = {}
		for node in route["nodes"]:
			for station in station_names(graph, node):
				if markers(graph, node):
					station_nodes.setdefault(station, []).append(node)
		rows_by_station = {}
		for row in route["rows"].itertuples():
			rows_by_station.setdefault(key(row.station), []).append(row)
		route_stations = list(station_nodes)
		for station_index, station in enumerate(route_stations):
			candidates = station_nodes[station]
			node = candidates[0 if station_index == 0 else min(1, len(candidates) - 1)]
			rows = rows_by_station.get(station, [])
			for row in rows:
				event_type = str(row.event).strip().casefold()
				target = V_departure if event_type in {"depart", "departure"} else V_arrival
				event_id = add_event(
					event_type,
					route_key,
					node,
					pd.to_timedelta(row.time),
				)
				target.append(event_id)
				station_activity_events.append({
					"route": route_key,
					"station": station,
					"event_type": event_type,
					"event_id": event_id,
				})
			if not rows:
				event_id = add_event(
					"run_through",
					route_key,
					node,
					time_at(route, node),
				)
				V_run_through.append(event_id)

	route_node_events = {}
	for route_key, route in route_data.items():
		route_node_events[route_key] = {}
		for node in route["nodes"]:
			event_id = add_event(
				"route_node",
				route_key,
				node,
				time_at(route, node),
			)
			route_node_events[route_key][node] = event_id

	E_running = [
		(
			route_node_events[route_key][edge["origin"]],
			route_node_events[route_key][edge["destination"]],
		)
		for route_key, route in route_data.items()
		for edge in route["edges"]
	]
	E_dwell = []
	for route_key in route_data:
		route_events = [
			item for item in station_activity_events
			if item["route"] == route_key
		]
		for arrival in route_events:
			if arrival["event_type"] not in {"arrive", "arrival"}:
				continue
			departures = [
				item for item in route_events
				if item["station"] == arrival["station"]
				and item["event_type"] in {"depart", "departure"}
				and item["event_id"] > arrival["event_id"]
			]
			if departures:
				E_dwell.append((
					arrival["event_id"],
					departures[0]["event_id"],
				))

	E_headway = [
		(first_event, second_event)
		for node in {
			node
			for route in route_node_events.values()
			for node in route
		}
		for first_key_index, first_key in enumerate(route_keys)
		for second_key in route_keys[first_key_index + 1:]
		if node in route_node_events[first_key]
		and node in route_node_events[second_key]
		for first_event, second_event in [(
			route_node_events[first_key][node],
			route_node_events[second_key][node],
		)]
	]
	event_count = len(Events)
	headway_time_min = np.full((event_count, event_count), -M)
	for first_event_id, second_event_id in E_headway:
		first_event = Events[first_event_id]
		second_event = Events[second_event_id]
		value = headway_value(first_event, second_event)
		if value is not None:
			headway_time_min[first_event_id, second_event_id] = value
	E_travel = E_running
	V_initial = [
		item.event_id for item in Events if item.is_initial
	]
	V_additional = [
		item.event_id for item in Events if not item.is_initial
	]

	V_rt = V_run_through
	V_initial = [
		item.event_id for item in Events if item.is_initial
	]
	V_additional = [
		item.event_id for item in Events if not item.is_initial
	]
	N_initial = len(V_initial)
	N_additional = len(V_additional)
	event_count = len(Events)
	I = tuple(range(event_count))
	pi = np.zeros(event_count)
	s = np.zeros(event_count)
	t = np.zeros(event_count)
	tw_min = np.full(event_count, -M)
	tw_max = np.full(event_count, M)
	for item in Events:
		s[item.event_id] = item.location_id
		t[item.event_id] = item.service_id_internal
		if item.initial_time is not None:
			pi[item.event_id] = item.initial_time
	q = np.equal.outer(s, s).astype(float)
	l = np.equal.outer(t, t).astype(float)
	priority_by_event = np.array([
		train_priorities.get(str(item.route_id).strip(), 0.0)
		for item in Events
	])
	p = (
		priority_by_event[:, np.newaxis]
		> priority_by_event[np.newaxis, :]
	).astype(float)
	running_time_max = np.full(
		(event_count, event_count, len(T)),
		M,
	)
	running_time_min = np.full(
		(event_count, event_count, len(T)),
		-M,
	)
	dwell_time_max = np.full(
		(event_count, event_count, len(T)),
		M,
	)
	dwell_time_min = np.full(
		(event_count, event_count, len(T)),
		-M,
	)
	rho = np.zeros((event_count, len(T)))
	return {
		"route_graphs": route_graphs,
		"Events": Events,
		"V_split": V_split,
		"V_merge": V_merge,
		"V_crossing": V_crossing,
		"V_run_through": V_run_through,
		"V_rt": V_rt,
		"V_departure": V_departure,
		"V_arrival": V_arrival,
		"V_initial": V_initial,
		"V_additional": V_additional,
		"E_dwell": E_dwell,
		"E_running": E_running,
		"E_travel": E_travel,
		"E_headway": E_headway,
		"train_priorities": train_priorities,
		"route_train_types": route_train_types,
		"startup_travel_times": startup_travel_times,
		"braking_times": braking_times,
		"headway_time_min": headway_time_min,
		"I": I,
		"event_times_are_variables": True,
	}


class event:
	def __init__(self, event_id, is_initial, service_id_internal, service_id_external, location_id, event_type, train_type=None, train_model=None, initial_time=None, route_id=None, related_route_id=None, node=None):
		self.event_id = event_id
		self.service_id_internal = service_id_internal
		self.service_id_external = service_id_external
		self.location_id = location_id
		self.event_type = event_type
		self.is_initial = is_initial
		self.train_type = train_type #maglev or rail
		self.train_model = train_model
		self.scheduled_time = initial_time
		self.initial_time = initial_time
		self.route_id = route_id
		self.related_route_id = related_route_id
		self.node = node

def populate_events(event_definitions):
	"""Create events and split their IDs into initial and additional events.

	``event_definitions`` must contain dictionaries with the fields accepted by
	the :class:`event` constructor: ``event_id``, ``is_initial``,
	``service_id_internal``, ``service_id_external``, ``location_id``, and
	``event_type``.
	"""
	global Events, V_initial, V_additional, N_initial, N_additional

	Events = [
		event(
			event_id=definition["event_id"],
			is_initial=definition["is_initial"],
			service_id_internal=definition["service_id_internal"],
			service_id_external=definition["service_id_external"],
			location_id=definition["location_id"],
			event_type=definition["event_type"],
		)
		for definition in event_definitions
	]
	V_initial = [current_event.event_id for current_event in Events if current_event.is_initial]
	V_additional = [
		current_event.event_id
		for current_event in Events
		if not current_event.is_initial
	]
	N_initial = len(V_initial)
	N_additional = len(V_additional)

	return Events, V_initial, V_additional

def populate_priority_matrix(train_services=None, priority_matrix=None):
	"""Set priority for every ordered pair of train services.

	For train services ``i`` and ``j``, ``p[i, j]`` is one when the priority
	of the service whose ``internal_id`` is ``i`` is greater than the priority
	of the service whose ``internal_id`` is ``j``. All other entries are reset
	to zero before the comparison.
	"""
	if train_services is None:
		train_services = T
	if priority_matrix is None:
		priority_matrix = p

	priority_matrix.fill(0)
	for train_i in train_services:
		internal_id_i = train_i["internal_id"]
		priority_i = train_i["priority"]
		for train_j in train_services:
			internal_id_j = train_j["internal_id"]
			priority_matrix[internal_id_i, internal_id_j] = int(
				priority_i > train_j["priority"]
			)

	return priority_matrix

def populate_event_locations(events=None, location_array=None):
	"""Set the location of each event at the index of its event ID."""
	global s

	if events is None:
		events = Events
	if location_array is None:
		required_size = max(
			(event_item.event_id for event_item in events),
			default=-1,
		) + 1
		if required_size > len(s):
			s = np.zeros(required_size)
		location_array = s

	location_array.fill(0)
	for event_item in events:
		event_id = event_item.event_id
		if not isinstance(event_id, (int, np.integer)) or event_id < 0:
			raise ValueError(
				f"Event ID {event_id!r} must be a non-negative integer"
			)
		if event_id >= len(location_array):
			raise IndexError(
				f"Event ID {event_id} is outside the location array"
			)
		location_array[event_id] = event_item.location_id

	return location_array

def populate_event_services(events=None, service_array=None):
	"""Set the internal train service of each event at its event ID index."""
	global t

	if events is None:
		events = Events
	if service_array is None:
		required_size = max(
			(event_item.event_id for event_item in events),
			default=-1,
		) + 1
		if required_size > len(t):
			t = np.zeros(required_size)
		service_array = t

	service_array.fill(0)
	for event_item in events:
		event_id = event_item.event_id
		if not isinstance(event_id, (int, np.integer)) or event_id < 0:
			raise ValueError(
				f"Event ID {event_id!r} must be a non-negative integer"
			)
		if event_id >= len(service_array):
			raise IndexError(
				f"Event ID {event_id} is outside the service array"
			)
		service_array[event_id] = event_item.service_id_internal

	return service_array

def populate_q(location_array=None, same_location_matrix=None):
	"""Set q[i, j] to one when events i and j have the same location."""
	global q

	if location_array is None:
		location_array = s
	location_array = np.asarray(location_array)
	if location_array.ndim != 1:
		raise ValueError("The location array must be one-dimensional")

	required_shape = (len(location_array), len(location_array))
	if same_location_matrix is None:
		if q.shape != required_shape:
			q = np.zeros(required_shape)
		same_location_matrix = q
	elif same_location_matrix.shape != required_shape:
		raise ValueError(
			f"The q matrix must have shape {required_shape}, "
			f"not {same_location_matrix.shape}"
		)

	same_location_matrix[:] = (
		location_array[:, np.newaxis] == location_array[np.newaxis, :]
	)
	return same_location_matrix

def populate_l(service_array=None, same_service_matrix=None):
	"""Set l[i, j] to one when events i and j use the same train service."""
	global l

	if service_array is None:
		service_array = t
	service_array = np.asarray(service_array)
	if service_array.ndim != 1:
		raise ValueError("The service array must be one-dimensional")

	required_shape = (len(service_array), len(service_array))
	if same_service_matrix is None:
		if l.shape != required_shape:
			l = np.zeros(required_shape)
		same_service_matrix = l
	elif same_service_matrix.shape != required_shape:
		raise ValueError(
			f"The l matrix must have shape {required_shape}, "
			f"not {same_service_matrix.shape}"
		)

	same_service_matrix[:] = (
		service_array[:, np.newaxis] == service_array[np.newaxis, :]
	)
	return same_service_matrix


switching_time = {
	"rail_rail": 2,
	"rail_maglev": 20,
	"maglev_maglev": 10,
	"maglev_rail": 20
}
sight_time = 2 #seconds
train_number_dict = {} #dictionary of real-world train IDs and their corresponding train IDs in the code
station_name_dict = {} #dictionary of real-world full station names and their UIC Codes
MAX_DEVIATION = 10*60 #maximum deviation from the initial schedule in seconds
RUNNING_TIME_MAX_FACTOR = 1.5 #ratio between min and max running time

M=1e6 #big M for constraints
Events = [] #set of events, list of event classes
V_initial = [] #set of initial events
V_additional = [] #set of additional events
N_initial = len(V_initial) #number of initial events
N_additional = len(V_additional) #number of additional events

#Chopping Block
V_departure = [] #set of departure events
V_arrival = [] #set of arrival events
E_dwell = []
E_running = []
E_travel = []
E_headway = []
V_split = [] #set of split events
V_merge = [] #set of merge events
V_crossing = [] #set of crossing events
V_run_through = [] #set of station run through events
###End chopping block

T = [] #set of train services, list of dictionaries
running_time_max = np.full((N_initial+N_additional,N_initial+N_additional, len(T)), M) #max duration of train service t between events i and j
running_time_min = np.full((N_initial+N_additional,N_initial+N_additional, len(T)), -M) #min duration of train service t between events i and j
dwell_time_max = np.full((N_initial+N_additional, N_initial+N_additional, len(T)), M) #max duration of train service t's dwell between events i and j
dwell_time_min = np.full((N_initial+N_additional, N_initial+N_additional, len(T)), -M) #min duration of train service t's dwell between events i and j
s = np.zeros(N_initial+N_additional) #s[i] = location of event i
t = np.zeros(N_initial+N_additional) #t[i] = train service of event i
headway_time_min = np.full((N_initial+N_additional, N_initial+N_additional), -M) #set of minimum headways between events i, j ASYMMETRICAL
pi = np.zeros(N_initial+N_additional) #time at which event i takes place in the initial schedule
p = np.zeros((N_initial+N_additional, N_initial+N_additional)) #=1 if train i has priority over train j, =0 otherwise
tw_min = np.zeros(N_additional) #set of beginning time windows for event i
tw_max = np.zeros(N_additional) #set of ending time windows for event i
rho = np.zeros((N_initial+N_additional, len(T))) #1 if train t stops at station i, 0 otherwise
U = [] #set of rolling stock types, pandas df
train_priorities = {} #route priority loaded from route_data.csv
startup_travel_times = {} #startup time by rolling-stock type in route_data.csv
braking_times = {} #braking_times[node_id][route_id] in seconds
q = np.zeros((N_initial+N_additional,N_initial+N_additional)) #1 if events i and j are at the same location, 0 otherwise
l = np.zeros((N_initial+N_additional, N_initial+N_additional)) #1 if events i and j are on the same train service, 0 otherwise

I = tuple(range(N_initial + N_additional))


def build_sequence_optimization_model(trip_max, dwell_min, dwell_max, trains=None,):
	"""Build the Gurobi model with event times represented by ``x[i]``.

	Event-set variables such as ``V_initial`` and ``V_additional`` contain
	integer event IDs. Schedule values in ``pi`` are parameters only; they do
	not fix ``x`` unless the caller adds an explicit equality constraint.
	"""
	if trains is None:
		trains = T
	model = gp.Model("Thesis Model")
	x = model.addVars(I, vtype=GRB.CONTINUOUS, name="x", lb=0)
	delta_t = model.addVars(I, vtype=GRB.CONTINUOUS, name="dt", lb=0)
	a = model.addVars(I, vtype=GRB.BINARY, name="a")
	ad = model.addVars(I, vtype=GRB.CONTINUOUS, name="ad")

	model.addConstrs( #additional trains must be after begining of time window
		(x[i] >= tw_min[i] for i in V_additional),
		name="min_time_window",
	)
	model.addConstrs( #additional trains must be before end of time window
		(x[i] <= tw_max[i] for i in V_additional),
		name="max_time_window",
	)
	model.addConstrs( #additional trains cannot be more than MAX_DEVIATION seconds later than the initial schedule
		x[i]-pi[i] <= MAX_DEVIATION for i in V_initial
	)
	model.addConstrs( #additional trains cannot be more than MAX_DEVIATION seconds earlier than the initial schedule
		pi[i]-x[i] <= MAX_DEVIATION for i in V_initial
	)

	model.addConstrs(
		(
			x[j] - x[i]
			>= running_time_min[i, j, m]
			for i, j in E_travel
			for m in trains
		),
		name="travel_time_min",
	)
	model.addConstrs(
		(
			x[j] - x[i]
			<= trip_max[i, j]
			for i, j in E_travel
			for m in trains
		),
		name="travel_time_max",
	)
	model.addConstrs(
		(
			x[j] - x[i] >= rho[i, m] * dwell_min[i, j]
			for i, j in E_dwell
			for m in trains
		),
		name="dwell_time_min",
	)
	model.addConstrs(
		(
			x[j] - x[i] <= rho[i, m] * dwell_max[i, j]
			for i, j in E_dwell
			for m in trains
		),
		name="dwell_time_max",
	)
	return model, {
		"x": x,
		"delta_t": delta_t,
		"a": a,
		"ad": ad,
	}
