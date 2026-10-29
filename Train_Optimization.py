import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import pickle
import csv
import re
from pathlib import Path
from typing import TypeAlias


EventKey: TypeAlias = tuple[int, int, str, bool, str, int, int]


def time_to_seconds(value):
	"""Convert a time-of-day value to integer seconds from midnight."""
	if value is None or (isinstance(value, str) and not value.strip()):
		raise ValueError("A time value is required")
	if isinstance(value, pd.Timedelta):
		seconds = value.total_seconds()
	elif isinstance(value, np.timedelta64):
		seconds = float(value / np.timedelta64(1, "s"))
	elif isinstance(value, (int, float, np.integer, np.floating)):
		seconds = float(value)
	else:
		text = str(value).strip()
		parts = text.split(":")
		if len(parts) not in {2, 3}:
			raise ValueError(
				f"Invalid time {value!r}; expected HH:MM or HH:MM:SS"
			)
		try:
			hours = int(parts[0])
			minutes = int(parts[1])
			seconds_part = float(parts[2]) if len(parts) == 3 else 0.0
		except ValueError as error:
			raise ValueError(f"Invalid time {value!r}") from error
		if (
			hours < 0
			or not 0 <= minutes < 60
			or not 0 <= seconds_part < 60
		):
			raise ValueError(f"Invalid time {value!r}")
		seconds = hours * 3600 + minutes * 60 + seconds_part
	if not np.isfinite(seconds) or seconds < 0 or seconds >= 24 * 3600:
		raise ValueError(
			f"Time {value!r} must be between 00:00:00 and 23:59:59"
		)
	if not float(seconds).is_integer():
		raise ValueError(f"Time {value!r} must resolve to whole seconds")
	return int(seconds)


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
	non_empty_columns = (
		schedule.iloc[1:].astype(str).to_numpy() != ""
	).any(axis=0)
	schedule = schedule.iloc[:, non_empty_columns]
	schedule_values = schedule.to_numpy(dtype=str)

	group_width = 3
	if schedule.shape[1] % 4 == 0:
		group_width = 4
	elif schedule.shape[1] % 3 != 0:
		raise ValueError(
			"The schedule CSV must contain groups of three or four columns"
		)

	stops = []

	for event_column in range(0, schedule.shape[1], group_width):
		station_column = event_column + 1
		time_column = event_column + 2
		latest_time_column = event_column + 3 if group_width == 4 else time_column
		train_id = schedule_values[0, station_column].strip()
		if not train_id:
			raise ValueError(
				f"Missing train ID in row 1, column {station_column + 1}"
			)

		for row in range(1, schedule.shape[0]):
			event = schedule_values[row, event_column].strip()
			station = schedule_values[row, station_column].strip()
			earliest_time = schedule_values[row, time_column].strip()
			latest_time = schedule_values[row, latest_time_column].strip()
			if not event and not station and not earliest_time and not latest_time:
				continue
			if (
				not event
				or not station
				or not earliest_time
				or not latest_time
			):
				raise ValueError(
					f"Event, station, and time must all be present at row "
					f"{row + 1}, columns {event_column + 1}-{time_column + 1}"
				)
			stops.append(
				{
					"train_id": train_id,
					"event": event,
					"station": station,
					"time": time_to_seconds(earliest_time),
					"earliest_time": time_to_seconds(earliest_time),
					"latest_time": time_to_seconds(latest_time),
				}
			)

	return pd.DataFrame(
		stops,
		columns=[
			"train_id",
			"event",
			"station",
			"time",
			"earliest_time",
			"latest_time",
		],
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
		max_speed_kmh = float(str(
			rolling_stock_data.at[property_rows["max speed (km/h)"], train_type]
		))
		brake_acceleration = float(str(
			rolling_stock_data.at[
				property_rows["brake acceleration (m/s^2)"], train_type
			]
		))
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
			result.append(float(str(value)))
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
		return event[2]

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
	global Events, V, V_initial, V_additional, N_initial, N_additional
	global I, pi, s, t, q, l, tw_min, tw_max
	global running_time_max, running_time_min, dwell_time_max, dwell_time_min
	global headway_time_min, p, rho
	global E_dwell, E_running, E_travel, E_headway
	global train_priorities, startup_travel_times, braking_times
	global T_initial, T_additional, T
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
	route_type_row = next(
		(
			row for row in route_data_rows
			if row and row[0].strip().casefold() == "route type"
		),
		None,
	)
	if route_type_row is None:
		raise ValueError(
			f"Route data file {route_data_file!r} has no Route Type row"
		)
	route_types = {}
	for column, train_id in enumerate(route_data_rows[0][1:], start=1):
		if not train_id.strip():
			continue
		if column >= len(route_type_row) or not route_type_row[column].strip():
			raise ValueError(
				f"Route data file {route_data_file!r} has no route type for "
				f"train {train_id!r}"
			)
		route_types[str(train_id).strip()] = route_type_row[column].strip()
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

	def graph_node(graph, node_id_value):
		for node, attributes in graph.nodes(data=True):
			if attributes.get("node_id") == node_id_value:
				return node
		raise KeyError(f"No graph node found for node ID {node_id_value!r}")

	def markers(graph, node):
		attributes = graph.nodes[graph_node(graph, node)]
		retained_value = attributes.get("retained_for", ())
		retained = {
			marker.strip()
			for marker in retained_value.split(",")
			if marker.strip()
		} if isinstance(retained_value, str) else set(retained_value)
		return retained & {"station_boundary", "platform_edge", "platform_end"}

	def station_names(graph, node):
		attributes = graph.nodes[graph_node(graph, node)]
		names = set(attributes.get("platform_stations", ()))
		station = attributes.get("station")
		if station not in (None, "None", ""):
			names.add(station)
		return {key(name) for name in names}

	def ordered_edges(graph):
		assigned = list(graph.graph.get("assigned_edges", ()))
		route_node_ids = list(graph.graph.get("route_node_ids", ()))
		records = []
		used = set()
		previous = route_node_ids[0] if route_node_ids else None
		for edge_id in assigned:
			candidates = [
				(first, second, data)
				for first, second, data in graph.edges(data=True)
				if str(data.get("edge_id")) == str(edge_id)
				and id(data) not in used
			]
			if not candidates:
				continue
			oriented_candidates = []
			for first, second, data in candidates:
				first_id = node_id(graph, first)
				second_id = node_id(graph, second)
				if first_id == previous:
					oriented_candidates.append((first, second, data))
				elif second_id == previous:
					oriented_candidates.append((second, first, data))
			first, second, data = (
				oriented_candidates[0]
				if oriented_candidates
				else candidates[0]
			)
			used.add(id(data))
			records.append({
				"edge_id": data.get("edge_id", edge_id),
				"origin": node_id(graph, first),
				"destination": node_id(graph, second),
				"speed_limit": data.get(
					"speed_limit_kmh",
					data.get("speed_limit", data.get("maxspeed")),
				),
			})
			previous = node_id(graph, second)
		if records:
			return records
		for first, second, data in graph.edges(data=True):
			records.append({
				"edge_id": data.get("edge_id"),
				"origin": node_id(graph, first),
				"destination": node_id(graph, second),
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
			nodes = list(graph.graph.get("route_node_ids", ()))
		distance = {
			node_id_value: index
			for index, node_id_value in enumerate(nodes)
		}
		route_rows = schedule_rows[
			schedule_rows["train_id"].map(lambda value: str(value).strip())
			== str(train_id).strip()
		]
		known_times = [
			time_to_seconds(row.time)
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
	service_ids = {
		route_id: index
		for index, route_id in enumerate(route_data)
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
		metadata_nodes = list(route["nodes"])
		for edge in route["edges"]:
			for node in (edge["origin"], edge["destination"]):
				if node not in metadata_nodes:
					metadata_nodes.append(node)
		for index, node in enumerate(metadata_nodes):
			ordered_incoming = (
				route["edges"][index - 1]
				if index > 0
				and index - 1 < len(route["edges"])
				else None
			)
			incoming = ordered_incoming
			if incoming is None or incoming["destination"] != node:
				incoming = next(
					(
						edge for edge in route["edges"]
						if edge["destination"] == node
					),
					None,
				)
			outgoing = next(
				(edge for edge in route["edges"]
				 if edge["origin"] == node),
				None,
			)
			attributes = graph.nodes[graph_node(graph, node)]
			if incoming is None:
				incoming_speed_limit = None
			else:
				incoming_speed_limit = speed_limit_mps(
					incoming["speed_limit"],
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
			metadata[node]["speed_change"] = False
		for index, node in enumerate(route["nodes"]):
			metadata[node]["speed_change"] = (
				index > 0
				and metadata[node]["speed_limit"] is not None
				and metadata[route["nodes"][index - 1]]["speed_limit"] is not None
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
			length += float(str(rolling_stock_data.at[row, matches[0]]))
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
		if first_node["incoming_edge"] is None:
			braking_time = 0.0
		else:
			braking_time = braking_times.get(
				first_event.location_id,
				{},
			).get(second_vehicle)
			if braking_time is None:
				raise KeyError(
					f"No braking time for node {first_event.location_id} and "
					f"route {second_vehicle!r}"
				)
			braking_time *= HEADWAY_TIME_SAFETY_FACTOR
		incoming_speed_limit = first_node["incoming_speed_limit"] or 0.0
		clearance_time = (
			vehicle_lengths[str(first_event.route_id).strip()]
			/ incoming_speed_limit
			if incoming_speed_limit > 0
			else 0.0
		)
		base = braking_time + sight_time + clearance_time
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
		return base

	def time_at(route, node):
		if route["start_time"] is None or route["end_time"] is None:
			return None
		total = max(len(route["nodes"]) - 1, 1)
		fraction = route["distance"].get(node, 0) / total
		elapsed = route["end_time"] - route["start_time"]
		return int(round(route["start_time"] + elapsed * fraction))

	def add_event(
		event_type,
		route_id,
		node,
		initial_time,
		related_route_id=None,
		earliest_time=None,
		latest_time=None,
	):
		event_id = len(Events)
		is_initial = (
			str(route_id).strip() in initial_routes
			or (
				related_route_id is not None
				and str(related_route_id).strip() in initial_routes
			)
		)
		initial_seconds = (
			time_to_seconds(initial_time)
			if is_initial and initial_time is not None
			else None
		)
		earliest_seconds = (
			time_to_seconds(earliest_time)
			if earliest_time is not None
			else None
		)
		latest_seconds = (
			time_to_seconds(latest_time)
			if latest_time is not None
			else None
		)
		item = event(
			event_id,
			is_initial,
			service_ids[str(route_id).strip()],
			route_id,
			node,
			event_type,
			train_type=route_train_types[str(route_id).strip()],
			initial_time=initial_seconds,
			route_id=route_id,
			related_route_id=related_route_id,
			node=node,
		)
		item.window_min = earliest_seconds
		item.window_max = latest_seconds
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
				first_graph_node = graph_node(first["graph"], shared_node)
				if first_graph_node not in first["graph"]:
					continue
				retained = set(
					first["graph"].nodes[first_graph_node].get(
						"retained_for",
						(),
					)
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
					if abs(first_time - second_time) > 1800:
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
	missing_terminal_platform_nodes = []
	for route_key, route in route_data.items():
		graph = route["graph"]
		platform_station_nodes = {}
		for node in route["nodes"]:
			attributes = graph.nodes[graph_node(graph, node)]
			retained_value = attributes.get("retained_for", ())
			retained = {
				marker.strip()
				for marker in retained_value.split(",")
			} if isinstance(retained_value, str) else set(retained_value)
			if not retained & {"platform_edge", "platform_end"}:
				continue
			for station in station_names(graph, node):
				platform_station_nodes.setdefault(station, []).append(node)
		for station, candidates in platform_station_nodes.items():
			platform_station_nodes[station] = sorted(
				candidates,
				key=lambda node: route["distance"].get(node, float("inf")),
			)
		final_station = (
			key(route["rows"].iloc[-1]["station"])
			if not route["rows"].empty
			else None
		)
		if final_station is not None and len(
			platform_station_nodes.get(final_station, [])
		) < 2:
			missing_terminal_platform_nodes.append(route_key)
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
			node = (
				candidates[1]
				if station == final_station and len(candidates) >= 2
				else candidates[0]
			)
			rows = rows_by_station.get(station, [])
			for row in rows:
				event_type = str(row.event).strip().casefold()
				target = V_departure if event_type in {"depart", "departure"} else V_arrival
				event_id = add_event(
					event_type,
					route_key,
					node,
					row.time,
					earliest_time=row.earliest_time,
					latest_time=row.latest_time,
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
		route_nodes = list(route["nodes"])
		for edge in route["edges"]:
			for node in (edge["origin"], edge["destination"]):
				if node not in route_nodes:
					route_nodes.append(node)
		for node in route_nodes:
			event_id = add_event(
				"route_node",
				route_key,
				node,
				time_at(route, node),
			)
			route_node_events[route_key][node] = event_id

	for route_key, route in route_data.items():
		if route_key in initial_routes:
			continue
		bounded_events = []
		for activity in station_activity_events:
			if activity["route"] != route_key:
				continue
			item = Events[activity["event_id"]]
			if item.window_min is None or item.window_max is None:
				continue
			bounded_events.append((
				route["distance"].get(item.location_id, 0),
				item.window_min,
				item.window_max,
			))
		if not bounded_events:
			station_nodes = {}
			for node in route["nodes"]:
				for station in station_names(route["graph"], node):
					station_nodes.setdefault(station, []).append(node)
			for row in route["rows"].itertuples():
				station = key(row.station)
				candidates = station_nodes.get(station, [])
				if not candidates:
					continue
				node = candidates[0]
				bounded_events.append((
					route["distance"].get(node, 0),
					time_to_seconds(row.earliest_time),
					time_to_seconds(row.latest_time),
				))
		bounded_events.sort(key=lambda value: value[0])
		if not bounded_events:
			continue

		def interpolated_window(distance, column):
			if distance <= bounded_events[0][0]:
				return bounded_events[0][column]
			if distance >= bounded_events[-1][0]:
				return bounded_events[-1][column]
			for previous, following in zip(
				bounded_events,
				bounded_events[1:],
			):
				if previous[0] <= distance <= following[0]:
					span = following[0] - previous[0]
					if span == 0:
						return previous[column]
					fraction = (distance - previous[0]) / span
					return round(
						previous[column]
						+ fraction * (following[column] - previous[column])
					)
			return None

		for item in Events:
			if (
				not item.is_initial
				and str(item.route_id).strip() == route_key
				and (
					item.window_min is None
					or item.window_max is None
				)
			):
				distance = route["distance"].get(item.location_id, 0)
				item.window_min = interpolated_window(distance, 1)
				item.window_max = interpolated_window(distance, 2)

	services = []
	for route_id, route in route_data.items():
		route_events = [
			item for item in station_activity_events
			if item["route"] == route_id
		]
		departures = [
			item["event_id"] for item in route_events
			if item["event_type"] in {"depart", "departure"}
		]
		arrivals = [
			item["event_id"] for item in route_events
			if item["event_type"] in {"arrive", "arrival"}
		]
		route_nodes = list(route["nodes"])
		services.append({
			"internal_id": service_ids[route_id],
			"external_id": route_id,
			"route_type": route_types[route_id],
			"priority": train_priorities.get(route_id, 0.0),
			"departure_event": (
				departures[0]
				if departures
				else route_node_events[route_id][route_nodes[0]]
			),
			"arrival_event": (
				arrivals[-1]
				if arrivals
				else route_node_events[route_id][route_nodes[-1]]
			),
		})
	T_initial = [
		service for service in services
		if service["external_id"] in initial_routes
	]
	T_additional = [
		service for service in services
		if service["external_id"] not in initial_routes
	]
	T = T_initial + T_additional

	E_running = []
	for route_key, route in route_data.items():
		graph = route["graph"]
		activity_events_by_node = {}
		for activity in station_activity_events:
			if activity["route"] != route_key:
				continue
			item = Events[activity["event_id"]]
			activity_events_by_node.setdefault(
				item.location_id,
				{},
			)[activity["event_type"]] = item.event_id
		for edge in route["edges"]:
			origin_stations = station_names(graph, edge["origin"])
			destination_stations = station_names(graph, edge["destination"])
			origin_id = route_node_events[route_key][edge["origin"]]
			origin_graph_node = graph_node(graph, edge["origin"])
			origin_node_id = graph.nodes[origin_graph_node].get(
				"node_id",
				edge["origin"],
			)
			origin_activity = activity_events_by_node.get(origin_node_id, {})
			origin_id = origin_activity.get(
				"depart",
				origin_activity.get("departure", origin_id),
			)
			destination_id = route_node_events[route_key][edge["destination"]]
			destination_graph_node = graph_node(graph, edge["destination"])
			destination_node_id = graph.nodes[destination_graph_node].get(
				"node_id",
				edge["destination"],
			)
			destination_activity = activity_events_by_node.get(
				destination_node_id,
				{},
			)
			destination_id = destination_activity.get(
				"arrive",
				destination_activity.get("arrival", destination_id),
			)
			if (
				origin_stations & destination_stations
				and origin_id != route_node_events[route_key][edge["origin"]]
				and destination_id != route_node_events[route_key][edge["destination"]]
			):
				continue
			E_running.append((
				origin_id,
				destination_id,
			))
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

	headway_event_ids = list(V_departure) + list(V_arrival)
	excluded_first_station_events = set()
	for route_key, route in route_data.items():
		station_nodes = {}
		for node in route["nodes"]:
			for station in station_names(route["graph"], node):
				retained = markers(route["graph"], node)
				if retained:
					station_nodes.setdefault(station, []).append(node)
		for candidates in station_nodes.values():
			candidates.sort(
				key=lambda node: route["distance"].get(node, float("inf"))
			)
			if candidates:
				excluded_first_station_events.add(
					route_node_events[route_key][candidates[0]]
				)
		headway_event_ids.extend(
			event_id
			for node, event_id in route_node_events[route_key].items()
			if event_id not in excluded_first_station_events
		)
	headway_event_ids = sorted(set(headway_event_ids))
	E_headway = [
		(first_id, second_id)
		for position, first_id in enumerate(headway_event_ids)
		for second_id in headway_event_ids[position + 1:]
		if Events[first_id].location_id == Events[second_id].location_id
		and (
			Events[first_id].service_id_internal
			!= Events[second_id].service_id_internal
		)
	]
	event_count = len(Events)
	headway_time_min = np.full((event_count, event_count), -M)
	for first_event_id, second_event_id in E_headway:
		first_event = Events[first_event_id]
		second_event = Events[second_event_id]
		value = headway_value(first_event, second_event)
		reverse_value = headway_value(second_event, first_event)
		headway_time_min[first_event_id, second_event_id] = value
		headway_time_min[second_event_id, first_event_id] = reverse_value
	E_travel = E_running
	V_initial = [
		item.event_id for item in Events if item.is_initial
	]
	V_additional = [
		item.event_id for item in Events if not item.is_initial
	]
	V = V_initial + V_additional

	V_rt = V_run_through
	V_initial = [
		item.event_id for item in Events if item.is_initial
	]
	V_additional = [
		item.event_id for item in Events if not item.is_initial
	]
	V = V_initial + V_additional
	build_event_key_mappings()
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
		if item.window_min is not None:
			tw_min[item.event_id] = item.window_min
		if item.window_max is not None:
			tw_max[item.event_id] = item.window_max
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
	populate_running_times(
		events=Events,
		edges=E_running,
		services=T,
	)
	populate_dwell_times(
		route_data_file=route_data_file,
		events=Events,
		edges=E_dwell,
		services=T,
	)
	populate_rho(
		existing_schedule_file=existing_schedule_file,
		additional_schedule_file=additional_path,
		events=Events,
		services=T,
		route_graphs=route_graphs,
	)
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
		"missing_terminal_platform_nodes": missing_terminal_platform_nodes,
		"train_priorities": train_priorities,
		"route_train_types": route_train_types,
		"startup_travel_times": startup_travel_times,
		"braking_times": braking_times,
		"headway_time_min": headway_time_min,
		"I": I,
		"event_times_are_variables": True,
	}




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

def populate_running_times(
	running_time_file="running_time_between_nodes.csv",
	events=None,
	edges=None,
	services=None,
):
	"""Populate minimum and maximum running times for route edges."""
	global running_time_min, running_time_max
	events = Events if events is None else events
	edges = E_running if edges is None else edges
	services = T if services is None else services
	event_by_id = {item.event_id: item for item in events}
	service_by_route = {
		str(service["external_id"]).strip(): service["internal_id"]
		for service in services
	}
	route_type_by_service = {
		service["internal_id"]: {
			"ic": "intercity",
		}.get(
			str(service["route_type"]).strip().casefold(),
			str(service["route_type"]).strip().casefold(),
		)
		for service in services
	}
	times = {}
	with Path(running_time_file).open(
		encoding="utf-8-sig",
		newline="",
	) as handle:
		for row in csv.DictReader(handle):
			try:
				train_id = str(row["Train ID"]).strip()
				origin = int(row["Origin Node"])
				destination = int(row["Destination Node"])
				minimum = float(row["Minimum Running Time (s)"])
			except (KeyError, TypeError, ValueError) as error:
				raise ValueError(
					f"Invalid running-time row: {row!r}"
				) from error
			if not np.isfinite(minimum) or minimum < 0:
				raise ValueError(
					f"Invalid minimum running time {minimum!r}"
				)
			times[(train_id, origin, destination)] = minimum

	running_time_min.fill(-M)
	running_time_max.fill(M)
	for edge_index, (first_id, second_id) in enumerate(edges):
		first = event_by_id[first_id]
		second = event_by_id[second_id]
		route_id = str(first.route_id).strip()
		service_id = service_by_route.get(route_id)
		if service_id is None:
			continue
		key = (route_id, int(first.location_id), int(second.location_id))
		if key not in times:
			reverse_key = (route_id, key[2], key[1])
			if reverse_key in times:
				first_id, second_id = second_id, first_id
				edges[edge_index] = (first_id, second_id)
				first = event_by_id[first_id]
				second = event_by_id[second_id]
				key = reverse_key
		if key not in times:
			raise KeyError(
				f"No running time for route {route_id!r}, "
				f"nodes {key[1]}->{key[2]}"
			)
		route_type = route_type_by_service[service_id]
		try:
			supplement = RUNNING_TIME_SUPPLEMENT[route_type]
		except KeyError as error:
			raise KeyError(
				f"No running-time supplement for route type {route_type!r}"
			) from error
		running_time_min[first_id, second_id, service_id] = (
			times[key] * supplement
		)
		running_time_max[first_id, second_id, service_id] = (
			times[key] * RUNNING_TIME_MAX_FACTOR
		)
	return running_time_min, running_time_max


def populate_dwell_times(
	route_data_file="route_data.csv",
	events=None,
	edges=None,
	services=None,
):
	"""Populate minimum and maximum dwell times for station dwell edges."""
	global dwell_time_min, dwell_time_max
	events = Events if events is None else events
	edges = E_dwell if edges is None else edges
	services = T if services is None else services
	route_ids = {
		str(service["external_id"]).strip(): service["internal_id"]
		for service in services
	}
	with Path(route_data_file).open(
		encoding="utf-8-sig",
		newline="",
	) as handle:
		rows = list(csv.reader(handle))
	if not rows:
		raise ValueError(f"Route data file {route_data_file!r} is empty")
	header = rows[0]
	duration_row = next(
		(
			row for row in rows
			if row and row[0].strip().casefold() == "stop duration (s)"
		),
		None,
	)
	if duration_row is None:
		raise ValueError(
			f"Route data file {route_data_file!r} has no stop-duration row"
		)
	dwell_by_route = {}
	for column, route_id in enumerate(header[1:], start=1):
		if not route_id.strip():
			continue
		try:
			dwell_by_route[str(route_id).strip()] = float(
				duration_row[column]
			)
		except (IndexError, ValueError) as error:
			raise ValueError(
				f"Invalid stop duration for route {route_id!r}"
			) from error

	dwell_time_min.fill(-M)
	dwell_time_max.fill(M)
	event_by_id = {item.event_id: item for item in events}
	for first_id, second_id in edges:
		route_id = str(event_by_id[first_id].route_id).strip()
		service_id = route_ids.get(route_id)
		if service_id is None or route_id not in dwell_by_route:
			continue
		minimum = dwell_by_route[route_id]
		if not np.isfinite(minimum) or minimum < 0:
			raise ValueError(
				f"Invalid dwell time {minimum!r} for route {route_id!r}"
			)
		dwell_time_min[first_id, second_id, service_id] = minimum
		dwell_time_max[first_id, second_id, service_id] = (
			minimum * DWELL_TIME_MAX_FACTOR
		)
	return dwell_time_min, dwell_time_max


def populate_rho(
	existing_schedule_file="existing_rail_routes.csv",
	additional_schedule_file: str | Path = "new_rail_routes.csv",
	events=None,
	services=None,
	route_graphs=None,
):
	"""Populate rho from the station stops in both schedule files."""
	global rho
	events = Events if events is None else events
	services = T if services is None else services
	schedule_ids = set()
	for filename in (existing_schedule_file, additional_schedule_file):
		schedule = load_train_schedule(filename)
		schedule_ids.update(
			str(train_id).strip()
			for train_id in schedule["train_id"]
		)
	service_by_route = {
		str(service["external_id"]).strip(): service["internal_id"]
		for service in services
	}
	missing_routes = schedule_ids - set(service_by_route)
	if missing_routes:
		raise KeyError(
			f"Schedule routes have no train service: {sorted(missing_routes)}"
		)
	stop_locations = {
		service_id: set() for service_id in service_by_route.values()
	}
	if route_graphs is not None:
		def station_key(value):
			return " ".join(str(value).split()).casefold()

		for route_id, service_id in service_by_route.items():
			graph = route_graphs.get(route_id, route_graphs.get(int(route_id)))
			if graph is None:
				continue
			stations = set()
			for schedule_file in (
				existing_schedule_file,
				additional_schedule_file,
			):
				schedule = load_train_schedule(schedule_file)
				rows = schedule[
					schedule["train_id"].map(lambda value: str(value).strip()) == route_id
				]
				stations.update(station_key(value) for value in rows["station"])
			for graph_node, attributes in graph.nodes(data=True):
				names = set(attributes.get("platform_stations", ()))
				names.add(attributes.get("station"))
				if any(
					name not in (None, "", "None")
					and station_key(name) in stations
					for name in names
				):
					location_id = attributes.get("node_id", graph_node)
					stop_locations[service_id].add(location_id)
	for item in events:
		if item.event_type in {"arrival", "arrive", "departure", "depart"}:
			stop_locations.setdefault(item.service_id_internal, set()).add(
				item.location_id
			)
	rho.fill(0)
	for item in events:
		for service_id, locations in stop_locations.items():
			if item.location_id in locations:
				rho[item.event_id, service_id] = 1
	return rho


def build_event_key_mappings(events=None):
	"""Build the bidirectional semantic-key/index mapping for events."""
	global event_keys, event_key_to_index, index_to_event_key

	if events is None:
		events = Events
	keys = tuple(item.event_key for item in events)
	if len(set(keys)) != len(keys):
		raise ValueError("Event keys must be unique")
	event_keys = keys
	event_key_to_index = {
		key: index for index, key in enumerate(event_keys)
	}
	index_to_event_key = {
		index: key for index, key in enumerate(event_keys)
	}
	return event_key_to_index, index_to_event_key


def event_index(key: EventKey) -> int:
	"""Return the dense NumPy index for an event key."""
	try:
		return event_key_to_index[key]
	except KeyError as error:
		raise KeyError(f"Unknown event key {key!r}") from error


def event_key(index: int) -> EventKey:
	"""Return the semantic event key for a dense NumPy index."""
	try:
		return index_to_event_key[index]
	except KeyError as error:
		raise KeyError(f"Unknown event index {index!r}") from error

def build_train_schedule_output(model, x, events=None, route_graphs=None):
	"""Build a station-only schedule report from the solved event-time variables."""
	if model.SolCount <= 0:
		raise ValueError("Cannot build a schedule report because no solution exists")
	if events is None:
		events = Events
	if route_graphs is None:
		route_graphs = {}

	def route_graph(route_id):
		graph = route_graphs.get(route_id)
		if graph is None:
			graph = route_graphs.get(str(route_id).strip())
		if graph is None:
			try:
				graph = route_graphs.get(int(route_id))
			except (TypeError, ValueError):
				graph = None
		return graph

	def station_label(route_id, location_id):
		graph = route_graph(route_id)
		if graph is not None:
			for node, attributes in graph.nodes(data=True):
				if attributes.get("node_id", node) != location_id:
					continue
				names = set(attributes.get("platform_stations", ()))
				station = attributes.get("station")
				if station not in (None, "", "None"):
					names.add(station)
				names = sorted(str(name) for name in names if name not in (None, ""))
				if names:
					return " / ".join(names)
		return str(location_id)

	def route_position(route_id, location_id):
		graph = route_graph(route_id)
		if graph is None:
			return (1, 0)
		route_node_ids = graph.graph.get("route_node_ids", ())
		try:
			return (0, route_node_ids.index(location_id))
		except ValueError:
			return (1, 0)

	def format_time(seconds):
		whole_seconds = int(round(seconds))
		hours, remainder = divmod(whole_seconds, 3600)
		minutes, seconds_part = divmod(remainder, 60)
		return f"{hours:02d}:{minutes:02d}:{seconds_part:02d}"

	def format_duration(seconds):
		if pd.isna(seconds):
			return ""
		sign = "-" if seconds < 0 else ""
		return f"{sign}{format_time(abs(seconds))}"

	def event_window(array, event_id, service_id):
		values = np.asarray(array)
		if values.ndim == 1:
			return values[event_id]
		return values[event_id, service_id]

	station_events = [
		item for item in events
		if item.event_type in {"arrival", "arrive", "departure", "depart"}
	]
	if not station_events:
		route_node_events = {
			(str(item.route_id), item.location_id): item
			for item in events
			if item.event_type == "route_node"
		}
		schedule_files = (
			"existing_rail_routes.csv",
			"additional_rail_routes.csv",
			"new_rail_routes.csv",
		)
		scheduled_rows = []
		for schedule_file in schedule_files:
			if Path(schedule_file).exists():
				scheduled_rows.append(load_train_schedule(schedule_file))
				if schedule_file != "existing_rail_routes.csv":
					break
		for schedule_index, schedule in enumerate(scheduled_rows):
			for row in schedule.itertuples():
				graph = route_graph(row.train_id)
				station_key = " ".join(str(row.station).split()).casefold()
				candidates = []
				for node_id in graph.graph.get("route_node_ids", ()):
						node = next(
							(
								node
								for node, attributes in graph.nodes(data=True)
								if attributes.get("node_id", node) == node_id
							),
							None,
						)
						if node is None:
							continue
						attributes = graph.nodes[node]
						names = set(attributes.get("platform_stations", ()))
						names.add(attributes.get("station"))
						retained_value = attributes.get("retained_for", ())
						retained = {
							marker.strip()
							for marker in retained_value.split(",")
						} if isinstance(retained_value, str) else set(retained_value)
						if (
							not retained & {"platform_edge", "platform_end"}
							or not any(
								name not in (None, "", "None")
								and " ".join(str(name).split()).casefold()
								== station_key
								for name in names
							)
						):
							continue
						candidates.append(node_id)
				if len(candidates) < 2:
						continue
				item = route_node_events.get(
						(str(row.train_id), candidates[1])
				)
				if item is not None:
						station_events.append(
							{
								"event": (
									"arrival"
									if row.event.casefold() in {"arrive", "arrival"}
									else "departure"
								),
								"event_id": item.event_id,
								"location_id": item.location_id,
								"route_id": item.route_id,
								"service_id_external": item.service_id_external,
								"event_key": item.event_key,
								"is_initial": schedule_index == 0,
								"service_id_internal": item.service_id_internal,
								"original_time": (
									row.time if schedule_index == 0 else None
								),
								"window_min": (
									row.earliest_time
									if schedule_index == 0
									else None
								),
								"window_max": (
									row.latest_time
									if schedule_index == 0
									else None
								),
							}
						)
	rows = []
	for item in station_events:
		event_type = item.event_type if hasattr(item, "event_type") else item["event"]
		event_id = item.event_id if hasattr(item, "event_id") else item["event_id"]
		event_key_value = (
			item.event_key if hasattr(item, "event_key") else event_key(event_id)
		)
		time_value = float(x[event_key_value].X)
		original_time = (
			item.initial_time
			if hasattr(item, "initial_time")
			else item.get("original_time")
		)
		is_initial = (
			item.is_initial
			if hasattr(item, "is_initial")
			else item.get("is_initial", False)
		)
		service_id = (
			item.service_id_internal
			if hasattr(item, "service_id_internal")
			else item.get("service_id_internal")
		)
		if is_initial and original_time is not None:
			window_min = original_time - MAX_DEVIATION
			window_max = original_time + MAX_DEVIATION
		elif not is_initial and service_id is not None:
			window_min = event_window(tw_min, event_id, service_id)
			window_max = event_window(tw_max, event_id, service_id)
		else:
			window_min = None
			window_max = None
		rows.append({
			"train": str(
				item.service_id_external
				if hasattr(item, "service_id_external")
				else item["service_id_external"]
			),
			"route_id": item.route_id if hasattr(item, "route_id") else item["route_id"],
			"event": (
				"arrival"
				if event_type in {"arrival", "arrive"}
				else "departure"
			),
			"station": station_label(
				item.route_id if hasattr(item, "route_id") else item["route_id"],
				item.location_id if hasattr(item, "location_id") else item["location_id"],
			),
			"location_id": (
				item.location_id if hasattr(item, "location_id")
				else item["location_id"]
			),
			"time_seconds": time_value,
			"original_time_seconds": original_time,
			"window_min_seconds": window_min,
			"window_max_seconds": window_max,
			"event_id": event_id,
		})

	def route_sort_key(route_id):
		try:
			return (0, int(route_id))
		except (TypeError, ValueError):
			return (1, str(route_id))

	rows.sort(key=lambda row: (
		route_sort_key(row["route_id"]),
		route_position(row["route_id"], row["location_id"]),
		row["event_id"],
	))
	last_departure = {}
	last_arrival = {}
	last_scheduled_departure = {}
	cumulative_scheduled_travel_time = {}
	for row in rows:
		route_id = row["route_id"]
		if row["event"] == "departure":
			arrival = last_arrival.get(route_id)
			last_departure[route_id] = row
			last_scheduled_departure[route_id] = row
			if (
				arrival is not None
				and arrival["location_id"] == row["location_id"]
			):
				row["travel_time_seconds"] = (
					row["time_seconds"] - arrival["time_seconds"]
				)
			else:
				row["travel_time_seconds"] = np.nan
			cumulative_scheduled_travel_time.setdefault(route_id, 0.0)
			row["cumulative_scheduled_travel_time_seconds"] = (
				cumulative_scheduled_travel_time[route_id]
			)
			continue
		departure = last_departure.get(route_id)
		if departure is None or departure["location_id"] == row["location_id"]:
			row["travel_time_seconds"] = np.nan
		else:
			row["travel_time_seconds"] = (
				row["time_seconds"] - departure["time_seconds"]
			)
		last_arrival[route_id] = row
		scheduled_departure = last_scheduled_departure.get(route_id)
		if (
			scheduled_departure is not None
			and row["original_time_seconds"] is not None
			and scheduled_departure["original_time_seconds"] is not None
		):
			cumulative_scheduled_travel_time[route_id] = (
				cumulative_scheduled_travel_time.get(route_id, 0.0)
				+ row["original_time_seconds"]
				- scheduled_departure["original_time_seconds"]
			)
		row["cumulative_scheduled_travel_time_seconds"] = (
			cumulative_scheduled_travel_time.get(route_id, np.nan)
		)

	cumulative_travel_time = {}
	for row in rows:
		route_id = row["route_id"]
		previous = cumulative_travel_time.get(route_id, 0.0)
		if not pd.isna(
			row["travel_time_seconds"]
		):
			previous += row["travel_time_seconds"]
		cumulative_travel_time[route_id] = previous
		row["cumulative_travel_time_seconds"] = previous

	schedule = pd.DataFrame(rows)
	if schedule.empty:
		return pd.DataFrame(columns=[
			"train", "event", "station", "time", "change_from_original",
			"travel_time", "cumulative_travel_time",
			"cumulative_scheduled_travel_time",
			"minimum_time_window", "maximum_time_window",
		])
	schedule["time"] = schedule["time_seconds"].map(format_time)
	schedule["change_from_original"] = schedule.apply(
		lambda row: format_duration(
			row["time_seconds"] - row["original_time_seconds"]
		)
		if not pd.isna(row["original_time_seconds"])
		else "",
		axis=1,
	)
	schedule["travel_time"] = schedule["travel_time_seconds"].map(
		format_duration
	)
	schedule["cumulative_travel_time"] = schedule[
		"cumulative_travel_time_seconds"
	].map(
		format_duration
	)
	schedule["cumulative_scheduled_travel_time"] = schedule[
		"cumulative_scheduled_travel_time_seconds"
	].map(
		format_duration
	)
	schedule["minimum_time_window"] = schedule["window_min_seconds"].map(
		lambda value: "" if pd.isna(value) else format_time(value)
	)
	schedule["maximum_time_window"] = schedule["window_max_seconds"].map(
		lambda value: "" if pd.isna(value) else format_time(value)
	)
	return schedule[[
		"train",
		"event",
		"station",
		"time",
		"change_from_original",
		"travel_time",
		"cumulative_travel_time",
		"cumulative_scheduled_travel_time",
		"minimum_time_window",
		"maximum_time_window",
	]]


def export_train_schedule_csv(schedule, output_file="train_schedule.csv"):
	"""Export the station-only schedule as horizontally separated train blocks."""
	block_columns = [
		"event",
		"station",
		"time",
		"change_from_original",
		"travel_time",
		"cumulative_travel_time",
		"cumulative_scheduled_travel_time",
		"minimum_time_window",
		"maximum_time_window",
	]
	blocks = []
	for train, train_schedule in schedule.groupby("train", sort=False):
		block = train_schedule[block_columns].copy()
		block.columns = [
			"Event",
			"Station",
			"Time",
			"Change from original",
			"Travel time",
			"Cumulative travel time",
			"Cumulative scheduled travel time",
			"Minimum time window",
			"Maximum time window",
		]
		block.loc[-1] = [
			f"Train {train}",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
			"",
		]
		block.index = block.index + 1
		block = block.sort_index()
		blocks.append(block.reset_index(drop=True))

	if not blocks:
		output = pd.DataFrame(columns=[
			"Event",
			"Station",
			"Time",
			"Change from original",
			"Travel time",
			"Cumulative travel time",
			"Cumulative scheduled travel time",
			"Minimum time window",
			"Maximum time window",
		])
	else:
		output_blocks = []
		for index, block in enumerate(blocks):
			output_blocks.append(block)
			if index < len(blocks) - 1:
				output_blocks.append(pd.DataFrame({"": [""] * len(block)}))
		output = pd.concat(output_blocks, axis=1)

	output.to_csv(output_file, index=False, na_rep="")
	return Path(output_file)


def export_minimum_headways_csv(
	events=None,
	headway_edges=None,
	headway_time_minimums=None,
	output_file="minimum_headways.csv",
):
	"""Export every calculated minimum headway between train event pairs."""
	events = Events if events is None else events
	headway_edges = E_headway if headway_edges is None else headway_edges
	headway_time_minimums = (
		headway_time_min
		if headway_time_minimums is None
		else headway_time_minimums
	)

	def format_time(value):
		if value is None or pd.isna(value):
			return ""
		whole_seconds = int(round(value))
		hours, remainder = divmod(whole_seconds, 3600)
		minutes, seconds = divmod(remainder, 60)
		return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

	rows = []
	for departure_id, arrival_id in headway_edges:
		minimum = headway_time_minimums[departure_id, arrival_id]
		if not np.isfinite(minimum) or minimum < 0:
			continue
		departure = events[departure_id]
		arrival = events[arrival_id]
		rows.append({
			"departure_train": str(departure.service_id_external),
			"arrival_train": str(arrival.service_id_external),
			"location_id": departure.location_id,
			"departure_event_id": departure_id,
			"arrival_event_id": arrival_id,
			"departure_event": departure.event_type,
			"arrival_event": arrival.event_type,
			"departure_time": format_time(departure.initial_time),
			"arrival_time": format_time(arrival.initial_time),
			"minimum_headway_seconds": minimum,
			"minimum_headway": format_time(minimum),
		})
		reverse_minimum = headway_time_minimums[arrival_id, departure_id]
		if np.isfinite(reverse_minimum) and reverse_minimum >= 0:
			rows.append({
				"departure_train": str(arrival.service_id_external),
				"arrival_train": str(departure.service_id_external),
				"location_id": arrival.location_id,
				"departure_event_id": arrival_id,
				"arrival_event_id": departure_id,
				"departure_event": arrival.event_type,
				"arrival_event": departure.event_type,
				"departure_time": format_time(arrival.initial_time),
				"arrival_time": format_time(departure.initial_time),
				"minimum_headway_seconds": reverse_minimum,
				"minimum_headway": format_time(reverse_minimum),
			})

	output = pd.DataFrame(rows, columns=[
		"departure_train",
		"arrival_train",
		"location_id",
		"departure_event_id",
		"arrival_event_id",
		"departure_event",
		"arrival_event",
		"departure_time",
		"arrival_time",
		"minimum_headway_seconds",
		"minimum_headway",
	])
	output.to_csv(output_file, index=False, na_rep="")
	return Path(output_file)


def main():
	"""Load the case study data, populate derived arrays, and build the model."""
	data = load_and_populate_route_events()
	headway_file = export_minimum_headways_csv(
		events=Events,
		headway_edges=E_headway,
		headway_time_minimums=headway_time_min,
	)
	data["minimum_headways_file"] = headway_file
	print(f"Minimum headways written to {headway_file}")
	if data["missing_terminal_platform_nodes"]:
		print(
			"Trains without a second platform-assigned node at their "
			"end station: "
			+ ", ".join(data["missing_terminal_platform_nodes"])
		)
	populate_priority_matrix()
	populate_event_locations()
	populate_event_services()
	populate_q()
	populate_l()

	event_count = len(I)
	running_time_maximums = running_time_max
	dwell_time_minimums = dwell_time_min
	dwell_time_maximums = dwell_time_max
	time_window_minimums = tw_min
	time_window_maximums = tw_max
	initial_schedule_times = pi
	running_time_minimums = running_time_min
	stop_matrix = rho

	if T:
		trip_max = running_time_maximums
		dwell_min = dwell_time_minimums
		dwell_max = dwell_time_maximums
	else:
		trip_max = np.full((event_count, event_count, 0), M)
		dwell_min = np.full((event_count, event_count, 0), -M)
		dwell_max = np.full((event_count, event_count, 0), M)

	model, variables = build_sequence_optimization_model(
		trip_max=trip_max,
		dwell_min=dwell_min,
		dwell_max=dwell_max,
		time_window_min=time_window_minimums,
		time_window_max=time_window_maximums,
		train_services=T,
		additional_train_services=T_additional,
		initial_event_ids=V_initial,
		additional_event_ids=V_additional,
		event_key_values=event_keys,
		initial_schedule_times=initial_schedule_times,
		running_time_minimums=running_time_minimums,
		stop_matrix=stop_matrix,
		travel_edges=E_travel,
		travel_edge_services=[
			Events[first_id].service_id_internal
			for first_id, _ in E_travel
		],
		dwell_edges=E_dwell,
		headway_edges=E_headway,
		headway_time_minimums=headway_time_min,
		objective_initial_deviation=c_1,
		objective_additional_duration=c_2,
		max_deviation=MAX_DEVIATION,
		big_m=M,
		event_key_fn=event_key,
	)

	model.optimize()
	if model.Status == GRB.INFEASIBLE:
		model.computeIIS()
		iis_file = "Train_Optimization_infeasible.ilp"
		model.write(iis_file)
		print(f"Model is infeasible; IIS written to {iis_file}")
	elif model.SolCount > 0:
		train_schedule = build_train_schedule_output(
			model=model,
			x=variables["x"],
			events=Events,
			route_graphs=data["route_graphs"],
		)
		data["train_schedule"] = train_schedule
		schedule_file = export_train_schedule_csv(train_schedule)
		data["train_schedule_file"] = schedule_file
		print(f"Train schedule written to {schedule_file}")
	return data, model, variables

# Size-independent constants and bootstrap collections.
M = 1e6  # Big-M value used for inactive constraints.
DWELL_TIME_MAX_FACTOR = 3  # Ratio between maximum and minimum dwell time.
MAX_DEVIATION = 10 * 60  # Maximum deviation from the initial schedule (s).
RUNNING_TIME_MAX_FACTOR = 1.5  # Ratio between maximum and minimum running time.
HEADWAY_TIME_SAFETY_FACTOR = 1.5  # Safety factor applied to running times.
RUNNING_TIME_SUPPLEMENT={
	"sprinter": 1.07,
	"intercity": 1.07,
	"maglev": 1.07,
	"freight": 1.07
}
switching_time = {  # Switching time by pair of train categories (s).
	"maglev_maglev": 10,
	"maglev_rail": 20,
	"rail_maglev": 20,
	"rail_rail": 2,
}
sight_time = 2  # Sight time required before a movement (s).
T_additional = []  # Set of additional train services.
T_initial = []  # Set of initial train services.
V_additional = []  # Set of events belonging to additional services.
V_initial = []  # Set of events belonging to initial services.
N_additional = len(V_additional)  # Number of additional events.
N_initial = len(V_initial)  # Number of initial events.
T = T_initial + T_additional  # Set of all train services.
V = V_initial + V_additional  # Set of all events.

# Direct optimization inputs
# Binary-valued data, alphabetically ordered.
rho = np.empty((0, 0))  # 1 when service t stops at the location of event i.

# Integer/float data, alphabetically ordered.
c_1 = 1  # Objective weight for deviation from the initial schedule.
c_2 = 1  # Objective weight for additional train duration.
dwell_time_max = np.empty((0, 0, 0))  # Maximum dwell duration between events (s).
dwell_time_min = np.empty((0, 0, 0))  # Minimum dwell duration between events (s).
pi = np.empty(0)  # Scheduled time of each initial event (s from midnight).
running_time_max = np.empty((0, 0, 0))  # Maximum running duration between events (s).
running_time_min = np.empty((0, 0, 0))  # Minimum running duration between events (s).
tw_max = np.empty(0)  # Latest permitted time for additional events (s).
tw_min = np.empty(0)  # Earliest permitted time for additional events (s).

# Event and service index collections used by the optimization.
E_dwell = []  # Event pairs connected by dwell constraints.
E_travel = []  # Event pairs connected by running-time constraints.
event_keys: tuple[EventKey, ...] = ()  # Semantic keys for all events.

# Intermediate-only data used to create optimization inputs.
# Binary-valued data, alphabetically ordered.
l = np.empty((0, 0))  # 1 when two events use the same train service.
p = np.empty((0, 0))  # 1 when the first service has priority over the second.
q = np.empty((0, 0))  # 1 when two events occur at the same location.

# Integer/float data, alphabetically ordered.
I = ()  # Dense integer indices for all events.
headway_time_min = np.empty((0, 0))  # Minimum headway between event pairs (s).
s = np.empty(0)  # Location ID of each event.
t = np.empty(0)  # Internal train-service ID of each event.

# String-keyed/data collections, alphabetically ordered.
braking_times = {}  # Braking times indexed by node ID and route ID (s).
station_name_dict = {}  # Mapping from station names to station identifiers.
startup_travel_times = {}  # Startup travel times by rolling-stock type (s).
train_priorities = {}  # Route priority values loaded from route_data.csv.
train_number_dict = {}  # Mapping from real-world to internal train IDs.
U = []  # Rolling-stock types and their properties.

# Event, route, and service collections, alphabetically ordered.
E_headway = []  # Event pairs subject to headway constraints.
E_running = []  # Event pairs connected by route edges.
Events = []  # Event objects.
V_arrival = []  # Arrival events.
V_crossing = []  # Crossing events.
V_departure = []  # Departure events.
V_merge = []  # Merge events.
V_run_through = []  # Station run-through events.
V_split = []  # Split events.
event_key_to_index: dict[EventKey, int] = {}  # Semantic key to dense index.
index_to_event_key: dict[int, EventKey] = {}  # Dense index to semantic key.

def build_sequence_optimization_model(
	trip_max,
	dwell_min,
	dwell_max,
	time_window_min: np.ndarray,
	time_window_max: np.ndarray,
	train_services,
	additional_train_services,
	initial_event_ids,
	additional_event_ids,
	event_key_values,
	initial_schedule_times: np.ndarray,
	running_time_minimums: np.ndarray,
	stop_matrix: np.ndarray,
	travel_edges,
	travel_edge_services,
	dwell_edges,
	headway_edges,
	headway_time_minimums: np.ndarray,
	objective_initial_deviation,
	objective_additional_duration,
	max_deviation,
	big_m,
	event_key_fn,
	trains=None,
):
	"""Build the Gurobi model with event times represented by ``x[i]``.

	Event-set variables such as ``V_initial`` and ``V_additional`` contain
	integer event IDs. Schedule values in ``pi`` are parameters only; they do
	not fix ``x`` unless the caller adds an explicit equality constraint.
	"""
	if trains is None:
		trains = tuple(range(len(train_services)))
	else:
		trains = tuple(
			train["internal_id"] if isinstance(train, dict) else train
			for train in trains
		)
	model = gp.Model("Thesis Model")

	#time of event i
	x = model.addVars(event_key_values, vtype=GRB.CONTINUOUS, name="x", lb=0,)

	#travel time of train t
	delta_t = model.addVars(range(len(train_services)), vtype=GRB.CONTINUOUS, name="dt", lb=0)
	event_key_pairs = [
		(first_key, second_key)
		for first_key in event_key_values
		for second_key in event_key_values
	]

	#=1 if i is before j, 0 otherwise
	a = model.addVars(event_key_pairs, vtype=GRB.BINARY, name="a")

	#difference between the scheduled and actual time of event i
	ad = model.addVars([event_key_fn(index) for index in initial_event_ids],vtype=GRB.CONTINUOUS,name="ad",lb=0)

	model.setObjective(
		gp.quicksum(objective_initial_deviation * ad[event_key_fn(i)] for i in initial_event_ids)
		+ objective_additional_duration * gp.quicksum(
			delta_t[service["internal_id"]]
			for service in additional_train_services
		)
	)

	services_by_id = {
		service["internal_id"]: service
		for service in train_services
	}
	model.addConstrs(
		(
			delta_t[service_id]
			== x[event_key_fn(services_by_id[service_id]["arrival_event"])]
			- x[event_key_fn(services_by_id[service_id]["departure_event"])]
			for service_id in range(len(train_services))
		),
		name="service_duration",
	)

	model.addConstrs( #additional trains must be after beginning of time window
		(
			x[event_key_fn(i)] >= time_window_min[i]
			for i in additional_event_ids
		),
		name="min_time_window_additional",
	)
	model.addConstrs( #additional trains must be before end of time window
		(
			x[event_key_fn(i)] <= time_window_max[i]
			for i in additional_event_ids
		),
		name="max_time_window_additional",
	)

	model.addConstrs( #initial trains cannot be more than MAX_DEVIATION seconds later than the initial schedule
		x[event_key_fn(i)] - initial_schedule_times[i] <= max_deviation
		for i in initial_event_ids
	)
	model.addConstrs( #initial trains cannot be more than MAX_DEVIATION seconds earlier than the initial schedule
		initial_schedule_times[i] - x[event_key_fn(i)] <= max_deviation
		for i in initial_event_ids
	)

	model.addConstrs(
		(
			x[event_key_fn(j)] - x[event_key_fn(i)]
			>= running_time_minimums[i, j, service_id]
			for (i, j), service_id in zip(
				travel_edges,
				travel_edge_services,
			)
		),
		name="travel_time_min",
	)
	model.addConstrs(
		(
			x[event_key_fn(j)] - x[event_key_fn(i)]
			<= trip_max[i, j, service_id]
			for (i, j), service_id in zip(
				travel_edges,
				travel_edge_services,
			)
		),
		name="travel_time_max",
	)
	model.addConstrs(
		(
			x[event_key_fn(j)] - x[event_key_fn(i)]
			>= stop_matrix[i, t] * dwell_min[i, j, t]
			for i, j in dwell_edges
			for t in trains
			if stop_matrix[i, t] > 0
		),
		name="dwell_time_min",
	)
	model.addConstrs(
		(
			x[event_key_fn(j)] - x[event_key_fn(i)]
			<= stop_matrix[i, t] * dwell_max[i, j, t]
			for i, j in dwell_edges
			for t in trains
			if stop_matrix[i, t] > 0
		),
		name="dwell_time_max",
	)
	model.addConstrs(
		(
			x[event_key_fn(j)] - x[event_key_fn(i)]
			>= headway_time_minimums[i, j]
			* a[event_key_fn(i), event_key_fn(j)]
			- big_m * (
				1 - a[event_key_fn(i), event_key_fn(j)]
			)
			for i, j in headway_edges
		),
		name="headway_constraint_1",
	)

	model.addConstrs(
		(
			x[event_key_fn(i)] - x[event_key_fn(j)]
			>= headway_time_minimums[j, i]
			* (1 - a[event_key_fn(i), event_key_fn(j)])
			- big_m * a[event_key_fn(i), event_key_fn(j)]
			for i, j in headway_edges
		),
		name="headway_constraint_2",
	)

	model.addConstrs(#part 1 of linearizing absolute value for deviation from initial schedule
		(
			ad[event_key_fn(i)] >= x[event_key_fn(i)] - initial_schedule_times[i]
			for i in initial_event_ids
		)
	)

	model.addConstrs(#part 2 of linearizing absolute value for deviation from initial schedule
		(
			ad[event_key_fn(i)] >= initial_schedule_times[i] - x[event_key_fn(i)]
			for i in initial_event_ids
		)
	)

	return model, {
		"x": x,
		"delta_t": delta_t,
		"a": a,
		"ad": ad,
	}

class event:
	def __init__(
		self,
		event_id,
		is_initial,
		service_id_internal,
		service_id_external,
		location_id,
		event_type,
		train_type=None,
		train_model=None,
		initial_time=None,
		route_id=None,
		related_route_id=None,
		node=None,
	):
		self.event_id = event_id
		self.service_id_internal = service_id_internal
		self.service_id_external = service_id_external
		self.location_id = location_id
		self.event_type = event_type
		self.is_initial = is_initial
		self.train_type = train_type
		self.train_model = train_model
		self.scheduled_time = initial_time
		self.initial_time = initial_time
		self.route_id = route_id
		self.related_route_id = related_route_id
		self.node = node
		self.window_min: int | None = None
		self.window_max: int | None = None

	@property
	def event_key(self) -> EventKey:
		if self.route_id is None:
			raise ValueError("An event must have a route ID")
		return (
			int(self.event_id),
			int(self.location_id),
			str(self.event_type),
			bool(self.is_initial),
			str(self.train_type),
			int(self.initial_time or 0),
			int(self.route_id),
		)

if __name__ == "__main__":
	main()
