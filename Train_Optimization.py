import gurobipy as gp
from gurobipy import GRB
import numpy as np
import pandas as pd
import pickle


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

class event:
	def __init__(self, event_id, is_initial, service_id_internal, 
	             service_id_external, location_id, event_type, train_type, train_model):
		self.event_id = event_id
		self.service_id_internal = service_id_internal
		self.service_id_external = service_id_external
		self.location_id = location_id
		self.event_type = event_type
		self.is_initial = is_initial
		self.train_type = train_type #maglev or rail
		self.train_model = train_model

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



M=1e6 #big M for constraints
Events = [] #set of events, list of event classes
V_initial = [] #set of initial events
V_additional = [] #set of additional events
N_initial = len(V_initial) #number of initial events
N_additional = len(V_additional) #number of additional events


#Chopping Block
V_departure = [] #set of departure events
V_arrival = [] #set of arrival events
V_split = [] #set of split events
V_merge = [] #set of merge events
V_crossing = [] #set of crossing events
V_rt = [] #set of station run through events
S = [] #set of locations
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
q = np.zeros((N_initial+N_additional, N_initial+N_additional)) #1 if events i and j are at the same location, 0 otherwise
l = np.zeros((N_initial+N_additional, N_initial+N_additional)) #1 if events i and j are on the same train service, 0 otherwise




















sequence_optimization_model = gp.Model("Thesis Model")

x=sequence_optimization_model.addVars(I, vtype=GRB.CONTINUOUS, name='x', lb=0) #time at which event i takes place
delta_t=sequence_optimization_model.addVars(I, vtype=GRB.CONTINUOUS, name='dt', lb=0) #travel time of train t
a=sequence_optimization_model.addVars(I, vtype=GRB.BINARY, name='a') #=1 if j is after or simultaneous with i, =0 otherwise
ad=sequence_optimization_model.addVars(I, vtype=GRB.CONTINUOUS, name='ad') #time adjustment of event i




sequence_optimization_model.addconstrs((x[i] >= tw_min[i] for i in V_additional), name='min_time_window')
sequence_optimization_model.addconstrs((x[i] <= tw_max[i] for i in V_additional), name='max_time_window')
sequence_optimization_model.addconstrs((x[j] - x[i] >= running_time_min[i,j] + rho[i]*acc[m] + rho[j]*decc[m] for i, j in E_travel for m in trains), name='travel_time_min')
sequence_optimization_model.addconstrs((x[j] - x[i] <= trip_max[i,j] + rho[i]*acc[m] + rho[j]*decc[m] for i, j in E_travel for m in trains), name='travel_time_max')
sequence_optimization_model.addconstrs((x[j] - x[i] >= rho[i]*dwell_min[i,j] for i, j in E_dwell), name='dwell_time_min')
sequence_optimization_model.addconstrs((x[j] - x[i] <= rho[i]*dwell_max[i,j] for i, j in E_dwell), name='dwell_time_max')
