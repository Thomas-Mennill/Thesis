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


rail_rail_switching_time = 2 #seconds
rail_maglev_switching_time = 20 #seconds
maglev_maglev_switching_time = 10 #seconds
maglev_rail_switching_time = 20 #seconds
V_initial = [] #set of initial events
V_additional = [] #set of additional events
V_departure = [] #set of departure events
V_arrival = [] #set of arrival events
E_dwell = [] #set of dwell edges
E_travel = [] #set of travel edges
E_headway = [] #set of headway edges
E_headway_restricted = [] #set of headway edges with timing restriction
T = [] #set of existing train services
T_additional = [] #set of additional train services
trip_max = [] #max duration of trip i, j
trip_min = [] #min duration of trip i, j
dwell_max = [] #max duration of dwell i, j
dwell_min = [] #min duration of dwell i, j
S = [] #set of stations
H = [] #set of minimum headways between events i, j ASYMMETRICAL
pi = [] #time at which event i takes place in the initial schedule
decc = [] #set of deceleration times
L = [] #=1 if travel from track segment i to track segment j is possible, =0 otherwise
P = [] #=1 if train i has priority over train j, =0 otherwise
tw_min = [] #set of beginning time windows for event i
tw_max = [] #set of ending time windows for event i
track_segments = {} #set of track segments
track_data = [] #data for track segment i
sw=[] #set of switches 
train_number_dict = {} #dictionary of real-world train IDs and their corresponding train IDs in the code
station_name_dict = {} #dictionary of real-world full station names and their UIC Codes
rho = [] #1 if train t stops at station j, 0 otherwise
trains = [] #set of train types
M=1e6 #big M for constraints
r = [] #1 if travel from track segment i to track segment j is possible, 0 otherwise






sequence_optimization_model = gp.Model("Thesis Model")

x=sequence_optimization_model.addVars(I, vtype=GRB.CONTINUOUS, name='x', lb=0) #time at which event i takes place
dt=sequence_optimization_model.addVars(I, vtype=GRB.CONTINUOUS, name='dt', lb=0) #travel time of train t
a=sequence_optimization_model.addVars(I, vtype=GRB.BINARY, name='a') #=1 if j is after or simultaneous with i, =0 otherwise
b=sequence_optimization_model.addVars(I, vtype=GRB.BINARY, name='b') #=1 if event i happens on track segment k, =0 otherwise
ad=sequence_optimization_model.addVars(I, vtype=GRB.CONTINUOUS, name='ad') #time adjustment of event i




sequence_optimization_model.addconstrs((x[i] >= tw_min[i] for i in V_additional), name='min_time_window')
sequence_optimization_model.addconstrs((x[i] <= tw_max[i] for i in V_additional), name='max_time_window')
sequence_optimization_model.addconstrs((x[j] - x[i] >= trip_min[i,j] + rho[i]*acc[m] + rho[j]*decc[m] for i, j in E_travel for m in trains), name='travel_time_min')
sequence_optimization_model.addconstrs((x[j] - x[i] <= trip_max[i,j] + rho[i]*acc[m] + rho[j]*decc[m] for i, j in E_travel for m in trains), name='travel_time_max')
sequence_optimization_model.addconstrs((x[j] - x[i] >= rho[i]*dwell_min[i,j] for i, j in E_dwell), name='dwell_time_min')
sequence_optimization_model.addconstrs((x[j] - x[i] <= rho[i]*dwell_max[i,j] for i, j in E_dwell), name='dwell_time_max')
#room for constraint 7 from paper

#headways depend on train order, but they can be precalcuated to keep the program MILP
sequence_optimization_model.addconstrs((x[j] - x[i] >= H[i,j] - (1-a[i,j])*M for i, j in E_headway), name='headway_min')
sequence_optimization_model.addconstrs((x[j] - x[i] <= H[i,j]*(1-a[i,j]) - M*a[i,j] for i, j in E_headway), name='headway_max')
sequence_optimization_model.addconstrs((a[i,j] == a[k,m] for i, k in E_headway for j, m in E_headway), name='headway_ordering')
#sequence_optimization_model.addconstrs((b[i,j] + b[i,k] <= 1 for i in V_additional for j, k in sw), name='switching_constraint')
sequence_optimization_model.addconstrs((a[i,j] - a[k,m] <= 0 for i, j in E_dwell for k, m in E_dwell), name='low_priority_overtaking')
sequence_optimization_model.addconstrs((a[i,j] - a[k,m] > 0 for i, j in E_dwell for k, m in E_dwell), name='low_priority_overtaking_2')
sequence_optimization_model.addconstrs((x[i] - pi[i] >= ad[i] for i in V_initial), name='initial_time_adjustment')
sequence_optimization_model.addconstrs((pi[i] - x[i] >= ad[i] for i in V_initial), name='initial_time_adjustment_2')
sequence_optimization_model.addconstrs((b[i,k])

# sum(b[i,k]*b[j,k])*something in headway, so that the acceptable minimum headway range becomes unlimited if the two events are not on the same track segment
# 