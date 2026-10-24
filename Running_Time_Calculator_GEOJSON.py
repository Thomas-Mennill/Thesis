"""
TO DO

Verify cant angle stuff
Verify slope angle stuff


"""


import csv
import pandas as pd
import math
import pickle
import re
from pathlib import Path
import numpy as np
import networkx as nx
import scipy.interpolate as interp
import geopandas as gpd
import folium
import matplotlib.pyplot as plt
from shapely.geometry import LineString

from Display_Routes import build_used_routes_network


ROUTE_GRAPH_FILE = "case_study_route_graphs.pkl"
EXISTING_ROUTE_STOPS_FILE = "existing_rail_routes.csv"
NEW_ROUTE_STOPS_FILE = "new_rail_routes.csv"
ROUTE_DATA_FILE = "route_data.csv"
ROLLING_STOCK_DATA_FILE = "rolling_stock_data.csv"
INFRASTRUCTURE_GRAPH_FILE = "case_study_network.gpickle"
RESULTS_FILE = "running_time_results.csv"
NODE_RESULTS_FILE = "running_time_between_nodes.csv"
BRAKING_TIME_FILE = "braking_times_by_node.csv"
ROUTE_CSV_FOLDER = "running_time_by_route"
ROUTE_MAP_FOLDER = "route_speed_maps"
train_type = "conventional"


def load_route_station_stops(csv_file):
    """Return station stops grouped by train ID from the timetable CSV."""
    with open(csv_file, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"Route file {csv_file!r} is empty")

    station_stops = {}
    for column, header in enumerate(rows[0]):
        train_id = header.strip()
        if not train_id.isdigit():
            continue
        train_number = int(train_id)
        stops = []
        for row in rows[1:]:
            if column >= len(row):
                continue
            station = row[column].strip()
            if station and (not stops or stops[-1] != station):
                stops.append(station)
        station_stops[train_number] = stops
    if not station_stops:
        raise ValueError(f"No train station stops found in {csv_file!r}")
    return station_stops


def load_route_station_stops_files(*csv_files):
    """Merge station stops from multiple timetable CSV files."""
    if not csv_files:
        raise ValueError("At least one station-stop CSV file is required")
    station_stops = {}
    for csv_file in csv_files:
        for train_id, stops in load_route_station_stops(csv_file).items():
            if train_id in station_stops:
                raise ValueError(
                    f"Train ID {train_id} appears in multiple station-stop "
                    f"files"
                )
            station_stops[train_id] = stops
    return station_stops


def load_route_stop_durations(csv_file=ROUTE_DATA_FILE):
    """Return the scheduled stop duration in seconds for each train route."""
    with open(csv_file, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"Route data file {csv_file!r} is empty")

    duration_row = next(
        (
            row for row in rows
            if row and row[0].strip().lower() == "stop duration (s)"
        ),
        None,
    )
    if duration_row is None:
        raise ValueError(
            f"Route data file {csv_file!r} has no stop duration row"
        )

    stop_durations = {}
    for column, route_id in enumerate(rows[0][1:], start=1):
        route_id = route_id.strip()
        if not route_id:
            continue
        try:
            train_id = int(route_id)
            stop_durations[train_id] = float(duration_row[column])
        except (IndexError, ValueError) as error:
            raise ValueError(
                f"Invalid stop duration for route {route_id!r}"
            ) from error
    if not stop_durations:
        raise ValueError(f"No route stop durations found in {csv_file!r}")
    return stop_durations


def load_route_train_types(csv_file=ROUTE_DATA_FILE):
    """Return the rolling-stock type for each train route."""
    with open(csv_file, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"Route data file {csv_file!r} is empty")
    type_row = next(
        (
            row for row in rows
            if row and row[0].strip().lower() == "train type"
        ),
        None,
    )
    if type_row is None:
        raise ValueError(f"Route data file {csv_file!r} has no train type row")
    train_types = {}
    for column, route_id in enumerate(rows[0][1:], start=1):
        route_id = route_id.strip()
        if not route_id:
            continue
        try:
            train_types[int(route_id)] = type_row[column].strip()
        except (IndexError, ValueError) as error:
            raise ValueError(
                f"Invalid train type for route {route_id!r}"
            ) from error
    if not train_types:
        raise ValueError(f"No route train types found in {csv_file!r}")
    return train_types


def load_rolling_stock_data(csv_file=ROLLING_STOCK_DATA_FILE):
    """Load rolling-stock attributes and their composite aggregation rules."""
    frame = pd.read_csv(csv_file, index_col=0, encoding="utf-8-sig")
    normalized_rows = {
        str(index).strip().casefold().replace("_", " "): index
        for index in frame.index
        if not pd.isna(index) and str(index).strip()
    }
    if "max speed (km/h)" not in normalized_rows:
        raise ValueError(
            f"Rolling-stock file {csv_file!r} has no maximum-speed row"
        )
    behaviour_column = next(
        (
            column for column in frame.columns
            if str(column).strip().casefold() == "combined behaviour"
        ),
        None,
    )
    if behaviour_column is None:
        raise ValueError(
            f"Rolling-stock file {csv_file!r} has no Combined Behaviour column"
        )
    return frame, normalized_rows, behaviour_column

    """Numerically calculate braking time from ``speed_limit`` to zero.

    The braking force is combined with Davis resistance for conventional
    trains and aerodynamic resistance for maglev trains.
    """
    attributes = resolve_rolling_stock_attributes(
        train_type,
        rolling_stock_data,
    )
    brake_acceleration = attributes.get("brake acceleration (m/s^2)")
    if brake_acceleration is None or brake_acceleration <= 0:
        raise ValueError(
            f"Rolling-stock data for {train_type!r} has no positive "
            "brake acceleration"
        )
    speed_limit = float(speed_limit)
    if not np.isfinite(speed_limit) or speed_limit < 0:
        raise ValueError("speed_limit must be finite and non-negative")
    if not np.isfinite(time_step) or time_step <= 0:
        raise ValueError("time_step must be finite and positive")
    if speed_limit == 0:
        return 0.0

    mass = attributes["mass (kg)"]
    rotating_factor = attributes.get("rotating mass factor", 1.0)
    brake_force = brake_acceleration * rotating_factor * mass
    normalized_type = str(train_type).strip().casefold()
    if normalized_type == "maglev":
        required = ("drag coefficient", "width (m)", "height (m)")
        if any(attributes.get(name) is None for name in required):
            raise ValueError(
                f"Maglev type {train_type!r} lacks aerodynamic data"
            )
        air_density = 1.204
        frontal_area = attributes["width (m)"] * attributes["height (m)"]

        def resistance(speed):
            return calculate_air_resistance(
                speed,
                attributes["drag coefficient"],
                air_density,
                frontal_area,
            )
    else:
        required = (
            "constant resistance (n)",
            "linear resistance (ns/m)",
            "quadratic resistance (ns^2/m^2)",
        )
        if any(attributes.get(name) is None for name in required):
            raise ValueError(
                f"Conventional type {train_type!r} lacks Davis resistance data"
            )

        def resistance(speed):
            return calculate_davis_resistance(attributes, speed)

    _velocities, _distances, times = get_brake_curve(
        0,
        speed_limit,
        0.0,
        brake_force,
        mass,
        9.81,
        np.array([0.0]),
        rotating_factor,
        resistance,
        dt=time_step,
        return_times=True,
    )
    return float(times[-1]) if times else 0.0
    """Return normalized-route nodes at every speed-limit change."""
    distances = np.asarray(results[7], dtype=float)
    speed_limits = np.asarray(results[10], dtype=float)
    change_points = find_speed_limit_changes(speed_limits)
    return [
        {
            "node": index + 1,
            "index": int(point),
            "distance": float(distances[point]),
            "previous_speed_limit": float(speed_limits[point - 1]),
        }
        for index, point in enumerate(change_points)
        if point > 0
    ]

#writes braking times to a CSV file
def write_braking_time_csv(route_data_by_train, train_types_by_train, rolling_stock_data, output_file=BRAKING_TIME_FILE,):
    """Write stopping times before every node in each train route.

    Rows are ordered route-node ordinals and columns are train routes. Each
    cell is the time in seconds to stop from the preceding route node's speed
    limit using that route's rolling-stock type. The first route node has no
    preceding node and therefore receives a braking time of zero.
    """
    route_nodes = {}
    for train_id, (_route_network, results) in route_data_by_train.items():
        train_type = train_types_by_train.get(train_id)
        if train_type is None:
            raise KeyError(f"No train type found for route {train_id!r}")
        speed_limits = np.asarray(results[10], dtype=float)
        alphas_uniform = np.asarray(results[8], dtype=float)
        distances = np.asarray(results[7], dtype=float)
        attributes = resolve_rolling_stock_attributes(
            train_type,
            rolling_stock_data,
        )
        mass = attributes["mass (kg)"]
        rotating_factor = attributes.get("rotating mass factor", 1.0)
        brake_force = (
            attributes["brake acceleration (m/s^2)"]
            * rotating_factor
            * mass
        )
        normalized_type = str(train_type).strip().casefold()
        if normalized_type == "maglev":
            air_density = 1.204
            frontal_area = attributes["width (m)"] * attributes["height (m)"]

            def resistance_function(speed):
                return calculate_air_resistance(
                    speed,
                    attributes["drag coefficient"],
                    air_density,
                    frontal_area,
                )
        else:

            def resistance_function(speed):
                return calculate_davis_resistance(attributes, speed)

        all_route_nodes = results[13]
        route_nodes[train_id] = []
        for node_index in range(len(all_route_nodes)):
            if node_index == 0:
                route_nodes[train_id].append(0.0)
                continue

            previous_node_distance = all_route_nodes[node_index - 1]["distance"]
            previous_speed_limit = float(np.interp(
                previous_node_distance,
                distances,
                speed_limits,
            ))
            brake_index = min(
                len(alphas_uniform) - 1,
                max(
                    1,
                    int(np.searchsorted(
                        distances,
                        previous_node_distance,
                        side="left",
                    )),
                ),
            )
            if previous_speed_limit <= 0:
                route_nodes[train_id].append(0.0)
            else:
                _velocities, _braking_distances, braking_times = get_brake_curve(
                    brake_index,
                    previous_speed_limit,
                    0.0,
                    brake_force,
                    mass,
                    9.81,
                    alphas_uniform,
                    rotating_factor,
                    resistance_function,
                    dt=0.1,
                    return_times=True,
                )
                route_nodes[train_id].append(float(braking_times[-1]))

    route_ids = list(route_nodes)
    row_count = max(
        (len(values) for values in route_nodes.values()),
        default=0,
    )
    with open(output_file, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Node"] + [
            f"Route {train_id}" for train_id in route_ids
        ])
        for row_index in range(row_count):
            writer.writerow([
                f"Node {row_index + 1}",
                *[
                    values[row_index] if row_index < len(values) else ""
                    for values in (
                        route_nodes[train_id] for train_id in route_ids
                    )
                ],
            ])

#handles multiple train sets being coupled together
def resolve_rolling_stock_attributes(vehicle_type, rolling_stock_data,):
    """Resolve a single or combined vehicle type using the CSV rules."""
    frame, normalized_rows, behaviour_column = rolling_stock_data
    is_combined_vehicle = "+" in str(vehicle_type)
    components = [
        component.strip()
        for component in str(vehicle_type).split("+")
        if component.strip()
    ]
    if not components:
        raise ValueError("Vehicle type cannot be empty")

    component_columns = {}
    for component in components:
        key = component.casefold()
        matches = [
            column for column in frame.columns
            if str(column).strip().casefold() == key
        ]
        if not matches:
            raise ValueError(
                f"Vehicle type {component!r} is not present in rolling-stock data"
            )
        component_columns[component] = matches[0]

    attributes = {}
    for normalized_row, row_name in normalized_rows.items():
        if normalized_row in {"combined behaviour", "sources"}:
            continue
        values = []
        for component in components:
            value = frame.at[row_name, component_columns[component]]
            if pd.isna(value) or not str(value).strip():
                continue
            try:
                values.append(float(value))
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid {row_name!r} value for vehicle {component!r}"
                ) from error
        if not values:
            continue
        if not is_combined_vehicle or len(values) == 1:
            combined_value = values[0]
        else:
            behaviour = str(
                frame.at[row_name, behaviour_column]
            ).strip().casefold()
            if behaviour == "sum":
                combined_value = sum(values)
            elif behaviour == "min":
                combined_value = min(values)
            elif behaviour == "max":
                combined_value = max(values)
            elif behaviour == "give error if not equal":
                if not np.allclose(values, values[0]):
                    raise ValueError(
                        f"Cannot combine {vehicle_type!r}: values for "
                        f"{row_name!r} are not equal"
                    )
                combined_value = values[0]
            elif behaviour == "weighted average, based on length":
                length_row = normalized_rows.get("length (m)")
                if length_row is None:
                    raise ValueError(
                        f"Cannot combine {vehicle_type!r}: length data is "
                        f"required for {row_name!r}"
                    )
                lengths = [
                    float(frame.at[length_row, component_columns[component]])
                    for component in components
                ]
                if any(length <= 0 for length in lengths):
                    raise ValueError(
                        f"Cannot combine {vehicle_type!r}: length data must "
                        f"be positive for {row_name!r}"
                    )
                combined_value = np.average(values, weights=lengths)
            elif pd.isna(frame.at[row_name, behaviour_column]) and np.allclose(
                values,
                values[0],
            ):
                combined_value = values[0]
            else:
                raise ValueError(
                    f"Unsupported Combined Behaviour {behaviour!r} "
                    f"for {row_name!r}"
                )
        attributes[normalized_row] = combined_value
    return attributes


def _route_points(route_network, include_speed_limits=False, include_route_segments=False,):
    """Extract ordered x/y/z points and optional infrastructure limits."""
    def same_point(first, second):
        return math.hypot(
            float(first[0]) - float(second[0]),
            float(first[1]) - float(second[1]),
        ) <= 1e-6

    waypoints = route_network.graph.get("route_node_coordinates")
    if not waypoints:
        raise ValueError("Route network has no ordered route nodes")

    coordinates = []
    infrastructure_speed_limits = []
    route_segments = []
    assigned_edges = route_network.graph.get("assigned_edges", [])
    for segment_index, (start, end) in enumerate(zip(waypoints, waypoints[1:])):
        try:
            path = nx.shortest_path(route_network, start, end)
        except nx.NetworkXNoPath as error:
            raise ValueError(
                f"No directed path between route nodes {start!r} and {end!r}"
            ) from error
        segment_start = len(coordinates)
        segment_station = None
        segment_edges = []
        for first, second in zip(path, path[1:]):
            edge_data = route_network.get_edge_data(first, second)
            if not edge_data:
                raise ValueError(
                    f"Missing edge data for route edge {first!r}->{second!r}"
                )
            attributes = next(iter(edge_data.values()))
            if segment_station is None:
                segment_station = attributes.get("station")
            geometry = attributes.get("geometry")
            if not geometry:
                geometry = [first, second]
            edge_speed_limits = (
                [parse_speed_limit(attributes.get("maxspeed"))]
                * len(geometry)
                if include_speed_limits
                else []
            )
            if same_point(geometry[-1], first):
                geometry = list(reversed(geometry))
                edge_speed_limits.reverse()
            if not same_point(geometry[0], first):
                raise ValueError(
                    f"Edge geometry does not start at route node {first!r}"
                )
            shares_previous_point = (
                bool(coordinates)
                and same_point(geometry[0], coordinates[-1])
            )
            edge_start = (
                len(coordinates) - 1
                if shares_previous_point
                else len(coordinates)
            )
            if shares_previous_point:
                geometry = geometry[1:]
                edge_speed_limits = edge_speed_limits[1:]
            coordinates.extend(geometry)
            if include_speed_limits:
                infrastructure_speed_limits.extend(edge_speed_limits)
            segment_edges.append({
                "edge_id": attributes.get("edge_id"),
                "origin": first,
                "destination": second,
                "start_index": edge_start,
                "end_index": len(coordinates) - 1,
            })

        if include_route_segments and len(coordinates) > segment_start:
            route_segments.append({
                "start_index": segment_start,
                "end_index": len(coordinates) - 1,
                "path_nodes": path,
                "station": segment_station,
                "edge_id": (
                    assigned_edges[segment_index]
                    if segment_index < len(assigned_edges)
                    else None
                ),
                "edges": segment_edges,
            })

    if len(coordinates) < 2:
        raise ValueError("Route network contains fewer than two input points")
    if include_speed_limits:
        result = coordinates, infrastructure_speed_limits
        if include_route_segments:
            return (*result, route_segments)
        return result
    return coordinates


def _station_stop_distances(route_network, route_segments, coordinates, distances, station_stops,):
    """Return stops at the second labelled node of each non-origin station.

    The first station in ``station_stops`` is the route origin and is not a
    stopping point. For every later requested station, traversal must reach
    the second node labelled with that station before the train stops. This
    deliberately uses ordered route nodes rather than segment lengths or the
    number of platform edges.
    """
    if not station_stops:
        return []

    origin_station = station_stops[0]
    requested_stations = set(station_stops[1:]) - {origin_station}
    if not requested_stations:
        return []

    def station_key(station):
        return " ".join(str(station).split()).casefold()

    occurrences = {
        station_key(station): []
        for station in requested_stations
    }
    seen_nodes = set()
    for segment in route_segments:
        start_index = segment["start_index"]
        end_index = segment["end_index"]
        if end_index < start_index:
            continue
        coordinate_slice = coordinates[start_index:end_index + 1]
        distance_slice = distances[start_index:end_index + 1]
        for node in segment["path_nodes"]:
            if node in seen_nodes:
                continue
            seen_nodes.add(node)
            station = route_network.nodes[node].get("station")
            normalized_station = station_key(station)
            if normalized_station not in occurrences:
                continue
            if not coordinate_slice:
                continue
            node_index = min(
                range(len(coordinate_slice)),
                key=lambda index: (
                    (float(coordinate_slice[index][0]) - float(node[0])) ** 2
                    + (float(coordinate_slice[index][1]) - float(node[1])) ** 2
                ),
            )
            occurrences[normalized_station].append(
                float(distance_slice[node_index])
            )

    stop_distances = []
    for station in station_stops[1:]:
        normalized_station = station_key(station)
        if normalized_station not in occurrences:
            continue
        station_occurrences = occurrences[normalized_station]
        if len(station_occurrences) < 2:
            raise ValueError(
                f"Station {station!r} has only "
                f"{len(station_occurrences)} labelled route node(s); "
                "the stopping rule requires the second labelled node."
            )
        stop_distances.append(station_occurrences[1])
    return sorted(stop_distances)


def _simplified_route_events(route_network, infrastructure_graph, route_segments, coordinates, distances,):
    """Return simplified station/junction events and their route distances."""
    events = []
    for segment in route_segments:
        start_index = segment["start_index"]
        end_index = segment["end_index"]
        for node in segment["path_nodes"]:
            station = route_network.nodes[node].get("station")
            if station not in (None, "None"):
                event_node = station
            elif infrastructure_graph.degree(node) > 2:
                event_node = node
            else:
                continue

            node_index = min(
                range(start_index, end_index + 1),
                key=lambda index: math.hypot(
                    float(coordinates[index][0]) - float(node[0]),
                    float(coordinates[index][1]) - float(node[1]),
                ),
            )
            if events and events[-1]["node"] == event_node:
                continue
            events.append({
                "node": event_node,
                "distance": float(distances[node_index]),
            })
    return events


def _route_node_events(route_network, route_segments, coordinates, distances):
    """Return every ordered route node and its distance along the route."""
    events = []
    seen_nodes = set()
    for segment in route_segments:
        start_index = segment["start_index"]
        end_index = segment["end_index"]
        coordinate_slice = coordinates[start_index:end_index + 1]
        distance_slice = distances[start_index:end_index + 1]
        for node in segment["path_nodes"]:
            if node in seen_nodes or not coordinate_slice:
                continue
            seen_nodes.add(node)
            coordinate_index = min(
                range(len(coordinate_slice)),
                key=lambda index: (
                    (float(coordinate_slice[index][0]) - float(node[0])) ** 2
                    + (float(coordinate_slice[index][1]) - float(node[1])) ** 2
                ),
            )
            events.append({
                "node": node,
                "station": route_network.nodes[node].get("station"),
                "distance": float(distance_slice[coordinate_index]),
            })
    events.sort(key=lambda event: event["distance"])
    return events


def _speed_color(speed_kmh, minimum_speed, maximum_speed):
    """Return a blue-to-green-to-red color for a speed value."""
    if maximum_speed <= minimum_speed:
        ratio = 0.0
    else:
        ratio = (speed_kmh - minimum_speed) / (maximum_speed - minimum_speed)
    ratio = max(0.0, min(1.0, ratio))
    if ratio <= 0.5:
        red = 0
        green = round(510 * ratio)
        blue = round(255 * (1 - 2 * ratio))
    else:
        red = round(510 * (ratio - 0.5))
        green = round(255 * (1 - 2 * (ratio - 0.5)))
        blue = 0
    return f"#{red:02x}{green:02x}{blue:02x}"


def write_first_train_speed_map(route_network, positions, velocities, train_id, output_file,):
    """Write an HTML map of the first train's simulated route speed."""
    coordinates = _route_points(route_network)
    projected_line = gpd.GeoSeries(
        [LineString(coordinates)], crs="EPSG:4326"
    ).to_crs(epsg=3310).iloc[0]
    projected_coordinates = list(projected_line.coords)
    route_distances = [0.0]
    for first, second in zip(projected_coordinates, projected_coordinates[1:]):
        route_distances.append(
            route_distances[-1]
            + math.hypot(second[0] - first[0], second[1] - first[1])
        )

    speeds_kmh = np.asarray(velocities) * 3.6
    minimum_speed = float(speeds_kmh.min())
    maximum_speed = float(speeds_kmh.max())
    center = [
        sum(float(coordinate[1]) for coordinate in coordinates) / len(coordinates),
        sum(float(coordinate[0]) for coordinate in coordinates) / len(coordinates),
    ]
    speed_map = folium.Map(location=center, zoom_start=10, control_scale=True)

    for index, (first, second) in enumerate(zip(coordinates, coordinates[1:])):
        midpoint_distance = (
            route_distances[index] + route_distances[index + 1]
        ) / 2
        speed = float(np.interp(midpoint_distance, positions, speeds_kmh))
        folium.PolyLine(
            locations=[
                [float(first[1]), float(first[0])],
                [float(second[1]), float(second[0])],
            ],
            color=_speed_color(speed, minimum_speed, maximum_speed),
            weight=5,
            opacity=0.9,
            tooltip=f"Train {train_id}: {speed:.1f} km/h",
        ).add_to(speed_map)

    legend = f"""
    <div style="position: fixed; bottom: 30px; left: 30px;
                width: 230px; z-index: 9999; background: white;
                border: 2px solid grey; padding: 8px;">
      <b>Train {train_id} speed</b><br>
      <div style="height: 12px; margin: 5px 0; background:
           linear-gradient(to right, blue, green, red);"></div>
      <span style="float: left;">{minimum_speed:.1f} km/h</span>
      <span style="float: right;">{maximum_speed:.1f} km/h</span>
    </div>
    """
    speed_map.get_root().html.add_child(folium.Element(legend))
    speed_map.fit_bounds([
        [min(float(coordinate[1]) for coordinate in coordinates),
         min(float(coordinate[0]) for coordinate in coordinates)],
        [max(float(coordinate[1]) for coordinate in coordinates),
         max(float(coordinate[0]) for coordinate in coordinates)],
    ])
    speed_map.save(output_file)

def get_brake_curve(i, v_max, v_target, F_brake, m, g, alphas_uniform, rotating_factor, resistance_function, dx=1, dt=0.1, return_times=False,):
    velocities = []
    distances = []
    times = []

    v_current = v_max
    x_current = 0
    elapsed = 0.0

    while(v_current > v_target): 
        distances.append(x_current)
        velocities.append(v_current)
        times.append(elapsed)

        # Same braking equation as main simulation
        resistance = resistance_function(v_current)
        dv = (
            F_brake
            - m * g * math.sin(alphas_uniform[i - 1])
            - resistance
        ) * dt / (rotating_factor * m)
        if not np.isfinite(dv) or dv <= 0:
            raise RuntimeError("Brake curve calculation did not converge")
        dv = min(dv, v_current - v_target)
        v_current -= dv
        
        # Calculate distance traveled in this time step
        x_current += v_current * dt
        elapsed += dt

    # Interpolate to get velocities at regular distance intervals
    if len(distances) > 1:
        brake_curve_interpolated = np.interp(np.arange(0, x_current, dx), distances, velocities)
    else:
        brake_curve_interpolated = np.array([v_max])
    
    if return_times:
        times.append(elapsed)
        return brake_curve_interpolated, np.asarray(distances), np.asarray(times)
    return brake_curve_interpolated

def find_speed_limit_changes(speed_limits):
    tolerance = 1E-6
    change_points = [
        i for i in range(1, len(speed_limits))
        if abs(speed_limits[i] - speed_limits[i - 1]) > tolerance
    ]

    return change_points

#speed can be limited in four ways:
#max speed of the train, max tolerable centripetal acceleration from a comfort standpoint
#max centripetal acceleration from track cant angle, and a train stopping for a station or having some other limit

#this needs to be modified because currently, if theta is 0, the speed limit will be 0. Small angle approximation?
def calculate_speed_limit(r, g, max_centripetal_acceleration, vehicle_max_speed, theta=0, vehicle_type=None, infrastructure_speed_limit=None,):
    """Return the applicable speed limit in metres per second.

    Non-maglev vehicles use a finite infrastructure limit in preference to
    the geometric limit.  The vehicle maximum remains an upper bound.
    """
    vehicle_type = train_type if vehicle_type is None else vehicle_type
    if (
        str(vehicle_type).lower() != "maglev"
        and infrastructure_speed_limit is not None
    ):
        if isinstance(
            infrastructure_speed_limit, (int, float, np.number)
        ):
            infrastructure_limit = float(infrastructure_speed_limit)
        else:
            infrastructure_limit = parse_speed_limit(
                infrastructure_speed_limit
            )
        if np.isfinite(infrastructure_limit):
            return min(infrastructure_limit, vehicle_max_speed)

    geometric_max = min(np.sqrt(max_centripetal_acceleration*g*r), math.sqrt(g*math.tan(theta)*r))
    return min(geometric_max, vehicle_max_speed)

def calculate_air_resistance(v, C_d, rho, A):
    return 0.5*C_d*rho*A*v**2

def parse_speed_limit(value):
    """Return a GeoJSON speed limit in m/s, or infinity when it is unavailable."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return float("inf")

    match = re.search(r"\d+(?:\.\d+)?", str(value))
    if match is None:
        return float("inf")

    return float(match.group()) / 3.6

def standardize_speed_limits(speed_limits, distances=None, section_length_m=2000, max_interval_samples=None):
    """Standardize speed limits by fixed track sections.

    If distances are supplied, each section is a block of track of length
    section_length_m (default 2000 m). The standardized limit for every point in a
    section is the minimum speed limit seen anywhere in that section.

    The legacy sample-based behavior is retained as a fallback for callers that do
    not provide distances.
    """
    speed_limits = np.asarray(speed_limits, dtype=float)
    standardized = speed_limits.copy()

    if speed_limits.size == 0:
        return standardized

    if distances is not None:
        distances = np.asarray(distances, dtype=float)
        if distances.shape[0] != speed_limits.shape[0]:
            raise ValueError("speed_limits and distances must have the same length")

        section_ids = np.floor(distances / section_length_m).astype(int)
        for section_id in np.unique(section_ids):
            idx = np.where(section_ids == section_id)[0]
            if idx.size == 0:
                continue
            minimum = speed_limits[idx].min()
            standardized[idx] = minimum
        return standardized

    if max_interval_samples is None:
        return standardized

    tolerance = 1E-6
    standardized = speed_limits.copy()
    if len(speed_limits) < 2:
        return standardized

    change_points = find_speed_limit_changes(speed_limits)

    if len(change_points) < 2:
        return standardized

    change_groups = []
    for change_point in change_points:
        if (not change_groups or
                change_point - change_groups[-1][-1] >= max_interval_samples):
            change_groups.append([change_point])
        else:
            change_groups[-1].append(change_point)

    for group in change_groups:
        if len(group) < 2:
            continue

        start = group[0]
        end = group[-1]
        minimum = min(speed_limits[change_point] for change_point in group)
        standardized[start:end + 1] = [minimum] * (end - start + 1)

    return standardized

def calculate_speed_limit_geojson(geojson, g, max_centripetal_acceleration, vehicle_max_speed):
    """Calculate speed limits for a GeoJSON track.

    The speed limit is the minimum of the geometric limit and the vehicle's maximum
    speed. The geometric limit is based on the radius of curvature and the maximum
    centripetal acceleration.
    """
    if not geojson or 'features' not in geojson:
        raise ValueError("Invalid GeoJSON data")

    features = geojson['features']
    if not features:
        raise ValueError("GeoJSON has no features")

    speed_limits = []
    for feature in features:
        geometry = feature.get('geometry')
        if not geometry or geometry.get('type') != 'LineString':
            continue

        coordinates = geometry.get('coordinates', [])
        if len(coordinates) < 2:
            continue


    x = coordinates['x'].to_numpy()
    y = coordinates['y'].to_numpy()
    z = coordinates['elevation'].to_numpy()

    # Calculate radius of curvature for each segment
    alphas = [0.0] #track slope at each point
    speed_limits = []
        
    #calculating distance along track for each point and radius of curvature of each point
    d = 0 #cumulative distance along track to make the loop work
    s = [0] #distance along track at each point
    for i in range(1, len(x)):
        d += math.sqrt((x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2)
                
        s.append(d)
        
        for i in range(1, len(z)):
            delta_s = s[i] - s[i - 1]
            delta_z = z[i] - z[i - 1]
            alphas.append(math.atan2(delta_z, delta_s))
        
        ###THIS CORRESPONDS TO s, but does not correspond to s after it is normalized
        alphas = np.array(alphas) #track angle at each point, in radians
                
        spline_x = interp.UnivariateSpline(s, x, k = 5)
        spline_y = interp.UnivariateSpline(s, y, k = 5)
        spline_alphas = interp.UnivariateSpline(s, alphas, k = 5)

        #normalizing size of s steps    
        S = np.linspace(s[0], s[-1], S_POINTS)
        alphas_uniform = spline_alphas(S)
        
        derivative_x_1 = spline_x(S, nu = 1)
        derivative_y_1 = spline_y(S, nu = 1)

        derivative_x_2 = spline_x(S, nu = 2)
        derivative_y_2 = spline_y(S, nu = 2)

        #both values are placeholders because I don't have data
        thetas_conventional = np.full(len(S), 0.25) #track cant angle at each point, in radians
        thetas_maglev = np.full(len(S), 0.1) #track cant angle at each point, in radians
                
        curvature = abs(
            derivative_x_1 * derivative_y_2
            - derivative_y_1 * derivative_x_2
        ) / np.maximum(
            (derivative_x_1**2 + derivative_y_1**2)**(3 / 2),
            1e-12,
        )
        radii_of_curvature = 1 / np.maximum(curvature, 1e-12)
        
        for i in range(len(radii_of_curvature)):
            speed_limits.append(calculate_speed_limit(radii_of_curvature[i], g, MAX_CENTRIPETAL_ACCELERATION, MAX_TRAIN_SPEED, thetas_conventional[i]))

    return speed_limits

def calculate_davis_resistance(rolling_stock_attributes, v,):
    """Calculate Davis resistance in newtons for speed ``v`` in m/s."""
    return (
        rolling_stock_attributes["constant resistance (n)"]
        + rolling_stock_attributes["linear resistance (ns/m)"] * v
        + rolling_stock_attributes["quadratic resistance (ns^2/m^2)"] * v**2
    )


def calculate_route_data(route_network, station_stops_for_route=(), infrastructure_graph=None, vehicle_type="conventional", rolling_stock_data=None,):
    """Calculate running data from one directed route network."""
    if rolling_stock_data is None:
        raise ValueError("Rolling-stock data is required for route calculation")
    attributes = resolve_rolling_stock_attributes(
        vehicle_type,
        rolling_stock_data,
    )
    required_attributes = [
        "mass (kg)",
        "power (w)",
        "brake acceleration (m/s^2)",
        "max tractive force (n)",
        "max speed (km/h)",
    ]
    normalized_vehicle_type = str(vehicle_type).strip().casefold()
    is_maglev = all(
        component.strip() == "maglev"
        for component in normalized_vehicle_type.split("+")
    )
    if is_maglev:
        required_attributes.extend([
            "width (m)",
            "height (m)",
            "drag coefficient",
        ])
    else:
        required_attributes.extend([
            "quadratic resistance (ns^2/m^2)",
            "constant resistance (n)",
        ])
    missing_attributes = [
        attribute for attribute in required_attributes
        if attribute not in attributes
    ]
    if missing_attributes:
        raise ValueError(
            f"Rolling-stock data for {vehicle_type!r} is missing: "
            + ", ".join(missing_attributes)
        )
    rotating_factor = attributes.get("rotating mass factor", 1.0)
    m = attributes["mass (kg)"]
    P_axle = attributes["power (w)"]
    brake_acceleration = attributes["brake acceleration (m/s^2)"]
    g = 9.81 #gravity constant, m/s^2
    rho = 1.204 #air density, kg/m^3
    MAX_CENTRIPETAL_ACCELERATION = 0.15 #g
    MAX_TRAIN_SPEED = attributes["max speed (km/h)"] / 3.6
    F_max = attributes["max tractive force (n)"]
    F_brake = brake_acceleration * rotating_factor * m
    if is_maglev:
        train_width = attributes["width (m)"]
        train_height = attributes["height (m)"]
        A = train_width * train_height
        drag_coefficient = attributes["drag coefficient"]
        resistance_function = lambda speed: calculate_air_resistance(
            speed,
            drag_coefficient,
            rho,
            A,
        )
    else:
        resistance_function = lambda speed: calculate_davis_resistance(
            attributes,
            speed,
        )
    S_POINTS = 50000

    coordinates, infrastructure_speed_limits, route_segments = _route_points(
        route_network,
        include_speed_limits=True,
        include_route_segments=True,
    )
    x = []
    y = []
    z = []
    projected_line = gpd.GeoSeries(
        [LineString(coordinates)], crs="EPSG:4326"
    ).to_crs(epsg=3310).iloc[0]
    for original_coordinate, projected_coordinate in zip(
        coordinates, projected_line.coords
    ):
        x.append(projected_coordinate[0])
        y.append(projected_coordinate[1])
        z.append(original_coordinate[2] if len(original_coordinate) > 2 else 0.0)

    if len(x) < 2:
        raise ValueError("GeoJSON must contain at least one LineString with two points")

    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)
    infrastructure_speed_limits = np.asarray(infrastructure_speed_limits)

    
    alphas = [0.0] #track slope at each point
    speed_limits = []

    #calculating distance along track for each point and radius of curvature of each point
    d = 0 #cumulative distance along track to make the loop work
    s = [0] #distance along track at each point
    for i in range(1, len(x)):
        d += math.sqrt((x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2)
        
        s.append(d)

    if infrastructure_graph is None:
        infrastructure_graph = route_network
    route_events = _simplified_route_events(
        route_network,
        infrastructure_graph,
        route_segments,
        coordinates,
        np.asarray(s),
    )
    route_node_events = _route_node_events(
        route_network,
        route_segments,
        coordinates,
        np.asarray(s),
    )
    route_edge_events = []
    for segment in route_segments:
        for edge in segment["edges"]:
            route_edge_events.append({
                "edge_id": edge["edge_id"],
                "origin": route_network.nodes[edge["origin"]].get(
                    "node_id",
                    edge["origin"],
                ),
                "destination": route_network.nodes[edge["destination"]].get(
                    "node_id",
                    edge["destination"],
                ),
                "start_distance": float(s[edge["start_index"]]),
                "end_distance": float(s[edge["end_index"]]),
            })

    for i in range(1, len(z)):
        delta_s = s[i] - s[i - 1]
        delta_z = z[i] - z[i - 1]
        alphas.append(math.atan2(delta_z, delta_s))

    ###THIS CORRESPONDS TO s, but does not correspond to s after it is normalized
    alphas = np.array(alphas) #track angle at each point, in radians
        
    curvature_line = LineString(np.column_stack((x, y))).simplify(
        5.0,
        preserve_topology=False,
    )
    curvature_coordinates = np.asarray(curvature_line.coords)
    curvature_x = curvature_coordinates[:, 0]
    curvature_y = curvature_coordinates[:, 1]
    curvature_s = np.r_[
        0.0,
        np.cumsum(np.hypot(
            np.diff(curvature_x),
            np.diff(curvature_y),
        )),
    ]
    spline_x = interp.UnivariateSpline(
        curvature_s,
        curvature_x,
        k=3,
        s=10000.0,
    )
    spline_y = interp.UnivariateSpline(
        curvature_s,
        curvature_y,
        k=3,
        s=10000.0,
    )
    spline_alphas = interp.UnivariateSpline(s, alphas, k = 5)

    #normalizing size of s steps    
    S = np.linspace(s[0], s[-1], S_POINTS)
    alphas_uniform = spline_alphas(S)

    curvature_S = np.linspace(curvature_s[0], curvature_s[-1], S_POINTS)
    derivative_x_1 = spline_x(curvature_S, nu=1)
    derivative_y_1 = spline_y(curvature_S, nu=1)
        
    derivative_x_2 = spline_x(curvature_S, nu=2)
    derivative_y_2 = spline_y(curvature_S, nu=2)

    #both values are placeholders because I don't have data
    thetas_conventional = np.full(len(S), 0.25) #track cant angle at each point, in radians
    thetas_maglev = np.full(len(S), 0.1) #track cant angle at each point, in radians
        
    curvature = abs(
        derivative_x_1 * derivative_y_2
        - derivative_y_1 * derivative_x_2
    ) / np.maximum(
        (derivative_x_1**2 + derivative_y_1**2)**(3 / 2),
        1e-12,
    )
    radii_of_curvature = 1 / np.maximum(curvature, 1e-12)

    infrastructure_limits_uniform = np.interp(
        S,
        np.asarray(s),
        infrastructure_speed_limits,
        left=infrastructure_speed_limits[0],
        right=infrastructure_speed_limits[-1],
    )
    curvature_speed_limits = []
    for i, radius in enumerate(radii_of_curvature):
           curvature_speed_limits.append(calculate_speed_limit(
               radius,
               g,
               MAX_CENTRIPETAL_ACCELERATION,
               MAX_TRAIN_SPEED,
               (
                   thetas_maglev[i]
                   if is_maglev
                   else thetas_conventional[i]
               ),
               vehicle_type="maglev" if is_maglev else normalized_vehicle_type,
               infrastructure_speed_limit=infrastructure_limits_uniform[i],
           ))

    speed_limits = np.interp(
        S,
        curvature_S,
        curvature_speed_limits,
    )
    speed_limits = standardize_speed_limits(
        speed_limits,
        distances=S,
        section_length_m=2000,
    )

    times = [0]
    positions = [0]
    velocities = [0]
    c_accelerations = [0]
    x_accelerations = [0]
    c_forces = [0]
    x_forces = [0]

    t=0
    x=S[0] #should always be equal to 0, but in case it isn't this ensures there is no error
    v=0
    dt = 0.1 #seconds
    i=0 #index for distance-based arrays
    k=1 #index for time-based arrays
    dv = 0
    iteration_count = 0

    station_stop_distances = _station_stop_distances(
        route_network,
        route_segments,
        coordinates,
        np.asarray(s),
        station_stops_for_route,
    )
    for stop_distance in station_stop_distances:
        stop_index = min(
            len(speed_limits) - 1,
            int(np.searchsorted(S, stop_distance, side="left")),
        )
        speed_limits[stop_index] = 0

    max_iterations = 100000  # Prevent infinite loops

    change_points = find_speed_limit_changes(speed_limits)

    brake_limits = np.full(len(S), float("inf"))

    for point in change_points:
        if speed_limits[point] < speed_limits[point-1]:
            dx = (S[-1] - S[0]) / (len(S) - 1)  # Distance spacing of S array
            brake_curve = get_brake_curve(
                point,
                speed_limits[point - 1],
                speed_limits[point],
                F_brake,
                m,
                g,
                alphas_uniform,
                rotating_factor,
                resistance_function,
                dx,
            )
            for j in range(len(brake_curve)):
                if point-len(brake_curve)+j < len(brake_limits) and brake_curve[j] < brake_limits[point-len(brake_curve)+j]:
                    brake_limits[point-len(brake_curve)+j] = brake_curve[j]

    while(x < S[-1]):
        iteration_count += 1
        if iteration_count > max_iterations:
            print(f"Warning: Max iterations ({max_iterations}) reached. Stopping simulation.")
            break
        
        # If very close to end and velocity is near 0, exit
        if S[-1] - x < 1.0 and v < 0.01:
            x = S[-1]
            break
        
        strategy = ""
        dv = 0  

        # Look ahead to the speed limit at the next position
        x_new = positions[k-1]+v*dt
        limit_at_new_position = np.interp(x_new, S, speed_limits)
        brake_limit_at_new_position = np.interp(x_new, S, brake_limits)
        
        # The effective speed limit is the minimum of speed limit and brake limit
        effective_limit_at_new_position = min(limit_at_new_position, brake_limit_at_new_position)

        if (effective_limit_at_new_position<velocities[k-1]) and strategy == "":
            strategy = "braking"
        elif(effective_limit_at_new_position-velocities[k-1])<0.05 and strategy == "":
            strategy = "max_speed"
        elif((velocities[k-1]-effective_limit_at_new_position)<0.05 and strategy == ""):
            strategy = "accelerating"

        if strategy == "accelerating" and velocities[k-1] !=0:
            dv = (
                min(P_axle / velocities[k - 1], F_max)
                - m * g * math.sin(alphas_uniform[i - 1])
                - resistance_function(velocities[k - 1])
            ) * dt / (rotating_factor * m)
            v_new = velocities[k-1]+dv
            # Cap acceleration to not exceed effective limit
            v_new = min(v_new, effective_limit_at_new_position)
            # Recalculate dv to match capped velocity
            dv = v_new - velocities[k-1]
        
        #handles initial case where velocity is 0
        elif strategy == "accelerating" and velocities[k-1] == 0:
            dv = (
                F_max
                - m * g * math.sin(alphas_uniform[i - 1])
                - resistance_function(velocities[k - 1])
            ) * dt / (rotating_factor * m)
            v_new = velocities[k-1]+dv
            # Cap acceleration to not exceed effective limit
            v_new = min(v_new, effective_limit_at_new_position)
            # Recalculate dv to match capped velocity
            dv = v_new - velocities[k-1]

        
        if strategy == "max_speed":
            v_new = v
            dv = 0

        elif strategy == "braking":
            dv = (
                F_brake
                - m * g * math.sin(alphas_uniform[i - 1])
                - resistance_function(velocities[k - 1])
            ) * dt / (rotating_factor * m)
            
            v_new = velocities[k-1]-dv

        positions.append(x_new)
        velocities.append(v_new)
        times.append(t + dt)
        c_accelerations.append(v_new**2/radii_of_curvature[i])
        x_accelerations.append(dv/dt)
        c_forces.append(
            m * v_new**2 / radii_of_curvature[i]
        )
        x_forces.append(m * dv / dt)

        x=x_new
        v=v_new
        t+=dt
        k+=1

        #update i so that it corresponds to different array shape for S, speed_limits, alphas
        while(x>S[i] and i < len(S)-1):
            i+=1

    
    return (
        times,
        positions,
        velocities,
        c_accelerations,
        x_accelerations,
        c_forces,
        x_forces,
        S,
        alphas_uniform,
        radii_of_curvature,
        speed_limits,
        brake_limits,
        route_events,
        route_node_events,
        route_edge_events,
    )


def _signature_value(value):
    """Convert model metadata into a stable, hashable signature value."""
    if isinstance(value, dict):
        return tuple(sorted(
            (key, _signature_value(item))
            for key, item in value.items()
        ))
    if isinstance(value, (list, tuple, set)):
        return tuple(_signature_value(item) for item in value)
    return value


def _route_calculation_signature(route_network, station_stops, vehicle_type,):
    """Return the inputs that determine a route's running-time calculation."""
    route_nodes = tuple(
        tuple(node) for node in route_network.graph.get(
            "route_node_coordinates", ()
        )
    )
    edge_ids = tuple(route_network.graph.get("assigned_edges", ()))
    model_attributes = tuple(
        (name, _signature_value(route_network.graph.get(name)))
        for name in ("train_model", "vehicle_type", "model", "train_type")
        if name in route_network.graph
    )
    return (
        str(vehicle_type).strip().casefold(),
        model_attributes,
        route_nodes,
        edge_ids,
        tuple(station_stops),
    )


def calculate_identical_route_data(route_graphs, station_stops_by_train, infrastructure_graph, train_types_by_train, rolling_stock_data,):
    """Calculate one result for each group of identical train routes.

    Routes are identical when they have the same ordered route nodes, the same
    ordered graph edge IDs, the same station-stop sequence, and the same train
    model. Every train ID is still returned, but the numerical simulation is
    performed only once per signature.
    """
    calculated = {}
    route_data_by_train = {}
    group_examples = {}
    for train_id, route_graph in route_graphs.items():
        route_network = build_used_routes_network({train_id: route_graph})
        station_stops = tuple(station_stops_by_train.get(train_id, ()))
        vehicle_type = train_types_by_train.get(train_id, "conventional")
        signature = _route_calculation_signature(
            route_network,
            station_stops,
            vehicle_type,
        )
        if signature not in calculated:
            calculated[signature] = (
                route_network,
                calculate_route_data(
                    route_network,
                    station_stops,
                    infrastructure_graph,
                    vehicle_type,
                    rolling_stock_data,
                ),
            )
            group_examples[signature] = train_id
        route_data_by_train[train_id] = calculated[signature]
    return route_data_by_train, group_examples


def _schedule_node_events(route_node_events, station_stops):
    """Match scheduled stations to their ordered labelled route nodes."""
    normalized_events = [
        (
            " ".join(str(event["station"]).split()).casefold(),
            event,
        )
        for event in route_node_events
        if event.get("station") not in (None, "None")
    ]
    events_by_station = {}
    for key, event in normalized_events:
        events_by_station.setdefault(key, []).append(event)

    schedule_events = []
    used_occurrences = {}
    for station_index, station in enumerate(station_stops):
        key = " ".join(str(station).split()).casefold()
        occurrences = events_by_station.get(key, [])
        occurrence_index = used_occurrences.get(key, 0)
        if occurrence_index >= len(occurrences):
            raise ValueError(
                f"Station {station!r} has no route node for schedule position "
                f"{station_index + 1}"
            )
        schedule_events.append(occurrences[occurrence_index])
        used_occurrences[key] = occurrence_index + 1
    return schedule_events


def _maximum_speed_between_distances(start_distance, end_distance, positions, velocities,):
    """Return the highest simulated speed reached between two route nodes."""
    lower_distance, upper_distance = sorted(
        (float(start_distance), float(end_distance))
    )
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    in_edge = (
        (positions >= lower_distance)
        & (positions <= upper_distance)
    )
    edge_velocities = velocities[in_edge]
    edge_velocities = np.append(
        edge_velocities,
        np.interp(
            [lower_distance, upper_distance],
            positions,
            velocities,
        ),
    )
    return float(np.max(edge_velocities) * 3.6)


def write_route_node_csvs(route_data_by_train, station_stops_by_train, output_folder=ROUTE_CSV_FOLDER,):
    """Write one schedule-ordered station travel-time CSV per train."""
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    for train_id, (_route_network, results) in route_data_by_train.items():
        station_stops = station_stops_by_train.get(train_id, ())
        schedule_events = _schedule_node_events(
            results[13],
            station_stops,
        )
        times = np.asarray(results[0])
        positions = np.asarray(results[1])
        with (output_path / f"train_{train_id}.csv").open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "Node",
                "Travel time to following node (s)",
            ])
            for index, event in enumerate(schedule_events):
                if index + 1 < len(schedule_events):
                    next_event = schedule_events[index + 1]
                    travel_time = float(np.interp(
                        next_event["distance"],
                        positions,
                        times,
                    ) - np.interp(
                        event["distance"],
                        positions,
                        times,
                    ))
                else:
                    travel_time = ""
                writer.writerow([station_stops[index], travel_time])


def display_speed_profile_graphs(route_data_by_train, group_examples,):
    """Display one full-route speed profile graph for each calculated group."""
    for group_number, (signature, example_train_id) in enumerate(
        group_examples.items(),
        start=1,
    ):
        _route_network, results = route_data_by_train[example_train_id]
        positions = np.asarray(results[1], dtype=float)
        velocities = np.asarray(results[2], dtype=float) * 3.6
        distances = np.asarray(results[7], dtype=float)
        speed_limits = np.asarray(results[10], dtype=float) * 3.6
        brake_limits = np.asarray(results[11], dtype=float) * 3.6
        group_train_ids = [
            train_id
            for train_id, data in route_data_by_train.items()
            if data is route_data_by_train[example_train_id]
        ]

        figure, axis = plt.subplots(figsize=(14, 7))
        axis.plot(
            positions / 1000,
            velocities,
            color="blue",
            linewidth=1.2,
            label="Actual speed",
        )
        axis.step(
            distances / 1000,
            speed_limits,
            where="post",
            color="red",
            linewidth=1.0,
            label="Speed limit",
        )
        finite_brake_limits = np.where(
            np.isfinite(brake_limits),
            brake_limits,
            np.nan,
        )
        axis.plot(
            distances / 1000,
            finite_brake_limits,
            color="green",
            linewidth=1.0,
            label="Braking curve",
        )
        axis.set_xlabel("Distance along route (km)")
        axis.set_ylabel("Speed (km/h)")
        axis.set_title(
            f"Train group {group_number} speed profile "
            f"(example train {example_train_id}; trains "
            f"{', '.join(map(str, group_train_ids))})"
        )
        axis.grid(True, alpha=0.3)
        axis.legend()
        figure.tight_layout()
    plt.show()


def main(route_graph_file=ROUTE_GRAPH_FILE, existing_route_stops_file=EXISTING_ROUTE_STOPS_FILE, new_route_stops_file=NEW_ROUTE_STOPS_FILE, route_data_file=ROUTE_DATA_FILE, rolling_stock_data_file=ROLLING_STOCK_DATA_FILE, results_file=RESULTS_FILE, node_results_file=NODE_RESULTS_FILE, infrastructure_graph_file=INFRASTRUCTURE_GRAPH_FILE, route_csv_folder=ROUTE_CSV_FOLDER, route_map_folder=ROUTE_MAP_FOLDER, braking_time_file=BRAKING_TIME_FILE,):
    """Calculate running times and grouped route outputs."""
    with open(route_graph_file, "rb") as handle:
        route_graphs = pickle.load(handle)
    station_stops_by_train = load_route_station_stops_files(
        existing_route_stops_file,
        new_route_stops_file,
    )
    stop_durations_by_train = load_route_stop_durations(route_data_file)
    train_types_by_train = load_route_train_types(route_data_file)
    rolling_stock_data = load_rolling_stock_data(
        rolling_stock_data_file
    )
    with open(infrastructure_graph_file, "rb") as handle:
        infrastructure_graph = pickle.load(handle)
    result_rows = []
    node_result_rows = []

    route_data_by_train, group_examples = calculate_identical_route_data(
        route_graphs,
        station_stops_by_train,
        infrastructure_graph,
        train_types_by_train,
        rolling_stock_data,
    )
    for train_id in route_graphs:
        route_network, results = route_data_by_train[train_id]
        station_stops_for_route = station_stops_by_train.get(train_id, [])
        times, positions, velocities = results[:3]
        route_edge_events = results[14]
        stop_duration = stop_durations_by_train.get(train_id, 0)
        total_dwell_time = max(
            stop_duration * (len(station_stops_for_route) - 2),
            0,
        )
        total_time = times[-1] + total_dwell_time
        total_distance = positions[-1]
        average_speed = total_distance / total_time * 3.6
        max_speed = max(velocities, default=0) * 3.6
        result_rows.append([
            train_id,
            total_time,
            total_distance,
            average_speed,
            max_speed,
        ])
        for edge in route_edge_events:
            start_distance = edge["start_distance"]
            end_distance = edge["end_distance"]
            node_result_rows.append([
                train_id,
                edge["edge_id"],
                edge["origin"],
                edge["destination"],
                end_distance - start_distance,
                float(np.interp(end_distance, positions, times))
                - float(np.interp(start_distance, positions, times)),
                _maximum_speed_between_distances(
                    start_distance,
                    end_distance,
                    positions,
                    velocities,
                ),
            ])
        print(
            f"Train {train_id}: {total_time / 60:.2f} minutes, "
            f"{total_distance / 1000:.2f} km, "
            f"{average_speed:.2f} km/h, "
            f"max {max_speed:.2f} km/h"
        )

    write_route_node_csvs(
        route_data_by_train,
        station_stops_by_train,
        route_csv_folder,
    )
    write_braking_time_csv(
        route_data_by_train,
        train_types_by_train,
        rolling_stock_data,
        braking_time_file,
    )
    display_speed_profile_graphs(
        route_data_by_train,
        group_examples,
    )
    map_folder = Path(route_map_folder)
    map_folder.mkdir(parents=True, exist_ok=True)
    for group_number, (signature, example_train_id) in enumerate(
        group_examples.items(),
        start=1,
    ):
        route_network, results = route_data_by_train[example_train_id]
        write_first_train_speed_map(
            route_network,
            results[1],
            results[2],
            example_train_id,
            map_folder / f"route_group_{group_number}_train_{example_train_id}.html",
        )

    with open(results_file, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "Train ID",
            "Travel Time (s)",
            "Distance (m)",
            "Average Speed (km/h)",
            "Max Speed (km/h)",
        ])
        writer.writerows(result_rows)

    with open(node_results_file, "w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "Train ID",
            "Edge ID",
            "Origin Node",
            "Destination Node",
            "Distance (m)",
            "Minimum Running Time (s)",
            "Maximum Speed (km/h)",
        ])
        writer.writerows(node_result_rows)



if __name__ == "__main__":
    main()
