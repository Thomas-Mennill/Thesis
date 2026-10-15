"""
TO DO

Accurate Stop Location
Verify cant angle stuff
Verify slope angle stuff
Optimize because certain sections have many routes doing the same thing and this will take forever
Make an abstract graph where segments are edges (or places between signals) and stations/switches are nodes
no need for physical properties after this step


"""


import csv
import pandas as pd
import math
import pickle
import re
import numpy as np
import networkx as nx
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import geopandas as gpd
import folium
from shapely.geometry import LineString

from Display_Routes import build_used_routes_network


ROUTE_GRAPH_FILE = "case_study_route_graphs.pkl"
ROUTE_STOPS_FILE = "rail_routes.csv"
ROUTE_DATA_FILE = "route_data.csv"
INFRASTRUCTURE_GRAPH_FILE = "case_study_network.gpickle"
RESULTS_FILE = "running_time_results.csv"
NODE_RESULTS_FILE = "running_time_between_nodes.csv"
SPEED_MAP_FILE = "first_train_speed_map.html"
train_type = "conventional"


def load_route_station_stops(csv_file=ROUTE_STOPS_FILE):
    """Return station stops grouped by train ID from the timetable CSV."""
    with open(csv_file, encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"Route file {csv_file!r} is empty")

    station_stops = {}
    # Each timetable route occupies event, station, time, and a separator
    # column. The separator is retained in the CSV between adjacent routes.
    for column in range(1, len(rows[0]), 4):
        train_id = rows[0][column].strip()
        if not train_id or train_id.lower() == "time":
            continue
        try:
            train_number = int(train_id)
        except ValueError as error:
            raise ValueError(
                f"Invalid train ID {train_id!r} in column {column + 1}"
            ) from error
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


def _route_points(
    route_network,
    include_speed_limits=False,
    include_route_segments=False,
):
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
    assigned_segments = route_network.graph.get("assigned_segments", [])
    for segment_index, (start, end) in enumerate(zip(waypoints, waypoints[1:])):
        try:
            path = nx.shortest_path(route_network, start, end)
        except nx.NetworkXNoPath as error:
            raise ValueError(
                f"No directed path between route nodes {start!r} and {end!r}"
            ) from error
        segment_start = len(coordinates)
        segment_station = None
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
            if coordinates and same_point(geometry[0], coordinates[-1]):
                geometry = geometry[1:]
                edge_speed_limits = edge_speed_limits[1:]
            coordinates.extend(geometry)
            if include_speed_limits:
                infrastructure_speed_limits.extend(edge_speed_limits)

        if include_route_segments and len(coordinates) > segment_start:
            route_segments.append({
                "start_index": segment_start,
                "end_index": len(coordinates) - 1,
                "path_nodes": path,
                "station": segment_station,
                "segment_id": (
                    assigned_segments[segment_index]
                    if segment_index < len(assigned_segments)
                    else None
                ),
            })

    if len(coordinates) < 2:
        raise ValueError("Route network contains fewer than two input points")
    if include_speed_limits:
        result = coordinates, infrastructure_speed_limits
        if include_route_segments:
            return (*result, route_segments)
        return result
    return coordinates


def _station_stop_distances(route_segments, distances, station_stops):
    """Return stopping distances at the longest segment of each station."""
    stop_distances = []
    requested_stations = set(station_stops)
    for station in requested_stations:
        candidates = [
            segment for segment in route_segments
            if segment["station"] == station
        ]
        if not candidates:
            continue
        longest_segment = max(
            candidates,
            key=lambda segment: (
                distances[segment["end_index"]]
                - distances[segment["start_index"]]
            ),
        )
        stop_distances.append(distances[longest_segment["end_index"]])
    return sorted(stop_distances)


def _simplified_route_events(
    route_network,
    infrastructure_graph,
    route_segments,
    coordinates,
    distances,
):
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


def write_first_train_speed_map(
    route_network,
    positions,
    velocities,
    train_id,
    output_file=SPEED_MAP_FILE,
):
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

def get_brake_curve(i, v_max, v_target, a_max, F_brake, m, g, alphas_uniform, rotating_factor, C_d, R_l, rho, A, dx=1):
    if a_max == 0:
        return float("inf")

    velocities = []
    distances = []

    v_current = v_max
    x_current = 0

    dt = 0.1 #seconds

    while(v_current > v_target): 
        distances.append(x_current)
        velocities.append(v_current)

        # Same braking equation as main simulation
        dv = min((F_brake - m*g*math.sin(alphas_uniform[i-1]) - 
                        calculate_air_resistance(v_current, C_d, rho, A) - 
                        v_current*R_l)*dt/(rotating_factor*m), a_max*dt) 
        v_current -= dv
        
        # Calculate distance traveled in this time step
        x_current += v_current * dt

    # Interpolate to get velocities at regular distance intervals
    if len(distances) > 1:
        brake_curve_interpolated = np.interp(np.arange(0, x_current, dx), distances, velocities)
    else:
        brake_curve_interpolated = np.array([v_max])
    
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
def calculate_speed_limit(
    r,
    g,
    max_centripetal_acceleration,
    vehicle_max_speed,
    theta=0,
    vehicle_type=None,
    infrastructure_speed_limit=None,
):
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
                
        curvature = abs(derivative_x_1 * derivative_y_2 - derivative_y_1 * derivative_x_2) / (derivative_x_1**2 + derivative_y_1**2)**(3/2)
        radii_of_curvature = 1 / curvature
        
        for i in range(len(radii_of_curvature)):
            speed_limits.append(calculate_speed_limit(radii_of_curvature[i], g, MAX_CENTRIPETAL_ACCELERATION, MAX_TRAIN_SPEED, thetas_conventional[i]))

    return speed_limits

#need to pin routes together for situations where a train does not stop after a route, to preserve velocity and acceleration data. 
def calculate_route_data(
    route_network,
    station_stops_for_route=(),
    infrastructure_graph=None,
):
    """Calculate running data from one directed route network."""
    rotating_factor = 1 #rotating mass factor
    m = 380000 #mass of train, kg. From TGV Duplex
    P_axle = 9200*1000*2 #power at axles, W. From TGV Duplex
    F_brake = 8*30000 #braking force, N. From TGV Duplex
    C_d = 0.0726 #drag coefficient, from TGV Duplex
    R_l = 0.2 #linear resistance coefficient
    train_width = 2.896 #metres
    train_height = 4.303 #metres
    g = 9.81 #gravity constant, m/s^2
    rho = 1.204 #air density, kg/m^3
    MAX_LINEAR_ACCELERATION = 1 #g
    MAX_CENTRIPETAL_ACCELERATION = 0.15 #g
    MAX_TRAIN_SPEED = 300/3.6 #m/s
    A=train_width*train_height #frontal area of train, m^2
    F_max = 220000 #maximum tractive force, N
    gauge_magrail = 1 #m
    gauge_standard = 1.435 #m
    MAX_BRAKING_DECELERATION = F_brake / (rotating_factor * m)
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
        
    curvature = abs(derivative_x_1 * derivative_y_2 - derivative_y_1 * derivative_x_2) / (derivative_x_1**2 + derivative_y_1**2)**(3/2)
    radii_of_curvature = 1 / curvature

    infrastructure_limits_uniform = np.interp(
        S,
        np.asarray(s),
        infrastructure_speed_limits,
        left=infrastructure_speed_limits[0],
        right=infrastructure_speed_limits[-1],
    )
    for i in range(len(radii_of_curvature)):
           speed_limits.append(calculate_speed_limit(
               radii_of_curvature[i],
               g,
               MAX_CENTRIPETAL_ACCELERATION,
               MAX_TRAIN_SPEED,
               thetas_conventional[i],
               vehicle_type=train_type,
               infrastructure_speed_limit=infrastructure_limits_uniform[i],
           ))

    speed_limits = standardize_speed_limits(speed_limits, distances=S, section_length_m=2000)

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
        route_segments,
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
            brake_curve = get_brake_curve(point, speed_limits[point-1], speed_limits[point], MAX_LINEAR_ACCELERATION, F_brake, m, g, alphas_uniform, rotating_factor, C_d, R_l, rho, A, dx)
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
            dv = (min(P_axle/velocities[k-1], F_max) - m*g*math.sin(alphas_uniform[i-1]) - 
                calculate_air_resistance(velocities[k-1], C_d, rho, A) - 
                velocities[k-1]*R_l)*dt/(rotating_factor*m)
            v_new = velocities[k-1]+dv
            # Cap acceleration to not exceed effective limit
            v_new = min(v_new, effective_limit_at_new_position)
            # Recalculate dv to match capped velocity
            dv = v_new - velocities[k-1]
        
        #handles initial case where velocity is 0
        elif strategy == "accelerating" and velocities[k-1] == 0:
            dv = (F_max - m*g*math.sin(alphas_uniform[i-1]) - 
                calculate_air_resistance(velocities[k-1], C_d, rho, A) - 
                velocities[k-1]*R_l)*dt/(rotating_factor*m)
            v_new = velocities[k-1]+dv
            # Cap acceleration to not exceed effective limit
            v_new = min(v_new, effective_limit_at_new_position)
            # Recalculate dv to match capped velocity
            dv = v_new - velocities[k-1]

        
        if strategy == "max_speed":
            v_new = v
            dv = 0

        elif strategy == "braking":
            dv = min((F_brake - m*g*math.sin(alphas_uniform[i-1]) - 
                calculate_air_resistance(velocities[k-1], C_d, rho, A) - 
                velocities[k-1]*R_l)*dt/(rotating_factor*m), MAX_BRAKING_DECELERATION*dt)
            
            v_new = velocities[k-1]-dv

        positions.append(x_new)
        velocities.append(v_new)
        times.append(t)
        c_accelerations.append(v_new**2/radii_of_curvature[i])
        x_accelerations.append(dv/dt)
        c_forces.append(m*v_new**2/radii_of_curvature[i])
        x_forces.append(m*dv/dt)

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
    )

def main(
    route_graph_file=ROUTE_GRAPH_FILE,
    route_stops_file=ROUTE_STOPS_FILE,
    route_data_file=ROUTE_DATA_FILE,
    results_file=RESULTS_FILE,
    node_results_file=NODE_RESULTS_FILE,
    infrastructure_graph_file=INFRASTRUCTURE_GRAPH_FILE,
    speed_map_file=SPEED_MAP_FILE,
):
    """Calculate running time separately for every constructed train route."""
    with open(route_graph_file, "rb") as handle:
        route_graphs = pickle.load(handle)
    station_stops_by_train = load_route_station_stops(route_stops_file)
    stop_durations_by_train = load_route_stop_durations(route_data_file)
    with open(infrastructure_graph_file, "rb") as handle:
        infrastructure_graph = pickle.load(handle)
    first_train_graph_data = None
    first_train_map_data = None
    result_rows = []
    node_result_rows = []

    for train_id, route_graph in route_graphs.items():
        route_network = build_used_routes_network({train_id: route_graph})
        station_stops_for_route = station_stops_by_train.get(train_id, [])
        results = calculate_route_data(
            route_network,
            station_stops_for_route,
            infrastructure_graph,
        )
        times, positions, velocities = results[:3]
        route_events = results[12]
        if first_train_graph_data is None:
            first_train_graph_data = (
                train_id,
                positions,
                velocities,
                results[7],
                results[10],
                results[11],
                route_events,
            )
            first_train_map_data = (
                train_id,
                route_network,
                positions,
                velocities,
            )
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
        event_times = [
            float(np.interp(event["distance"], positions, times))
            for event in route_events
        ]
        for first, second, first_time, second_time in zip(
            route_events,
            route_events[1:],
            event_times,
            event_times[1:],
        ):
            node_result_rows.append([
                train_id,
                first["node"],
                second["node"],
                second["distance"] - first["distance"],
                second_time - first_time,
            ])
        print(
            f"Train {train_id}: {total_time / 60:.2f} minutes, "
            f"{total_distance / 1000:.2f} km, "
            f"{average_speed:.2f} km/h, "
            f"max {max_speed:.2f} km/h"
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
            "Origin Node",
            "Destination Node",
            "Distance (m)",
            "Minimum Running Time (s)",
        ])
        writer.writerows(node_result_rows)

    if first_train_map_data is not None:
        write_first_train_speed_map(
            first_train_map_data[1],
            first_train_map_data[2],
            first_train_map_data[3],
            first_train_map_data[0],
            speed_map_file,
        )

    if first_train_graph_data is not None:
        (
            train_id,
            positions,
            velocities,
            distances,
            speed_limits,
            brake_limits,
            route_events,
        ) = first_train_graph_data
        for first_event, second_event in zip(
            route_events,
            route_events[1:],
        ):
            start_distance = first_event["distance"]
            end_distance = second_event["distance"]
            distance_mask = (
                (distances >= start_distance)
                & (distances <= end_distance)
            )
            position_mask = (
                (np.asarray(positions) >= start_distance)
                & (np.asarray(positions) <= end_distance)
            )
            interval_distances = distances[distance_mask] - start_distance
            interval_positions = (
                np.asarray(positions)[position_mask] - start_distance
            )

            plt.figure(figsize=(12, 6))
            plt.step(
                interval_distances,
                np.asarray(speed_limits)[distance_mask] * 3.6,
                where="post",
                color="red",
                label="Speed limit",
            )
            plt.step(
                interval_distances,
                np.asarray(brake_limits)[distance_mask] * 3.6,
                where="post",
                color="green",
                label="Brake limits",
            )
            plt.plot(
                interval_positions,
                np.asarray(velocities)[position_mask] * 3.6,
                color="blue",
                label="Actual speed",
            )
            plt.xlabel("Distance from origin node (m)")
            plt.ylabel("Speed (km/h)")
            plt.title(
                f"Train {train_id}: {first_event['node']} to "
                f"{second_event['node']}"
            )
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()

        plt.show(block=True)


if __name__ == "__main__":
    main()