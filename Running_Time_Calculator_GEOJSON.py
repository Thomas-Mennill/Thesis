"""
TO DO

Accurate Stop Location
Make sure speed limits are used if they exist and the trains are conventional
Verify cant angle stuff
Verify linear angle stuff
Make sure the travel time calculation skips stations if the route skips stations
and doesn't if the route stops at stations
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
from shapely.geometry import LineString

from Display_Routes import build_used_routes_network


ROUTE_GRAPH_FILE = "case_study_route_graphs.pkl"
ROUTE_STOPS_FILE = "rail_routes.csv"
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


def _route_points(route_network):
    """Extract ordered x/y/z points from one directed route network."""
    def same_point(first, second):
        return math.hypot(
            float(first[0]) - float(second[0]),
            float(first[1]) - float(second[1]),
        ) <= 1e-6

    waypoints = route_network.graph.get("route_node_coordinates")
    if not waypoints:
        raise ValueError("Route network has no ordered route nodes")

    coordinates = []
    for start, end in zip(waypoints, waypoints[1:]):
        try:
            path = nx.shortest_path(route_network, start, end)
        except nx.NetworkXNoPath as error:
            raise ValueError(
                f"No directed path between route nodes {start!r} and {end!r}"
            ) from error
        for first, second in zip(path, path[1:]):
            edge_data = route_network.get_edge_data(first, second)
            if not edge_data:
                raise ValueError(
                    f"Missing edge data for route edge {first!r}->{second!r}"
                )
            attributes = next(iter(edge_data.values()))
            geometry = attributes.get("geometry")
            if not geometry:
                geometry = [first, second]
            if same_point(geometry[-1], first):
                geometry = list(reversed(geometry))
            if not same_point(geometry[0], first):
                raise ValueError(
                    f"Edge geometry does not start at route node {first!r}"
                )
            if coordinates and same_point(geometry[0], coordinates[-1]):
                geometry = geometry[1:]
            coordinates.extend(geometry)

    if len(coordinates) < 2:
        raise ValueError("Route network contains fewer than two input points")
    return coordinates

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
def calculate_speed_limit(r, g, max_centripetal_acceleration, vehicle_max_speed, theta=0):
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
def calculate_route_data(route_network, station_stops_for_route=()):
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
    TOLERANCE = 1E-6 #tolerance for equals
    MAX_LINEAR_ACCELERATION = 1 #g
    MAX_CENTRIPETAL_ACCELERATION = 0.15 #g
    MAX_TRAIN_SPEED = 300/3.6 #m/s
    A=train_width*train_height #frontal area of train, m^2
    progress_step_percent = 5
    F_max = 220000 #maximum tractive force, N
    gauge_magrail = 1 #m
    gauge_standard = 1.435 #m
    MAX_BRAKING_DECELERATION = F_brake / (rotating_factor * m)
    S_POINTS = 50000

    station_stop = bool(station_stops_for_route)
    coordinates = _route_points(route_network)
    x = []
    y = []
    z = []
    infrastructure_speed_limits = []
    projected_line = gpd.GeoSeries(
        [LineString(coordinates)], crs="EPSG:4326"
    ).to_crs(epsg=3310).iloc[0]
    for original_coordinate, projected_coordinate in zip(
        coordinates, projected_line.coords
    ):
        x.append(projected_coordinate[0])
        y.append(projected_coordinate[1])
        z.append(original_coordinate[2] if len(original_coordinate) > 2 else 0.0)
        infrastructure_speed_limits.append(float("inf"))

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

    speed_limits = standardize_speed_limits(speed_limits, distances=S, section_length_m=2000)

    if train_type.lower() != "maglev" and np.isfinite(infrastructure_speed_limits).any():
        infrastructure_limits_uniform = np.interp(
            S,
            np.asarray(s),
            infrastructure_speed_limits,
            left=infrastructure_speed_limits[0],
            right=infrastructure_speed_limits[-1],
        )
        speed_limits = np.minimum(speed_limits, infrastructure_limits_uniform)



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

    #If the train stops at a station at the end of this segment, set the speed limit to 0 for the last point
    if station_stop:    
        speed_limits[-1] = 0

    next_progress_percent = progress_step_percent
    print("Progress: 0%")
    last_reported_progress = 0
    max_iterations = 100000  # Prevent infinite loops

    change_points = find_speed_limit_changes(speed_limits)

    brake_limits = np.full(len(S), float("inf"))

    for point in change_points:
        if speed_limits[point] < speed_limits[point-1]:
            print(f"Point {point} is a speed limit change.")
            dx = (S[-1] - S[0]) / (len(S) - 1)  # Distance spacing of S array
            brake_curve = get_brake_curve(point, speed_limits[point-1], speed_limits[point], MAX_LINEAR_ACCELERATION, F_brake, m, g, alphas_uniform, rotating_factor, C_d, R_l, rho, A, dx)
            for j in range(len(brake_curve)):
                if point-len(brake_curve)+j < len(brake_limits) and brake_curve[j] < brake_limits[point-len(brake_curve)+j]:
                    brake_limits[point-len(brake_curve)+j] = brake_curve[j]

    # Display speed limits and brake curves before simulation
    plt.figure(figsize=(12, 6))
    plt.step(S, np.array(speed_limits)*3.6, where="post", color="red", label="Speed limit")
    plt.step(S, np.array(brake_limits)*3.6, where="post", color="green", label="Brake limits")
    plt.xlabel("Distance along route (m)")
    plt.ylabel("Speed (km/h)")
    plt.title("Speed Limits and Brake Curves")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show(block=True)

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

        if v_new - limit_at_new_position > TOLERANCE:
            print(x_new, v_new, limit_at_new_position, v_new - limit_at_new_position)
        
        route_progress = min(100, int((x / S[-1]) * 100))
        route_progress_fine = min(100, (x / S[-1]) * 100)
        
        if route_progress_fine >= 95:
            # Beyond 95%: report every 0.1%
            if route_progress_fine - last_reported_progress >= 0.1:
                print(f"Progress: {route_progress_fine:.1f}%")
                last_reported_progress = int(route_progress_fine * 10) / 10
        else:
            # Up to 95%: report every 5%
            if route_progress >= next_progress_percent:
                print(f"Progress: {next_progress_percent}%")
                next_progress_percent += progress_step_percent
                last_reported_progress = next_progress_percent - progress_step_percent
        

    if last_reported_progress < 100:
        print("Progress: 100%")


    
    return times, positions, velocities, c_accelerations, x_accelerations, c_forces, x_forces, S, alphas_uniform, radii_of_curvature, speed_limits, brake_limits

def main(
    route_graph_file=ROUTE_GRAPH_FILE,
    route_stops_file=ROUTE_STOPS_FILE,
):
    """Calculate running time separately for every constructed train route."""
    with open(route_graph_file, "rb") as handle:
        route_graphs = pickle.load(handle)
    station_stops_by_train = load_route_station_stops(route_stops_file)

    for train_id, route_graph in route_graphs.items():
        route_network = build_used_routes_network({train_id: route_graph})
        station_stops_for_route = station_stops_by_train.get(train_id, [])
        results = calculate_route_data(
            route_network,
            station_stops_for_route,
        )
        times, positions, velocities = results[:3]
        dwell_time = 60 * max(0, len(station_stops_for_route) - 2)
        total_time = times[-1] + dwell_time
        total_distance = positions[-1]
        print(
            f"Train {train_id}: {total_time / 60:.2f} minutes, "
            f"{total_distance / 1000:.2f} km, "
            f"{total_distance / total_time * 3.6:.2f} km/h"
        )


if __name__ == "__main__":
    main()