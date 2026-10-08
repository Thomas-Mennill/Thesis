"""
Created on Sat Apr 20 22:49:46 2024

@author: Joseph Thomas
"""

import geopandas as gpd
from shapely.ops import split
import shapely
import math
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import UnivariateSpline, CubicSpline

# Levitation in tons/m*train
dpi_value = 200
show_loading_bar = True
levitator_specs = np.loadtxt("levitator_specs.csv", delimiter = ",")

curviness_factor = 2.5 # back of the envelope for cars because you need to brake and then speed up again the whole time.

# folder = "Intercity_3100"
# routes = [
#     "DenHaagCentraal_LeidenCentraal",
#     "LeidenCentraal_Schiphol",
#     "Schiphol_AmsterdamZuid",
#     "AmsterdamZuid_AmsterdamBijlmerArenA",
#     "AmsterdamBijlmerArenA_UtrechtCentraal",
#     "UtrechtCentraal_EdeWageningen",
#     "EdeWageningen_ArnhemCentraal",
#     "ArnhemCentraal_Nijmegen"
# ]

# folder = "Sprinter_4300"
# routes = [
#     "DenHaagCentraal_LaanVanNOI",
#     "LaanVanNOI_Mariahoeve",
#     "Mariahoeve_Voorschoten",
#     "Voorschoten_DeVink",
#     "DeVink_LeidenCentraal",
#     "LeidenCentraal_Sassenheim",
#     "Sassenheim_NieuwVennep",
#     "NieuwVennep_Hoofddorp",
#     "Hoofddorp_Schiphol",
#     "Schiphol_AmsterdamZuid",
#     "AmsterdamZuid_AmsterdamRAI",
#     "AmsterdamRAI_Duivendrecht",
#     "Duivendrecht_DiemenZuid",
#     "DiemenZuid_Weesp",
#     "Weesp_AlmerePoort",
#     "AlmerePoort_AlmereMuziekwijk",
#     "AlmereMuziekwijk_AlmereCentrum",
#     "AlmereCentrum_AlmereParkwijk",
#     "AlmereParkwijk_AlmereBuiten",
#     "AlmereBuiten_AlmereOostvaarders",
#     "AlmereOostvaarders_LelystadCentrum"
# ]

folder = ""
routes = [
    "Paris_Bordeaux"
]

train_type = "D"
num_wagons = 4
seat_belts = "N"
num_passengers = 300
region = "EU"

MAXIMUM_ACCELERATION_SEAT_BELTS = 3
MAXIMUM_CENTRIFUGAL_G_FORCE_SEAT_BELTS = 0.2

MAXIMUM_ACCELERATION_NORMAL = 1 #0.5
MAXIMUM_CENTRIFUGAL_G_FORCE_NORMAL = 0.15 #0.05

MAXIMUM_SPEED_PASSENGERS = 600 # km/h

MAXIMUM_ACCELERATION_FREIGHT = 4 # m/s^2
MAXIMUM_CENTRIFUGAL_G_FORCE_FREIGHT = 0.8

MAXIMUM_SPEED_FREIGHT = 300 # km/h

PLOT = False

if train_type == "S" or train_type == "D":
    top_design_speed = MAXIMUM_SPEED_PASSENGERS / 3.6 # m/s
    
    if seat_belts == "Y":
        a_max = MAXIMUM_ACCELERATION_SEAT_BELTS # m/s^2
        max_centrifugal_g_force = MAXIMUM_CENTRIFUGAL_G_FORCE_SEAT_BELTS
    else:
        a_max = MAXIMUM_ACCELERATION_NORMAL # m/s^2
        max_centrifugal_g_force = MAXIMUM_CENTRIFUGAL_G_FORCE_NORMAL
elif train_type == "F":
    top_design_speed = MAXIMUM_SPEED_FREIGHT / 3.6 # m/s
    
    a_max = MAXIMUM_ACCELERATION_FREIGHT # m/s^2
    max_centrifugal_g_force = MAXIMUM_CENTRIFUGAL_G_FORCE_FREIGHT

if train_type == "S":
    max_num_passengers = 100 * num_wagons
elif train_type == "D":
    max_num_passengers = 180 * num_wagons
else:
    max_num_passengers = 0

if train_type == "F":
    max_cargo = 14 * 25 * num_wagons # tons
    
    print(f"How many (metric) tons of cargo? (0 - {max_cargo} tons)")
    cargo_mass = float(input("Answer: ")) * 1e3
else:
    cargo_mass = 0

train_length = num_wagons * 25
specific_mass = 1.25e3 # kg/m*train
train_mass = specific_mass * train_length + cargo_mass + num_passengers * 70

if train_type == "F":
    train_width = 3.02 # m
    train_height = 4.67 # m
    K1 = 1.84
    K2 = 0.0067
elif train_type == "D":
    train_width = 3.02 # m
    train_height = 4.67 # m
    K1 = 0.14
    K2 = 0.0039
elif train_type == "S":
    train_width = 3.02 # m
    train_height = 2.5 # m
    K1 = 0.14
    K2 = 0.0039

normalized_length = train_length * (2 * train_width + 2 * train_height) / (train_width * train_height)
drag_coefficient = K1 + K2 * normalized_length
frontal_area = train_width * train_height

motor_efficiency = 0.8
regenerative_braking_efficiency = 0.4
levitator_power = np.interp(train_mass / train_length / 1e3, levitator_specs[:, 0], levitator_specs[:, 1]) * train_length

print("")
print("--- TRAIN SPECIFICATIONS ---")
print(f"drag coefficient: {drag_coefficient}")
print(f"train mass: {train_mass / 1e3} tons")
print(f"train length: {train_length} m")
print(f"train mass per m train: {train_mass / 1e3 / train_length} tons/m*train")
print(f"levitator power: {levitator_power} W")

def calculate_route_data(file_name):
    if ".gpx" not in file_name:
        file_name = file_name + ".gpx"
    
    if folder != "":
        file_name = "Routes/" + folder + "/" + file_name
    else:
        file_name = "Routes/" + file_name
    
    if region == "EU":
        currency = "€"
        energy_price = 0.32 # €/kWh
        benzine_price = 2.109 # €/L
    elif region == "US":
        currency = "US$"
        energy_price = 0.178 # US$/kWh
        benzine_price = 0.97 # US$/L
    
    # Load the GPX data files.
    track_data = gpd.read_file(file_name, layer='tracks')
    station_data = gpd.read_file(file_name, layer='waypoints')
    
    # Convert coordinate system.
    #track = track_data.geometry.explode(index_parts = False).to_crs(epsg=27562)
    #stations = station_data.geometry.to_crs(epsg=27562)
    track = track_data.geometry.explode(index_parts = False).to_crs(epsg=3310)
    stations = station_data.geometry.to_crs(epsg=3310)
    
    # Convert to Pandas Data Frame
    track_coordinates = track.get_coordinates()
    station_coordinates = stations.get_coordinates()
    
    x = track_coordinates['x'].to_numpy()
    y = track_coordinates['y'].to_numpy()
    
    # Notes x and y as a function of the curvilinear abscissa.
    d = 0
    s = [0]
    for i in range(1, len(x)):
        d += math.sqrt((x[i] - x[i - 1])**2 + (y[i] - y[i - 1])**2)
        
        s.append(d)
    
    spline_x = UnivariateSpline(s, x, k = 5)
    spline_y = UnivariateSpline(s, y, k = 5)
    
    s = np.linspace(s[0], s[-1], 10000)
    
    vx = spline_x(s, nu = 1)
    vy = spline_y(s, nu = 1)
    
    ax = spline_x(s, nu = 2)
    ay = spline_y(s, nu = 2)
    
    curvature = abs(vx * ay - vy * ax) / (vx**2 + vy**2)**(3/2)
    radius_of_curvature = 1 / curvature
    
    '''plt.figure(dpi = dpi_value)
    plt.scatter(spline_x(s) / 1000, spline_y(s) / 1000, c = curvature, s = 1)
    #plt.plot(x, y, 'k.')
    plt.grid()
    plt.xlabel("x (km)")
    plt.ylabel("y (km)")
    plt.axis('equal')
    plt.show()
    
    plt.figure(dpi = dpi_value)
    plt.plot(s / 1000, curvature, '.', markersize = 1)
    plt.xlabel("curvilinear abscissa (km)")
    plt.ylabel("curvature ($m^{-1}$)")
    plt.yscale('log')
    plt.grid()
    plt.show()
    
    plt.figure(dpi = dpi_value)
    plt.plot(s / 1000, 1e-3 * radius_of_curvature, '.', markersize = 1)
    plt.xlabel("curvilinear abscissa (km)")
    plt.ylabel("turning radius (km)")
    plt.yscale('log')
    plt.grid()
    plt.show()'''
    
    print(file_name)
    print("Number of Data Points: ", len(x))
    print("Total Distance: ", d * 1e-3, "km")
    print("Maximum Curvature Radius: ", 1e-3 / curvature.min(), "km")
    print("Minimum Curvature Radius: ", 1e-3 / curvature.max(), "km")
    print("Mean Curvature Radius: ", 1e-3 / curvature.mean(), "km")
    
    print("")
    print("--- REVENUE ---")
    
    if train_type == "F":
        revenue = 0.08 * (cargo_mass / 1e3) * (d / 1e3)
        print(f"revenue: {currency} {revenue:,}")
    elif train_type == "S" or train_type == "D":
        ticket_price = 0.05 * (d / 1e3)
        revenue = ticket_price * num_passengers
        print(f"ticket price: {currency} {np.round(ticket_price, 2)}")
        print(f"revenue: {currency} {revenue:,}")
    
    #print("")
    #print("IDEAL OPERATION")
    
    g = 9.81 # m/s^2
    freight_train_velocity = 80/3.6 # m/s
    
    # Calculates freight train balancing angle.
    def calculate_balancing_angle(v, r):
        gravity_force = g
        centripetal_force = v**2 / r
        
        return np.arctan(centripetal_force / gravity_force)
    
    # Calculates maximum velocity at a turn.
    def calculate_maximum_velocity(theta, r):
        return np.sqrt(max_centrifugal_g_force * g * r)
        
        #return np.sqrt(np.abs(np.tan(theta)) * g * r)
    
    radii = np.linspace(500, 10000)
    balancing_angles = calculate_balancing_angle(freight_train_velocity, radii)
    
    '''plt.figure(dpi = dpi_value)
    plt.plot(radii, np.degrees(balancing_angles))
    plt.xlabel("turning radius (m)")
    plt.ylabel("balancing angle (deg)")
    plt.show()'''
    
    # Calculates maximum speed if the train can turn a number of degrees.
    maximum_train_tilt_angle = np.radians(10) # deg
    
    velocities = calculate_maximum_velocity(balancing_angles + maximum_train_tilt_angle, radii)
    
    '''plt.figure(dpi = dpi_value)
    plt.plot(radii, velocities * 3.6)
    plt.xlabel("turning radius (m)")
    plt.ylabel("maximum velocity (km/h)")
    plt.show()'''
    
    # Adds the maximum velocities to the railway.
    track_angles = calculate_balancing_angle(freight_train_velocity, radius_of_curvature)
    ideal_velocity_profile = calculate_maximum_velocity(track_angles + maximum_train_tilt_angle, radius_of_curvature)
    for i in range(len(ideal_velocity_profile)):
        if ideal_velocity_profile[i] > top_design_speed:
            ideal_velocity_profile[i] = top_design_speed
    
    '''plt.figure(dpi = dpi_value)
    plt.plot(s / 1000, ideal_velocity_profile * 3.6, '.', markersize = 1)
    plt.xlabel("curvilinear abscissa (km)")
    plt.ylabel("maximum velocity (km/h)")
    plt.grid()
    plt.show()'''
    
    def calculate_g_force(theta, v, r):
        tilt_angle = calculate_balancing_angle(freight_train_velocity, r) + maximum_train_tilt_angle
        gravity_force = g
        centripetal_force = v**2 / r
        
        net_vertical_force = np.sin(tilt_angle) * centripetal_force \
            + np.cos(tilt_angle) * gravity_force
        
        return net_vertical_force / g
    
    #print("Mean Maximum Velocity: ", ideal_velocity_profile.mean() * 3.6, "km/h")
    #print("Travel Time: ", (d / (ideal_velocity_profile.mean())) / 60, "min")
    
    #g_forces = calculate_g_force(track_angles + maximum_train_tilt_angle, ideal_velocity_profile, radius_of_curvature)
    #print("Mean G Force: ", g_forces.mean())
    #print("Max G Force: ", g_forces.max())
    
    # Calculates Newtonian operation statistics.
    # Looks at next sampling point of the velocity profile.
    # Determines when the difference in velocity is too great.
    # Then changes target velocity.
    print("")
    print("-- OPERATION --")
    
    # Maximum acceleration of the train. (Also used for deceleration.)
    forward_vision = 1000 # amount of indices the train looks ahead to see whether it can stay under the maximum velocity.
    
    def calculate_deceleration_distance(v0, v1):
        T = abs(v1 - v0) / a_max
        return v0 * T - 0.5 * a_max * T**2
    
    def F_air_resistance(v):
        air_density = 1.204 # kg/m^3
        
        return 0.5 * drag_coefficient * frontal_area * air_density * v**2
    
    # NOTE! This is reusing an existing variable.
    x = s[0]
    v = 0
    a = 0
    t = 0
    dt = 0.1
    i = 0
    p = 0
    
    turning_radii = []
    times = []
    positions = []
    velocities = []
    centrifugal_forces = []
    long_forces = []
    
    motor_energy_loss = 0
    
    loading_bar_counter = 0
    
    while x < d:
        # Update i.
        while x > s[i]:
            i += 1
        
        # Check for slow down impossibility.
        slow_down = False
        for j in range(forward_vision):
            if i + j >= len(s):
                break
            if j == 0 or ideal_velocity_profile[i + j] >= v:
                continue
            
            elif calculate_deceleration_distance(v, ideal_velocity_profile[i + j]) >= s[i + j] - s[i]:
                slow_down = True
    
        if calculate_deceleration_distance(v, 0) >= s[len(s) - 1] - s[i]:
            slow_down = True
            
        if slow_down == False:
            a = max(min(ideal_velocity_profile[i] - v, a_max), -a_max)
        else:
            a = -a_max
        
        if a < 0:
            p = (-np.abs(regenerative_braking_efficiency * train_mass * a) + F_air_resistance(v) / motor_efficiency) * v
        else:
            p = (train_mass * a + F_air_resistance(v)) * v / motor_efficiency
        
        #p = F_air_resistance(v) * v / motor_efficiency
        
        motor_energy_loss += p * dt
        
        positions.append(x)
        velocities.append(v)
        times.append(t)
        turning_radii.append(radius_of_curvature[i])
        
        centrifugal_forces.append(v**2 / radius_of_curvature[i])
        long_forces.append(a)
        
        v += a * dt
        x += v * dt
        
        t += dt
        
        if show_loading_bar:
            loading_bar_counter += 1
            if loading_bar_counter >= 1000:
                print(f"PROGRESS: {np.round(x / d * 100, 2)} %")
                loading_bar_counter = 0
    
    times = np.array(times)
    positions = np.array(positions)
    velocities = np.array(velocities)
    
    '''plt.figure(dpi = dpi_value)
    plt.plot(times, long_forces)
    plt.xlabel("t (s)")
    plt.ylabel("longitudinal acceleration (m/s^2)")
    plt.show()
    
    plt.figure(dpi = dpi_value)
    plt.plot(times, centrifugal_forces)
    plt.xlabel("t (s)")
    plt.ylabel("centrifugal acceleration (m/s^2)")
    plt.show()
    
    plt.figure(dpi = dpi_value)
    plt.plot(times[0:3000], long_forces[0:3000])
    plt.xlabel("t (s)")
    plt.ylabel("longitudinal acceleration (m/s^2)")
    plt.show()
    
    plt.figure(dpi = dpi_value)
    plt.plot(times[0:3000], centrifugal_forces[0:3000])
    plt.xlabel("t (s)")
    plt.ylabel("centrifugal acceleration (m/s^2)")
    plt.show()
    
    def plot_dft(x, y, sampling_rate, title = "", is_log = False, cut_off = 0.05):
        sp = np.fft.fft(y)
        frequencies = np.fft.fftfreq(x.shape[-1], 1 / sampling_rate)
        powers = abs(sp)
        
        maximum_power = np.max(powers)
        
        c_freq = []
        c_powers = []
        for i in range(len(powers)):
            if powers[i] > cut_off * maximum_power and frequencies[i] >= 0:
                c_freq.append(frequencies[i])
                c_powers.append(powers[i])
        
        plt.figure(dpi = 400)
        plt.stem(c_freq, c_powers, markerfmt=" ")
        plt.title(title)
        plt.xlabel("frequency (Hz)")
        if is_log:
            plt.yscale('log')
        plt.show()
        
    plot_dft(times, centrifugal_forces, 1 / dt, "Centrifugal Forces (DFT)", False, 0.05)
    plot_dft(times, long_forces, 1 / dt, "Longitudinal Forces (DFT)", False, 0.05)
    
    plot_dft(times, centrifugal_forces, 1 / dt, "Centrifugal Forces (DFT)", True, 0)
    plot_dft(times, long_forces, 1 / dt, "Longitudinal Forces (DFT)", True, 0)
    
    plt.figure(dpi = dpi_value)
    plt.plot(times, turning_radii)
    plt.xlabel("t (s)")
    plt.ylabel("turning radius (m)")
    plt.yscale('log')
    plt.show()
    
    plt.figure(dpi = dpi_value)
    plt.plot(times, velocities * 3.6, '.', markersize = 1)
    plt.xlabel("t (s)")
    plt.ylabel("velocity (km/h)")
    plt.grid()
    plt.show()'''
    
    if (PLOT == True):
        plt.figure(dpi = dpi_value)
        plt.plot(positions / 1000, velocities * 3.6, '.', markersize = 1)
        plt.plot(s / 1000, ideal_velocity_profile * 3.6, '.', markersize = 1)
        plt.xlabel("curvilinear abscissa (km)")
        plt.ylabel("velocity (km/h)")
        plt.grid()
        plt.show()
    
    print("Travel Time: ", t / 60, "min")
    print("Mean Velocity (over time): ", velocities.mean() * 3.6, "km/h")
    print("Max Velocity: ", velocities.max() * 3.6, "km/h")
    
    levitator_energy_loss = levitator_power * t
    
    #tilt_power_coefficeint = 1e3 # W/angle
    #tilt_power = 0
    tilt_energy_loss = 0
    
    total_energy_loss = levitator_energy_loss + motor_energy_loss + tilt_energy_loss
    total_energy_cost = total_energy_loss / (3.6e6) * energy_price
    
    print(f"Total Levitator Energy Usage: {levitator_energy_loss / (3.6e6)} kWh")
    print(f"Total Motor Energy Usage: {motor_energy_loss / (3.6e6)} kWh")
    print(f"Total Energy Usage: {total_energy_loss / (3.6e6)} kWh")
    print(f"Total Energy Cost: {currency} {np.round(total_energy_cost, 2):,}")
    
    if train_type != "F":
        print("")
        print(f"Energy Cost per Passenger-100km: {currency} {np.round(100 * total_energy_cost / num_passengers / (d / 1000), 2)}")
        print(f"Energy Cost per Passenger-100km (BMW 330e, 5 people): {currency} {np.round(curviness_factor * 4.9 * benzine_price / 5, 2)}") # 4.9 L/100km
        print(f"Energy Cost per Passenger-100km (Tesla Model 3, 5 people): {currency} {np.round(curviness_factor * 14.2 * energy_price / 5, 2)}") # 14.2 kWh/100km
        print("")
        print(f"Cost per Passenger: {currency} {np.round(ticket_price, 2)}")
        print(f"Cost per Passenger (BMW 330e, 5 people): {currency} {np.round(curviness_factor * 4.9 * benzine_price / 5 * (d / (100e3)), 2)}") # 4.9 L/100km
        print(f"Cost per Passenger (Tesla Model 3, 5 people): {currency} {np.round(curviness_factor * 14.2 * energy_price / 5 * (d / (100e3)), 2)}") # 14.2 kWh/100km
    
    cost_per_double_track = 5.1e6 # US$/km*double track
    
    print("")
    print("--- ECONOMICS ---")
    print(f"Investment Cost (Bi-directional): US$ {np.round(cost_per_double_track * d / 1000):,}")
    print(f"Cost per Train: US$ {10e6:,}")
    
    profit = revenue - total_energy_cost
    
    print("")
    print("--- CASH FLOW ---")
    print(f"revenue: US$ {revenue:,}")
    print(f"profit: US$ {profit:,}")
    print(f"profit margin: {np.round(profit / revenue * 100, 2)} %")
    
    return (t, s[-1], velocities, total_energy_loss)

t_total = 0
total_distance = 0
distances = []
velocities = []
total_energy_cost = 0
for route in routes:
    (_t, _d, _v, _e) = calculate_route_data(route)
    
    t_total += _t
    total_distance += _d
    total_energy_cost += _e

    distances.append(_d)
    velocities += _v.tolist()
    
# Adds station dwell time (1 minute per station)
t_dwell_total = 60 * (len(routes) - 2)

print("")
print(f"--- FINAL REPORT FOR TOTAL JOURNEY ---")
print(f"total distance: {total_distance / 1e3} km")
print(f"total travel time: {(t_total + t_dwell_total) / 60} min")
print(f"total energy cost: {total_energy_cost * 2.77778e-7} kWh")
print(f"average speed: {(np.average(velocities) * t_total + 0) / (t_total + t_dwell_total) * 3.6} km/h")
print(f"top speed: {np.max(velocities) * 3.6} km/h")
print(f"average distance between stations: {np.average(distances) / 1e3} km")

