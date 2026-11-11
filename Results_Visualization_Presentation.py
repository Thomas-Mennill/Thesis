"""Display a railway string diagram from the complete event-times Parquet file.

The horizontal axis is time and the vertical axis is the ordered list of
stations. Each coloured line represents one train. Arrival/departure events
are shown as circles and pass-through events as crosses.

Example:
    python Results_Visualization_Presentation.py train_event_times.parquet
"""

from __future__ import annotations

import argparse
import csv
import pickle
import statistics
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import folium
import pandas as pd


STATION_ORDER = (
    "Amsterdam Zuid",
    "Amsterdam Bijlmer ArenA",
    "Amsterdam Holendrecht",
    "Abcoude",
    "Breukelen",
    "Maarssen",
    "Utrecht Zuilen",
    "Utrecht Centraal",
)

ALTERNATE_STATION_ORDER = (
    "Amsterdam Centraal",
    "Amsterdam Muiderpoort",
    "Amsterdam Amstel",
    "Duivendrecht",
    "Amsterdam Bijlmer ArenA",
    "Amsterdam Holendrecht",
    "Abcoude",
    "Breukelen",
    "Maarssen",
    "Utrecht Zuilen",
    "Utrecht Centraal",
)

TRAIN_TYPE_COLOURS = {
    "sprinter": "red",
    "freight": "purple",
    "intercity": "blue",
    "ic": "blue",
    "ice": "blue",
    "maglev": "green",
}
TRAIN_CLASSES = ("sprinter", "intercity", "ice", "maglev", "freight")
MAP_STATION_ORDER = ALTERNATE_STATION_ORDER
TRAIN_TYPE_ALIASES = {"ic": "intercity"}
TRAIN_TYPE_LABELS = {"intercity": "IC"}


@dataclass(frozen=True)
class StopEvent:
    """One arrival or departure event in a train's schedule."""

    event: str
    station: str
    time: datetime
    location_id: str


@dataclass(frozen=True)
class TrainRoute:
    """The ordered events for one train."""

    train_id: str
    events: tuple[StopEvent, ...]
    train_type: str = "unknown"


def parse_time(value: str) -> datetime:
    """Parse the schedule's ``HH:MM:SS`` time representation."""
    for format_string in ("%H:%M:%S", "%H:%M"):
        try:
            return datetime.strptime(value.strip(), format_string)
        except ValueError:
            continue
    raise ValueError(f"Invalid schedule time {value!r}")


def load_event_times(
    parquet_file: Path,
    time_column: str = "time",
) -> list[TrainRoute]:
    """Read train events from the long-format event-times Parquet file."""
    frame = pd.read_parquet(parquet_file)
    required_columns = {
        "train", "event_id", "event_type", "location_id", "station",
        time_column,
    }
    missing_columns = required_columns - set(frame.columns)
    if missing_columns:
        raise ValueError(
            f"Event-times Parquet is missing columns: {sorted(missing_columns)}"
        )

    grouped_events: dict[str, list[tuple[int, StopEvent]]] = {}
    for row_number, row in enumerate(frame.to_dict("records"), start=2):
        train_id = str(row["train"]).strip()
        station = str(row["station"]).strip()
        if not train_id or not station:
            raise ValueError(f"Missing train or station on Parquet row {row_number}")
        try:
            event_id = int(row["event_id"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid event ID on Parquet row {row_number}: "
                f"{row['event_id']!r}"
            ) from error
        time_value = row[time_column]
        if pd.isna(time_value) or not str(time_value).strip():
            continue
        event = StopEvent(
            event=str(row["event_type"]).strip().lower(),
            station=station,
            time=parse_time(str(time_value)),
            location_id=str(row["location_id"]).strip(),
        )
        grouped_events.setdefault(train_id, []).append((event_id, event))

    routes = [
        TrainRoute(
            train_id=train_id,
            events=tuple(event for _, event in sorted(events)),
        )
        for train_id, events in grouped_events.items()
    ]

    if not routes:
        raise ValueError(f"Event-times Parquet contains no train events: {parquet_file}")
    return routes


def load_train_types(route_data_file: Path) -> dict[str, str]:
    """Read train categories from the route metadata CSV."""
    with route_data_file.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"Route metadata CSV is empty: {route_data_file}")
    try:
        route_row = next(row for row in rows if row and row[0].strip() == "Route")
        type_row = next(
            row for row in rows if row and row[0].strip() == "Route Type"
        )
    except StopIteration as error:
        raise ValueError(
            f"Route metadata must contain Route and Route Type rows: "
            f"{route_data_file}"
        ) from error
    return {
        route_id.strip(): TRAIN_TYPE_ALIASES.get(
            train_type.strip().casefold(),
            train_type.strip().casefold(),
        )
        for route_id, train_type in zip(route_row[1:], type_row[1:])
        if route_id.strip() and train_type.strip()
    }


def load_station_positions(
    route_graph_file: Path,
    station_order: tuple[str, ...] = STATION_ORDER,
) -> dict[str, float]:
    """Return corridor station positions measured along the route network."""
    with route_graph_file.open("rb") as handle:
        route_graphs = pickle.load(handle)
    positions: dict[str, list[float]] = {
        station: [] for station in station_order
    }
    for graph in route_graphs.values():
        route_node_ids = graph.graph.get("route_node_ids", ())
        assigned_edges = graph.graph.get("assigned_edges", ())
        cumulative = [0.0]
        for edge_id in assigned_edges:
            lengths = [
                attributes.get("length_m")
                for _, _, attributes in graph.edges(data=True)
                if str(attributes.get("edge_id")) == str(edge_id)
            ]
            if not lengths or lengths[0] is None:
                raise ValueError(
                    f"Route graph {graph.graph.get('train_id', '')!r} "
                    f"has no length for edge {edge_id!r}"
                )
            cumulative.append(cumulative[-1] + float(lengths[0]))
        for index, node_id in enumerate(route_node_ids):
            for node, attributes in graph.nodes(data=True):
                if attributes.get("node_id", node) != node_id:
                    continue
                station = attributes.get("station")
                if station in positions:
                    positions[station].append(cumulative[index])

    missing = [station for station, values in positions.items() if not values]
    if missing:
        raise ValueError(
            f"Route network contains no positions for stations: {missing}"
        )
    baseline = min(
        statistics.median(values) for values in positions.values()
    )
    return {
        station: statistics.median(values) - baseline
        for station, values in positions.items()
    }


def _speed_colour(speed_kmh: float) -> str:
    """Return a map colour for an average segment speed."""
    if speed_kmh < 40:
        return "#2166ac"
    if speed_kmh < 70:
        return "#67a9cf"
    if speed_kmh < 100:
        return "#fddbc7"
    return "#b2182b"


def create_average_speed_map(
    routes: list[TrainRoute],
    route_graph_file: Path,
    output_file: Path,
    train_type: str,
) -> bool:
    """Create an HTML speed map for one train class."""
    with route_graph_file.open("rb") as handle:
        route_graphs = pickle.load(handle)

    included_events = {"run_through", "arrive", "arrival", "depart", "departure"}
    segment_samples: dict[str, list[float]] = {}
    segment_geometries: dict[str, dict] = {}
    all_coordinates: list[tuple[float, float]] = []

    for route in routes:
        if route.train_type != train_type:
            continue
        graph = route_graphs.get(int(route.train_id))
        if graph is None:
            continue
        route_node_ids = list(graph.graph.get("route_node_ids", ()))
        assigned_edges = list(graph.graph.get("assigned_edges", ()))
        edges_by_id = {
            str(attributes.get("edge_id")): (first, second, attributes)
            for first, second, attributes in graph.edges(data=True)
        }
        events_by_node: dict[str, list[StopEvent]] = {}
        for event in route.events:
            events_by_node.setdefault(event.location_id, []).append(event)

        def event_time(node_id: int, destination: bool) -> float | None:
            events = events_by_node.get(str(node_id), [])
            preferred = (
                {"arrive", "arrival", "run_through"}
                if destination
                else {"depart", "departure", "run_through"}
            )
            candidates = [
                event.time for event in events if event.event in preferred
            ]
            if not candidates:
                candidates = [
                    event.time for event in events
                    if event.event in included_events or event.event == "route_node"
                ]
            return (
                candidates[0].hour * 3600
                + candidates[0].minute * 60
                + candidates[0].second
                if candidates
                else None
            )

        for index, edge_id in enumerate(assigned_edges):
            edge = edges_by_id.get(str(edge_id))
            if edge is None or index + 1 >= len(route_node_ids):
                continue
            _, _, attributes = edge
            geometry = attributes.get("geometry", [])
            length_m = float(attributes.get("length_m", 0.0))
            if len(geometry) < 2 or length_m <= 0:
                continue
            key = str(edge_id)
            if key not in segment_geometries:
                coordinates = [
                    (float(point[0]), float(point[1])) for point in geometry
                ]
                segment_geometries[key] = {
                    "key": key,
                    "coordinates": coordinates,
                    "distance_m": length_m,
                }
                all_coordinates.extend(coordinates)
            origin_time = event_time(route_node_ids[index], False)
            destination_time = event_time(route_node_ids[index + 1], True)
            elapsed_seconds = (
                destination_time - origin_time
                if origin_time is not None and destination_time is not None
                else None
            )
            if elapsed_seconds is None or elapsed_seconds <= 0:
                continue
            speed_kmh = length_m / elapsed_seconds * 3.6
            segment_samples.setdefault(key, []).append(speed_kmh)

    if not segment_geometries:
        return False

    average_speeds = {
        key: statistics.mean(values)
        for key, values in segment_samples.items()
    }
    center_lon = statistics.mean(point[0] for point in all_coordinates)
    center_lat = statistics.mean(point[1] for point in all_coordinates)
    speed_map = folium.Map(location=[center_lat, center_lon], zoom_start=11)

    for key, segment in segment_geometries.items():
        average_speed = average_speeds.get(key)
        if average_speed is None:
            colour = "#808080"
            popup = (
                f"Network segment: {key}<br>"
                "Average speed: unavailable<br>"
                f"Segment distance: {segment['distance_m'] / 1000:.2f} km"
            )
        else:
            colour = _speed_colour(average_speed)
            popup = (
                f"Network segment: {key}<br>"
                f"Average speed: {average_speed:.1f} km/h<br>"
                f"Segment distance: {segment['distance_m'] / 1000:.2f} km<br>"
                f"Samples: {len(segment_samples[key])}"
            )
        folium.PolyLine(
            locations=[(latitude, longitude) for longitude, latitude in segment["coordinates"]],
            color=colour,
            weight=5,
            opacity=0.8,
            popup=folium.Popup(popup, max_width=300),
        ).add_to(speed_map)

    folium.LayerControl().add_to(speed_map)
    speed_map.save(output_file)
    return True


def plot_string_diagram(
    routes: list[TrainRoute],
    station_positions: dict[str, float],
    station_order: tuple[str, ...] = STATION_ORDER,
    output_file: Path | None = None,
    show: bool = True,
    close: bool = True,
    schedule_label: str = "Final",
) -> None:
    """Plot and optionally save a filtered railway string diagram.

    Only ``run_through``, arrival, and departure events at stations in
    ``STATION_ORDER`` are retained.
    """
    station_positions = {
        station: station_positions[station] for station in station_order
    }
    title = (
        f"{station_order[0]} to {station_order[-1]} "
        f"({schedule_label} schedule)"
    )
    included_events = {"run_through", "arrive", "arrival", "depart", "departure"}
    filtered_routes = [
        TrainRoute(
            train_id=route.train_id,
            events=tuple(
                event
                for event in route.events
                if event.station in station_positions
                and event.event in included_events
            ),
            train_type=route.train_type,
        )
        for route in routes
    ]
    filtered_routes = [route for route in filtered_routes if route.events]
    if not filtered_routes:
        raise ValueError("No train events match the configured station order")

    figure_height = max(6.0, 0.45 * len(station_order) + 2.0)
    figure, axis = plt.subplots(figsize=(14, figure_height))
    for route in filtered_routes:
        x_values = [event.time for event in route.events]
        y_values = [station_positions[event.station] for event in route.events]
        colour = TRAIN_TYPE_COLOURS.get(route.train_type, "black")
        train_type_label = TRAIN_TYPE_LABELS.get(
            route.train_type,
            route.train_type.title(),
        )
        axis.plot(
            x_values,
            y_values,
            color=colour,
            linewidth=1.8,
            label=f"{route.train_id} ({train_type_label})",
        )
        for marker, event_types in (
            ("o", {"arrive", "arrival", "depart", "departure"}),
            ("x", {"run_through"}),
        ):
            marked_events = [
                event for event in route.events if event.event in event_types
            ]
            if marked_events:
                axis.scatter(
                    [event.time for event in marked_events],
                    [station_positions[event.station] for event in marked_events],
                    color=colour,
                    marker=marker,
                    s=28 if marker == "o" else 34,
                    linewidths=1.2,
                )

    axis.set_yticks(list(station_positions.values()))
    axis.set_yticklabels(list(station_positions.keys()))
    axis.invert_yaxis()
    axis.set_ylabel("Station")
    axis.set_xlabel("Time")
    axis.set_title(title)
    axis.grid(axis="both", linestyle=":", linewidth=0.7, alpha=0.7)
    axis.legend(
        title="Train",
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        borderaxespad=0,
    )
    figure.autofmt_xdate()
    figure.tight_layout()

    if output_file is not None:
        figure.savefig(output_file, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    elif close:
        plt.close(figure)


def main() -> None:
    """Load the schedule and display both railway string diagrams."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "schedule",
        nargs="?",
        type=Path,
        default=Path("train_event_times.parquet"),
        help="Complete event-times Parquet file (default: train_event_times.parquet)",
    )
    parser.add_argument(
        "--route-graphs",
        type=Path,
        default=Path("case_study_route_graphs.pkl"),
        help="Route-network graph pickle used for station distances",
    )
    parser.add_argument(
        "--route-data",
        type=Path,
        default=Path("route_data.csv"),
        help="Route metadata CSV containing train categories",
    )
    parser.add_argument(
        "--speed-map",
        type=Path,
        default=Path("train_average_speeds_map.html"),
        help="HTML map colored by average speed per station segment",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("train_schedule_string_diagram.png"),
        help="Image file to create (default: train_schedule_string_diagram.png)",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the image without opening an interactive plot window",
    )
    args = parser.parse_args()
    routes = load_event_times(args.schedule)
    initial_routes = load_event_times(args.schedule, time_column="original_time")
    train_types = load_train_types(args.route_data)
    routes = [
        TrainRoute(
            train_id=route.train_id,
            events=route.events,
            train_type=train_types.get(route.train_id, "unknown"),
        )
        for route in routes
    ]
    initial_routes = [
        TrainRoute(
            train_id=route.train_id,
            events=route.events,
            train_type=train_types.get(route.train_id, "unknown"),
        )
        for route in initial_routes
    ]
    station_positions = load_station_positions(
        args.route_graphs,
        STATION_ORDER,
    )
    plot_string_diagram(
        routes,
        station_positions=station_positions,
        station_order=STATION_ORDER,
        output_file=args.output,
        show=False,
        close=False,
        schedule_label="Final",
    )

    central_corridor_output = args.output.with_name(
        f"{args.output.stem}_central_corridor{args.output.suffix}"
    )
    central_corridor_positions = load_station_positions(
        args.route_graphs,
        ALTERNATE_STATION_ORDER,
    )
    plot_string_diagram(
        routes,
        station_positions=central_corridor_positions,
        station_order=ALTERNATE_STATION_ORDER,
        output_file=central_corridor_output,
        show=False,
        close=False,
        schedule_label="Final",
    )

    initial_outputs = []
    if initial_routes:
        initial_output = args.output.with_name(
            f"{args.output.stem}_initial{args.output.suffix}"
        )
        plot_string_diagram(
            initial_routes,
            station_positions=station_positions,
            station_order=STATION_ORDER,
            output_file=initial_output,
            show=False,
            close=False,
            schedule_label="Initial",
        )
        initial_central_output = args.output.with_name(
            f"{args.output.stem}_initial_central_corridor{args.output.suffix}"
        )
        plot_string_diagram(
            initial_routes,
            station_positions=central_corridor_positions,
            station_order=ALTERNATE_STATION_ORDER,
            output_file=initial_central_output,
            show=False,
            close=False,
            schedule_label="Initial",
        )
        initial_outputs.extend([initial_output, initial_central_output])
    speed_map_outputs = []
    skipped_speed_maps = []
    for train_class in TRAIN_CLASSES:
        speed_map_output = args.speed_map.with_name(
            f"{args.speed_map.stem}_{train_class}{args.speed_map.suffix}"
        )
        map_created = create_average_speed_map(
            routes,
            route_graph_file=args.route_graphs,
            output_file=speed_map_output,
            train_type=train_class,
        )
        if map_created:
            speed_map_outputs.append(speed_map_output)
        else:
            skipped_speed_maps.append(train_class)

    if not args.no_show:
        plt.show()

    print(
        f"Generated diagrams for {len(routes)} trains: "
        f"{args.output} and {central_corridor_output}"
    )
    if initial_outputs:
        print(
            f"Generated initial-schedule diagrams for {len(initial_routes)} trains: "
            + " and ".join(str(output) for output in initial_outputs)
        )
    print("Generated average-speed maps: " + ", ".join(
        str(output) for output in speed_map_outputs
    ))
    if skipped_speed_maps:
        print(
            "Skipped average-speed maps without valid route data: "
            + ", ".join(skipped_speed_maps)
        )


if __name__ == "__main__":
    main()