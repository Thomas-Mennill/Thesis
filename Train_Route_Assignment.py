"""Assign railway tracks and complete routes for scheduled trains.

The input graph is a NetworkX graph.  Nodes may have a ``station`` attribute
and edges may provide ``length`` (metres), ``switches`` (an integer), and
``track_index``/``track_count`` attributes.  A station track is identified by
``station``, ``parallel_group`` and ``track_index``.  Missing optional
attributes have safe defaults, which also makes the functions useful with the
graphs produced by the other scripts in this repository.

Train definitions can be loaded directly from the timetable blocks used by
``existing_rail_routes.csv`` and ``new_rail_routes.csv``.
They contain ``id``, ``category``, ``priority``, ``stations`` and optional
``platforms``.  ``stations`` is never reordered: a route that visits a
different station is rejected.
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import networkx as nx


SPRINTER = "sprinter"
NON_SPRINTER_CATEGORIES = {"intercity", "freight", "maglev", "ice"}


def _normalise_category(value: Any) -> str:
    category = str(value).strip().casefold().replace("-", "").replace("_", "")
    return {
        "ic": "intercity",
        "intercitytrain": "intercity",
        "freighttrain": "freight",
        "maglevtrain": "maglev",
        "icetrain": "ice",
    }.get(category, category)


def _integer_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _time_hour(train: Mapping[str, Any]) -> int:
    value = train.get("departure_time", train.get("time", ""))
    text = str(value).strip()
    if not text:
        return 0
    try:
        return int(text.split(":", 1)[0]) % 24
    except (ValueError, IndexError) as error:
        raise ValueError(
            f"Train {train.get('id', '<unknown>')} has invalid departure time "
            f"{value!r}"
        ) from error


def _edge_length(first: Any, second: Any, data: Mapping[str, Any]) -> float:
    value = data.get("length", data.get("length_meters"))
    if value is not None:
        length = float(value)
        if not math.isfinite(length) or length < 0:
            raise ValueError(f"Invalid edge length {value!r}")
        return length
    if isinstance(first, (tuple, list)) and isinstance(second, (tuple, list)):
        return math.dist(first[:2], second[:2])
    return 1.0


def _edge_records(graph: nx.Graph, first: Any, second: Any):
    if graph.is_multigraph():
        return graph[first][second].items()
    return [(None, graph[first][second])]


def _station(node: Any, attributes: Mapping[str, Any]) -> str | None:
    value = attributes.get("station")
    if value in (None, "", "None"):
        return None
    return str(value).strip()


def _platform_value(value: Any) -> str | None:
    if value in (None, ""):
        return None
    text = str(value).strip()
    return text.casefold() if text else None


def _platforms_for_train(train: Mapping[str, Any]) -> Mapping[str, Any]:
    platforms = train.get("platforms", {})
    if isinstance(platforms, Mapping):
        return platforms
    if isinstance(platforms, (list, tuple)):
        stations = train.get("stations", ())
        return dict(zip(stations, platforms))
    return {}


def _edge_platforms(edge: Mapping[str, Any], node_data: Mapping[str, Any]) -> set[str]:
    values = {
        edge.get("platform"),
        edge.get("platform_id"),
        edge.get("platform_number"),
        node_data.get("platform"),
        node_data.get("platform_id"),
        node_data.get("platform_number"),
    }
    return {
        platform
        for value in values
        if (platform := _platform_value(value)) is not None
    }


def _edge_track(data: Mapping[str, Any], station: str | None) -> tuple:
    group = data.get("parallel_group", data.get("track_group", "default"))
    index = data.get("track_index", data.get("track_number"))
    if index is None:
        index = 0
    try:
        index = int(index)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid track index {index!r}") from error
    return station or str(data.get("station", "")), str(group), index


def right_hand_track_index(track_count: int, direction: str = "forward") -> int:
    """Return the index nearest the right-hand side of a parallel bundle."""
    if track_count < 1:
        raise ValueError("track_count must be positive")
    if direction.casefold() in {"reverse", "backward", "down"}:
        return track_count - 1
    return 0


def _right_hand_penalty(data: Mapping[str, Any]) -> int:
    count = data.get("track_count")
    index = data.get("track_index", data.get("track_number"))
    if count is None or index is None:
        return 0
    expected = right_hand_track_index(int(count), data.get("direction", "forward"))
    return abs(int(index) - expected)


def _is_platform_edge(data: Mapping[str, Any]) -> bool:
    return bool(
        data.get("has_platform", data.get("platform", data.get("platform_id")))
    )


def _ordered_trains(trains: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    def key(train):
        category = _normalise_category(train.get("category", train.get("route_type", "")))
        return (
            0 if category == SPRINTER else 1,
            -float(train.get("priority", 0)),
            _time_hour(train),
            str(train.get("id", "")),
        )

    return sorted(trains, key=key)


def _candidate_edges(graph: nx.Graph, node: Any, station: str, *, sprinter: bool, used_sprinter_tracks: set[tuple], force_sprinter_track: bool, requested_platforms: Mapping[str, Any],):
    candidates = []
    for neighbour in graph.neighbors(node):
        neighbour_station = _station(neighbour, graph.nodes[neighbour])
        if neighbour_station is not None and neighbour_station != station:
            continue
        for key, data in _edge_records(graph, node, neighbour):
            edge = dict(data)
            track = _edge_track(edge, station)
            for endpoint in (node, neighbour):
                endpoint_station = _station(endpoint, graph.nodes[endpoint])
                requested = _platform_value(
                    requested_platforms.get(endpoint_station)
                )
                endpoint_platforms = _edge_platforms(
                    edge, graph.nodes[endpoint]
                )
                if requested is not None and requested not in endpoint_platforms:
                    break
            else:
                requested = None
            if requested is not None:
                continue
            has_platform = _is_platform_edge(edge) or bool(
                graph.nodes[node].get("has_platform")
                or graph.nodes[neighbour].get("has_platform")
            )
            if sprinter and not has_platform:
                continue
            if force_sprinter_track and track not in used_sprinter_tracks:
                continue
            edge["_track"] = track
            edge["_key"] = key
            edge["_neighbour"] = neighbour
            edge["_right_hand_penalty"] = _right_hand_penalty(edge)
            edge["_sprinter_overlap"] = int(track in used_sprinter_tracks)
            candidates.append(edge)

    # For the supported even-sized bundles, do not silently choose an
    # off-side track when the right-hand track is available.
    for edge in candidates:
        count = edge.get("track_count")
        index = edge.get("track_index", edge.get("track_number"))
        try:
            supported_bundle = int(count) in (2, 4, 6)
        except (TypeError, ValueError):
            supported_bundle = False
        if supported_bundle and index is not None:
            expected = right_hand_track_index(
                int(count), edge.get("direction", "forward")
            )
            if any(
                _integer_or_none(other.get("track_count")) == int(count)
                and _integer_or_none(
                    other.get("track_index", other.get("track_number"))
                ) == expected
                for other in candidates
            ) and _integer_or_none(index) != expected:
                continue
        yield edge


def _route_for_train(graph: nx.Graph, train: Mapping[str, Any], used_sprinter_tracks: set[tuple], force_sprinter_track: bool,) -> dict[str, Any]:
    stations = [str(item).strip() for item in train.get("stations", ())]
    if len(stations) < 2 or any(not station for station in stations):
        raise ValueError(f"Train {train.get('id', '<unknown>')} needs two stations")
    starts = [node for node, data in graph.nodes(data=True)
              if _station(node, data) == stations[0]]
    if not starts:
        raise ValueError(f"Station {stations[0]!r} is absent from the graph")

    # The tuple prioritizes track separation for non-sprinters and the
    # right-hand track for sprinters; switches and distance follow.
    queue = []
    best = {}
    previous = {}
    counter = 0
    for node in starts:
        state = (node, 1, None)
        best[state] = (0, 0.0, 0, 0)
        heapq.heappush(queue, (best[state], counter, state))
        counter += 1

    while queue:
        cost, _, state = heapq.heappop(queue)
        if cost != best.get(state):
            continue
        node, next_station, previous_track = state
        if next_station == len(stations):
            break
        target = stations[next_station]
        requested_platforms = _platforms_for_train(train)
        sprinter = _normalise_category(
            train.get("category", train.get("route_type", ""))
        ) == SPRINTER
        for edge in _candidate_edges(
            graph, node, target, sprinter=sprinter,
            used_sprinter_tracks=used_sprinter_tracks,
            force_sprinter_track=force_sprinter_track,
            requested_platforms=requested_platforms,
        ):
            neighbour = edge["_neighbour"]
            reached = _station(neighbour, graph.nodes[neighbour])
            new_index = next_station
            if reached is not None:
                if reached != target:
                    continue
                new_index += 1
            track = edge["_track"]
            switches = int(edge.get("switches", edge.get("switch_count", 0)))
            if previous_track is not None and track != previous_track:
                switches += 1
            distance = cost[2] + _edge_length(node, neighbour, edge)
            if sprinter:
                new_cost = (
                    cost[0] + edge["_right_hand_penalty"],
                    cost[1] + switches,
                    distance,
                    cost[3],
                )
            else:
                new_cost = (
                    cost[0] + (0 if edge["_sprinter_overlap"] else 1),
                    cost[1] + switches,
                    distance,
                    cost[3] + edge["_right_hand_penalty"],
                )
            new_state = (neighbour, new_index, track)
            if new_cost < best.get(new_state, (math.inf, math.inf, math.inf, math.inf)):
                best[new_state] = new_cost
                previous[new_state] = (state, edge)
                heapq.heappush(queue, (new_cost, counter, new_state))
                counter += 1

    final = [state for state in best if state[1] == len(stations)]
    if not final:
        raise ValueError(
            f"No route for train {train.get('id', '<unknown>')} follows stations "
            f"exactly: {stations}"
        )
    final_state = min(final, key=best.__getitem__)
    edges = []
    states = [final_state]
    while states[-1] in previous:
        old_state, edge = previous[states[-1]]
        edges.append(edge)
        states.append(old_state)
    edges.reverse()
    tracks = tuple(edge["_track"] for edge in edges if edge["_track"][0])
    return {
        "train_id": train.get("id"),
        "stations": stations,
        "track_assignments": list(dict.fromkeys(tracks)),
        "path": [state[0] for state in reversed(states)],
        "cost": {
            "switches": best[final_state][1],
            "length": best[final_state][2],
        },
    }


def assign_train_routes(graph: nx.Graph, trains: Iterable[Mapping[str, Any]], *, non_sprinter_limit: int = 12,) -> list[dict[str, Any]]:
    """Assign tracks first, then complete every route in service order."""
    if non_sprinter_limit < 0:
        raise ValueError("non_sprinter_limit must be non-negative")
    ordered = _ordered_trains(trains)
    used_sprinter_tracks: set[tuple] = set()
    non_sprinter_by_hour = defaultdict(int)
    assignments = []
    for train in ordered:
        category = _normalise_category(
            train.get("category", train.get("route_type", ""))
        )
        if category not in {SPRINTER} | NON_SPRINTER_CATEGORIES:
            raise ValueError(f"Unsupported train category {train.get('category')!r}")
        hour = _time_hour(train)
        force_sprinter = (
            category != SPRINTER
            and non_sprinter_by_hour[hour] >= non_sprinter_limit
        )
        assignment = _route_for_train(
            graph, train, used_sprinter_tracks, force_sprinter
        )
        assignment["category"] = category
        assignment["forced_sprinter_track"] = force_sprinter
        assignments.append(assignment)
        if category == SPRINTER:
            used_sprinter_tracks.update(assignment["track_assignments"])
        else:
            non_sprinter_by_hour[hour] += 1
    return assignments


def _split_station_platform(station: str, platform: str = "") -> tuple[str, str | None]:
    station = station.strip()
    platform = platform.strip()
    if platform:
        return station, platform
    for separator in ("|", "@"):
        if separator in station:
            name, value = station.rsplit(separator, 1)
            return name.strip(), value.strip() or None
    return station, None


def load_timetable(path: str | Path) -> list[dict[str, Any]]:
    """Load the event/station/time/platform blocks used by the route CSVs."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        raise ValueError(f"Timetable {path!r} is empty")

    trains = []
    event_columns = []
    for column in range(max(map(len, rows))):
        if any(
            column < len(row)
            and row[column].strip().casefold() in {"depart", "departure", "arrive", "arrival"}
            for row in rows[1:]
        ):
            event_columns.append(column)

    for column in event_columns:
        station_column = column + 1
        time_column = column + 2
        platform_column = column + 3
        if station_column >= len(rows[0]):
            continue
        train_id = rows[0][station_column].strip()
        if not train_id or train_id.casefold() in {"time", "platform"}:
            continue
        stations = []
        platforms = {}
        events = []
        for row in rows[1:]:
            if station_column >= len(row):
                continue
            event = row[column].strip() if column < len(row) else ""
            station = row[station_column].strip()
            time = row[time_column].strip() if time_column < len(row) else ""
            platform = (
                row[platform_column].strip()
                if platform_column < len(row)
                else ""
            )
            if not event and not station and not time and not platform:
                continue
            if not event or not station:
                raise ValueError(
                    f"Incomplete event for train {train_id!r} in {path!r}"
                )
            station, explicit_platform = _split_station_platform(
                station, platform
            )
            if not stations or stations[-1] != station:
                stations.append(station)
            if explicit_platform:
                platforms[station] = explicit_platform
            events.append({"event": event, "station": station, "time": time})
        if not events:
            raise ValueError(f"Train {train_id!r} has no events in {path!r}")
        departure = next(
            event for event in events
            if event["event"].casefold() in {"depart", "departure"}
        )
        trains.append({
            "id": train_id,
            "stations": stations,
            "platforms": platforms,
            "departure_time": departure["time"] or "00:00:00",
            "events": events,
        })
    if not trains:
        raise ValueError(f"No train blocks found in {path!r}")
    return trains


def load_route_metadata(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load route_data.csv-style metadata keyed by route/train ID."""
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))
    if len(rows) < 2:
        return {}
    metadata = {}
    for column in range(1, len(rows[0])):
        train_id = rows[0][column].strip()
        if not train_id:
            continue
        metadata[train_id] = {
            rows[row][0].strip(): rows[row][column].strip()
            for row in range(1, len(rows))
            if rows[row] and column < len(rows[row]) and rows[row][0].strip()
        }
    return metadata


def load_trains(path: str | Path, metadata_path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load timetable blocks, optionally enriching them with route metadata."""
    if Path(path).suffix.casefold() == ".json":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("Train JSON must contain a list")
        trains = data
    else:
        trains = load_timetable(path)
    metadata = load_route_metadata(metadata_path) if metadata_path else {}
    for train in trains:
        values = metadata.get(str(train.get("id")), {})
        train.setdefault("category", values.get("Route Type", ""))
        train.setdefault("priority", float(values.get("Priority", 0) or 0))
    return trains


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--network", default="case_study_network.gpickle")
    parser.add_argument("--trains", required=True)
    parser.add_argument("--metadata", default=None)
    parser.add_argument("--output", default="assigned_train_routes.json")
    args = parser.parse_args()
    with Path(args.network).open("rb") as handle:
        graph = pickle.load(handle)
    assignments = assign_train_routes(
        graph, load_trains(args.trains, args.metadata)
    )
    Path(args.output).write_text(
        json.dumps(assignments, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
