"""Export NS train stops at selected stations to a CSV file.

The NS Reisinformatie API key must be supplied through the NS_API_KEY
environment variable.  The station input file may contain station names,
station codes, or UIC codes, one per line.  Empty lines and lines beginning
with '#' are ignored.

Example:
    $env:NS_API_KEY = "your-key"
    python NS_Station_Schedule_To_CSV.py stations.txt -o selected_trains.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


DEFAULT_BASE_URL = "https://ns.nl"
DEFAULT_TIMEOUT = 30
OUTPUT_FIELDS = (
    "train_number",
    "vehicle",
    "event_time",
    "event_type",
    "station",
    "platform",
)


def normalise(value: Any) -> str:
    """Return a case-insensitive representation suitable for matching."""
    return str(value).strip().casefold() if value is not None else ""


def load_stations(path: Path) -> set[str]:
    """Read station names/codes from a text file."""
    with path.open(encoding="utf-8-sig") as station_file:
        stations = {
            normalise(line.split("#", 1)[0])
            for line in station_file
            if line.split("#", 1)[0].strip()
        }
    if not stations:
        raise ValueError(f"Station file contains no stations: {path}")
    return stations


def request_json(
    base_url: str,
    endpoint: str,
    api_key: str,
    params: dict[str, str],
    timeout: int,
) -> dict[str, Any]:
    """Request and decode one NS API response, with actionable errors."""
    query = f"?{urlencode(params)}" if params else ""
    url = f"{base_url.rstrip('/')}/{endpoint.lstrip('/')}{query}"
    request = Request(
        url,
        headers={
            "Accept": "application/json",
            "Ocp-Apim-Subscription-Key": api_key,
            "X-Api-Key": api_key,
        },
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        raise RuntimeError(
            f"NS API request failed ({error.code}) for {endpoint}: "
            f"{error.reason}"
        ) from error
    except URLError as error:
        raise RuntimeError(f"Could not reach NS API for {endpoint}: {error.reason}") from error
    except json.JSONDecodeError as error:
        raise RuntimeError(f"NS API returned invalid JSON for {endpoint}") from error

    if not isinstance(payload, dict):
        raise RuntimeError(f"NS API returned an unexpected response for {endpoint}")
    return payload


def payload_of(response: dict[str, Any]) -> dict[str, Any]:
    payload = response.get("payload", response)
    return payload if isinstance(payload, dict) else {}


def departures_from(response: dict[str, Any]) -> list[dict[str, Any]]:
    departures = payload_of(response).get("departures", [])
    return [departure for departure in departures if isinstance(departure, dict)]


def train_number(departure: dict[str, Any]) -> str | None:
    product = departure.get("product")
    if isinstance(product, dict):
        number = product.get("number") or product.get("trainNumber")
        if number:
            return str(number)
    for key in ("trainNumber", "number"):
        if departure.get(key):
            return str(departure[key])
    return None


def station_values(stop: dict[str, Any]) -> set[str]:
    station = stop.get("station", {})
    values: set[str] = set()
    if isinstance(station, dict):
        values.update(
            normalise(station.get(key))
            for key in ("name", "code", "uicCode", "UICCode")
            if station.get(key) is not None
        )
    elif station:
        values.add(normalise(station))
    values.update(
        normalise(stop.get(key))
        for key in ("stationName", "stationCode", "uicCode")
        if stop.get(key) is not None
    )
    return {value for value in values if value}


def station_name(stop: dict[str, Any]) -> str:
    station = stop.get("station", {})
    if isinstance(station, dict):
        return str(station.get("name") or station.get("code") or "")
    return str(station or stop.get("stationName") or "")


def first_value(value: Any, *keys: str) -> Any:
    if isinstance(value, dict):
        for key in keys:
            if value.get(key) not in (None, ""):
                return value[key]
    return None


def event_time(event: dict[str, Any]) -> str | None:
    return first_value(event, "plannedTime", "actualTime", "time")


def event_platform(stop: dict[str, Any], event: dict[str, Any]) -> str:
    platform = first_value(event, "plannedTrack", "actualTrack", "track")
    if platform is None:
        platform = first_value(stop, "plannedTrack", "actualTrack", "track")
    return "" if platform is None else str(platform)


def display_vehicle(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float)):
        return str(value)
    if isinstance(value, list):
        return ", ".join(
            item for item in (display_vehicle(entry) for entry in value) if item
        )
    if isinstance(value, dict):
        for keys in (
            ("number", "type"),
            ("name", "type"),
            ("name",),
            ("type",),
        ):
            values = [str(value[key]) for key in keys if value.get(key) not in (None, "")]
            if values:
                return " ".join(values)
    return ""


def route_vehicle(route_payload: dict[str, Any]) -> str:
    for key in ("actualStock", "stock", "vehicle", "vehicles", "material", "composition"):
        vehicle = display_vehicle(route_payload.get(key))
        if vehicle:
            return vehicle
    return ""


def rows_for_route(
    train: str,
    route_response: dict[str, Any],
    selected_stations: set[str],
) -> list[dict[str, str]]:
    payload = payload_of(route_response)
    stops = payload.get("stops", [])
    if not isinstance(stops, list):
        return []
    vehicle = route_vehicle(payload)
    rows: list[dict[str, str]] = []
    for stop in stops:
        if not isinstance(stop, dict) or not station_values(stop) & selected_stations:
            continue
        for event_type, key in (("arrival", "arrivals"), ("departure", "departures")):
            events = stop.get(key, [])
            if not isinstance(events, list):
                continue
            for event in events:
                if not isinstance(event, dict) or event_time(event) is None:
                    continue
                rows.append(
                    {
                        "train_number": train,
                        "vehicle": vehicle,
                        "event_time": str(event_time(event)),
                        "event_type": event_type,
                        "station": station_name(stop),
                        "platform": event_platform(stop, event),
                    }
                )
    return rows


def export_schedule(
    stations_file: Path,
    output_file: Path,
    api_key: str,
    base_url: str = DEFAULT_BASE_URL,
    delay: float = 0.2,
    timeout: int = DEFAULT_TIMEOUT,
) -> int:
    selected_stations = load_stations(stations_file)
    train_numbers: set[str] = set()
    for station in sorted(selected_stations):
        response = request_json(base_url, "/departures", api_key, {"station": station}, timeout)
        train_numbers.update(
            number for departure in departures_from(response)
            if (number := train_number(departure)) is not None
        )
        if delay:
            time.sleep(delay)

    rows: list[dict[str, str]] = []
    for train in sorted(train_numbers):
        route = request_json(base_url, "/vehicles", api_key, {"id": train}, timeout)
        rows.extend(rows_for_route(train, route, selected_stations))
        if delay:
            time.sleep(delay)

    rows.sort(key=lambda row: (row["event_time"], row["train_number"], row["station"], row["event_type"]))
    with output_file.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export NS train arrivals and departures at stations from a text file."
    )
    parser.add_argument("stations_file", type=Path, help="Text file with one station name/code per line")
    parser.add_argument("-o", "--output", type=Path, default=Path("ns_station_schedule.csv"))
    parser.add_argument("--base-url", default=os.getenv("NS_API_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds between API requests")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_key = os.getenv("NS_API_KEY")
    if not api_key:
        raise SystemExit("Set the NS_API_KEY environment variable before running this script.")
    try:
        row_count = export_schedule(
            args.stations_file,
            args.output,
            api_key,
            args.base_url,
            args.delay,
            args.timeout,
        )
    except (OSError, ValueError, RuntimeError) as error:
        raise SystemExit(f"Error: {error}") from error
    print(f"Wrote {row_count} schedule events to {args.output}")


if __name__ == "__main__":
    main()
