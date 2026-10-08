import datetime
import json
import time
import requests

# 1. Setup Configuration
API_KEY = "524af9b154c946308a6b61f45be6c6ce"  # Replace with your actual key
STATIONS_LIST = ["DT", "ASD"]  # Delft and Amsterdam Centraal

HEADERS = {
    "Ocp-Apim-Subscription-Key": API_KEY,
    "Accept": "application/json",
}

# NS requires distinct versions for distinct APIs
BASE_URL_V2 = "https://ns.nl"
BASE_URL_V3 = "https://gateway.apiportal.ns.nl/reisinformatie-api/api/v3"


def get_train_numbers(station_code):
    """Fetches unique train run numbers passing through a station using V2."""
    url = f"{BASE_URL_V2}/departures"
    params = {"station": station_code}
    train_numbers = set()

    response = requests.get(url, headers=HEADERS, params=params)

    # Confirm we have content before decoding
    if response.status_code == 200 and response.text.strip():
        try:
            data = response.json()
            departures = (
                data.get("payload", {}).get("departures", [])
                if isinstance(data, dict)
                else []
            )
            for train in departures:
                train_no = train.get("product", {}).get("number")
                if train_no:
                    train_numbers.add(train_no)
        except Exception:
            pass  # Handles edge cases where response text isn't JSON JSON format
    return train_numbers


def get_full_route(train_number):
    """Fetches vehicle path details using V2."""
    url = f"{BASE_URL_V2}/vehicles"
    params = {"id": train_number}

    response = requests.get(url, headers=HEADERS, params=params)

    if response.status_code == 200 and response.text.strip():
        try:
            payload = response.json().get("payload", {})
            stops = payload.get("stops", [])
            route = []
            for stop in stops:
                route.append(
                    {
                        "station": stop.get("station", {}).get("name"),
                        "arr": stop.get("arrivals", [{}])[0].get(
                            "plannedTime", "Origin"
                        ),
                        "dep": stop.get("departures", [{}])[0].get(
                            "plannedTime", "Destination"
                        ),
                    }
                )
            return route
        except Exception:
            return None
    return None


if __name__ == "__main__":
    all_trains = set()
    print("Connecting to NS API and gathering train codes...")

    for station in STATIONS_LIST:
        nums = get_train_numbers(station)
        all_trains.update(nums)
        time.sleep(0.2)

    print(f"Success! Found {len(all_trains)} trains.")

    # Compile the final schedule dictionary
    master_schedule = {}
    for count, t_num in enumerate(sorted(all_trains), 1):
        print(f"[{count}/{len(all_trains)}] Fetching schedule for #{t_num}")
        route = get_full_route(t_num)
        if route:
            master_schedule[t_num] = route
        time.sleep(0.2)

    # Save to a structured file
    with open("ns_24h_schedule.json", "w", encoding="utf-8") as f:
        json.dump(master_schedule, f, indent=4, ensure_ascii=False)

    print("\nProcessing Complete! File saved as 'ns_24h_schedule.json'")
