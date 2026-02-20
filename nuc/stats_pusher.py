#!/usr/bin/env python3
"""
Polls bird_monitor.py on the Pi and pushes stats to the Laravel API.
Laravel broadcasts via Reverb to all connected browsers.

Env vars:
    PI_URL          - Pi bird monitor base URL (default: http://192.168.1.176:8888)
    LARAVEL_URL     - Laravel API base URL (default: https://ekstremedia.no)
    LARAVEL_TOKEN   - Bearer token for Laravel API (optional, for future auth)
    POLL_INTERVAL   - Seconds between polls (default: 5)
"""

import json
import os
import sys
import time

import requests

PI_URL = os.getenv("PI_URL", "http://192.168.1.176:8888")
LARAVEL_URL = os.getenv("LARAVEL_URL", "https://ekstremedia.no")
LARAVEL_TOKEN = os.getenv("LARAVEL_TOKEN", "")
POLL_INTERVAL = int(os.getenv("POLL_INTERVAL", "5"))

STATS_ENDPOINT = f"{PI_URL}/api/stats"
BIRDS_ENDPOINT = f"{PI_URL}/api/birds"
PUSH_ENDPOINT = f"{LARAVEL_URL}/api/birdcam/stats"


def fetch_pi_data():
    stats = requests.get(STATS_ENDPOINT, timeout=5).json()
    birds = requests.get(BIRDS_ENDPOINT, timeout=5).json()
    return stats, birds


def push_to_laravel(stats, birds):
    headers = {"Content-Type": "application/json", "Accept": "application/json"}
    if LARAVEL_TOKEN:
        headers["Authorization"] = f"Bearer {LARAVEL_TOKEN}"

    resp = requests.post(
        PUSH_ENDPOINT,
        json={"stats": stats, "birds": birds},
        headers=headers,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def main():
    print(f"Stats pusher started")
    print(f"  Pi: {PI_URL}")
    print(f"  Laravel: {PUSH_ENDPOINT}")
    print(f"  Interval: {POLL_INTERVAL}s")
    print()

    backoff = POLL_INTERVAL
    max_backoff = 60

    while True:
        try:
            stats, birds = fetch_pi_data()
            push_to_laravel(stats, birds)

            bird_count = stats.get("current_birds", 0)
            visits = stats.get("today_visits", 0)
            fps = stats.get("fps", 0)
            print(f"Pushed: {bird_count} fugler, {visits} besøk, {fps} FPS")

            backoff = POLL_INTERVAL

        except requests.exceptions.ConnectionError as e:
            print(f"Tilkoblingsfeil: {e}")
            backoff = min(backoff * 2, max_backoff)
            print(f"  Prøver igjen om {backoff}s...")

        except requests.exceptions.Timeout:
            print("Tidsavbrudd — prøver igjen...")
            backoff = min(backoff * 2, max_backoff)

        except Exception as e:
            print(f"Feil: {e}")
            backoff = min(backoff * 2, max_backoff)

        time.sleep(backoff)


if __name__ == "__main__":
    main()
