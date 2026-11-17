#!/usr/bin/env python3
"""Forecast GTA VI's launch-day Twitch viewers hour by hour.

Reads one case per file from the input directory (MAKEFUN_INPUT_DIR, else
/input) and writes one forecast per case to the output directory
(MAKEFUN_OUTPUT_DIR, else /output).

Method ("console-shape median"): the case's comparables hold past big
launches' hourly Twitch category viewers.  Launches that unlocked the way
GTA VI does -- console only, rolling midnight local, no PC simultaneous
unlock -- share one release-day shape, while PC launches, early-access
titles and tiny categories do not.  So the forecast takes, per hour, the
median across the big console-only comparables of ln(viewers_h / viewers_1),
shrunk 0.85 toward persistence, and spreads quantiles with the pool's
cross-launch dispersion (IQR sigma, floored by a horizon-growing term).

Standard library only.  No network, no clock, no randomness: the same input
bytes always produce the same output bytes.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone

Z_VALUES = (-1.6448536269514722, -0.6744897501960817, 0.0,
            0.6744897501960817, 1.6448536269514722)
LEVELS = (0.05, 0.25, 0.5, 0.75, 0.95)
MAX_VIEWERS = 1_000_000_000.0
MIN_POOL = 5
BIG_HOUR1 = 20000.0
SHRINK = 0.85


def parse_time(text):
    """Parse an ISO-8601 UTC timestamp into an aware datetime, or None."""
    if not isinstance(text, str):
        return None
    try:
        moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment


def median(values):
    """Median of a non-empty list of numbers."""
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def quartile(values, q):
    """Linear-interpolation quartile (q in 0..1) of a non-empty list."""
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = q * (len(ordered) - 1)
    low = int(math.floor(pos))
    high = int(math.ceil(pos))
    if low == high:
        return ordered[low]
    frac = pos - low
    return ordered[low] * (1.0 - frac) + ordered[high] * frac


def iqr_sigma(values):
    """Robust sigma estimate from the interquartile range."""
    if len(values) < 2:
        return 0.0
    return (quartile(values, 0.75) - quartile(values, 0.25)) / 1.349


def is_console_only(platforms):
    """True when the platform string names no PC-style platform."""
    text = (platforms or "").upper()
    for token in ("PC", "STEAM", "STADIA"):
        if token in text:
            return False
    return True


def release_day_hours(comparable):
    """Hour -> viewers for a comparable's release day, or None if incomplete.

    A release day counts only when all 24 hours are present with viewers
    above zero; sparse SullyGnome sampling leaves the rest out.
    """
    day_start = parse_time(comparable.get("day_start"))
    if day_start is None:
        return None
    hours = {}
    for row in comparable.get("hourly") or []:
        moment = parse_time(row.get("from"))
        viewers = row.get("viewers")
        if moment is None or not isinstance(viewers, (int, float)):
            continue
        if viewers <= 0:
            continue
        index = int((moment - day_start).total_seconds() // 3600) + 1
        if 1 <= index <= 24:
            hours[index] = float(viewers)
    if len(hours) != 24:
        return None
    return hours


def build_pool(case):
    """Pick the comparable launches whose shape this case should follow.

    Preference order: big console-only launches, then console-only launches
    of any size, then big launches of any kind, then any launch with a full
    release day.  The case's own game never enters its own pool.
    """
    game_name = ""
    game = case.get("game") or {}
    if isinstance(game, dict):
        game_name = (game.get("name") or "").strip().lower()
    candidates = []
    for comparable in case.get("comparables") or []:
        if not isinstance(comparable, dict):
            continue
        name = (comparable.get("name") or "").strip().lower()
        if game_name and name == game_name:
            continue
        hours = release_day_hours(comparable)
        if hours is None:
            continue
        candidates.append((comparable, hours))
    big_console = [(c, h) for c, h in candidates
                   if is_console_only(c.get("platforms")) and h[1] >= BIG_HOUR1]
    if len(big_console) >= MIN_POOL:
        return [h for _, h in big_console]
    console = [(c, h) for c, h in candidates
               if is_console_only(c.get("platforms"))]
    if len(console) >= MIN_POOL:
        return [h for _, h in console]
    big = [(c, h) for c, h in candidates if h[1] >= BIG_HOUR1]
    if len(big) >= MIN_POOL:
        return [h for _, h in big]
    return [h for _, h in candidates]


def hour1_print(case):
    """Hour 1's viewer count, with fallbacks if `known` is missing."""
    known = case.get("known") or []
    for row in known:
        if isinstance(row, dict) and row.get("hour") == 1:
            viewers = row.get("viewers")
            if isinstance(viewers, (int, float)) and viewers > 0:
                return float(viewers)
    for key in ("history_hourly", "history"):
        rows = case.get(key) or []
        for row in reversed(rows):
            if isinstance(row, dict):
                viewers = row.get("viewers")
                if isinstance(viewers, (int, float)) and viewers > 0:
                    return float(viewers)
    return 1.0


def forecast_hours(case):
    """Return the list of 23 forecast rows for hours 2..24."""
    base = hour1_print(case)
    if base < 1.0:
        base = 1.0
    # Always emit all 23 hours: the scorer requires exactly hours 2..24,
    # so never let a short or odd `forecast` list in the case shrink the
    # output.
    wanted = list(range(2, 25))

    pool = build_pool(case)
    medians = {}
    sigmas = {}
    for hour in wanted:
        if pool:
            ratios = [math.log(day[hour] / day[1]) for day in pool]
            medians[hour] = median(ratios)
            floor = 0.20 + 0.02 * (hour - 1)
            sigmas[hour] = max(iqr_sigma(ratios), floor)
        else:
            medians[hour] = 0.0
            sigmas[hour] = 0.15 * math.sqrt(hour - 1) + 0.20

    rows = []
    for hour in wanted:
        centre = base * math.exp(SHRINK * medians[hour])
        spread = sigmas[hour]
        quantiles = []
        previous = 0.0
        for z in Z_VALUES:
            value = centre * math.exp(z * spread)
            value = round(value, 2)
            if value <= 0.0:
                value = 0.01
            if value > MAX_VIEWERS:
                value = MAX_VIEWERS
            if value < previous:
                value = previous
            previous = value
            quantiles.append(value)
        rows.append({"hour": hour, "q": quantiles})
    return rows


def forecast_case(case):
    """Forecast one parsed case object into the output document."""
    return {"hours": forecast_hours(case)}


def persistence_rows(case):
    """Fallback forecast: every quantile at hour 1's print."""
    base = hour1_print(case)
    if base < 1.0:
        base = 1.0
    value = round(min(base, MAX_VIEWERS), 2)
    if value <= 0.0:
        value = 0.01
    return {"hours": [{"hour": h, "q": [value] * 5} for h in range(2, 25)]}


def run(input_dir, output_dir):
    """Forecast every *.in file in the input dir, in name order."""
    os.makedirs(output_dir, exist_ok=True)
    names = sorted(name for name in os.listdir(input_dir)
                   if name.endswith(".in"))
    for name in names:
        path = os.path.join(input_dir, name)
        case_id = name[:-3]
        try:
            with open(path, "r", encoding="utf-8") as handle:
                case = json.load(handle)
            if isinstance(case, dict) and case.get("id"):
                case_id = str(case["id"])
            document = forecast_case(case)
        except Exception:
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    case = json.load(handle)
                document = persistence_rows(case if isinstance(case, dict) else {})
            except Exception:
                document = {"hours": [{"hour": h, "q": [1.0] * 5}
                                      for h in range(2, 25)]}
        out_path = os.path.join(output_dir, case_id + ".out")
        with open(out_path, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(document, separators=(",", ":")) + "\n")


def main(argv):
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR") or "/input"
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR") or "/output"
    if len(argv) > 1:
        input_dir = argv[1]
    if len(argv) > 2:
        output_dir = argv[2]
    run(input_dir, output_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
