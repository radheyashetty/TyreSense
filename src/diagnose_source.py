import os
import logging
import requests
import fastf1
from fastf1.exceptions import DataNotLoadedError


class WarningCaptureHandler(logging.Handler):
    """Logging handler to capture warning messages from FastF1."""
    def __init__(self):
        super().__init__()
        self.warnings = []

    def emit(self, record):
        if record.levelno >= logging.WARNING:
            self.warnings.append(self.format(record))


def probe_rcm_http(api_path):
    """Directly probe the Formula 1 live timing static endpoint for RCM stream."""
    path = api_path if not api_path.startswith("/static/") else api_path[len("/static/"):]
    url = f"https://livetiming.formula1.com/static/{path}RaceControlMessages.jsonStream"
    try:
        resp = requests.get(url, timeout=10)
        return resp.status_code, len(resp.content), url
    except Exception as err:
        return -1, 0, f"{url} (error: {err})"


def count_session_rows(session):
    """Safely count laps, weather, and race control messages rows."""
    try:
        laps_cnt = len(session.laps) if session.laps is not None else 0
    except (DataNotLoadedError, Exception):
        laps_cnt = 0
    try:
        weather_df = session.laps.get_weather_data() if session.laps is not None else None
        weather_cnt = len(weather_df) if weather_df is not None else 0
    except (DataNotLoadedError, Exception):
        weather_cnt = 0
    try:
        rcm_cnt = len(session.race_control_messages) if session.race_control_messages is not None else 0
    except (DataNotLoadedError, Exception):
        rcm_cnt = -1
    return laps_cnt, weather_cnt, rcm_cnt


def run_session_load(year, round_num, with_messages, warn_handler):
    """Load a session with or without messages and record warnings and row counts."""
    warn_handler.warnings.clear()
    session = fastf1.get_session(year, round_num, "R")
    try:
        session.load(telemetry=False, weather=True, messages=with_messages)
    except Exception as err:
        warn_handler.warnings.append(f"Load exception: {err}")
    laps_cnt, weather_cnt, rcm_cnt = count_session_rows(session)
    warnings = list(warn_handler.warnings)
    return session, laps_cnt, weather_cnt, rcm_cnt, warnings


def diagnose_target(year, round_num, warn_handler):
    """Diagnose a target race session across both load configurations and HTTP probe."""
    print(f"\n--- Diagnosing {year} Round {round_num} ---")
    lines = [f"\n=== {year} Round {round_num} ==="]
    
    # Load with messages=False
    _, laps_no_m, wea_no_m, rcm_no_m, warns_no_m = run_session_load(year, round_num, False, warn_handler)
    lines.append(f"Load [messages=False]: laps={laps_no_m}, weather={wea_no_m}, rcm={rcm_no_m}")
    lines.append(f"  Warnings ({len(warns_no_m)}): {'; '.join(warns_no_m) if warns_no_m else 'None'}")
    
    # Load with messages=True
    session, laps_m, wea_m, rcm_m, warns_m = run_session_load(year, round_num, True, warn_handler)
    lines.append(f"Load [messages=True]: laps={laps_m}, weather={wea_m}, rcm={rcm_m}")
    lines.append(f"  Warnings ({len(warns_m)}): {'; '.join(warns_m) if warns_m else 'None'}")
    
    # HTTP probe
    status_code, byte_len, url = probe_rcm_http(session.api_path)
    lines.append(f"HTTP Probe: status={status_code}, bytes={byte_len}, url={url}")
    return lines


def main():
    """Run diagnosis across target sessions and output reports/diagnose_source.txt."""
    fastf1.Cache.enable_cache("fastf1_cache")
    warn_handler = WarningCaptureHandler()
    logging.getLogger("fastf1").addHandler(warn_handler)
    
    targets = [(2022, 1), (2023, 1), (2023, 10), (2024, 1), (2025, 1)]
    output_lines = ["DIAGNOSIS REPORT FOR F1 RACE CONTROL MESSAGES & LAPS"]
    for y, r in targets:
        lines = diagnose_target(y, r, warn_handler)
        for l in lines:
            print(l)
        output_lines.extend(lines)
    
    os.makedirs("reports", exist_ok=True)
    report_path = os.path.join("reports", "diagnose_source.txt")
    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(output_lines) + "\n")
    print(f"\nSaved diagnosis output to {report_path}")


if __name__ == "__main__":
    main()
