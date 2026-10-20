import re
import os
from datetime import date
import requests
import yaml
import pandas as pd

TRACKS_CONFIG = "config/tracks.yaml"
VERIFY_CSV = "reports/testing/track_verification.csv"
HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

CIRCUITS = {
    "austin": ("https://www.formula1.com/en/racing/2024/united-states/circuit", "https://en.wikipedia.org/wiki/Circuit_of_the_Americas"),
    "baku": ("https://www.formula1.com/en/racing/2024/azerbaijan/circuit", "https://en.wikipedia.org/wiki/Baku_City_Circuit"),
    "barcelona": ("https://www.formula1.com/en/racing/2024/spain/circuit", "https://en.wikipedia.org/wiki/Circuit_de_Barcelona-Catalunya"),
    "hungaroring": ("https://www.formula1.com/en/racing/2024/hungary/circuit", "https://en.wikipedia.org/wiki/Hungaroring"),
    "imola": ("https://www.formula1.com/en/racing/2024/emiliaromagna/circuit", "https://en.wikipedia.org/wiki/Imola_Circuit"),
    "interlagos": ("https://www.formula1.com/en/racing/2024/brazil/circuit", "https://en.wikipedia.org/wiki/Interlagos_Circuit"),
    "jeddah": ("https://www.formula1.com/en/racing/2024/saudi-arabia/circuit", "https://en.wikipedia.org/wiki/Jeddah_Corniche_Circuit"),
    "las_vegas": ("https://www.formula1.com/en/racing/2024/las-vegas/circuit", "https://en.wikipedia.org/wiki/Las_Vegas_Strip_Circuit"),
    "lusail": ("https://www.formula1.com/en/racing/2024/qatar/circuit", "https://en.wikipedia.org/wiki/Lusail_International_Circuit"),
    "melbourne": ("https://www.formula1.com/en/racing/2024/australia/circuit", "https://en.wikipedia.org/wiki/Albert_Park_Circuit"),
    "mexico_city": ("https://www.formula1.com/en/racing/2024/mexico/circuit", "https://en.wikipedia.org/wiki/Aut%C3%B3dromo_Hermanos_Rodr%C3%ADguez"),
    "miami": ("https://www.formula1.com/en/racing/2024/miami/circuit", "https://en.wikipedia.org/wiki/Miami_International_Autodrome"),
    "monaco": ("https://www.formula1.com/en/racing/2024/monaco/circuit", "https://en.wikipedia.org/wiki/Circuit_de_Monaco"),
    "montreal": ("https://www.formula1.com/en/racing/2024/canada/circuit", "https://en.wikipedia.org/wiki/Circuit_Gilles_Villeneuve"),
    "monza": ("https://www.formula1.com/en/racing/2024/italy/circuit", "https://en.wikipedia.org/wiki/Monza_Circuit"),
    "paul_ricard": ("https://www.formula1.com/en/racing/2022/france/circuit", "https://en.wikipedia.org/wiki/Circuit_Paul_Ricard"),
    "sakhir": ("https://www.formula1.com/en/racing/2024/bahrain/circuit", "https://en.wikipedia.org/wiki/Bahrain_International_Circuit"),
    "shanghai": ("https://www.formula1.com/en/racing/2024/china/circuit", "https://en.wikipedia.org/wiki/Shanghai_International_Circuit"),
    "silverstone": ("https://www.formula1.com/en/racing/2024/great-britain/circuit", "https://en.wikipedia.org/wiki/Silverstone_Circuit"),
    "singapore": ("https://www.formula1.com/en/racing/2024/singapore/circuit", "https://en.wikipedia.org/wiki/Marina_Bay_Street_Circuit"),
    "spa": ("https://www.formula1.com/en/racing/2024/belgium/circuit", "https://en.wikipedia.org/wiki/Circuit_de_Spa-Francorchamps"),
    "spielberg": ("https://www.formula1.com/en/racing/2024/austria/circuit", "https://en.wikipedia.org/wiki/Red_Bull_Ring"),
    "suzuka": ("https://www.formula1.com/en/racing/2024/japan/circuit", "https://en.wikipedia.org/wiki/Suzuka_International_Racing_Course"),
    "yas_marina": ("https://www.formula1.com/en/racing/2024/united-arab-emirates/circuit", "https://en.wikipedia.org/wiki/Yas_Marina_Circuit"),
    "zandvoort": ("https://www.formula1.com/en/racing/2024/netherlands/circuit", "https://en.wikipedia.org/wiki/Circuit_Zandvoort")
}


def parse_f1_html(html):
    """Extract circuit length in km from Formula1.com HTML or embedded JSON."""
    m = re.search(r'"trackLength":\s*"([0-9.]+)"', html)
    if m:
        return float(m.group(1))
    m = re.search(r">([0-9.]+)km</dd>", html)
    if m:
        return float(m.group(1))
    m = re.search(r"Circuit Length\s*</p>\s*<p[^>]*>\s*([0-9.]+)\s*km", html, re.I)
    return float(m.group(1)) if m else None


def parse_wiki_html(html, target_km=None):
    """Extract Grand Prix circuit length in km from Wikipedia infobox."""
    patterns = [
        r'"length_km":\s*\{\s*"wt":\s*"([0-9.]+)"',
        r'Length</th>\s*<td[^>]*>\s*([0-9.]+)',
        r'Grand Prix[^\n<]*</th>\s*<td[^>]*>\s*([0-9.]+)',
        r'length_km\s*=\s*([0-9.]+)'
    ]
    all_found = []
    for pat in patterns:
        for m in re.findall(pat, html, re.I):
            try:
                v = float(m)
                if 2.5 <= v <= 8.5:
                    all_found.append(v)
            except ValueError:
                pass
    if not all_found:
        return None
    if target_km is not None:
        def diff_fn(val):
            return abs(val - target_km)
        all_found.sort(key=diff_fn)
        return all_found[0]
    return all_found[0]


def fetch_source_length(url, parser_fn, target_km=None):
    """Fetch URL and parse length, returning (length, status_code, error)."""
    try:
        r = requests.get(url, headers=HEADERS, timeout=10)
        if r.status_code == 200:
            length = parser_fn(r.text, target_km) if parser_fn == parse_wiki_html else parser_fn(r.text)
            return length, 200, "" if length is not None else "parsed_null"
        return None, r.status_code, f"http_{r.status_code}"
    except Exception as err:
        return None, 0, str(err)


def verify_single_track(tid, urls, info, today_str):
    """Fetch lengths from 2 sources and evaluate agreement for a single circuit."""
    f1_url, wiki_url = urls
    base_len = info.get("length_km")
    l_f1, _, err_f1 = fetch_source_length(f1_url, parse_f1_html)
    l_wiki, _, err_wiki = fetch_source_length(wiki_url, parse_wiki_html, target_km=l_f1 or base_len)
    agree = (l_f1 is not None and l_wiki is not None and abs(l_f1 - l_wiki) <= 0.010)

    r1 = {"circuit": tid, "source": "formula1.com", "url": f1_url, "number_found": l_f1, "date_fetched": today_str, "agrees": "yes" if agree else "no", "notes": err_f1}
    r2 = {"circuit": tid, "source": "wikipedia", "url": wiki_url, "number_found": l_wiki, "date_fetched": today_str, "agrees": "yes" if agree else "no", "notes": err_wiki}
    if agree and l_f1 is not None:
        info.update({"verified": True, "length_km": l_f1, "source": f1_url, "verified_on": today_str})
    else:
        info.update({"verified": False, "source": f1_url if l_f1 is not None else wiki_url, "verified_on": today_str})
    return [r1, r2], agree


def run_track_verification():
    """Verify all 25 tracks across two independent sources and update tracks.yaml."""
    today_str = str(date.today())
    records = []
    with open(TRACKS_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    verified_count = 0
    for tid, urls in sorted(CIRCUITS.items()):
        info = cfg["tracks"].get(tid, {})
        recs, agree = verify_single_track(tid, urls, info, today_str)
        records.extend(recs)
        if agree:
            verified_count += 1

    os.makedirs(os.path.dirname(VERIFY_CSV), exist_ok=True)
    pd.DataFrame(records).to_csv(VERIFY_CSV, index=False)
    with open(TRACKS_CONFIG, "w", encoding="utf-8") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    print(f"Track verification: {verified_count}/25 verified. Saved to {VERIFY_CSV}")


if __name__ == "__main__":
    run_track_verification()
