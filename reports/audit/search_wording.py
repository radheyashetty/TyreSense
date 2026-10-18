import re

words = ['Bayesian', 'Kalman', 'temperature scaling', 'guarantee', 'proves', 'causal', 'measured wear']
files_to_search = ['reports/final_report.md', 'reports/project_overview.md', 'README.md', 'app.py']

for path in files_to_search:
    print(f"=== Searching in {path} ===")
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    for w in words:
        matches = [m.start() for m in re.finditer(re.escape(w), text, re.IGNORECASE)]
        if matches:
            for idx in matches:
                start = max(0, idx - 40)
                end = min(len(text), idx + len(w) + 40)
                snippet = text[start:end].replace("\n", " ")
                print(f"  [{w}] at {idx}: \"...{snippet}...\"")
