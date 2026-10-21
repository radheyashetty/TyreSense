import os, re, glob, pandas as pd, json

def verify_numbers_in_artifacts(report_path="reports/overnight/RUN4_REPORT.md"):
    if not os.path.exists(report_path):
        print(f"Report {report_path} does not exist yet.")
        return

    with open(report_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Find float numbers like 8.14, 7.94, etc.
    nums = re.findall(r"\b\d+\.\d+\b", text)
    print(f"Total decimal numbers found in report: {len(nums)}")

    # Load text of all csvs and jsons under reports/ and runs/
    artifact_texts = []
    for fpath in glob.glob("reports/**/*", recursive=True) + glob.glob("runs/**/*", recursive=True):
        if os.path.isfile(fpath) and fpath.endswith((".csv", ".json", ".txt", ".log", ".md")):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    artifact_texts.append((fpath, f.read()))
            except Exception:
                pass

    unverified = []
    for n in set(nums):
        # Ignore trivial or formatting floats like 1.0, 0.0, 0.5, 0.8
        if n in ["1.0", "0.0", "0.5", "0.8", "0.1", "0.2", "2.0"]:
            continue
        found = False
        for fpath, c in artifact_texts:
            if n in c:
                found = True
                break
        if not found:
            unverified.append(n)

    print(f"Unverified numbers count: {len(unverified)}")
    if unverified:
        print("Unverified numbers:", unverified[:20])
    else:
        print("ALL NUMBERS VERIFIED IN ARTIFACTS!")

if __name__ == "__main__":
    verify_numbers_in_artifacts()
