"""
Step1: LabQar is QA format so we should reformat it for RAG 
"""

import json
import re
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
OUT_DIR = Path(__file__).parent / "artifacts"
OUT_DIR.mkdir(exist_ok=True)

RE_PARAM = re.compile(r"lab test '([^']+)'")
RE_UNIT = re.compile(r"measuring in '([^']+)'")
RE_SPECIMEN = re.compile(r"in Specimen '([^']+)'")
RE_GENDER = re.compile(r"for '([^']+)' and")
RE_AGE = re.compile(r"and '([^']+)'")
RE_CATEGORY = re.compile(r"in the category '([^']+)'")
RE_CONDITION = re.compile(r"with the condition '([^']+)'")
# LabQAR renders this clause with *curly* quotes ('reference type '..'')
RE_REFTYPE = re.compile(r"reference type [\u2018']([^\u2019']+)[\u2019']")


def parse_question(q: str) -> dict:
    def grab(pattern):
        m = pattern.search(q)
        return m.group(1) if m else None
    return {
        "parameter": grab(RE_PARAM),
        "unit": grab(RE_UNIT),
        "specimen": grab(RE_SPECIMEN),
        "gender": grab(RE_GENDER),
        "age_group": grab(RE_AGE),
        "category": grab(RE_CATEGORY),
        "condition": grab(RE_CONDITION),
        "reference_type": grab(RE_REFTYPE),
    }


def range_type(lower, upper):
    if lower is not None and upper is not None:
        return "two_sided"
    if lower is not None and upper is None:
        return "lower_only" 
    if lower is None and upper is not None:
        return "upper_only"
    return "undefined"


def classify(value, lower, upper):
    if lower is not None and value < lower:
        return "Low"
    if upper is not None and value > upper:
        return "High"
    return "Normal"


def main():
    set1 = json.loads((DATA_DIR / "Set_1.json").read_text(encoding="utf-8"))
    set2 = {r["ID"]: r for r in json.loads((DATA_DIR / "Set_2.json").read_text(encoding="utf-8"))}

    registry_rows = []
    parse_failures = 0

    for r1 in set1:
        rid = r1["ID"]
        r2 = set2.get(rid)
        if r2 is None:
            continue

        meta = parse_question(r1["Question"])
        if any(meta[k] is None for k in ("parameter", "unit", "specimen", "gender", "age_group")):
            parse_failures += 1
            continue

        rr = r2["reference_range"]
        lower = rr.get("lower_bound")
        upper = rr.get("upper_bound")
        unit = rr.get("unit", meta["unit"])

        registry_rows.append({
            "ref_id": rid,
            "parameter": meta["parameter"],
            "category": meta["category"],
            "condition": meta["condition"],
            "reference_type": meta["reference_type"],
            "specimen": meta["specimen"],
            "gender": "all" if meta["gender"] == "any gender" else meta["gender"],
            "age_group": "all" if meta["age_group"] == "any age group" else meta["age_group"],
            "unit": unit,
            "lower_bound": lower,
            "upper_bound": upper,
            "range_type": range_type(lower, upper),
            "source": "LabQAR_Set1_Set2",
        })

    print(f"Registry rows built: {len(registry_rows)} (parse failures: {parse_failures})")

    with open(OUT_DIR / "step1_reference_registry.json", "w", encoding="utf-8") as f:
        json.dump(registry_rows, f, indent=2, ensure_ascii=False)

    import csv
    with open(OUT_DIR / "step1_reference_registry.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(registry_rows[0].keys()))
        writer.writeheader()
        writer.writerows(registry_rows)

    val_pattern = re.compile(r"is (?:'([0-9.]+)'|([0-9.]+))\.")
    corrected, mismatches = [], []

    for r2 in set2.values():
        m = val_pattern.search(r2["Question"])
        if not m:
            continue
        value = float(m.group(1) or m.group(2))
        rr = r2["reference_range"]
        true_label = classify(value, rr.get("lower_bound"), rr.get("upper_bound"))

        choices = {}
        for line in r2["Choices"].strip().split("\n"):
            letter, label = line.split(":", 1)
            choices[letter.strip()] = label.strip()
        raw_label = choices.get(r2["Answer"])

        row = {
            "ID": r2["ID"],
            "value": value,
            "lower_bound": rr.get("lower_bound"),
            "upper_bound": rr.get("upper_bound"),
            "raw_label": raw_label,
            "corrected_label": true_label,
            "was_mismatch": raw_label != true_label,
        }
        corrected.append(row)
        if row["was_mismatch"]:
            mismatches.append(row)

    with open(OUT_DIR / "step1_set2_corrected_labels.json", "w", encoding="utf-8") as f:
        json.dump(corrected, f, indent=2)

    print(f"\nSet_2 label audit: {len(mismatches)}/{len(corrected)} "
          f"({100*len(mismatches)/len(corrected):.1f}%) raw 'Answer' labels "
          f"disagree with reference_range-derived ground truth.")
    for row in mismatches[:5]:
        print(f"  ID {row['ID']}: value={row['value']} range=[{row['lower_bound']},{row['upper_bound']}] "
              f"raw='{row['raw_label']}' corrected='{row['corrected_label']}'")

    print(f"\nWrote:")
    print(f"  artifacts/step1_reference_registry.json/.csv  ({len(registry_rows)} rows)")
    print(f"  artifacts/step1_set2_corrected_labels.json     ({len(corrected)} rows)")


if __name__ == "__main__":
    main()
