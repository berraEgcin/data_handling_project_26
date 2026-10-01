"""
2 evaluation: retrieval evaluation, classification evaluation:
"""

import json
import re
from pathlib import Path

from step3_retriever import ReferenceRangeRetriever
from step4_range_checker_rag import RangeCheckerRAG

DATA_DIR = Path(__file__).parent / "data"
ART_DIR = Path(__file__).parent / "artifacts"

RE_PARAM = re.compile(r"lab test '([^']+)'")
RE_SPECIMEN = re.compile(r"in Specimen '([^']+)'")
RE_GENDER = re.compile(r"for '([^']+)' and")
RE_AGE = re.compile(r"and '([^']+)'")
RE_CATEGORY = re.compile(r"in the category '([^']+)'")
RE_UNIT = re.compile(r"measuring in '([^']+)'")
RE_CONDITION = re.compile(r"with the condition '([^']+)'")
RE_REFTYPE = re.compile(r"reference type [\u2018']([^\u2019']+)[\u2019']")
VAL_PATTERN = re.compile(r"is (?:'([0-9.]+)'|([0-9.]+))\.")


def parse_query_fields(question: str):
    param = RE_PARAM.search(question).group(1)
    specimen = RE_SPECIMEN.search(question).group(1)
    gender = RE_GENDER.search(question).group(1)
    age = RE_AGE.search(question).group(1)
    cat = RE_CATEGORY.search(question)
    unit = RE_UNIT.search(question)
    condition = RE_CONDITION.search(question)
    reftype = RE_REFTYPE.search(question)
    return param, specimen, ("all" if gender == "any gender" else gender), \
        ("all" if age == "any age group" else age), (cat.group(1) if cat else None), \
        (unit.group(1) if unit else None), (condition.group(1) if condition else None), \
        (reftype.group(1) if reftype else None)


def eval_retrieval():
    retriever = ReferenceRangeRetriever()
    set1 = json.loads((DATA_DIR / "Set_1.json").read_text(encoding="utf-8"))

    exact_matches, range_matches, total = 0, 0, 0
    misses = []

    for r in set1:
        param, specimen, gender, age_group, category, unit, condition, reftype = parse_query_fields(r["Question"])
        results = retriever.retrieve(param, specimen, gender, age_group, category, unit,
                                      condition=condition, reference_type=reftype, top_k=1)
        total += 1
        if not results:
            misses.append((r["ID"], "no_result"))
            continue

        m = results[0].doc["metadata"]
        exact = results[0].method in ("exact", "exact_via_alias")
        if exact:
            exact_matches += 1
        else:
            misses.append((r["ID"], results[0].method))

    print("=" * 70)
    print("SET 1 -- RETRIEVAL EVALUATION")
    print("=" * 70)
    print(f"Total queries:         {total}")
    print(f"Exact structured hit:  {exact_matches} ({100*exact_matches/total:.1f}%)")
    print(f"Fell back to fuzzy/other: {total - exact_matches}")
    if misses:
        print("Sample non-exact retrievals:")
        for mid, reason in misses[:10]:
            print(f"  ID {mid}: {reason}")


def eval_classification():
    checker = RangeCheckerRAG()
    set2 = json.loads((DATA_DIR / "Set_2.json").read_text(encoding="utf-8"))
    corrected = {row["ID"]: row for row in json.loads((ART_DIR / "step1_set2_corrected_labels.json").read_text(encoding="utf-8"))}

    correct_vs_raw, correct_vs_fixed, total = 0, 0, 0

    for r in set2:
        gold = corrected.get(r["ID"])
        if gold is None:
            continue
        param, specimen, gender, age_group, category, unit, condition, reftype = parse_query_fields(r["Question"])
        result = checker.evaluate(param, gold["value"], specimen, gender, age_group, category, unit,
                                   condition=condition, reference_type=reftype)
        total += 1
        if result["status"] == gold["raw_label"]:
            correct_vs_raw += 1
        if result["status"] == gold["corrected_label"]:
            correct_vs_fixed += 1

    print("\n" + "=" * 70)
    print("SET 2 -- CLASSIFICATION EVALUATION (deterministic range_checker)")
    print("=" * 70)
    print(f"Total queries: {total}")
    print(f"Accuracy vs RAW  'Answer' field:       {correct_vs_raw}/{total} ({100*correct_vs_raw/total:.1f}%)")
    print(f"Accuracy vs CORRECTED gold label:      {correct_vs_fixed}/{total} ({100*correct_vs_fixed/total:.1f}%)")
    print("(The corrected label is the right one to trust -- see step1_build_registry.py.)")


if __name__ == "__main__":
    eval_retrieval()
    eval_classification()
