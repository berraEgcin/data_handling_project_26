import json
import re
from pathlib import Path

from step3_retriever import ReferenceRangeRetriever
from step3d_hybrid_retriever import HybridRetriever
from step4_range_checker_rag import RangeCheckerRAG
from step6_llm_explain import Explainer

ART_DIR = Path(__file__).parent / "artifacts"

# A. Retrieval evaluation
def _lexical_baseline_search(structured_retriever: ReferenceRangeRetriever, query: str, top_k: int):
    matches = structured_retriever._fuzzy_param_match(query, top_k=top_k)
    return [name for name, _score in matches]


def evaluate_retrieval(top_k: int = 3):
    template_queries = json.loads((ART_DIR / "step5b_eval_queries.json").read_text())
    hard_queries = json.loads((ART_DIR / "step5b_hard_queries.json").read_text())

    structured = ReferenceRangeRetriever()
    hybrid = HybridRetriever(structured=structured)

    def run(queries, label):
        semantic_hits_at_1, semantic_hits_at_k, semantic_rr = 0, 0, 0.0
        lexical_hits_at_1, lexical_hits_at_k, lexical_rr = 0, 0, 0.0
        for q in queries:
            gold = q["gold_parameter_norm"]

            results = hybrid.retrieve_freetext(q["query"], top_k=top_k)
            ranked = [r.doc["metadata"]["parameter_norm"] for r in results]
            if ranked and ranked[0] == gold:
                semantic_hits_at_1 += 1
            if gold in ranked:
                semantic_hits_at_k += 1
                semantic_rr += 1.0 / (ranked.index(gold) + 1)

            lex_ranked = _lexical_baseline_search(structured, q["query"], top_k)
            if lex_ranked and lex_ranked[0] == gold:
                lexical_hits_at_1 += 1
            if gold in lex_ranked:
                lexical_hits_at_k += 1
                lexical_rr += 1.0 / (lex_ranked.index(gold) + 1)

        n = len(queries)
        print(f"\n--- {label} (n={n}) ---")
        print(f"{'method':<12} {'Acc@1':>8} {'Acc@' + str(top_k):>8} {'MRR':>8}")
        print(f"{'semantic':<12} {100*semantic_hits_at_1/n:>7.1f}% {100*semantic_hits_at_k/n:>7.1f}% {semantic_rr/n:>8.3f}")
        print(f"{'lexical':<12} {100*lexical_hits_at_1/n:>7.1f}% {100*lexical_hits_at_k/n:>7.1f}% {lexical_rr/n:>8.3f}")
        return {
            "semantic": {"acc@1": semantic_hits_at_1/n, f"acc@{top_k}": semantic_hits_at_k/n, "mrr": semantic_rr/n},
            "lexical": {"acc@1": lexical_hits_at_1/n, f"acc@{top_k}": lexical_hits_at_k/n, "mrr": lexical_rr/n},
        }

    print("=" * 70)
    print("RETRIEVAL EVALUATION")
    print("=" * 70)
    template_scores = run(template_queries, "sanity check")
    hard_scores = run(hard_queries, "Hard queries")

    if hybrid.embedder.backend == "tfidf_svd_fallback":
    return {"template": template_scores, "hard": hard_scores}

# B. Generation evaluation
NUM_RE = re.compile(r"-?\d+\.?\d*")
STATUS_WORDS = {
    "High": ["high", "elevated", "above"],
    "Low": ["low", "below", "deficient"],
    "Normal": ["normal", "within", "expected range"],
}


def _verdict_consistency(explanation: str, status: str) -> bool:
    text = explanation.lower()
    other_statuses = [s for s in STATUS_WORDS if s != status]
    for other in other_statuses:
        for kw in STATUS_WORDS[other]:
            if kw in text:
                return False
    return True


def _numeric_groundedness(explanation: str, result: dict, value: float, tolerance=0.05):
    known = [v for v in (value, result.get("lower_bound"), result.get("upper_bound")) if v is not None]
    found = [float(x) for x in NUM_RE.findall(explanation)]
    unmatched = []
    for f in found:
        if not any(abs(f - k) <= max(tolerance * abs(k), tolerance) for k in known):
            unmatched.append(f)
    return (len(unmatched) == 0), unmatched


def evaluate_generation(n_cases: int = 20, seed: int = 42):
    import random
    rng = random.Random(seed)

    checker = RangeCheckerRAG()
    explainer = Explainer()

    registry = json.loads((ART_DIR / "step1_reference_registry.json").read_text(encoding="utf-8"))
    two_sided = [r for r in registry if r["lower_bound"] is not None and r["upper_bound"] is not None]
    sample = rng.sample(two_sided, min(n_cases, len(two_sided)))

    consistent, grounded, results_log = 0, 0, []
    for row in sample:
        mid = (row["lower_bound"] + row["upper_bound"]) / 2
        value = row["upper_bound"] * 1.3 if rng.random() < 0.5 else mid

        result = checker.evaluate(row["parameter"], value, specimen=row["specimen"],
                                   gender=row["gender"], age_group=row["age_group"])
        explanation = explainer.explain(row["parameter"], value, row["unit"], result)

        ok_verdict = _verdict_consistency(explanation, result["status"])
        ok_numbers, unmatched = _numeric_groundedness(explanation, result, value)
        consistent += ok_verdict
        grounded += ok_numbers
        results_log.append({
            "parameter": row["parameter"], "value": value, "status": result["status"],
            "explanation": explanation, "verdict_consistent": ok_verdict,
            "numerically_grounded": ok_numbers, "unmatched_numbers": unmatched,
        })

    n = len(sample)
    print("\n" + "=" * 70)
    print(f"GENERATION EVALUATION -- Step 6 explanations (n={n}, LLM backend="
          f"{'live API' if explainer._client else 'template fallback)'})")
    print("=" * 70)
    print(f"Verdict-consistent:   {consistent}/{n} ({100*consistent/n:.1f}%)")
    print(f"Numerically grounded: {grounded}/{n} ({100*grounded/n:.1f}%)")
    failures = [r for r in results_log if not (r["verdict_consistent"] and r["numerically_grounded"])]
    if failures:
        print("\nFailing cases:")
        for f in failures[:5]:
            print(f"  {f['parameter']} = {f['value']:.2f} -> {f['status']}: {f['explanation']!r}")
            if f["unmatched_numbers"]:
                print(f"    unmatched numbers: {f['unmatched_numbers']}")
    return results_log


if __name__ == "__main__":
    evaluate_retrieval(top_k=3)
    evaluate_generation(n_cases=20)
