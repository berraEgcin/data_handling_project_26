import os
from step3d_hybrid_retriever import HybridRetriever

SYSTEM_PROMPT = """You are a lab-reference-range Q&A assistant. You are
given a user's question and a set of retrieved passages from LabQAR (a
curated reference-range dataset). Answer using ONLY the retrieved passages.

Rules:
- If the passages don't contain enough information to answer, say so
  plainly instead of guessing or using outside knowledge.
- Cite which retrieved parameter/context you're drawing from when relevant.
- Keep the answer to a few sentences.
- Do not give a diagnosis or treatment recommendation.
"""


def _format_context(hits) -> str:
    lines = []
    for i, h in enumerate(hits, 1):
        lines.append(f"[{i}] {h.doc['text']}")
    return "\n".join(lines)


class RagQA:
    def __init__(self, retriever: HybridRetriever = None, model: str = "gemini-3.8-flash",
                 api_key: str = None):
        self.retriever = retriever or HybridRetriever()
        self.model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        self._client = None
        if self.api_key:
            try:
                from google import genai
                self._client = genai.Client(api_key=self.api_key)
            except ImportError:
                print("WARNING: google-genai package not installed -- pip install google-genai")

    def answer(self, question: str, top_k: int = 3, metadata_filter: dict = None) -> dict:
        hits = self.retriever.retrieve_freetext(question, top_k=top_k, metadata_filter=metadata_filter)
        context = _format_context(hits)

        if self._client is None:
            answer_text = ("[No GOOGLE_API_KEY set -- showing retrieved context only, "
                            "no generation step run]\n" + context)
        else:
            user_msg = f"Question: {question}\n\nRetrieved passages:\n{context}\n\nAnswer the question."
            try:
                from google.genai import types
                resp = self._client.models.generate_content(
                    model=self.model,
                    contents=user_msg,
                    config=types.GenerateContentConfig(
                        system_instruction=SYSTEM_PROMPT,
                    ),
                )
                answer_text = resp.text.strip()
            except Exception as e:
                answer_text = f"[LLM call failed: {e}]\n" + context

        return {"question": question, "answer": answer_text, "retrieved": hits}


if __name__ == "__main__":
    qa = RagQA()
    questions = [
        "what's a healthy fasting glucose range for an adult?",
        "is a TSH of 6.2 mIU/L normal for an adult?",
        "what does it mean if my LDL cholesterol is elevated?",
    ]
    for q in questions:
        result = qa.answer(q, top_k=3)
        print(f"\nQ: {q}")
        print(f"A: {result['answer']}")
        print("Retrieved:")
        for h in result["retrieved"]:
            print(f"  [{h.score:.3f}] {h.doc['metadata']['parameter']}")
