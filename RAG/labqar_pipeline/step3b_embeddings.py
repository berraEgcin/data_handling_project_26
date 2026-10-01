import json
import numpy as np
from pathlib import Path

ART_DIR = Path(__file__).parent / "artifacts"
DEFAULT_MODEL = "all-MiniLM-L6-v2"

class Embedder:
    def __init__(self, model_name: str = DEFAULT_MODEL):
        self.model_name = model_name
        self.backend = None
        self._model = None
        self._svd = None
        self._tfidf = None
        self.dim = None
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(model_name)
            self.backend = "sentence-transformers"
            self.dim = self._model.get_sentence_embedding_dimension()
        except ImportError:
            print("WARNING: sentence-transformers not installed")
            self.backend = "tfidf_svd_fallback"

    def fit_fallback(self, texts):
        if self.backend != "tfidf_svd_fallback":
            return
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.decomposition import TruncatedSVD
        self._tfidf = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
        X = self._tfidf.fit_transform(texts)
        n_components = min(128, X.shape[0] - 1, X.shape[1] - 1)
        self._svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._svd.fit(X)
        self.dim = n_components

    def encode(self, texts) -> np.ndarray:
        if isinstance(texts, str):
            texts = [texts]
        if self.backend == "sentence-transformers":
            return np.asarray(self._model.encode(texts, normalize_embeddings=True))
        else:
            if self._tfidf is None:
                raise RuntimeError("Call fit_fallback(corpus_texts)")
            X = self._tfidf.transform(texts)
            vecs = self._svd.transform(X)
            norms = np.linalg.norm(vecs, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return vecs / norms


def build_embeddings(corpus_path: Path = ART_DIR / "step2_corpus.jsonl",
                      out_path: Path = ART_DIR / "step3b_embeddings.npz",
                      model_name: str = DEFAULT_MODEL) -> Embedder:
    docs = [json.loads(line) for line in open(corpus_path)]
    texts = [d["text"] for d in docs]
    doc_ids = [d["doc_id"] for d in docs]

    embedder = Embedder(model_name)
    embedder.fit_fallback(texts)
    vectors = embedder.encode(texts)

    np.savez_compressed(out_path, doc_ids=np.array(doc_ids), vectors=vectors,
                         backend=embedder.backend, model_name=model_name)
    print(f"Encoded {len(texts)} documents -> {vectors.shape} ({embedder.backend}, "
          f"dim={embedder.dim})")
    print(f"Wrote {out_path}")
    return embedder


if __name__ == "__main__":
    build_embeddings()
