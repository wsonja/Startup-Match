import json
import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def normalize_text_list(items):
    return " ".join(str(x) for x in (items or []) if x)


def get_svd_doc(company):
    return " ".join([
        normalize_text_list(company.get("aggregated_skills", [])),
        normalize_text_list(company.get("inferred_roles", [])),
        normalize_text_list(company.get("tags", [])),
        normalize_text_list(company.get("categories", [])),
        company.get("sector", "") or "",
        company.get("subsector", "") or "",
        company.get("short_description", "") or "",
    ]).strip().lower()


def load_companies():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    json_file_path = os.path.join(current_directory, "enriched_init.json")
    with open(json_file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data["companies"]


def build_svd_term_space(companies):
    docs = [get_svd_doc(company) for company in companies]
    docs = [doc for doc in docs if doc.strip()]

    if len(docs) < 3:
        raise ValueError("Not enough non-empty docs to build SVD space.")

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        token_pattern=r"(?u)\b\w[\w.+#-]*\b",
    )
    X = vectorizer.fit_transform(docs)

    if X.shape[0] < 3 or X.shape[1] < 3:
        raise ValueError(f"Matrix too small for SVD: shape={X.shape}")

    n_components = min(20, X.shape[0] - 1, X.shape[1] - 1)
    if n_components < 2:
        raise ValueError(f"n_components too small: {n_components}")

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    svd.fit(X)

    feature_names = vectorizer.get_feature_names_out()
    term_vectors = svd.components_.T
    vocab = {term: idx for idx, term in enumerate(feature_names)}

    return {
        "vectorizer": vectorizer,
        "svd": svd,
        "feature_names": feature_names,
        "term_vectors": term_vectors,
        "vocab": vocab,
        "components": svd.components_,
        "n_components": n_components,
    }


def main():
    current_directory = os.path.dirname(os.path.abspath(__file__))
    out_path = os.path.join(current_directory, "svd_space.joblib")

    companies = load_companies()
    artifact = build_svd_term_space(companies)

    joblib.dump(artifact, out_path)
    print(f"Saved SVD artifact to {out_path}")


if __name__ == "__main__":
    main()