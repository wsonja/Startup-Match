import json
import os
import re

import joblib
from dotenv import load_dotenv
from infosci_spark_client import LLMClient

load_dotenv()

BANNED_GEO_TERMS = {
    "india", "canada", "latin america", "latam", "america", "europe",
    "asia", "china", "mexico", "us", "uk", "africa", "brazil",
    "japan", "korea", "singapore"
}


def get_top_terms(svd_space, dim_idx, top_n=12):
    components = svd_space["components"]
    feature_names = svd_space["feature_names"]

    weights = components[dim_idx]
    top_indices = weights.argsort()[::-1][:top_n]

    terms = []
    for idx in top_indices:
        term = str(feature_names[idx]).strip()
        if term:
            terms.append(term)

    return terms


def extract_label(response):
    if isinstance(response, dict):
        label = response.get("content") or response.get("text") or ""
    else:
        label = str(response)

    label = label.strip().strip('"').strip("'")
    label = re.sub(r"[^A-Za-z0-9 /&+-]", "", label)
    label = re.sub(r"\s+", " ", label).strip()

    return label


def has_geo_term(label):
    lower = label.lower()
    return any(term in lower for term in BANNED_GEO_TERMS)


def fallback_label_from_terms(terms):
    filtered = []
    for term in terms:
        lower = term.lower()
        if any(geo in lower for geo in BANNED_GEO_TERMS):
            continue
        if term not in filtered:
            filtered.append(term)

    if not filtered:
        return "Startup Operations"

    return " / ".join(filtered[:2]).title()


def ask_llm_for_label(client, dim_idx, terms):
    messages = [
        {
            "role": "system",
            "content": (
                "You name latent SVD dimensions for a startup matching system.\n"
                "Return only one short label, no punctuation, no explanation.\n"
                "The label must be 2 to 4 words.\n"
                "Make labels specific and distinct.\n"
                "Do not use geographic labels or region labels.\n"
                "Never include India, Canada, Latin America, LatAm, Europe, US, UK, Asia, China, Mexico, or other locations.\n"
                "If location terms appear, ignore them and name the business, product, role, or technical theme instead.\n"
                "Avoid generic labels like Data, Technology, Software, Business, Analytics.\n"
                "Good examples: AI Developer Tools, Healthcare Operations, Frontend Web Apps, Fintech Infrastructure, Supply Chain Automation."
            ),
        },
        {
            "role": "user",
            "content": f"Dimension {dim_idx} top terms: {', '.join(terms)}",
        },
    ]

    response = client.chat(messages, stream=False)
    label = extract_label(response)

    if not label or has_geo_term(label):
        label = fallback_label_from_terms(terms)

    return label


def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    svd_path = os.path.join(current_dir, "svd_space.joblib")

    svd_space = joblib.load(svd_path)
    n_dims = svd_space["components"].shape[0]

    api_key = os.getenv("SPARK_API_KEY")
    if not api_key:
        raise RuntimeError("SPARK_API_KEY not found")

    client = LLMClient(api_key=api_key)

    labels = {}

    for dim_idx in range(n_dims):
        terms = get_top_terms(svd_space, dim_idx)
        label = ask_llm_for_label(client, dim_idx, terms)

        labels[dim_idx] = label
        print(f"{dim_idx}: {label} | {terms}")

    print("\n\nPaste this into routes.py:\n")
    print("DIMENSION_LABELS = ")
    print(json.dumps(labels, indent=4))


if __name__ == "__main__":
    main()