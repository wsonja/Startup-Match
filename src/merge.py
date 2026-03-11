#!/usr/bin/env python3

import ast
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parent
YC_PATH = ROOT / "2023-07-13-yc-companies.csv"
STARTUPS_PATH = ROOT / "Startups.csv"
AI_PATH = ROOT / "ai_startup_funding_database_2014_2025.csv"

OUT_JSON = ROOT / "merged_companies.json"
OUT_CSV = ROOT / "merged_companies.csv"


def is_missing(x):
    if x is None:
        return True
    if isinstance(x, float) and math.isnan(x):
        return True
    if pd.isna(x):
        return True
    s = str(x).strip()
    return s == "" or s.lower() == "nan"


def clean_str(x):
    if is_missing(x):
        return None
    return str(x).strip()


def parse_list_like(x):
    if is_missing(x):
        return []
    if isinstance(x, list):
        return [str(i).strip() for i in x if str(i).strip()]

    s = str(x).strip()
    if not s:
        return []

    # Try Python-list style first, like "['AI', 'SaaS']"
    if s.startswith("[") and s.endswith("]"):
        try:
            parsed = ast.literal_eval(s)
            if isinstance(parsed, list):
                return [str(i).strip() for i in parsed if str(i).strip()]
        except Exception:
            pass

    # Fallback: comma-separated
    return [part.strip() for part in s.split(",") if part.strip()]


def normalize_name(name):
    if is_missing(name):
        return None
    s = str(name).lower().strip()
    s = s.replace("&", " and ")
    s = re.sub(r"[^a-z0-9]+", " ", s)
    s = re.sub(r"\b(inc|llc|corp|corporation|company|co)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s or None


def normalize_domain(url):
    if is_missing(url):
        return None
    s = str(url).strip().lower()
    s = re.sub(r"^https?://", "", s)
    s = re.sub(r"^www\d*\.", "", s)
    s = s.split("/")[0]
    s = s.strip()
    return s or None


def first_nonempty(*vals):
    for v in vals:
        if not is_missing(v):
            return v
    return None


def merge_lists(*lists_):
    seen = set()
    out = []
    for lst in lists_:
        if not lst:
            continue
        for item in lst:
            val = str(item).strip()
            if val and val.lower() not in seen:
                seen.add(val.lower())
                out.append(val)
    return out


def parse_mapping_location(s):
    if is_missing(s):
        return None, None, None
    parts = [p.strip() for p in str(s).split("-")]
    if len(parts) >= 3:
        city = parts[0]
        state = parts[1]
        country = parts[2]
        return city, state, country
    return None, None, None


def build_retrieval_document(company):
    pieces = [
        company.get("canonical_name"),
        company.get("short_description"),
        company.get("long_description"),
        " ".join(company.get("tags", [])),
        " ".join(company.get("categories", [])),
        company.get("sector"),
        company.get("subsector"),
        company.get("location"),
        company.get("city"),
        company.get("state"),
        company.get("country"),
        " ".join(company.get("aggregated_skills", [])),
    ]
    return " ".join([p for p in pieces if p]).strip()


def yc_row_to_company(row):
    tags = parse_list_like(row["tags"])
    founders = parse_list_like(row["founders_names"])

    return {
        "canonical_name": clean_str(row["company_name"]),
        "aliases": [],
        "normalized_name": normalize_name(row["company_name"]),
        "website": clean_str(row["website"]),
        "domain": normalize_domain(row["website"]),
        "short_description": clean_str(row["short_description"]),
        "long_description": clean_str(row["long_description"]),
        "tags": tags,
        "categories": [],
        "sector": None,
        "subsector": None,
        "is_yc": True,
        "company_id_yc": None if is_missing(row["company_id"]) else int(row["company_id"]),
        "yc_batch": clean_str(row["batch"]),
        "yc_year": None,
        "yc_session": None,
        "status": clean_str(row["status"]),
        "year_founded": None if is_missing(row["year_founded"]) else int(float(row["year_founded"])),
        "location": clean_str(row["location"]),
        "city": None,
        "state": None,
        "country": clean_str(row["country"]),
        "team_size": None if is_missing(row["team_size"]) else int(float(row["team_size"])),
        "num_founders": None if is_missing(row["num_founders"]) else int(row["num_founders"]),
        "founders_names": founders,
        "cb_url": clean_str(row["cb_url"]),
        "linkedin_url": clean_str(row["linkedin_url"]),
        "funding_summary": {},
        "aggregated_skills": [],
        "source_datasets": ["yc_2023"],
    }


def startups_row_to_company(row):
    city = clean_str(row["Headquarters (City)"])
    state = clean_str(row["Headquarters (US State)"])
    country = clean_str(row["Headquarters (Country)"])

    if not city and not state and not country:
        city2, state2, country2 = parse_mapping_location(row["Mapping Location"])
        city = city or city2
        state = state or state2
        country = country or country2

    website = clean_str(row["Website"])
    categories = parse_list_like(row["Categories"])
    founders = parse_list_like(row["Founders"])

    yc_year = None if is_missing(row["Y Combinator Year"]) else int(row["Y Combinator Year"])
    yc_session = clean_str(row["Y Combinator Session"])
    yc_batch = None
    if yc_year and yc_session:
        yc_batch = f"{yc_session[0].upper()}{str(yc_year)[-2:]}"

    return {
        "canonical_name": clean_str(row["Company"]),
        "aliases": [],
        "normalized_name": normalize_name(row["Company"]),
        "website": website,
        "domain": normalize_domain(website),
        "short_description": None,
        "long_description": clean_str(row["Description"]),
        "tags": [],
        "categories": categories,
        "sector": None,
        "subsector": None,
        "is_yc": True,
        "company_id_yc": None,
        "yc_batch": yc_batch,
        "yc_year": yc_year,
        "yc_session": yc_session,
        "status": clean_str(row["Satus"]),
        "year_founded": None if is_missing(row["Year Founded"]) else int(float(row["Year Founded"])),
        "location": clean_str(row["Mapping Location"]),
        "city": city,
        "state": state,
        "country": country,
        "team_size": None,
        "num_founders": None,
        "founders_names": founders,
        "cb_url": clean_str(row["Crunchbase / Angel List Profile"]),
        "linkedin_url": None,
        "funding_summary": {
            "investors": parse_list_like(row["Investors"]),
            "amounts_raised_raw": clean_str(row["Amounts raised in different funding rounds"]),
        },
        "aggregated_skills": [],
        "source_datasets": ["startups_2005_2014"],
    }


def ai_group_to_company(company_name, group):
    latest = group.sort_values("deal_date").iloc[-1]
    total_funding = group["amount_usd_millions"].fillna(0).sum()

    return {
        "canonical_name": clean_str(company_name),
        "aliases": [],
        "normalized_name": normalize_name(company_name),
        "website": None,
        "domain": None,
        "short_description": None,
        "long_description": clean_str(latest["key_products"]),
        "tags": [],
        "categories": [],
        "sector": clean_str(latest["sector"]),
        "subsector": clean_str(latest["subsector"]),
        "is_yc": False,
        "company_id_yc": None,
        "yc_batch": None,
        "yc_year": None,
        "yc_session": None,
        "status": None,
        "year_founded": None if is_missing(latest["founded_year"]) else int(latest["founded_year"]),
        "location": None,
        "city": clean_str(latest["hq_city"]),
        "state": None,
        "country": clean_str(latest["hq_country"]),
        "team_size": None if is_missing(latest["employees_approx"]) else int(latest["employees_approx"]),
        "num_founders": None,
        "founders_names": [],
        "cb_url": None,
        "linkedin_url": None,
        "funding_summary": {
            "deal_count": int(len(group)),
            "latest_round_type": clean_str(latest["round_type"]),
            "latest_amount_usd_millions": None if is_missing(latest["amount_usd_millions"]) else float(latest["amount_usd_millions"]),
            "latest_post_money_valuation_millions": None if is_missing(latest["post_money_valuation_millions"]) else float(latest["post_money_valuation_millions"]),
            "total_funding_usd_millions": float(total_funding),
            "lead_investors": merge_lists(*[parse_list_like(x) for x in group["lead_investors"].tolist()]),
            "other_investors": merge_lists(*[parse_list_like(x) for x in group["other_investors"].tolist()]),
            "ipo_status": clean_str(latest["ipo_status"]),
            "ceo": clean_str(latest["ceo"]),
            "annual_revenue_millions": None if is_missing(latest["annual_revenue_millions"]) else float(latest["annual_revenue_millions"]),
            "profitable": None if is_missing(latest["profitable"]) else bool(latest["profitable"]),
            "open_source": None if is_missing(latest["open_source"]) else bool(latest["open_source"]),
        },
        "aggregated_skills": [],
        "source_datasets": ["ai_funding_2014_2025"],
    }


def merge_company_records(base, incoming):
    if incoming["canonical_name"] and incoming["canonical_name"] != base["canonical_name"]:
        base["aliases"] = merge_lists(base.get("aliases", []), [incoming["canonical_name"]])

    base["website"] = first_nonempty(base.get("website"), incoming.get("website"))
    base["domain"] = first_nonempty(base.get("domain"), incoming.get("domain"))
    base["short_description"] = first_nonempty(base.get("short_description"), incoming.get("short_description"))
    base["long_description"] = first_nonempty(base.get("long_description"), incoming.get("long_description"))

    base["tags"] = merge_lists(base.get("tags", []), incoming.get("tags", []))
    base["categories"] = merge_lists(base.get("categories", []), incoming.get("categories", []))

    base["sector"] = first_nonempty(base.get("sector"), incoming.get("sector"))
    base["subsector"] = first_nonempty(base.get("subsector"), incoming.get("subsector"))

    base["is_yc"] = bool(base.get("is_yc") or incoming.get("is_yc"))
    base["company_id_yc"] = first_nonempty(base.get("company_id_yc"), incoming.get("company_id_yc"))
    base["yc_batch"] = first_nonempty(base.get("yc_batch"), incoming.get("yc_batch"))
    base["yc_year"] = first_nonempty(base.get("yc_year"), incoming.get("yc_year"))
    base["yc_session"] = first_nonempty(base.get("yc_session"), incoming.get("yc_session"))

    base["status"] = first_nonempty(base.get("status"), incoming.get("status"))
    base["year_founded"] = first_nonempty(base.get("year_founded"), incoming.get("year_founded"))

    base["location"] = first_nonempty(base.get("location"), incoming.get("location"))
    base["city"] = first_nonempty(base.get("city"), incoming.get("city"))
    base["state"] = first_nonempty(base.get("state"), incoming.get("state"))
    base["country"] = first_nonempty(base.get("country"), incoming.get("country"))

    base["team_size"] = first_nonempty(base.get("team_size"), incoming.get("team_size"))
    base["num_founders"] = first_nonempty(base.get("num_founders"), incoming.get("num_founders"))
    base["founders_names"] = merge_lists(base.get("founders_names", []), incoming.get("founders_names", []))

    base["cb_url"] = first_nonempty(base.get("cb_url"), incoming.get("cb_url"))
    base["linkedin_url"] = first_nonempty(base.get("linkedin_url"), incoming.get("linkedin_url"))

    base["source_datasets"] = merge_lists(base.get("source_datasets", []), incoming.get("source_datasets", []))

    # Merge funding summary shallowly
    base_funding = base.get("funding_summary", {}) or {}
    incoming_funding = incoming.get("funding_summary", {}) or {}
    for k, v in incoming_funding.items():
        if k not in base_funding or is_missing(base_funding[k]) or base_funding[k] in ({}, [], None):
            base_funding[k] = v
        elif isinstance(base_funding[k], list) and isinstance(v, list):
            base_funding[k] = merge_lists(base_funding[k], v)
    base["funding_summary"] = base_funding

    base["normalized_name"] = first_nonempty(base.get("normalized_name"), incoming.get("normalized_name"))
    return base


def make_key(record):
    return record.get("domain") or record.get("normalized_name")


def main():
    yc = pd.read_csv(YC_PATH)
    startups = pd.read_csv(STARTUPS_PATH)
    ai = pd.read_csv(AI_PATH)

    merged = {}
    matched_startups = 0
    matched_ai = 0

    # 1) Start with YC 2023 as backbone
    for _, row in yc.iterrows():
        company = yc_row_to_company(row)
        key = make_key(company)
        if key:
            merged[key] = company

    # 2) Merge old YC startups
    for _, row in startups.iterrows():
        incoming = startups_row_to_company(row)

        candidate_keys = [incoming.get("domain"), incoming.get("normalized_name")]
        matched = None

        for key in candidate_keys:
            if key and key in merged:
                matched = key
                break

        if matched:
            merged[matched] = merge_company_records(merged[matched], incoming)
            matched_startups += 1
        else:
            key = make_key(incoming)
            if key:
                merged[key] = incoming

    # 3) Merge AI funding grouped by company
    for company_name, group in ai.groupby("company"):
        incoming = ai_group_to_company(company_name, group)

        candidate_keys = [incoming.get("domain"), incoming.get("normalized_name")]
        matched = None

        for key in candidate_keys:
            if key and key in merged:
                matched = key
                break

        if matched:
            merged[matched] = merge_company_records(merged[matched], incoming)
            matched_ai += 1
        else:
            key = make_key(incoming)
            if key:
                merged[key] = incoming

    # 4) Final cleanup
    companies = []
    for company in merged.values():
        company["retrieval_document"] = build_retrieval_document(company)
        companies.append(company)

    companies.sort(key=lambda x: (x.get("canonical_name") or "").lower())

    # JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"companies": companies}, f, indent=2, ensure_ascii=False)

    # Flat CSV for debugging
    flat_rows = []
    for c in companies:
        flat_rows.append({
            "canonical_name": c.get("canonical_name"),
            "website": c.get("website"),
            "domain": c.get("domain"),
            "short_description": c.get("short_description"),
            "long_description": c.get("long_description"),
            "tags": "|".join(c.get("tags", [])),
            "categories": "|".join(c.get("categories", [])),
            "sector": c.get("sector"),
            "subsector": c.get("subsector"),
            "is_yc": c.get("is_yc"),
            "company_id_yc": c.get("company_id_yc"),
            "yc_batch": c.get("yc_batch"),
            "yc_year": c.get("yc_year"),
            "yc_session": c.get("yc_session"),
            "status": c.get("status"),
            "year_founded": c.get("year_founded"),
            "location": c.get("location"),
            "city": c.get("city"),
            "state": c.get("state"),
            "country": c.get("country"),
            "team_size": c.get("team_size"),
            "num_founders": c.get("num_founders"),
            "founders_names": "|".join(c.get("founders_names", [])),
            "cb_url": c.get("cb_url"),
            "linkedin_url": c.get("linkedin_url"),
            "source_datasets": "|".join(c.get("source_datasets", [])),
            "aggregated_skills": "|".join(c.get("aggregated_skills", [])),
            "retrieval_document": c.get("retrieval_document"),
            "funding_summary_json": json.dumps(c.get("funding_summary", {}), ensure_ascii=False),
        })

    pd.DataFrame(flat_rows).to_csv(OUT_CSV, index=False)

    print(f"Backbone YC rows: {len(yc)}")
    print(f"Matched Startups.csv rows into YC backbone: {matched_startups}")
    print(f"Matched AI funding companies into existing records: {matched_ai}")
    print(f"Final merged company count: {len(companies)}")
    print(f"Wrote {OUT_JSON}")
    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()