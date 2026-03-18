import json

INPUT_FILE = "enriched_init.json"
OUTPUT_FILE = "enriched_init.json"  # overwrite; change if you want a new file


ROLE_KEYWORDS = {
    "Frontend Software Engineer": [
        "frontend", "front end", "react", "ui developer"
    ],
    "Backend Software Engineer": [
        "backend", "back end", "api", "server", "database"
    ],
    "Full Stack Developer": [
        "full stack", "full-stack"
    ],
    "Machine Learning Engineer": [
        "machine learning", "ml engineer", "ai engineer"
    ],
    "Data Scientist": [
        "data science", "data scientist"
    ],
    "Data Analyst": [
        "data analyst", "analytics", "business analyst"
    ],
    "Product Manager": [
        "product manager"
    ],
    "Product Designer": [
        "product design"
    ],
    "UI/UX Designer": [
        "ui/ux", "ux designer", "ui designer"
    ],
    "Graphic Designer": [
        "graphic design"
    ],
    "Marketing Manager": [
        "marketing manager"
    ],
    "Marketing Assistant": [
        "marketing assistant", "marketing"
    ],
    "Social Media Marketing Coordinator": [
        "social media", "content", "digital marketing"
    ],
    "Business Development Associate": [
        "business development", "sales", "client servicing"
    ],
    "HR Manager": [
        "hr manager", "hr business partner"
    ],
    "HR Assistant": [
        "hr assistant", "human resource", "recruiting"
    ],
    "Operations Associate": [
        "operations", "projects", "production", "procurement"
    ],
    "Finance Analyst": [
        "finance"
    ],
}


# Optional: prioritize stronger roles over weaker ones
PRIORITY = {
    "Marketing Manager": 2,
    "Marketing Assistant": 1,
    "HR Manager": 2,
    "HR Assistant": 1,
}


def extract_inferred_roles(text):
    text = (text or "").lower()
    found = []

    for role, keywords in ROLE_KEYWORDS.items():
        for k in keywords:
            if k in text:
                found.append(role)
                break

    # Deduplicate
    found = list(set(found))

    # Apply priority filtering
    final_roles = []
    seen_groups = {}

    for role in found:
        group = None
        if "Marketing" in role:
            group = "marketing"
        elif "HR" in role:
            group = "hr"

        if group:
            prev = seen_groups.get(group)
            if not prev or PRIORITY.get(role, 1) > PRIORITY.get(prev, 1):
                seen_groups[group] = role
        else:
            final_roles.append(role)

    final_roles.extend(seen_groups.values())

    return sorted(set(final_roles))


def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    companies = data.get("companies", [])

    for company in companies:
        job_text = company.get("job_posting_text", "")
        company["inferred_roles"] = extract_inferred_roles(job_text)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("Done: inferred_roles added.")


if __name__ == "__main__":
    main()