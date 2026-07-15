import pandas as pd

from services.config import ENTERPRISE_WORKFORCE_DATASET_PATH
from services.nlp_service import TECH_SKILLS, canonicalize_skill, categorize_skills, match_skills, normalize_skill


CAREER_CLUSTERS = {
    "Full Stack Engineering": {
        "signals": {"react", "javascript", "typescript", "node.js", "firebase", "mysql", "postgresql", "html", "css", "fastapi", "django", "flask"},
        "recommendations": [
            "typescript",
            "node.js",
            "express.js",
            "docker",
            "aws",
            "ci/cd",
            "system design",
            "redis",
            "postgresql",
            "kubernetes",
            "testing",
            "cloud deployment",
        ],
    },
    "AI/ML Engineering": {
        "signals": {"python", "machine learning", "tensorflow", "pytorch", "scikit-learn", "nlp", "deep learning"},
        "recommendations": ["mlops", "docker", "fastapi", "aws", "model monitoring", "feature store", "kubernetes", "spark"],
    },
    "Cloud Data Engineering": {
        "signals": {"python", "sql", "spark", "hadoop", "airflow", "dbt", "databricks", "snowflake"},
        "recommendations": ["airflow", "spark", "dbt", "snowflake", "databricks", "kafka", "docker", "aws", "terraform"],
    },
    "Business Intelligence": {
        "signals": {"sql", "power bi", "tableau", "excel", "data analytics", "business intelligence"},
        "recommendations": ["python", "statistics", "power bi", "tableau", "dbt", "snowflake", "data engineering"],
    },
    "Cloud/DevOps Engineering": {
        "signals": {"aws", "azure", "gcp", "docker", "kubernetes", "terraform", "jenkins", "ci/cd", "linux"},
        "recommendations": ["kubernetes", "terraform", "prometheus", "grafana", "github actions", "aws", "linux", "system design"],
    },
}

BLOCKED_CROSS_DOMAIN = {
    "quantum algorithms",
    "linear algebra",
    "climate data analysis",
    "energy modeling",
    "renewable energy",
    "quantum computing",
}


def _market_skill_pool(limit=80):
    if not ENTERPRISE_WORKFORCE_DATASET_PATH.exists():
        return TECH_SKILLS

    try:
        df = pd.read_csv(ENTERPRISE_WORKFORCE_DATASET_PATH, usecols=["skill", "jobs_count"])
        top_skills = (
            df.groupby("skill", as_index=False)["jobs_count"]
            .sum()
            .sort_values("jobs_count", ascending=False)
            .head(limit)["skill"]
            .tolist()
        )
        curated = [canonicalize_skill(skill) for skill in top_skills]
        return [skill for skill in curated if skill] or TECH_SKILLS
    except Exception:
        return TECH_SKILLS


def _skill_demand_map():
    if not ENTERPRISE_WORKFORCE_DATASET_PATH.exists():
        return {}

    try:
        df = pd.read_csv(ENTERPRISE_WORKFORCE_DATASET_PATH, usecols=["skill", "jobs_count", "skill_growth_rate", "avg_salary_usd"])
        summary = (
            df.groupby("skill", as_index=False)
            .agg(
                jobs_count=("jobs_count", "sum"),
                skill_growth_rate=("skill_growth_rate", "mean"),
                avg_salary_usd=("avg_salary_usd", "mean"),
            )
        )
        max_jobs = max(float(summary["jobs_count"].max()), 1.0)
        return {
            normalize_skill(row["skill"]): {
                "jobs_count": float(row["jobs_count"]),
                "skill_growth_rate": float(row["skill_growth_rate"]),
                "avg_salary_usd": float(row["avg_salary_usd"]),
                "market_percentile": min(1.0, float(row["jobs_count"]) / max_jobs),
            }
            for _, row in summary.iterrows()
        }
    except Exception:
        return {}


def _infer_career_cluster(resume_skills):
    current = {normalize_skill(canonicalize_skill(skill) or skill) for skill in resume_skills}
    best_role = "Full Stack Engineering"
    best_score = -1
    for role, profile in CAREER_CLUSTERS.items():
        overlap = len(current.intersection(profile["signals"]))
        category_bonus = 0
        categories = categorize_skills(current)
        if role == "Full Stack Engineering":
            category_bonus = len(categories.get("frontend_backend", [])) + len(categories.get("databases", []))
        elif role == "AI/ML Engineering":
            category_bonus = len(categories.get("ai_ml", []))
        elif role == "Cloud Data Engineering":
            category_bonus = len(categories.get("data_engineering", []))
        elif role == "Cloud/DevOps Engineering":
            category_bonus = len(categories.get("devops_mlops", [])) + len(categories.get("cloud_platforms", []))
        score = overlap * 3 + category_bonus
        if score > best_score:
            best_role = role
            best_score = score
    return best_role


def _candidate_pool(resume_skills, market_skills=None):
    role = _infer_career_cluster(resume_skills)
    current = {normalize_skill(canonicalize_skill(skill) or skill) for skill in resume_skills}
    adjacent = CAREER_CLUSTERS[role]["recommendations"]
    market = market_skills or _market_skill_pool()
    normalized_market = [canonicalize_skill(skill) or normalize_skill(skill) for skill in market]
    pool = []
    for skill in [*adjacent, *normalized_market]:
        canonical = canonicalize_skill(skill)
        if not canonical:
            continue
        normalized = normalize_skill(canonical)
        if normalized in current or normalized in BLOCKED_CROSS_DOMAIN:
            continue
        if role == "Full Stack Engineering" and normalized in BLOCKED_CROSS_DOMAIN:
            continue
        if canonical not in pool:
            pool.append(canonical)
    return role, pool


def _interpret_demand(demand):
    percentile = float(demand.get("market_percentile", 0.0))
    growth = float(demand.get("skill_growth_rate", 0.0))
    if percentile >= 0.70 and growth >= 0:
        return "Very High Demand"
    if percentile >= 0.45:
        return "High Demand"
    if growth >= 0.18:
        return "Emerging"
    if percentile >= 0.20:
        return "Moderate Growth"
    return "Stable"


def recommend_skills(resume_skills: list[str], market_skills: list[str] | None = None, top_k: int = 10):
    """Return career-coherent missing skills ranked by adjacency, semantic fit, and demand."""
    clean_resume = [canonicalize_skill(skill) or normalize_skill(skill) for skill in resume_skills if canonicalize_skill(skill) or normalize_skill(skill)]
    role, candidate_skills = _candidate_pool(clean_resume, market_skills)
    demand_map = _skill_demand_map()
    semantic_matches = match_skills(clean_resume, candidate_skills, max(top_k, min(30, len(candidate_skills)))) if clean_resume else []
    semantic_map = {normalize_skill(item["skill"]): float(item["score"]) for item in semantic_matches}
    adjacency_order = {normalize_skill(skill): index for index, skill in enumerate(CAREER_CLUSTERS[role]["recommendations"])}

    enriched_matches = []
    current = {normalize_skill(skill) for skill in clean_resume}
    for skill in candidate_skills:
        normalized = normalize_skill(skill)
        if normalized in current:
            continue
        demand = demand_map.get(normalized, {})
        semantic_distance = 1.0 - semantic_map.get(normalized, 0.45)
        adjacency_score = 1.0 - min(adjacency_order.get(normalized, 12), 12) / 12.0
        market_score = float(demand.get("market_percentile", 0.35))
        growth_score = min(1.0, max(0.0, float(demand.get("skill_growth_rate", 0.0)) + 0.45))
        is_adjacent = normalized in adjacency_order
        priority = round(
            adjacency_score * 0.70
            + semantic_distance * 0.08
            + market_score * 0.10
            + growth_score * 0.05
            + (0.12 if is_adjacent else -0.10),
            4,
        )
        demand_label = _interpret_demand(demand)
        enriched_matches.append(
            {
                "skill": skill,
                "score": round(1.0 - semantic_distance, 4),
                "recommended": True,
                "career_track": role,
                "market_demand": round(demand.get("jobs_count", 0.0), 2),
                "growth_signal": round(demand.get("skill_growth_rate", 0.0), 4),
                "avg_salary_usd": round(demand.get("avg_salary_usd", 0.0), 2),
                "priority_score": priority,
                "future_demand_level": demand_label,
                "reason": (
                    f"{skill} is a realistic next-step skill for {role}, improving adjacent capability, "
                    f"delivery maturity, and {demand_label.lower()} market positioning."
                ),
            }
        )

    enriched_matches.sort(key=lambda item: item["priority_score"], reverse=True)
    matches = enriched_matches[:top_k]
    return {
        "matches": matches,
        "recommended_skills": [item["skill"] for item in matches],
        "market_skills_used": candidate_skills[:top_k],
        "career_track": role,
    }
