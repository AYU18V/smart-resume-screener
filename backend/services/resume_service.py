from __future__ import annotations

import re
from functools import lru_cache

import numpy as np
import pandas as pd
import PyPDF2

from services.config import CLEAN_FORECAST_DATASET_PATH, ENTERPRISE_WORKFORCE_DATASET_PATH
from services.nlp_service import (
    TECH_SKILLS,
    categorize_skills,
    extract_resume_entities,
    extract_skills_from_text,
    match_skills,
    normalize_skill,
)
from services.recommendation_service import recommend_skills


ROLE_PROFILES = {
    "AI/Cloud Engineering": ["python", "machine learning", "mlops", "docker", "kubernetes", "aws", "fastapi", "model monitoring"],
    "AI/ML Engineering": ["python", "machine learning", "deep learning", "tensorflow", "pytorch", "mlops", "docker", "aws"],
    "Cloud Data Engineering": ["python", "sql", "aws", "spark", "airflow", "dbt", "docker", "kubernetes"],
    "Full Stack Engineering": ["javascript", "typescript", "react", "node.js", "fastapi", "postgresql", "docker", "aws"],
    "Business Intelligence": ["sql", "python", "power bi", "tableau", "excel", "business intelligence", "data analytics"],
}

ROLE_SALARY_BASE = {
    "Full Stack Engineering": {"india_lpa": (7, 18), "us_usd": (85000, 135000)},
    "AI/Cloud Engineering": {"india_lpa": (10, 24), "us_usd": (105000, 165000)},
    "AI/ML Engineering": {"india_lpa": (10, 26), "us_usd": (110000, 175000)},
    "Cloud Data Engineering": {"india_lpa": (9, 22), "us_usd": (100000, 158000)},
    "Business Intelligence": {"india_lpa": (6, 16), "us_usd": (78000, 125000)},
    "General Technology": {"india_lpa": (5, 12), "us_usd": (70000, 110000)},
}


SECTION_KEYWORDS = {
    "summary": ["summary", "profile", "objective"],
    "experience": ["experience", "employment", "work history", "projects"],
    "skills": ["skills", "technical skills", "technologies"],
    "education": ["education", "degree", "university"],
}


def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    chunks = []

    for page in reader.pages:
        chunks.append(page.extract_text() or "")

    return "\n".join(chunks)


@lru_cache(maxsize=1)
def _market_profile():
    rows = []
    for path in [CLEAN_FORECAST_DATASET_PATH, ENTERPRISE_WORKFORCE_DATASET_PATH]:
        if path.exists():
            try:
                usecols = ["skill", "jobs_count"]
                if path == ENTERPRISE_WORKFORCE_DATASET_PATH:
                    usecols += ["skill_growth_rate", "avg_salary_usd", "emerging_flag"]
                df = pd.read_csv(path, usecols=lambda column: column in usecols)
                rows.append(df)
            except Exception:
                continue

    if not rows:
        return {}

    df = pd.concat(rows, ignore_index=True)
    for column in ["jobs_count", "skill_growth_rate", "avg_salary_usd", "emerging_flag"]:
        if column not in df:
            df[column] = 0.0

    summary = (
        df.groupby("skill", as_index=False)
        .agg(
            demand=("jobs_count", "sum"),
            growth=("skill_growth_rate", "mean"),
            salary=("avg_salary_usd", "mean"),
            emerging=("emerging_flag", "max"),
        )
        .sort_values("demand", ascending=False)
    )
    max_demand = max(float(summary["demand"].max()), 1.0)
    profile = {}
    for _, row in summary.iterrows():
        key = normalize_skill(row["skill"])
        demand_index = min(100.0, float(row["demand"]) / max_demand * 100.0)
        growth_index = min(100.0, max(0.0, (float(row["growth"]) + 1.0) * 50.0))
        profile[key] = {
            "skill": row["skill"],
            "demand_index": demand_index,
            "growth_index": growth_index,
            "salary": float(row["salary"]),
            "emerging": bool(row["emerging"]),
            "market_score": round(demand_index * 0.55 + growth_index * 0.30 + (15.0 if row["emerging"] else 0.0), 2),
        }
    return profile


def _keyword_score(text, skills):
    lowered = text.lower()
    action_verbs = ["built", "designed", "deployed", "optimized", "automated", "implemented", "led", "improved"]
    metric_hits = len(re.findall(r"\b\d+%|\b\d+\+?\s*(?:users|records|models|apis|dashboards|projects)\b", lowered))
    verb_hits = sum(1 for verb in action_verbs if verb in lowered)
    skill_density = min(46, len(skills) * 4.5)
    return int(min(100, 32 + skill_density + verb_hits * 4 + metric_hits * 5))


def _formatting_score(text):
    lowered = text.lower()
    section_score = sum(1 for keywords in SECTION_KEYWORDS.values() if any(keyword in lowered for keyword in keywords)) * 18
    length_score = 20 if 600 <= len(text) <= 6000 else 14 if len(text) > 250 else 8
    contact_score = 12 if re.search(r"[\w.+-]+@[\w-]+\.[\w.-]+", text) else 0
    noise_penalty = 12 if len(re.findall(r"[^\w\s.,:/@%+#-]", text)) > 80 else 0
    return int(max(0, min(100, section_score + length_score + contact_score - noise_penalty)))


def _role_alignment(skills):
    skill_set = {normalize_skill(skill) for skill in skills}
    ranked = []
    for role, required in ROLE_PROFILES.items():
        overlap = len(skill_set.intersection(required))
        adjacent_bonus = min(25, len(skill_set) * 3)
        score = round(min(92, 35 + overlap / max(len(required), 1) * 55 + adjacent_bonus))
        ranked.append({"role": role, "score": score, "matched_skills": sorted(skill_set.intersection(required))})
    ranked.sort(key=lambda item: item["score"], reverse=True)
    return ranked


def _career_roadmap(current_skills, recommended):
    current = {normalize_skill(skill) for skill in current_skills}
    roadmap = []
    phases = [
        ("Foundation", ["sql", "python", "git", "statistics"]),
        ("Production AI", ["docker", "fastapi", "mlops", "scikit-learn"]),
        ("Cloud Scale", ["aws", "kubernetes", "airflow", "spark"]),
        ("Enterprise AI", ["llm", "rag", "langchain", "monitoring"]),
    ]
    recommended_set = {normalize_skill(item["skill"] if isinstance(item, dict) else item) for item in recommended}
    for phase, skills in phases:
        targets = [skill for skill in skills if skill not in current]
        if targets:
            prioritized = sorted(targets, key=lambda skill: skill not in recommended_set)
            roadmap.append(
                {
                    "phase": phase,
                    "skills": prioritized[:4],
                    "outcome": f"Build credible {phase.lower()} capability for higher market alignment.",
                }
            )
    return roadmap[:4]


def _market_scores(skills):
    profile = _market_profile()
    normalized = [normalize_skill(skill) for skill in skills]
    signals = [profile[skill] for skill in normalized if skill in profile]
    if not signals:
        baseline = 34 + min(12, len(skills) * 2)
        return baseline, baseline, []
    future_raw = float(np.mean([item["growth_index"] + (10 if item["emerging"] else 0) for item in signals]))
    competitiveness_raw = float(np.mean([item["market_score"] for item in signals]))
    density_bonus = min(16, len(skills) * 2.2)
    future = 38 + future_raw * 0.48 + density_bonus
    competitiveness = 36 + competitiveness_raw * 0.50 + density_bonus
    return int(min(92, max(42, future))), int(min(92, max(40, competitiveness))), signals


def _salary_intelligence(skills, role, years_experience=0.0):
    signals = [item for item in _market_profile().values() if normalize_skill(item["skill"]) in {normalize_skill(skill) for skill in skills}]
    salaries = [item["salary"] for item in signals if item.get("salary", 0) > 0]
    role_base = ROLE_SALARY_BASE.get(role, ROLE_SALARY_BASE["General Technology"])
    experience_multiplier = 1.0 + min(0.45, max(0.0, years_experience) * 0.055)
    if salaries:
        midpoint = float(np.mean(salaries))
    else:
        midpoint = float(np.mean(role_base["us_usd"]))
    growth = float(np.mean([item.get("growth_index", 0) for item in signals])) if signals else 0.0
    india_low, india_high = role_base["india_lpa"]
    us_low, us_high = role_base["us_usd"]
    india_low = round(india_low * experience_multiplier, 1)
    india_high = round(india_high * experience_multiplier, 1)
    us_low = int(round(max(us_low * experience_multiplier, midpoint * 0.78)))
    us_high = int(round(max(us_high * experience_multiplier, midpoint * 1.16)))
    return {
        "expected_salary_range_usd": {
            "low": us_low,
            "mid": int(round(midpoint)),
            "high": us_high,
        },
        "india_market_estimate": f"₹{india_low:g}L - ₹{india_high:g}L annually",
        "us_market_equivalent": f"${int(us_low / 1000)}k - ${int(us_high / 1000)}k annually",
        "role_context": role,
        "experience_context": "Entry-level" if years_experience < 2 else "Mid-level" if years_experience < 6 else "Senior",
        "future_salary_growth": "High" if growth >= 65 else "Moderate" if growth >= 45 else "Conservative",
        "hiring_competitiveness": "High" if len(skills) >= 8 else "Moderate" if len(skills) >= 4 else "Developing",
        "market_demand_level": "High Demand" if growth >= 55 or len(signals) >= 4 else "Moderate Growth",
        "industry_salary_trend": "Compensation outlook improves with cloud deployment, production engineering, scalable backend, and measurable project impact.",
    }


def _score_band(raw, floor=30, ceiling=92):
    return int(max(floor, min(ceiling, round(raw))))


def _resume_strength_analysis(text, skills, categories, role):
    lowered = text.lower()
    frontend = len(categories.get("frontend_backend", []))
    data = len(categories.get("data_engineering", [])) + len(categories.get("databases", []))
    cloud = len(categories.get("cloud_platforms", []))
    devops = len(categories.get("devops_mlops", []))
    ai = len(categories.get("ai_ml", []))
    project_complexity = min(100, 38 + len(re.findall(r"\b(project|built|developed|deployed|implemented|designed)\b", lowered)) * 8 + len(skills) * 2)
    system_design = min(100, 35 + sum(term in lowered for term in ["scalable", "microservices", "architecture", "system design", "api", "authentication"]) * 10)
    deployment = min(100, 30 + (cloud * 14) + (devops * 10) + (18 if "deployed" in lowered else 0))
    stack_maturity = min(100, 40 + frontend * 7 + data * 6 + cloud * 8 + devops * 7 + ai * 5)

    strengths = []
    weaknesses = []
    if frontend >= 2:
        strengths.append("Strong frontend or application engineering foundation.")
    if data >= 2:
        strengths.append("Good database and data handling fundamentals.")
    if cloud or devops:
        strengths.append("Evidence of deployment or production engineering readiness.")
    if ai:
        strengths.append("AI/data capability improves future workforce adaptability.")
    if not strengths:
        strengths.append(f"Found a coherent {role.lower()} direction with room to add stronger project proof.")

    if cloud == 0:
        weaknesses.append("Missing cloud deployment evidence.")
    if devops == 0:
        weaknesses.append("Missing containerization, CI/CD, or production operations signals.")
    if system_design < 55:
        weaknesses.append("Add scalable backend, API architecture, or system design examples.")
    if project_complexity < 58:
        weaknesses.append("Add measurable project outcomes and business impact metrics.")

    return {
        "project_complexity_score": _score_band(project_complexity, 30, 95),
        "technology_stack_maturity": _score_band(stack_maturity, 30, 95),
        "cloud_readiness": _score_band(deployment, 25, 95),
        "frontend_backend_balance": _score_band(45 + frontend * 8 + data * 5, 30, 92),
        "ai_readiness": _score_band(35 + ai * 13, 30, 92),
        "deployment_readiness": _score_band(deployment, 25, 95),
        "system_design_maturity": _score_band(system_design, 30, 92),
        "strengths": strengths,
        "weaknesses": weaknesses[:4],
    }


def _recommended_skills(detected_skills, top_k=8):
    recommendations = recommend_skills(detected_skills, top_k=max(12, top_k))
    recommended = [item for item in recommendations.get("matches", []) if item.get("recommended")]
    market_profile = _market_profile()
    for item in recommended:
        market = market_profile.get(normalize_skill(item["skill"]), {})
        item["future_demand_score"] = round(market.get("market_score", 0.0), 2)
        item["reason"] = (
            f"{item['skill']} improves semantic market fit and is linked to future workforce demand."
            if market
            else item.get("reason", "This skill improves alignment with target job families.")
        )
    recommended.sort(key=lambda item: (item.get("priority_score", 0), item.get("future_demand_score", 0)), reverse=True)
    return recommended[:top_k], recommendations.get("matches", [])


def _future_demand_level(score, growth_signal=0.0):
    score = float(score or 0)
    growth_signal = float(growth_signal or 0)
    if score >= 75:
        return "Very High Demand"
    if score >= 58:
        return "High Demand"
    if growth_signal >= 0.18:
        return "Emerging"
    if score >= 42:
        return "Moderate Growth"
    if score >= 25:
        return "Stable"
    return "Specialized"


def analyze_resume_text(text: str):
    entities = extract_resume_entities(text)
    technical_skills = entities["technical_skills"]
    soft_skills = entities["soft_skills"]
    certifications = entities["certifications"]
    detected_skills = sorted(set(technical_skills))
    semantic_matches = match_skills(detected_skills, TECH_SKILLS, top_k=12)
    recommended, all_matches = _recommended_skills(detected_skills, top_k=10)
    role_matches = _role_alignment(detected_skills)
    top_role = role_matches[0] if role_matches else {"role": "General Technology", "score": 0}
    skill_categories = entities.get("skill_categories") or categorize_skills(detected_skills)

    keyword_score = _keyword_score(text, detected_skills)
    formatting_score = _formatting_score(text)
    recruiter_score = _score_band(42 + len(detected_skills) * 3.2 + len(soft_skills) * 4 + len(certifications) * 5, 35, 92)
    semantic_match_score = _score_band(42 + (np.mean([item["score"] for item in semantic_matches]) * 46 if semantic_matches else len(detected_skills) * 2), 35, 92)
    future_readiness_score, market_competitiveness_score, market_signals = _market_scores(detected_skills)
    ats_score = _score_band(keyword_score * 0.34 + formatting_score * 0.22 + recruiter_score * 0.24 + semantic_match_score * 0.20, 32, 94)
    industry_alignment_score = _score_band(top_role["score"] * 0.68 + semantic_match_score * 0.32, 34, 92)
    future_employability_score = int(round(future_readiness_score * 0.62 + market_competitiveness_score * 0.38))
    ai_career_growth_score = int(round(min(100, future_employability_score + len(recommended) * 2.5)))
    salary_intelligence = _salary_intelligence(detected_skills, top_role["role"], entities["years_experience"])
    strength_analysis = _resume_strength_analysis(text, detected_skills, skill_categories, top_role["role"])
    resume_strength_score = _score_band(
        ats_score * 0.25
        + future_employability_score * 0.22
        + recruiter_score * 0.18
        + strength_analysis["technology_stack_maturity"] * 0.18
        + strength_analysis["project_complexity_score"] * 0.17,
        34,
        94,
    )

    missing_skills = [item["skill"] for item in recommended[:8]]
    career_roadmap = _career_roadmap(detected_skills, recommended)
    strengths = []
    if strength_analysis["strengths"]:
        strengths.extend(strength_analysis["strengths"])
    if top_role["score"] >= 40:
        strengths.append(f"Strongest role alignment: {top_role['role']} ({top_role['score']}%).")
    if certifications:
        strengths.append("Certifications add credibility for ATS and recruiter screening.")
    if not strengths:
        strengths.append("Resume has enough text for baseline ATS analysis, but needs clearer technical evidence.")

    improvement_suggestions = [
        "Add measurable impact bullets using metrics such as latency, cost, accuracy, users, or automation time saved.",
        "Group skills into Tools, Cloud, Frameworks, Databases, and AI/ML so ATS parsers can classify them cleanly.",
        "Add project outcomes that connect your skills to business or workforce impact.",
    ]
    if missing_skills:
        improvement_suggestions.insert(0, f"Add market-relevant keywords: {', '.join(missing_skills[:5])}.")
    if formatting_score < 70:
        improvement_suggestions.append("Improve ATS formatting by adding clear sections for Summary, Skills, Experience, Projects, and Education.")

    recommended_skills = [
        {
            "skill": item["skill"],
            "reason": item["reason"],
            "priority_score": item.get("priority_score", 0),
            "future_demand_score": item.get("future_demand_score", 0),
            "future_demand_level": item.get("future_demand_level") or _future_demand_level(item.get("future_demand_score", 0), item.get("growth_signal", 0)),
            "growth_interpretation": (
                f"{item['skill']} is classified as "
                f"{item.get('future_demand_level') or _future_demand_level(item.get('future_demand_score', 0), item.get('growth_signal', 0))} "
                "based on workforce demand, adjacent career fit, and adoption momentum."
            ),
            "market_momentum": "High" if item.get("priority_score", 0) >= 0.75 else "Moderate" if item.get("priority_score", 0) >= 0.5 else "Selective",
            "technology_longevity": "Long-term" if item.get("future_demand_score", 0) >= 45 else "Specialized",
            "adoption_velocity": "Accelerating" if item.get("growth_signal", 0) >= 0.12 else "Steady",
            "avg_salary_usd": item.get("avg_salary_usd", 0),
        }
        for item in recommended
    ]

    return {
        "ats_score": ats_score,
        "keyword_optimization_score": keyword_score,
        "formatting_quality_score": formatting_score,
        "recruiter_compatibility_score": recruiter_score,
        "future_readiness_score": future_readiness_score,
        "future_employability_score": future_employability_score,
        "ai_career_growth_score": ai_career_growth_score,
        "resume_strength_score": resume_strength_score,
        "market_competitiveness": market_competitiveness_score,
        "market_competitiveness_score": market_competitiveness_score,
        "industry_alignment": top_role["role"],
        "industry_alignment_score": industry_alignment_score,
        "semantic_match_score": semantic_match_score,
        "detected_skills": detected_skills,
        "technical_skills": detected_skills,
        "skill_categories": skill_categories,
        "soft_skills": soft_skills,
        "certifications": certifications,
        "years_experience": entities["years_experience"],
        "missing_skills": missing_skills,
        "recommended_skills": recommended_skills,
        "career_roadmap": career_roadmap,
        "resume_strengths": strengths,
        "resume_weaknesses": strength_analysis["weaknesses"],
        "resume_strength_analysis": strength_analysis,
        "improvement_suggestions": improvement_suggestions,
        "role_alignment": role_matches,
        "semantic_matches": semantic_matches,
        "matches": all_matches,
        "workforce_demand_matches": market_signals[:8],
        "salary_intelligence": salary_intelligence,
        "market_signal_summary": (
            "Resume skills were compared against normalized workforce demand, growth, emerging-skill, and salary signals."
        ),
        "ats_breakdown": {
            "keyword_optimization_score": keyword_score,
            "formatting_quality_score": formatting_score,
            "recruiter_compatibility_score": recruiter_score,
            "semantic_match_score": semantic_match_score,
        },
        "career_recommendations": recommended_skills[:5],
        "summary": (
            "Enterprise ATS analysis combines Hugging Face entity extraction, semantic embeddings, "
            "resume structure scoring, and market demand signals from the workforce forecasting dataset."
        ),
    }


def analyze_resume_file(file):
    text = extract_text_from_pdf(file)
    return analyze_resume_text(text)
