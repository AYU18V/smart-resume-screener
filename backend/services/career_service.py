from __future__ import annotations

from functools import lru_cache

import pandas as pd

from services.config import CLEAN_FORECAST_DATASET_PATH, ENTERPRISE_WORKFORCE_DATASET_PATH
from services.nlp_service import extract_skills_from_text, normalize_skill
from services.recommendation_service import recommend_skills


@lru_cache(maxsize=1)
def _career_market_frame():
    frames = []
    for path in [ENTERPRISE_WORKFORCE_DATASET_PATH, CLEAN_FORECAST_DATASET_PATH]:
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            if "skill" in df and "jobs_count" in df:
                frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()

    df = pd.concat(frames, ignore_index=True)
    for column in ["avg_salary_usd", "skill_growth_rate", "emerging_flag", "trend_volatility", "hiring_velocity"]:
        if column not in df:
            df[column] = 0.0
    return df


def _skill_market_summary():
    df = _career_market_frame()
    if df.empty:
        return {}
    summary = (
        df.groupby("skill", as_index=False)
        .agg(
            demand=("jobs_count", "sum"),
            avg_salary_usd=("avg_salary_usd", "mean"),
            growth=("skill_growth_rate", "mean"),
            emerging=("emerging_flag", "max"),
            volatility=("trend_volatility", "mean"),
            hiring_velocity=("hiring_velocity", "mean"),
        )
        .sort_values("demand", ascending=False)
    )
    max_demand = max(float(summary["demand"].max()), 1.0)
    result = {}
    for _, row in summary.iterrows():
        demand_score = min(100.0, float(row["demand"]) / max_demand * 100.0)
        growth_score = min(100.0, max(0.0, (float(row["growth"]) + 1.0) * 50.0))
        longevity = max(0.0, min(100.0, 70.0 + growth_score * 0.25 - float(row["volatility"]) * 20.0))
        result[normalize_skill(row["skill"])] = {
            "skill": row["skill"],
            "demand_score": round(demand_score, 2),
            "growth_score": round(growth_score, 2),
            "avg_salary_usd": round(float(row["avg_salary_usd"]), 2),
            "hiring_velocity": round(float(row["hiring_velocity"]), 3),
            "emerging": bool(row["emerging"]),
            "technology_longevity_score": round(longevity, 2),
        }
    return result


def build_career_intelligence(skills: list[str] | None = None, text: str | None = None):
    current_skills = skills or extract_skills_from_text(text or "")
    current_normalized = {normalize_skill(skill) for skill in current_skills}
    market = _skill_market_summary()
    recommendations = recommend_skills(current_skills, top_k=12)
    recommended = [item for item in recommendations.get("matches", []) if item.get("recommended")][:8]

    enriched_recommendations = []
    for item in recommended:
        signal = market.get(normalize_skill(item["skill"]), {})
        enriched_recommendations.append(
            {
                "skill": item["skill"],
                "reason": (
                    f"{item['skill']} is recommended because it improves your semantic market fit "
                    "and connects to future workforce demand."
                ),
                "demand_score": signal.get("demand_score", 0),
                "growth_score": signal.get("growth_score", 0),
                "technology_longevity_score": signal.get("technology_longevity_score", 0),
                "avg_salary_usd": signal.get("avg_salary_usd", item.get("avg_salary_usd", 0)),
            }
        )

    current_signals = [market[skill] for skill in current_normalized if skill in market]
    demand_score = round(sum(item["demand_score"] for item in current_signals) / max(len(current_signals), 1), 2)
    growth_score = round(sum(item["growth_score"] for item in current_signals) / max(len(current_signals), 1), 2)
    longevity_score = round(
        sum(item["technology_longevity_score"] for item in current_signals) / max(len(current_signals), 1),
        2,
    )
    salary_values = [item["avg_salary_usd"] for item in current_signals if item["avg_salary_usd"] > 0]
    avg_salary = round(sum(salary_values) / max(len(salary_values), 1), 2) if salary_values else 0

    roadmap = [
        {
            "phase": "0-30 days",
            "focus": "Close ATS keyword gaps",
            "skills": [item["skill"] for item in enriched_recommendations[:3]],
        },
        {
            "phase": "30-60 days",
            "focus": "Build deployable projects",
            "skills": [item["skill"] for item in enriched_recommendations[3:6]],
        },
        {
            "phase": "60-90 days",
            "focus": "Add cloud, MLOps, and portfolio proof",
            "skills": [item["skill"] for item in enriched_recommendations[6:8]],
        },
    ]

    saturation = "Low" if demand_score >= 60 and growth_score >= 55 else "Moderate" if demand_score >= 35 else "High"
    employability = round(demand_score * 0.4 + growth_score * 0.35 + longevity_score * 0.25, 2)

    salary_low = round(avg_salary * 0.82, 2) if avg_salary else 0
    salary_high = round(avg_salary * 1.22, 2) if avg_salary else 0
    demand_trend = "Accelerating" if growth_score >= 65 else "Steady" if growth_score >= 45 else "Softening"

    return {
        "current_skills": current_skills,
        "skill_gap_analysis": enriched_recommendations,
        "learning_roadmap": roadmap,
        "salary_intelligence": {
            "estimated_avg_salary_usd": avg_salary,
            "expected_salary_range_usd": {"low": salary_low, "mid": avg_salary, "high": salary_high},
            "salary_growth_signal": growth_score,
            "future_salary_growth": "High" if growth_score >= 65 else "Moderate" if growth_score >= 45 else "Conservative",
            "salary_note": "Salary values are estimated from available workforce datasets and should be treated as directional.",
        },
        "workforce_demand_intelligence": {
            "future_demand_score": demand_score,
            "future_demand_trend": demand_trend,
            "market_saturation": saturation,
            "hiring_velocity": round(sum(item["hiring_velocity"] for item in current_signals) / max(len(current_signals), 1), 3),
        },
        "career_forecasting": {
            "future_employability_score": employability,
            "growth_prediction": "Increasing" if employability >= 65 else "Stable" if employability >= 45 else "Needs upskilling",
            "technology_longevity_score": longevity_score,
            "market_competitiveness_score": round(demand_score * 0.45 + growth_score * 0.35 + len(enriched_recommendations) * 1.5, 2),
        },
        "recommended_skills": enriched_recommendations,
        "future_proof_stack": [item["skill"] for item in enriched_recommendations[:6]],
        "certification_recommendations": [
            "AWS Certified Solutions Architect" if any(normalize_skill(item["skill"]) == "aws" for item in enriched_recommendations) else "Cloud fundamentals certification",
            "Certified Kubernetes Administrator" if any(normalize_skill(item["skill"]) == "kubernetes" for item in enriched_recommendations) else "Docker and Kubernetes practitioner badge",
            "Databricks Data Engineer Associate" if any(normalize_skill(item["skill"]) in {"spark", "databricks"} for item in enriched_recommendations) else "Applied MLOps specialization",
        ],
    }
