from __future__ import annotations

import math
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd


BACKEND_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BACKEND_DIR / "data" / "raw"
PROCESSED_DIR = BACKEND_DIR / "data" / "processed"
ENTERPRISE_OUTPUT_PATH = PROCESSED_DIR / "enterprise_workforce_dataset.csv"
FINAL_SKILL_OUTPUT_PATH = PROCESSED_DIR / "final_skill_forecast_dataset.csv"


SKILL_SYNONYMS = {
    "ai": "Artificial Intelligence",
    "artificial intelligence": "Artificial Intelligence",
    "js": "JavaScript",
    "javascript": "JavaScript",
    "node": "Node.js",
    "nodejs": "Node.js",
    "node.js": "Node.js",
    "py": "Python",
    "python": "Python",
    "sql": "SQL",
    "aws": "AWS",
    "amazon web services": "AWS",
    "powerbi": "Power BI",
    "power bi": "Power BI",
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "dl": "Deep Learning",
    "deep learning": "Deep Learning",
    "nlp": "NLP",
    "natural language processing": "NLP",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "scikit learn": "Scikit-learn",
    "scikit-learn": "Scikit-learn",
    "k8s": "Kubernetes",
    "kubernetes": "Kubernetes",
    "reactjs": "React",
    "react.js": "React",
    "c sharp": "C#",
}


STRATEGIC_SKILLS = {
    "Python", "SQL", "AWS", "Java", "JavaScript", "React", "Node.js",
    "Docker", "Kubernetes", "TensorFlow", "PyTorch", "Machine Learning",
    "Deep Learning", "NLP", "Data Analysis", "Power BI", "Tableau",
    "Spark", "Hadoop", "Azure", "GCP", "Artificial Intelligence",
    "Scikit-learn", "Pandas", "NumPy", "MLOps", "DevOps", "Agile",
    "Project Management", "Business Development", "Customer Service",
    "Sales", "Communication", "Communication Skills", "Management",
    "Troubleshooting", "Accounting", "Recruitment", "Sap", "Oracle",
    "C#", "Css", "Automation", "Climate Data Analysis", "Energy Modeling",
    "Linear Algebra", "Quantum Algorithms", "Qiskit",
}


NOISY_TERMS = {
    "sr", "jr", "skills", "skill", "years", "year", "experience",
    "preferred", "candidate", "good", "excellent", "knowledge",
    "strong", "foundation", "ability", "work", "team", "required",
}


def clean_text(value: object) -> str:
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", str(value).strip())


def normalize_skill(value: object) -> str | None:
    text = clean_text(value)
    text = text.strip(" .;:/\\[]{}()")
    if not text:
        return None

    key = re.sub(r"\s+", " ", text.lower())
    skill = SKILL_SYNONYMS.get(key, text.title())
    letters = re.findall(r"[A-Za-z]", skill)
    words = re.findall(r"[A-Za-z+#.]+", skill)

    if len(letters) < 2 or len(skill) > 35:
        return None
    if re.match(r"^[#\-\d]", skill) or re.fullmatch(r"\d+", skill):
        return None
    if len(words) > 4 and skill not in STRATEGIC_SKILLS:
        return None
    if any(word.lower() in NOISY_TERMS for word in words) and skill not in STRATEGIC_SKILLS:
        return None
    return skill


def split_skills(value: object) -> list[str]:
    if pd.isna(value):
        return []
    raw_parts = re.split(r"[,;|]", str(value))
    skills = [normalize_skill(part) for part in raw_parts]
    return [skill for skill in skills if skill]


def parse_date(value: object) -> pd.Timestamp | None:
    if pd.isna(value):
        return None
    text = str(value).strip()
    relative = re.match(r"(\d+)\s+days?\s+ago", text, flags=re.IGNORECASE)
    if relative:
        return pd.Timestamp(datetime.today() - timedelta(days=int(relative.group(1))))
    parsed = pd.to_datetime(text, errors="coerce")
    return None if pd.isna(parsed) else parsed


def salary_to_usd(value: object, currency: object = "USD") -> float:
    if value is None or pd.isna(value):
        return np.nan
    currency = str(currency or "USD").upper()
    value = float(value)
    if value <= 0:
        return np.nan
    return value / 83.0 if currency == "INR" else value


def append_rows(rows: list[dict], *, date, skills, source, title="", company="", location="", salary=np.nan, remote=np.nan, industry="", experience=np.nan, emerging=0):
    if date is None or not skills:
        return
    month = pd.Timestamp(date).strftime("%Y-%m")
    for skill in sorted(set(skills)):
        rows.append(
            {
                "date": month,
                "skill": skill,
                "jobs_count": 1.0,
                "job_openings": 1.0,
                "avg_salary_usd": salary,
                "remote_pct": remote,
                "python_pct": np.nan,
                "aws_pct": np.nan,
                "gdp_growth": np.nan,
                "vc_funding_tech_bn": np.nan,
                "emerging_flag": float(emerging),
                "experience_level": experience,
                "location": clean_text(location).title(),
                "company": clean_text(company).title(),
                "industry": clean_text(industry).title(),
                "job_title": clean_text(title).title(),
                "source": source,
            }
        )


def load_jobs_dataset(rows: list[dict]):
    path = RAW_DIR / "jobs_dataset.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).drop_duplicates()
    for _, row in df.iterrows():
        append_rows(
            rows,
            date=parse_date(row.get("date_posted")),
            skills=split_skills(row.get("skills_list")),
            source=path.name,
            title=row.get("job_title"),
            company=row.get("company"),
            location=row.get("location"),
        )


def load_future_jobs_dataset(rows: list[dict]):
    path = RAW_DIR / "future_jobs_dataset.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).drop_duplicates()
    for _, row in df.iterrows():
        append_rows(
            rows,
            date=parse_date(row.get("posting_date")),
            skills=split_skills(row.get("skills_required")),
            source=path.name,
            title=row.get("job_title"),
            location=row.get("location"),
            salary=row.get("salary_usd"),
            remote=100.0 if str(row.get("remote_option")).lower() == "yes" else 0.0,
            industry=row.get("industry"),
            emerging=1,
        )


def load_job_trend_dataset(rows: list[dict]):
    path = RAW_DIR / "job_trend.csv"
    if not path.exists():
        return
    df = pd.read_csv(path).drop_duplicates()
    for _, row in df.iterrows():
        date = pd.Timestamp(year=int(row["year"]), month=int(row["month"]), day=1)
        openings = float(row.get("job_openings", 0) or 0)
        for skill, pct_column in {"Python": "python_pct", "AWS": "aws_pct"}.items():
            rows.append(
                {
                    "date": date.strftime("%Y-%m"),
                    "skill": skill,
                    "jobs_count": max(1.0, openings * float(row.get(pct_column, 0)) / 100.0),
                    "job_openings": openings,
                    "avg_salary_usd": row.get("avg_salary_usd"),
                    "remote_pct": row.get("remote_pct"),
                    "python_pct": row.get("python_pct"),
                    "aws_pct": row.get("aws_pct"),
                    "gdp_growth": row.get("gdp_growth"),
                    "vc_funding_tech_bn": row.get("vc_funding_tech_bn"),
                    "emerging_flag": row.get("emerging_flag"),
                    "experience_level": np.nan,
                    "location": "Global",
                    "company": "",
                    "industry": "Technology",
                    "job_title": clean_text(row.get("job_role")).title(),
                    "source": path.name,
                }
            )


def load_indian_market_dataset(rows: list[dict]):
    path = RAW_DIR / "indian-job-market-dataset-2025.xlsx"
    if not path.exists():
        return
    df = pd.read_excel(path).drop_duplicates(subset=["jobId"])
    for _, row in df.iterrows():
        min_salary = salary_to_usd(row.get("minimumSalary"), row.get("currency"))
        max_salary = salary_to_usd(row.get("maximumSalary"), row.get("currency"))
        salary_values = [value for value in [min_salary, max_salary] if not pd.isna(value)]
        exp_values = [row.get("minimumExperience"), row.get("maximumExperience")]
        exp_values = [float(value) for value in exp_values if not pd.isna(value)]
        append_rows(
            rows,
            date=parse_date(row.get("jobUploaded")),
            skills=split_skills(row.get("tagsAndSkills")),
            source=path.name,
            title=row.get("title"),
            company=row.get("companyName"),
            location=row.get("location"),
            salary=float(np.mean(salary_values)) if salary_values else np.nan,
            experience=float(np.mean(exp_values)) if exp_values else np.nan,
        )


def filter_forecastable_skills(raw: pd.DataFrame) -> pd.DataFrame:
    skill_summary = raw.groupby("skill").agg(
        total_demand=("jobs_count", "sum"),
        active_months=("date", "nunique"),
        sources=("source", "nunique"),
    )
    keep = skill_summary[
        (skill_summary["total_demand"] >= 80)
        | (skill_summary.index.isin(STRATEGIC_SKILLS) & (skill_summary["total_demand"] >= 8))
        | ((skill_summary["active_months"] >= 4) & (skill_summary["sources"] >= 2))
    ].index
    return raw[raw["skill"].isin(keep)].copy()


def add_feature_engineering(monthly: pd.DataFrame) -> pd.DataFrame:
    monthly["date_dt"] = pd.to_datetime(monthly["date"], format="%Y-%m")
    monthly = monthly.sort_values(["skill", "date_dt"])

    for column in ["avg_salary_usd", "remote_pct", "python_pct", "aws_pct", "gdp_growth", "vc_funding_tech_bn", "experience_level"]:
        monthly[column] = monthly[column].fillna(monthly[column].median()).fillna(0)

    monthly["month"] = monthly["date_dt"].dt.month
    monthly["month_sin"] = np.sin(2 * math.pi * monthly["month"] / 12)
    monthly["month_cos"] = np.cos(2 * math.pi * monthly["month"] / 12)
    monthly["semantic_skill_score"] = (
        monthly["emerging_flag"] * 0.30
        + monthly["avg_salary_usd"].rank(pct=True) * 0.25
        + monthly["jobs_count"].rank(pct=True) * 0.25
        + monthly["source_count"].rank(pct=True) * 0.20
    )

    engineered = []
    for _, group in monthly.groupby("skill", sort=False):
        full_index = pd.date_range(group["date_dt"].min(), group["date_dt"].max(), freq="MS")
        group = group.set_index("date_dt").sort_index().reindex(full_index)
        group["skill"] = group["skill"].ffill().bfill()
        group["date"] = group.index.strftime("%Y-%m")
        numeric_columns = [col for col in group.columns if col not in {"date", "skill"}]
        group[numeric_columns] = group[numeric_columns].fillna(0)

        group["jobs_lag_1"] = group["jobs_count"].shift(1).fillna(0)
        group["jobs_lag_3"] = group["jobs_count"].shift(3).fillna(0)
        group["jobs_lag_6"] = group["jobs_count"].shift(6).fillna(0)
        group["jobs_lag_12"] = group["jobs_count"].shift(12).fillna(0)
        group["rolling_avg_3"] = group["jobs_count"].rolling(3, min_periods=1).mean()
        group["rolling_avg_6"] = group["jobs_count"].rolling(6, min_periods=1).mean()
        group["rolling_avg_12"] = group["jobs_count"].rolling(12, min_periods=1).mean()
        group["trend_volatility"] = group["jobs_count"].rolling(6, min_periods=2).std().fillna(0)
        group["hiring_velocity"] = group["jobs_count"].diff().fillna(0)
        group["demand_acceleration"] = group["hiring_velocity"].diff().fillna(0)
        group["demand_momentum"] = group["jobs_count"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        group["workforce_growth_rate"] = group["rolling_avg_6"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        group["skill_growth_rate"] = group["jobs_count"].pct_change(3).replace([np.inf, -np.inf], 0).fillna(0)
        group["technology_adoption_rate"] = group["emerging_flag"] * (1 + group["demand_momentum"].clip(-1, 3))
        group["trend_slope_6"] = group["jobs_count"].rolling(6, min_periods=2).apply(
            lambda values: np.polyfit(np.arange(len(values)), values, 1)[0],
            raw=True,
        ).fillna(0)
        engineered.append(group.reset_index(drop=True))

    result = pd.concat(engineered, ignore_index=True)
    result = result.replace([np.inf, -np.inf], 0).fillna(0)
    result["forecast_weight"] = np.where(result["source_count"] > 1, 1.2, 1.0)
    return result.drop(columns=["month"], errors="ignore")


def build_enterprise_workforce_dataset() -> pd.DataFrame:
    rows = []
    load_jobs_dataset(rows)
    load_future_jobs_dataset(rows)
    load_job_trend_dataset(rows)
    load_indian_market_dataset(rows)
    if not rows:
        raise FileNotFoundError(f"No supported raw datasets found in {RAW_DIR}")

    raw = pd.DataFrame(rows).dropna(subset=["date", "skill"])
    raw = filter_forecastable_skills(raw)
    monthly = raw.groupby(["date", "skill"], as_index=False).agg(
        jobs_count=("jobs_count", "sum"),
        job_openings=("job_openings", "sum"),
        avg_salary_usd=("avg_salary_usd", "mean"),
        remote_pct=("remote_pct", "mean"),
        python_pct=("python_pct", "mean"),
        aws_pct=("aws_pct", "mean"),
        gdp_growth=("gdp_growth", "mean"),
        vc_funding_tech_bn=("vc_funding_tech_bn", "mean"),
        emerging_flag=("emerging_flag", "max"),
        experience_level=("experience_level", "mean"),
        location_demand=("location", "nunique"),
        industry_growth=("industry", "nunique"),
        source_count=("source", "nunique"),
    )
    enterprise = add_feature_engineering(monthly)
    enterprise = enterprise[enterprise["jobs_count"].notna()]
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    enterprise.to_csv(ENTERPRISE_OUTPUT_PATH, index=False)
    enterprise[["date", "skill", "jobs_count"]].to_csv(FINAL_SKILL_OUTPUT_PATH, index=False)
    return enterprise


if __name__ == "__main__":
    dataset = build_enterprise_workforce_dataset()
    print(f"Generated {ENTERPRISE_OUTPUT_PATH}")
    print(f"Rows: {len(dataset)}")
    print(f"Skills: {dataset['skill'].nunique()}")
    print(dataset.head().to_string(index=False))
