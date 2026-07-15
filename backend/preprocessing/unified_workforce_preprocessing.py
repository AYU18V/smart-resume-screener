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
UNIFIED_OUTPUT_PATH = PROCESSED_DIR / "unified_workforce_dataset.csv"
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
    "powerbi": "Power BI",
    "power bi": "Power BI",
    "ml": "Machine Learning",
    "machine learning": "Machine Learning",
    "dl": "Deep Learning",
    "deep learning": "Deep Learning",
    "nlp": "NLP",
    "tensorflow": "TensorFlow",
    "pytorch": "PyTorch",
    "scikit learn": "Scikit-learn",
    "scikit-learn": "Scikit-learn",
}


NOISY_SKILLS = {
    "sr",
    "jr",
    "years",
    "year",
    "skills",
    "experience",
    "preferred",
    "candidate",
    "good",
    "excellent",
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
    if not re.search(r"[A-Za-z]", skill):
        return None
    if re.match(r"^[#\-\d]", skill):
        return None
    if len(re.findall(r"[A-Za-z]", skill)) < 2:
        return None
    if len(skill) < 2 or len(skill) > 45 or skill.lower() in NOISY_SKILLS:
        return None
    if re.fullmatch(r"\d+", skill):
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
    if pd.isna(parsed):
        return None
    return parsed


def salary_to_usd(value: float | int | None, currency: str = "USD") -> float:
    if value is None or pd.isna(value):
        return np.nan
    currency = str(currency or "USD").upper()
    if currency == "INR":
        return float(value) / 83.0
    return float(value)


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
    emerging_cutoff = df["posting_date"].max() if "posting_date" in df.columns else None
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
        skill_signals = {
            "Python": openings * float(row.get("python_pct", 0)) / 100.0,
            "AWS": openings * float(row.get("aws_pct", 0)) / 100.0,
        }
        for skill, count in skill_signals.items():
            rows.append(
                {
                    "date": date.strftime("%Y-%m"),
                    "skill": skill,
                    "jobs_count": max(1.0, count),
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
        salary_values = [value for value in [min_salary, max_salary] if not pd.isna(value) and value > 0]
        avg_salary = float(np.mean(salary_values)) if salary_values else np.nan
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
            salary=avg_salary,
            experience=float(np.mean(exp_values)) if exp_values else np.nan,
            emerging=0,
        )


def add_feature_engineering(monthly: pd.DataFrame) -> pd.DataFrame:
    monthly["date_dt"] = pd.to_datetime(monthly["date"], format="%Y-%m")
    monthly = monthly.sort_values(["skill", "date_dt"])

    global_salary = monthly["avg_salary_usd"].median()
    monthly["avg_salary_usd"] = monthly["avg_salary_usd"].fillna(global_salary).fillna(0)
    for column in ["remote_pct", "python_pct", "aws_pct", "gdp_growth", "vc_funding_tech_bn", "experience_level"]:
        monthly[column] = monthly[column].fillna(monthly[column].median()).fillna(0)

    monthly["month"] = monthly["date_dt"].dt.month
    monthly["month_sin"] = np.sin(2 * math.pi * monthly["month"] / 12)
    monthly["month_cos"] = np.cos(2 * math.pi * monthly["month"] / 12)
    monthly["semantic_skill_score"] = (
        monthly["emerging_flag"] * 0.35
        + (monthly["avg_salary_usd"].rank(pct=True) * 0.35)
        + (monthly["jobs_count"].rank(pct=True) * 0.30)
    )

    engineered = []
    for _, group in monthly.groupby("skill", sort=False):
        group = group.sort_values("date_dt").copy()
        group["jobs_lag_1"] = group["jobs_count"].shift(1)
        group["jobs_lag_3"] = group["jobs_count"].shift(3)
        group["jobs_lag_6"] = group["jobs_count"].shift(6)
        group["rolling_avg_3"] = group["jobs_count"].rolling(3, min_periods=1).mean()
        group["rolling_avg_6"] = group["jobs_count"].rolling(6, min_periods=1).mean()
        group["rolling_avg_12"] = group["jobs_count"].rolling(12, min_periods=1).mean()
        group["volatility_6"] = group["jobs_count"].rolling(6, min_periods=2).std()
        group["hiring_velocity"] = group["jobs_count"].diff().fillna(0)
        group["demand_momentum"] = group["jobs_count"].pct_change().replace([np.inf, -np.inf], 0).fillna(0)
        group["trend_slope_6"] = group["jobs_count"].rolling(6, min_periods=2).apply(
            lambda values: np.polyfit(np.arange(len(values)), values, 1)[0],
            raw=True,
        )
        engineered.append(group)

    result = pd.concat(engineered, ignore_index=True)
    fill_columns = [
        "jobs_lag_1",
        "jobs_lag_3",
        "jobs_lag_6",
        "volatility_6",
        "trend_slope_6",
    ]
    for column in fill_columns:
        result[column] = result[column].fillna(0)

    result["forecast_weight"] = np.where(result["source_count"] > 1, 1.15, 1.0)
    result = result.drop(columns=["date_dt", "month"])
    return result


def build_unified_workforce_dataset() -> pd.DataFrame:
    rows = []
    load_jobs_dataset(rows)
    load_future_jobs_dataset(rows)
    load_job_trend_dataset(rows)
    load_indian_market_dataset(rows)

    if not rows:
        raise FileNotFoundError(f"No raw datasets found in {RAW_DIR}")

    raw = pd.DataFrame(rows)
    raw = raw.dropna(subset=["date", "skill"])
    raw = raw[raw["skill"].map(lambda value: isinstance(value, str) and 2 <= len(value) <= 45)]

    grouped = raw.groupby(["date", "skill"], as_index=False).agg(
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

    unified = add_feature_engineering(grouped)
    unified["jobs_count"] = unified["jobs_count"].clip(lower=0)
    unified = unified.replace([np.inf, -np.inf], 0).fillna(0)

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    unified.to_csv(UNIFIED_OUTPUT_PATH, index=False)
    unified[["date", "skill", "jobs_count"]].to_csv(FINAL_SKILL_OUTPUT_PATH, index=False)
    return unified


if __name__ == "__main__":
    dataset = build_unified_workforce_dataset()
    print(f"Generated {UNIFIED_OUTPUT_PATH}")
    print(f"Rows: {len(dataset)}")
    print(f"Columns: {list(dataset.columns)}")
    print(dataset.head().to_string(index=False))
