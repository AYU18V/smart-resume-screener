from pathlib import Path
import sys

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ---------------------------------------------------
# ADD BACKEND ROOT TO PYTHON PATH
# ---------------------------------------------------

BACKEND_DIR = Path(__file__).resolve().parents[1]

if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

# ---------------------------------------------------
# IMPORT PROJECT MODULES
# ---------------------------------------------------

from analysis.skill_analysis import (
    get_live_jobs,
    get_skill_summary,
    get_top_jobs,
    get_skill_trends
)

from services.config import CORS_ORIGINS

from services.career_service import build_career_intelligence

from services.forecast_service import (
    forecast_skills,
    get_dataset_insights,
    get_model_explainability,
    get_workforce_analytics,
    predict_demand
)

from services.recommendation_service import recommend_skills

from services.resume_service import analyze_resume_file

from services.skill_extraction_service import extract_skills

# ---------------------------------------------------
# FASTAPI APP
# ---------------------------------------------------

app = FastAPI(
    title="Computational Workforce Intelligence API",
    version="1.0.0"
)

# ---------------------------------------------------
# CORS CONFIGURATION
# ---------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# REQUEST MODELS
# ---------------------------------------------------

class TextRequest(BaseModel):
    text: str = Field(..., min_length=1)


class MatchSkillsRequest(BaseModel):
    resume_skills: list[str] = Field(default_factory=list)
    market_skills: list[str] | None = None
    top_k: int = Field(default=10, ge=1, le=50)


class PredictDemandRequest(BaseModel):
    skill: str = Field(..., min_length=1)
    months: int = Field(default=6, ge=1, le=24)


class CareerIntelligenceRequest(BaseModel):
    skills: list[str] = Field(default_factory=list)
    text: str | None = None

# ---------------------------------------------------
# ROOT ROUTE
# ---------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "Computational Workforce Intelligence API Running"
    }

# ---------------------------------------------------
# BASIC ANALYTICS ROUTES
# ---------------------------------------------------

@app.get("/skills")
def skills():
    return get_skill_summary()


@app.get("/top-jobs")
def top_jobs():
    return get_top_jobs()


@app.get("/skill-trends")
def skill_trends():
    return get_skill_trends()

# ---------------------------------------------------
# RESUME ANALYZER
# ---------------------------------------------------

@app.post("/analyze-resume")
async def analyze_resume(file: UploadFile = File(...)):

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a PDF resume."
        )

    try:
        result = analyze_resume_file(file.file)
        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Resume analysis failed: {exc}"
        ) from exc

# ---------------------------------------------------
# SKILL EXTRACTION
# ---------------------------------------------------

@app.post("/extract-skills")
def extract_skill_payload(payload: TextRequest):

    try:
        skills = extract_skills(payload.text)

        return {
            "skills": skills
        }

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Skill extraction failed: {exc}"
        ) from exc

# ---------------------------------------------------
# SKILL MATCHING
# ---------------------------------------------------

@app.post("/match-skills")
def match_skill_payload(payload: MatchSkillsRequest):

    try:
        result = recommend_skills(
            payload.resume_skills,
            payload.market_skills,
            payload.top_k
        )

        return result

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Skill matching failed: {exc}"
        ) from exc


@app.post("/career-intelligence")
def career_intelligence_payload(payload: CareerIntelligenceRequest):

    try:
        return build_career_intelligence(
            skills=payload.skills,
            text=payload.text
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Career intelligence failed: {exc}"
        ) from exc

# ---------------------------------------------------
# FORECAST SKILLS
# ---------------------------------------------------

@app.get("/forecast-skills")
def forecast(
    months: int = 6,
    skills: str | None = None
):

    try:
        selected_skills = (
            [skill.strip() for skill in skills.split(",")]
            if skills else None
        )

        return forecast_skills(
            months=months,
            skills=selected_skills
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Forecast failed: {exc}"
        ) from exc

# ---------------------------------------------------
# PREDICT DEMAND
# ---------------------------------------------------

@app.post("/predict-demand")
def predict_demand_payload(payload: PredictDemandRequest):

    try:
        return predict_demand(
            payload.skill,
            payload.months
        )

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}"
        ) from exc

# ---------------------------------------------------
# WORKFORCE ANALYTICS
# ---------------------------------------------------

@app.get("/workforce-analytics")
def workforce_analytics():

    try:
        return get_workforce_analytics()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Analytics failed: {exc}"
        ) from exc


@app.get("/dataset-insights")
def dataset_insights():

    try:
        return get_dataset_insights()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Dataset insight generation failed: {exc}"
        ) from exc


@app.get("/model-explainability")
def model_explainability():

    try:
        return get_model_explainability()

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Model explainability failed: {exc}"
        ) from exc

# ---------------------------------------------------
# LIVE JOBS
# ---------------------------------------------------

@app.get("/live-jobs")
def live_jobs():

    try:
        return get_live_jobs()

    except Exception as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Live jobs API failed: {exc}"
        ) from exc
