import re
from difflib import get_close_matches

from services.model_loader import load_ner_pipeline, load_sentence_model


TECH_SKILLS = [
    "python",
    "java",
    "javascript",
    "typescript",
    "sql",
    "aws",
    "azure",
    "gcp",
    "docker",
    "kubernetes",
    "mlops",
    "devops",
    "ci/cd",
    "tensorflow",
    "pytorch",
    "machine learning",
    "deep learning",
    "artificial intelligence",
    "generative ai",
    "react",
    "next.js",
    "node.js",
    "node",
    "fastapi",
    "django",
    "flask",
    "pandas",
    "numpy",
    "scikit-learn",
    "tableau",
    "power bi",
    "spark",
    "hadoop",
    "nlp",
    "langchain",
    "llm",
    "rag",
    "airflow",
    "git",
    "linux",
    "rest api",
    "microservices",
    "mongodb",
    "postgresql",
    "mysql",
    "redis",
    "snowflake",
    "databricks",
    "dbt",
    "kafka",
    "elasticsearch",
    "terraform",
    "ansible",
    "jenkins",
    "github actions",
    "gitlab ci",
    "prometheus",
    "grafana",
    "model monitoring",
    "feature store",
    "vector database",
    "pinecone",
    "weaviate",
    "llamaindex",
    "hugging face",
    "transformers",
    "computer vision",
    "data engineering",
    "data science",
    "data analytics",
    "business intelligence",
    "rest api",
    "graphql",
    "spring boot",
    "go",
    "rust",
    "c++",
    "c#",
    "excel",
    "looker",
    "salesforce",
    "sap",
    "oracle",
    "firebase",
    "system design",
    "tailwind css",
    "html",
    "css",
    "redux",
    "express.js",
    "authentication",
    "api design",
    "cloud deployment",
    "testing",
    "jest",
    "vite",
    "figma",
    "ui/ux",
    "statistics",
]


SKILL_ALIASES = {
    "js": "javascript",
    "nodejs": "node.js",
    "node js": "node.js",
    "reactjs": "react",
    "react js": "react",
    "postgres": "postgresql",
    "postgre sql": "postgresql",
    "k8s": "kubernetes",
    "kubernetes": "kubernetes",
    "amazon web services": "aws",
    "google cloud": "gcp",
    "ms azure": "azure",
    "powerbi": "power bi",
    "scikit learn": "scikit-learn",
    "sklearn": "scikit-learn",
    "gen ai": "generative ai",
    "genai": "generative ai",
    "large language models": "llm",
    "retrieval augmented generation": "rag",
    "tailwind": "tailwind css",
    "express": "express.js",
    "ci cd": "ci/cd",
}


INVALID_ENTITY_TERMS = {
    "",
    "a",
    "an",
    "and",
    "or",
    "the",
    "of",
    "for",
    "to",
    "in",
    "on",
    "with",
    "by",
    "from",
    "at",
    "as",
    "da",
    "daund",
    "mit",
    "mit a",
    "& al",
    "al",
    "etc",
    "project",
    "projects",
    "team",
    "work",
    "college",
    "university",
}


SKILL_CANONICAL = sorted({*TECH_SKILLS, *SKILL_ALIASES.values()})


SOFT_SKILLS = [
    "communication",
    "leadership",
    "collaboration",
    "problem solving",
    "stakeholder management",
    "mentoring",
    "analytical thinking",
    "agile",
    "scrum",
]


CERTIFICATIONS = [
    "aws certified",
    "azure fundamentals",
    "google cloud certified",
    "cka",
    "ckad",
    "pmp",
    "scrum master",
    "tensorflow developer",
    "databricks",
    "aws solutions architect",
    "aws cloud practitioner",
    "azure data engineer",
    "azure ai engineer",
    "google professional data engineer",
    "google professional cloud architect",
    "certified kubernetes administrator",
    "certified kubernetes application developer",
    "snowpro",
    "power bi data analyst",
]


SKILL_CATEGORIES = {
    "programming_languages": ["python", "java", "javascript", "typescript", "go", "rust", "c++", "c#"],
    "ai_ml": [
        "machine learning",
        "deep learning",
        "artificial intelligence",
        "generative ai",
        "tensorflow",
        "pytorch",
        "scikit-learn",
        "nlp",
        "llm",
        "rag",
        "hugging face",
        "transformers",
        "computer vision",
    ],
    "cloud_platforms": ["aws", "azure", "gcp", "snowflake", "databricks"],
    "devops_mlops": [
        "docker",
        "kubernetes",
        "mlops",
        "devops",
        "ci/cd",
        "terraform",
        "jenkins",
        "github actions",
        "prometheus",
        "grafana",
        "model monitoring",
    ],
    "data_engineering": ["sql", "spark", "hadoop", "airflow", "dbt", "kafka", "feature store"],
    "frontend_backend": ["react", "next.js", "node.js", "fastapi", "django", "flask", "spring boot", "graphql"],
    "analytics_tools": ["tableau", "power bi", "excel", "looker", "business intelligence"],
    "databases": ["postgresql", "mysql", "mongodb", "redis", "elasticsearch", "vector database", "pinecone", "weaviate"],
}


def normalize_skill(skill):
    cleaned = str(skill or "").lower()
    cleaned = cleaned.replace("–", "-").replace("—", "-")
    cleaned = re.sub(r"[\u2022•|]+", " ", cleaned)
    cleaned = re.sub(r"[^a-z0-9+#./\-\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" -_.,:/")
    return SKILL_ALIASES.get(cleaned, cleaned)


SKILL_LOOKUP = {normalize_skill(skill): skill for skill in SKILL_CANONICAL}


def _is_valid_skill_phrase(skill):
    normalized = normalize_skill(skill)
    if not normalized or normalized in INVALID_ENTITY_TERMS:
        return False
    if len(normalized) < 2:
        return False
    if len(normalized) <= 3 and normalized not in {"go", "c#", "c++", "sql", "aws", "gcp", "rag", "llm", "nlp", "css"}:
        return False
    if re.fullmatch(r"[\W_]+", normalized):
        return False
    if re.search(r"\b(?:an|da|mit|al)\b", normalized) and normalized not in SKILL_LOOKUP:
        return False
    if normalized in SKILL_LOOKUP:
        return True
    return False


def canonicalize_skill(skill, semantic_check=False):
    normalized = normalize_skill(skill)
    if normalized in SKILL_LOOKUP:
        return SKILL_LOOKUP[normalized]

    close = get_close_matches(normalized, SKILL_LOOKUP.keys(), n=1, cutoff=0.88)
    if close:
        return SKILL_LOOKUP[close[0]]

    if semantic_check and len(normalized) >= 4:
        try:
            model = load_sentence_model()
            from sentence_transformers import util

            candidate_embedding = model.encode([normalized], convert_to_tensor=True)
            skill_embeddings = model.encode(list(SKILL_LOOKUP.keys()), convert_to_tensor=True)
            scores = util.cos_sim(candidate_embedding, skill_embeddings)[0].detach().cpu().tolist()
            best_index = max(range(len(scores)), key=lambda index: scores[index])
            if float(scores[best_index]) >= 0.78:
                return SKILL_LOOKUP[list(SKILL_LOOKUP.keys())[best_index]]
        except Exception:
            pass

    return None


def categorize_skills(skills):
    normalized = {normalize_skill(skill) for skill in skills}
    categorized = {category: [] for category in SKILL_CATEGORIES}
    categorized["tools"] = []

    for skill in sorted(normalized):
        placed = False
        for category, category_skills in SKILL_CATEGORIES.items():
            if skill in category_skills:
                categorized[category].append(skill)
                placed = True
        if not placed:
            categorized["tools"].append(skill)

    return {category: values for category, values in categorized.items() if values}


def extract_skills_from_text(text):
    if not text or not text.strip():
        return []

    lowered = text.lower()
    detected = set()
    for skill in SKILL_LOOKUP:
        pattern = rf"(?<![a-z0-9+#.]){re.escape(skill)}(?![a-z0-9+#.])"
        if re.search(pattern, lowered):
            detected.add(SKILL_LOOKUP[skill])

    for alias, canonical in SKILL_ALIASES.items():
        pattern = rf"(?<![a-z0-9+#.]){re.escape(alias)}(?![a-z0-9+#.])"
        if re.search(pattern, lowered):
            detected.add(canonical)

    try:
        ner_pipeline = load_ner_pipeline()
        entities = ner_pipeline(text[:5000])
        for entity in entities:
            word = normalize_skill(entity.get("word", ""))
            canonical = canonicalize_skill(word, semantic_check=True)
            if canonical and _is_valid_skill_phrase(canonical):
                detected.add(canonical)
    except Exception:
        # The curated dictionary keeps the API useful if the model is not downloaded yet.
        pass

    clean = {canonicalize_skill(skill) or normalize_skill(skill) for skill in detected}
    return sorted(skill for skill in clean if _is_valid_skill_phrase(skill))


def extract_resume_entities(text):
    lowered = text.lower()
    technical = extract_skills_from_text(text)
    soft = sorted({skill for skill in SOFT_SKILLS if skill in lowered})
    certifications = sorted({cert for cert in CERTIFICATIONS if cert in lowered})
    experience_matches = re.findall(r"(\d+(?:\.\d+)?)\+?\s*(?:years|yrs)\b", lowered)
    years_experience = max([float(value) for value in experience_matches], default=0.0)
    return {
        "technical_skills": technical,
        "skill_categories": categorize_skills(technical),
        "soft_skills": soft,
        "certifications": certifications,
        "years_experience": years_experience,
    }


def match_skills(resume_skills, market_skills=None, top_k=10):
    market_skills = market_skills or TECH_SKILLS
    clean_resume = [canonicalize_skill(skill) or normalize_skill(skill) for skill in resume_skills if _is_valid_skill_phrase(skill)]
    clean_market = [canonicalize_skill(skill) or normalize_skill(skill) for skill in market_skills if _is_valid_skill_phrase(skill)]

    if not clean_resume or not clean_market:
        return []

    try:
        model = load_sentence_model()
        resume_embeddings = model.encode(clean_resume, convert_to_tensor=True)
        market_embeddings = model.encode(clean_market, convert_to_tensor=True)

        from sentence_transformers import util

        score_matrix = util.cos_sim(resume_embeddings, market_embeddings)
        best_scores = score_matrix.max(dim=0).values.detach().cpu().tolist()
    except Exception:
        resume_set = set(clean_resume)
        best_scores = [1.0 if skill in resume_set else 0.0 for skill in clean_market]

    matches = [
        {
            "skill": skill,
            "score": round(float(score), 4),
            "recommended": float(score) < 0.72,
        }
        for skill, score in zip(clean_market, best_scores)
    ]

    matches.sort(key=lambda item: item["score"])
    return matches[:top_k]
