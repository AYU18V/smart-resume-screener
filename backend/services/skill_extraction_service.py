from services.nlp_service import extract_skills_from_text


def extract_skills(text: str) -> list[str]:
    """Extract technical skills from free text using cached NLP models and rules."""
    return extract_skills_from_text(text)
