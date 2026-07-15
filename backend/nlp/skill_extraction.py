import pandas as pd

# skill dictionary
skills_list = [
    "python","java","sql","aws","docker","kubernetes",
    "tensorflow","pytorch","machine learning","deep learning",
    "react","node.js","pandas","numpy","scikit-learn",
    "tableau","power bi","spark","hadoop","nlp"
]

def extract_skills(text):
    text = text.lower()
    found_skills = []

    for skill in skills_list:
        if skill in text:
            found_skills.append(skill)

    return found_skills


def process_dataset():

    df = pd.read_csv("data/raw/jobs_dataset.csv")

    df["extracted_skills"] = df["description"].apply(extract_skills)

    print(df[["job_title","extracted_skills"]].head())

    df.to_csv("data/processed/jobs_with_skills.csv", index=False)


if __name__ == "__main__":
    process_dataset()