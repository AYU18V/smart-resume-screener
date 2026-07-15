from enterprise_workforce_preprocessing import FINAL_SKILL_OUTPUT_PATH, build_enterprise_workforce_dataset


def build_final_dataset():
    enterprise = build_enterprise_workforce_dataset()
    return enterprise[["date", "skill", "jobs_count"]]


if __name__ == "__main__":
    dataset = build_final_dataset()
    print(f"Generated {FINAL_SKILL_OUTPUT_PATH}")
    print(f"Rows: {len(dataset)}")
    print(dataset.head().to_string(index=False))
