import numpy as np
import pandas as pd


def extract_user_features(logs: pd.DataFrame, num_skills: int) -> pd.DataFrame:
    """
    Extract per-user features from interaction logs.

    Returns:
        DataFrame with one row per user.
    """
    user_features = []

    for user_id, user_logs in logs.groupby("user_id"):
        # -----------------------------
        # Performance statistics
        # -----------------------------
        success_rate = user_logs["success"].mean()
        avg_quiz = user_logs["quiz_score"].mean()
        std_quiz = user_logs["quiz_score"].std()
        avg_time = user_logs["time_spent"].mean()
        std_time = user_logs["time_spent"].std()

        # -----------------------------
        # Skill-level statistics
        # -----------------------------
        # Create skill mastery approximations from skill_match and success
        skill_mastery = np.zeros(num_skills)

        for _, row in user_logs.iterrows():
            item_skills = row.get("item_skills", None)
            skill_mastery += row["skill_match"] * np.ones(num_skills)

        # Normalize to number of interactions
        skill_mastery /= len(user_logs)

        # Fraction of skills above threshold (e.g., mastery ≥ 0.8)
        fraction_mastered = np.mean(skill_mastery >= 0.8)

        # -----------------------------
        # Aggregate features
        # -----------------------------
        features = {
            "user_id": user_id,
            "success_rate": success_rate,
            "avg_quiz": avg_quiz,
            "std_quiz": std_quiz,
            "avg_time": avg_time,
            "std_time": std_time,
            "fraction_mastered": fraction_mastered,
            "num_attempts": len(user_logs),
            "avg_skill_mastery": skill_mastery.mean()
        }

        user_features.append(features)

    return pd.DataFrame(user_features)


