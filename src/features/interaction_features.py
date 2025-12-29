import pandas as pd
import numpy as np
from typing import Dict

def extract_interaction_features(
    logs: pd.DataFrame,
    users: Dict[int, Dict],
    items: Dict[int, Dict],
) -> pd.DataFrame:
    """
    Generate features for each user-item pair.

    Returns:
        DataFrame with one row per user-item interaction
    """
    records = []

    for user_id, user in users.items():
        user_mastery = user["mastery"]

        for item_id, item in items.items():
            item_skills = item["skills"]
            item_difficulty = item["difficulty"]

            # -----------------------------
            # Skill-level features
            # -----------------------------
            relevant_skills = item_skills.astype(bool)
            if relevant_skills.any():
                skill_gap = np.mean(1.0 - user_mastery[relevant_skills])
                fraction_skills_mastered = np.mean(user_mastery[relevant_skills] >= 0.8)
            else:
                skill_gap = 0.0
                fraction_skills_mastered = 0.0

            # Difficulty gap: item difficulty vs user's average mastery
            avg_user_mastery = np.mean(user_mastery)
            difficulty_gap = item_difficulty - avg_user_mastery * 5  # scale mastery to 0-5

            # -----------------------------
            # User historical features
            # -----------------------------
            user_logs = logs[logs["user_id"] == user_id]

            success_rate = user_logs["success"].mean() if not user_logs.empty else 0.0
            avg_quiz = user_logs["quiz_score"].mean() if not user_logs.empty else 0.0
            avg_time = user_logs["time_spent"].mean() if not user_logs.empty else 0.0
            num_attempts = len(user_logs)

            # -----------------------------
            # Item historical features
            # -----------------------------
            item_logs = logs[logs["item_id"] == item_id]
            item_avg_success = item_logs["success"].mean() if not item_logs.empty else 0.0
            item_avg_quiz = item_logs["quiz_score"].mean() if not item_logs.empty else 0.0
            item_avg_time = item_logs["time_spent"].mean() if not item_logs.empty else 0.0

            # -----------------------------
            # Combine features
            # -----------------------------
            record = {
                "user_id": user_id,
                "item_id": item_id,
                "skill_gap": skill_gap,
                "fraction_skills_mastered": fraction_skills_mastered,
                "difficulty_gap": difficulty_gap,
                "user_success_rate": success_rate,
                "user_avg_quiz": avg_quiz,
                "user_avg_time": avg_time,
                "user_num_attempts": num_attempts,
                "item_avg_success": item_avg_success,
                "item_avg_quiz": item_avg_quiz,
                "item_avg_time": item_avg_time,
                "item_num_skills": int(np.sum(item_skills)),
                "item_num_prerequisites": len(item["prerequisites"]),
                "item_difficulty": item_difficulty
            }

            records.append(record)

    return pd.DataFrame(records)
