import pandas as pd
import numpy as np
from typing import Dict

def extract_interaction_features(
    logs: pd.DataFrame,
    users: Dict[int, Dict],
    items: Dict[int, Dict],
) -> pd.DataFrame:
    """
    Generate features for each user-item interaction in the logs.

    Returns:
        DataFrame with one row per interaction, augmented with computed features
    """
    # Start with a copy of the logs
    features_df = logs.copy()

    # Add computed features
    features_list = []
    for idx, row in logs.iterrows():
        user_id = row["user_id"]
        item_id = row["item_id"]
        
        user = users[user_id]
        item = items[item_id]
        
        user_mastery = user["mastery"]
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
        # User historical features (up to but not including this interaction)
        # -----------------------------
        # Note: For simplicity, using all user logs; in practice, you might want to filter by timestamp
        user_logs = logs[(logs["user_id"] == user_id) & (logs.index < idx)]  # Exclude current interaction

        success_rate = user_logs["success"].mean() if not user_logs.empty else 0.0
        avg_quiz = user_logs["quiz_score"].mean() if not user_logs.empty else 0.0
        avg_time = user_logs["time_spent"].mean() if not user_logs.empty else 0.0
        num_attempts = len(user_logs)

        # -----------------------------
        # Item historical features
        # -----------------------------
        item_logs = logs[(logs["item_id"] == item_id) & (logs.index < idx)]  # Exclude current interaction
        item_avg_success = item_logs["success"].mean() if not item_logs.empty else 0.0
        item_avg_quiz = item_logs["quiz_score"].mean() if not item_logs.empty else 0.0
        item_avg_time = item_logs["time_spent"].mean() if not item_logs.empty else 0.0

        # -----------------------------
        # Combine features
        # -----------------------------
        features = {
            "skill_gap": skill_gap,
            "fraction_skills_mastered": fraction_skills_mastered,
            "difficulty_gap": difficulty_gap,  # Overwrites the one from logs (different calculation)
            "user_success_rate": success_rate,
            "user_avg_quiz": avg_quiz,
            "user_avg_time": avg_time,
            "user_num_attempts": num_attempts,
            "item_avg_success": item_avg_success,
            "item_avg_quiz": item_avg_quiz,
            "item_avg_time": item_avg_time,
            "item_num_skills": int(np.sum(item_skills))
            # Note: Skipping item_num_prerequisites and item_difficulty as they duplicate logs columns
        }
        features_list.append(features)

    # Add features to the DataFrame
    features_df = logs.copy()
    features_df = pd.concat([features_df.drop(columns=list(set(logs.columns) & set(pd.DataFrame(features_list).columns))), pd.DataFrame(features_list)], axis=1)
    
    return features_df
