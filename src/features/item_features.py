import pandas as pd
import numpy as np
from typing import Dict

def extract_item_features(logs: pd.DataFrame, items: Dict[int, Dict]) -> pd.DataFrame:
    """
    Extract per-item features from interaction logs and item definitions.

    Args:
        logs: DataFrame returned by the simulator
        items: Dictionary of item_id -> item dicts

    Returns:
        DataFrame with one row per item
    """
    item_features = []

    for item_id, item in items.items():
        # Static features
        num_skills = int(np.sum(item["skills"]))
        num_prerequisites = len(item["prerequisites"])
        difficulty = float(item["difficulty"])

        # Logs for this item
        item_logs = logs[logs["item_id"] == item_id]

        if not item_logs.empty:
            avg_success = item_logs["success"].mean()
            avg_quiz = item_logs["quiz_score"].mean()
            avg_time = item_logs["time_spent"].mean()
            num_attempts = len(item_logs)
        else:
            # If no interactions yet, default values
            avg_success = 0.0
            avg_quiz = 0.0
            avg_time = 0.0
            num_attempts = 0

        features = {
            "item_id": item_id,
            "difficulty": difficulty,
            "num_skills": num_skills,
            "num_prerequisites": num_prerequisites,
            "avg_success": avg_success,
            "avg_quiz": avg_quiz,
            "avg_time": avg_time,
            "num_attempts": num_attempts
        }

        item_features.append(features)

    return pd.DataFrame(item_features)
