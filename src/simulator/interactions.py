import numpy as np
from typing import Dict

def sigmoid(x: float) -> float:
    """Numerically stable sigmoid"""
    return 1.0 / (1.0 + np.exp(-x))


def simulate_interaction(
        user: Dict,
        item: Dict,
        rng: np.random.Generator
) -> Dict:
    """
    Simulate a single user-item interaction

    Returns an interaction record containing:
        - Success
        - Quiz score
        - Time spend
        - Updated mastery
    """

    # Get user parameters
    mastery              = user["mastery"]
    learning_rate        = user["learning_rate"]
    difficulty_tolerance = user["difficulty_tolerance"]
    dropout_sensitivity  = user["dropout_sensitivity"]

    # Get item parameters
    item_skills          = item["skills"]
    difficulty           = item["difficulty"]
    estimated_time       = item["estimated_time"]
    prerequisites        = item["prerequisites"]



    # ----------------
    # Skill match
    # ---------------
    skill_count = np.sum(item_skills)
    if skill_count == 0:
        skill_match = 0.0
    else:
        skill_match = np.dot(mastery, item_skills) / skill_count


    # ------------------
    # Difficulty gap
    # ------------------
    difficulty_gap = difficulty / difficulty_tolerance


    # -------------------
    # Success probability
    # ------------------
    alpha = 10.0 # Weight on mastery
    beta = 1.0  # penalty for difficulty

    logit = alpha * skill_match - beta * difficulty_gap
    success_prob = sigmoid(logit)

    success = rng.random() < success_prob


    # ------------------------
    # Quiz score
    # ------------------------
    if success:
        quiz_score = rng.normal(loc=80 + 20 * skill_match, scale=5)
    else:
        quiz_score = rng.normal(loc=40 + 20 * skill_match, scale=10)

    quiz_score = float(np.clip(quiz_score, 0, 100))

    # --------------------
    # Time spent
    # -------------------
    time_noise = rng.normal(0, 2)

    if success:
        time_spent = estimated_time * (1.0 + 0.1 * difficulty_gap) + time_noise
    else:
        time_spent = estimated_time * (1.5 + 0.2 * difficulty_gap) + time_noise

    time_spent = max(1.0, time_spent)


    # -------------------
    # Mastery update
    # -------------------
    mastery_before = user["mastery"].copy()
    
    if success:
        mastery_update = learning_rate * item_skills * (1.0 - mastery)
        user["mastery"] = np.clip(mastery + mastery_update, 0.0, 1.0)
    
    mastery_after = user["mastery"].copy()
    
    # Calculate skill gain (average change in mastery for relevant skills)
    relevant_skills = item_skills.astype(bool)
    if relevant_skills.any():
        skill_gain = np.mean(mastery_after[relevant_skills] - mastery_before[relevant_skills])
    else:
        skill_gain = 0.0
    
    # Normalize skill_gain to [0, 1]
    skill_gain = float(np.clip(skill_gain, 0.0, 1.0))

    interaction = {
        "user_id": user["user_id"],
        "item_id": item["item_id"],
        "success": int(success),
        "quiz_score": quiz_score,
        "time_spent": time_spent,
        "skill_match": skill_match,
        "difficulty_gap": difficulty_gap,
        "difficulty": difficulty,
        "dropout_sensitivity": dropout_sensitivity,
        "num_prerequisites": len(prerequisites),
        "estimated_time": estimated_time,
        "skill_gain": skill_gain
    }

    return interaction

