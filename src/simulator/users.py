import numpy as np
from typing import Dict

def generate_users(
    num_users: int,
    num_skills: int,
    random_seed: int = 42
) -> Dict[int, Dict]:
    """
    Generate a population of learners

    Each user has:
        - A master vector of skills
        - A learning rate controlling mastery updates
        - A tolerance to difficulty
        - A dropout sensitivity parameter

    Returns:
        Dict[user_id, user_dict]
    """

    rng = np.random.default_rng(random_seed)
    users = {}

    # Parameters for initial mastery distribution
    alpha = 2.0
    beta = 5.0

    for user_id in range(num_users):
        # --------------
        # Initial mastery
        # --------------
        mastery = rng.beta(alpha, beta, size=num_skills).astype(np.float32)

        # ----------------------
        # Learning behavior traits
        # ----------------------
        learning_rate = rng.uniform(0.05, 0.3)
        difficulty_tolerance = rng.uniform(0.5, 1.5)
        dropout_sensitivity = rng.uniform(0.0, 1.0)

        user = {
            "user_id": user_id,
            "mastery": mastery,
            "learning_rate": learning_rate,
            "difficulty_tolerance": difficulty_tolerance,
            "dropout_sensitivity": dropout_sensitivity
        }

        users[user_id] = user

    return users

# -------- Sanity check --------- #
# users = generate_users(3, 5)
# for u in users.values():
#     print(u)