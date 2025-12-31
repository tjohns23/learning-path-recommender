import numpy as np
import pandas as pd
from typing import Dict, List

from .interactions import simulate_interaction

def prerequisites_satisfied(user: Dict, item: Dict, threshold: float = 0.6) -> bool:
        """
        Check whether a user's mastery satisfies an item's prerequisite skills
        """

        mastery = user["mastery"]
        for skill_id in item["prerequisites"]:
            if mastery[skill_id] < threshold:
                  return False
        return True


def select_item(user: Dict, items: Dict[int, Dict], rng: np.random.Generator) -> Dict:
    """
    Select the next item for a user based on current mastery and difficulty
    """
    candidates: List[Dict] = []

    for item in items.values():
        if prerequisites_satisfied(user, item):
             candidates.append(item)

    # If nothing satisfies prepreqs
    if not candidates:
         candidates = list(items.values())

    scores = []
    mastery = user["mastery"]

    for item in candidates:
        item_skills = item["skills"]
        skill_count = np.sum(item_skills)

        if skill_count == 0:
            avg_mastery = 0.0
        else:
            avg_mastery = np.dot(mastery, item_skills) / skill_count

        # Perfer slightly challenging items
        difficulty_gap = abs(item["difficulty"] - (avg_mastery * 5 + 1))
        score = np.exp(-difficulty_gap)
        scores.append(score)

    scores = np.array(scores)
    probs = scores / scores.sum()

    return rng.choice(candidates, p=probs)

def run_simulation_core(
    users: Dict[int, Dict],
    items: Dict[int, Dict],
    max_steps: int = 50,
    seed: int = 42
) -> pd.DataFrame:
    """
    Simulate learning interactions for all users

    Returns: 
        DaataFrame of interaction logs
    """

    rng = np.random.default_rng(seed)
    logs = []

    for user in users.values():
        consecutive_failures = 0

        for step in range(max_steps):
            item = select_item(user, items, rng)

            interaction = simulate_interaction(user, item, rng)
            interaction["step"] = step
            logs.append(interaction)

            # Update failure count
            if interaction["success"] == 0:
                consecutive_failures += 1
            else:
                consecutive_failures = 0
            
            # Dropout check
            dropout_prob = min(
                1.0,
                0.1 * consecutive_failures * user["dropout_sensitivity"]
            )

            if rng.random() < dropout_prob:
                break
    
    return pd.DataFrame(logs)


from .users import generate_users
from .items import generate_items

def run_simulation(
    num_users: int,
    num_items: int,
    steps_per_user: int = 50,
    seed: int = 42
):
    users = generate_users(num_users)
    items = generate_items(num_items)

    logs = run_simulation_core(
        users=users,
        items=items,
        max_steps=steps_per_user,
        seed=seed
    )

    return users, items, logs

# --------- Sanity Check ---------- #
# from items import generate_items
# from users import generate_users

# items = generate_items(20, 5)
# users = generate_users(3, 5)

# df = run_simulation(users, items, max_steps=30)
# print(df.head())
# print(df.tail())
# print(df.shape)

