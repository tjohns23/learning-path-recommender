import numpy as np
from typing import Dict, List

def generate_items(
    num_items: int,
    num_skills: int,
    random_seed: int = 42
) -> Dict[int, Dict]:
    
    """
    Generate a catalog of learning items.

    Each item has:
        - A binary skill coverage vector
        - A discrete difficulty level
        - Prerequisite skills
        - An estimated completion time

    Returns:
        Dict[item_id, item_dict]
    """

    rng = np.random.default_rng(random_seed)
    items = {}

    for item_id in range(num_items):
        # -----------------
        # Difficulty
        # -----------------
        difficulty = rng.integers(1, 6) # 1 to 5 inclusive


        # -----------------
        # Skill coverage
        # -----------------
        
        # Pick random skills
        num_item_skills = rng.integers(1, min(4, num_skills + 1)) 
        skill_indices = rng.choice(num_skills, size=num_item_skills, replace=False)
    
        skill_vector = np.zeros(num_skills, dtype=np.float32)
        skill_vector[skill_indices] = 1.0

        # ----------------
        # Prerequisites
        # ----------------
        prerequisites: List[int] = []
        if difficulty >= 3:
            for s in skill_indices:
                # Only allow prerequisites from earlier skills
                if s > 0 and rng.random() < 0.5:
                    prereq_skill = rng.integers(0, s)
                    prerequisites.append(prereq_skill)

        prerequisites = list(set(prerequisites)) # remove duplicates

        # ----------------
        # Estimated time
        # ----------------
        base_time = 10 # minutes
        estimated_time = base_time * difficulty + rng.normal(0, 2)

        item = {
            "item_id": item_id,
            "skills": skill_vector,
            "difficulty": difficulty,
            "prerequisites": prerequisites,
            "estimated_time": max(5.0, estimated_time) 
        }

        items[item_id] = item

    return items


items = generate_items(5, 6)
for item in items.values():
    print(item)